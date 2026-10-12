"""Recipe-v1 evidence, continuous MALA and strict checkpoint selection."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.mala import mala_sample
from launchers import eval_peptide_ts as launcher
from common.sample_stages import ENERGY_CUTOFFS_KT, sample_stages, resampling_indices


@pytest.mark.parametrize("name,steps", [
    ("ace_a_nme", 4), ("aaa", 4), ("ace_aaa_nme", 8), ("aaaaaa", 16),
    ("chignolin", 32),
])
def test_recipe_steps_accept_current_config_names(name, steps, tmp_path):
    args = launcher.parse_args(['--cfg_path', f'configs.{name}', '--ckpt_path', 'unused.pt',
                               '--out_dir', str(tmp_path), '--eval_recipe', 'v1'])
    assert args.steps == [steps]


def test_recipe_defaults_and_strict_saved_ema(tmp_path):
    base = ['--cfg_path', 'configs.ace_aaa_nme', '--ckpt_path', 'unused.pt',
            '--out_dir', str(tmp_path), '--eval_recipe', 'v1']
    args = launcher.parse_args(base)
    assert (args.steps, args.n_generate_samples, args.n_eval_samples) == ([8], 1000000, 10000)
    assert (args.posthoc_mala_steps, args.mala_adapt_steps, args.mala_checkpoint_interval) == (1024, 512, 128)
    with pytest.raises(SystemExit):
        launcher.parse_args(base + ['--no_ema'])
    from configs.ace_a_nme import get_config
    from common.flow_map import TorchFlowMap
    from common.state_utils import load_evaluation_model
    cfg = get_config(0)
    cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
    net = TorchFlowMap(cfg.network)
    path = tmp_path / 'model.pt'
    torch.save({'params': net.state_dict(), 'step': 100000}, path)
    with pytest.raises(ValueError, match='saved EMA'):
        load_evaluation_model(cfg, path, 'cpu', require_saved_ema=True, ema_factor=.999)
    ema = {k: torch.full_like(v, 7) for k, v in net.state_dict().items()}
    torch.save({'params': net.state_dict(), 'ema_params': {.999: ema}, 'step': 100000}, path)
    loaded, step, factor = load_evaluation_model(cfg, path, 'cpu', require_saved_ema=True, ema_factor=.999)
    assert (step, factor) == (100000, .999)
    assert all(torch.equal(v, ema[k]) for k, v in loaded.state_dict().items())


def test_recipe_snapshots_indices_filtering_and_continuous_rng():
    torch.manual_seed(9)
    x = torch.randn(100, 3, 3)
    e = x.square().sum((1, 2)) / 2
    e[-1] = torch.inf
    logw = torch.linspace(-2, 2, 100)
    target = SimpleNamespace(energy_and_grad=lambda x: (x.square().sum((1, 2)) / 2, x))
    artifacts = {}
    stages, stats = sample_stages(x, e, logw, clip_fraction=.2, energy=target,
        mala_steps=16, mala_step_size=.03, n_eval_samples=20, energy_cutoff=None,
        mala_adapt_steps=8, recipe_v1=True, checkpoint_interval=2, artifacts=artifacts)
    assert list(stages) == ['proposal', 'resampled'] + [f'relaxed-{k}' for k in range(2, 17, 2)]
    assert all(len(v) == 20 for v in stages.values())
    assert len(artifacts['proposal_indices'].unique()) == 20
    for stage, index in [('proposal', 'proposal_indices'), ('resampled', 'resampled_indices')]:
        torch.testing.assert_close(stages[stage].samples, x[artifacts[index]])
    kept = artifacts['filtered_indices']
    threshold = torch.quantile(logw[:99], .8)
    torch.testing.assert_close(kept, torch.where(logw[:99] <= threshold)[0])
    p = torch.softmax(logw[kept], 0)
    assert stats['resampled']['ess'] == pytest.approx(float(1 / p.square().sum()))
    assert stats['resampled']['n-after-filter'] == len(kept)
    assert stats['proposal']['n-nonfinite'] == 1
    torch.set_rng_state(artifacts['rng_before_mala'])
    initial = stages['resampled'].samples
    initial = initial - initial.mean(1, keepdim=True)
    final, info = mala_sample(initial, lambda x: (-x.square().sum((1, 2)) / 2, -x),
        16, .03, project_com=True, adapt=True, adapt_steps=8, checkpoint_steps=list(range(2, 17, 2)))
    for k in range(2, 17, 2):
        torch.testing.assert_close(stages[f'relaxed-{k}'].samples, info['checkpoints'][k], rtol=0, atol=0)
        torch.testing.assert_close(stages[f'relaxed-{k}'].E_target, -info['checkpoint_logp'][k])
    assert torch.equal(torch.get_rng_state(), artifacts['rng_after_mala'])
    assert torch.equal(info['step_size'][8], info['step_size'][-1])
    assert stats['relaxed-8']['mala-includes-adaptation']
    assert not stats['relaxed-10']['mala-includes-adaptation']
    with pytest.raises(ValueError, match='Fewer finite'):
        sample_stages(x[:3], e[:3], logw[:3], clip_fraction=.2, energy=target,
            mala_steps=2, mala_step_size=.03, n_eval_samples=4, energy_cutoff=None, recipe_v1=True)


@pytest.mark.parametrize("name", ["ace_a_nme", "chignolin"])
@pytest.mark.parametrize("mode", ["dlogpdt", "hutchinson"])
def test_recipe_launcher_evidence_and_stage_first_logging(monkeypatch, tmp_path, name, mode):
    from common import datasets, state_utils, ts_systems, tsx
    from common.flow_map import TorchFlowMap
    class Energy:
        def numpy(self, x): return ((x - x.mean(1, keepdims=True)) ** 2).sum((1, 2))
        def energy_and_grad(self, x): return self.numpy(x), 2 * (x - x.mean(1, keepdim=True))
        def close(self): pass
    monkeypatch.setattr(ts_systems, 'ParallelEnergy', lambda *a, **kw: Energy())
    monkeypatch.setattr(ts_systems, 'ensure_data', lambda *a, **kw: {'source': 'analytic'})
    monkeypatch.setattr(ts_systems, 'train_std', lambda *a, **kw: .25)
    setup_target = datasets.setup_target
    monkeypatch.setattr(datasets, 'setup_target', lambda cfg, **kw: setup_target(cfg, train_std=.25))
    def setup(cfg, *args, **kwargs):
        cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
        net = TorchFlowMap(cfg.network)
        return net, 100000, .999
    monkeypatch.setattr(state_utils, 'load_evaluation_model', setup)
    monkeypatch.setattr(ts_systems, 'build_eval_context', lambda system, *a, **kw:
        SimpleNamespace(true_data=tsx.SamplesData(torch.zeros(7, system.num_atoms, 3), torch.arange(7.))))
    class Evaluator:
        def __init__(self, **kw): assert kw['fix_symmetry'] and not kw['drop_unfixable_symmetry']
        def evaluate(self, stages, ctx, prefix, **kw):
            name, data = next(iter(stages.items()))
            assert len(data) == 7 and data.logw is None
            data.samples.zero_()  # metric mutation must not alter saved evidence
            return {f'{prefix}/{name}/energy-w2': 2.}
    monkeypatch.setattr(tsx, 'PeptideEnsembleEvaluator', Evaluator)
    import wandb
    logged = []
    run = SimpleNamespace(url='https://wandb.ai/test', id='evaltest', log=lambda row: logged.append(row), finish=lambda **kw: None)
    monkeypatch.setattr(wandb, 'init', lambda **kw: run)
    ckpt = tmp_path / 'mock.pt'
    ckpt.write_bytes(b'mock')
    monkeypatch.setattr('sys.argv', ['eval', '--cfg_path', f'configs.{name}', '--ckpt_path', str(ckpt),
        '--out_dir', str(tmp_path), '--device', 'cpu', '--no_plots', '--n_generate_samples', '40',
        '--n_eval_samples', '7', '--posthoc_mala_steps', '16', '--mala_checkpoint_interval', '2',
        '--mode', mode, '--steps', '2',
        '--eval_recipe', 'v1', '--wandb_project', 'test', '--wandb_id', 'evaltest'])
    launcher.main()
    expected = ['proposal', 'resampled'] + [f'relaxed-{k}' for k in range(2, 17, 2)]
    assert {k for row in logged for k in row if k.endswith('energy-w2')} == {
        f'{stage}/{split}/energy-w2' for stage in expected for split in ['val', 'test']}
    pool = torch.load(tmp_path / 'proposal-pool.pt', weights_only=False)
    saved = torch.load(tmp_path / 'resampling-mala.pt', weights_only=False)
    assert len(pool['x']) == 40 and set(saved['mala']['checkpoints']) == set(range(2, 17, 2))
    assert saved['mala']['checkpoints'][2].abs().sum() > 0
    assert json.loads((tmp_path / 'wandb-status.json').read_text())['finish_succeeded']
