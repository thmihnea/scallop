"""Regression tests for TS-only setup and likelihood-free pretraining evaluation."""
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transferable_samplers")
pytest.importorskip("mdtraj")

from common import datasets, ts_data, ts_eval, ts_systems
from common.tsx import SamplesData, TargetEnergy


def config(mode="pretrain_flowmap", target="ace_a_nme"):
    module = {"ala3_implicit": "aaa", "ala6_implicit": "aaaaaa"}.get(target, target)
    cfg = importlib.import_module(f"configs.{module}").get_config(0)
    cfg.network.load_path = ""
    cfg.teacher.load_path = ""
    cfg.network.divergence_mode = mode
    # This fixture bypasses setup_config_dict, which resolves the CLI sentinel 0.
    cfg.network.divergence_rescale = 1.0
    cfg.network.predict_divergence = mode not in ("pretrain_v", "pretrain_flowmap")
    cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
    cfg.logging.plot_bs = 8
    cfg.logging.visual_freq = 1
    cfg.logging.do_plots = False
    return cfg


@pytest.mark.parametrize("target", sorted(ts_systems.TS_SYSTEMS))
def test_setup_needs_scale_and_shape_only(monkeypatch, target):
    cfg = config(target=target)
    def forbidden(*args, **kwargs):
        raise AssertionError("Network setup must not load topology, data or create an energy context")
    for name in ("topology", "build_potential", "ensure_data", "train_std"):
        monkeypatch.setattr(ts_systems, name, forbidden)
    cfg, ex = datasets.setup_target(cfg, train_std=0.25)
    assert ex.shape == (ts_systems.resolve(target).num_atoms, 3)
    assert ex.dtype == np.float32
    assert cfg.network.unscale_factor == 0.25
    assert "atom_type_labels" not in cfg.network and "bonds" not in cfg.network
    from common.flow_map import TorchFlowMap
    net = TorchFlowMap(cfg.network)
    assert not any("atom_type" in key or "bond" in key for key in net.state_dict())
    assert not (Path(__file__).parents[1] / "common" / "distr").exists()


def test_setup_validates_shape_and_uses_requested_cache(monkeypatch, tmp_path):
    cfg = config()
    cfg.problem.ts_data_dir = str(tmp_path)
    seen = []
    monkeypatch.setattr(ts_systems, "train_std", lambda system, data_dir: seen.append(data_dir) or 0.2)
    datasets.setup_target(cfg)
    assert seen == [str(tmp_path)]
    cfg.network.n_particles += 1
    with pytest.raises(ValueError, match="shape"):
        datasets.setup_target(cfg, train_std=0.2)


@pytest.mark.parametrize("mode", ["pretrain_v", "pretrain_flowmap", "dlogpdt"])
def test_training_builds_ts_evaluation_for_every_mode(monkeypatch, mode):
    from launchers import learn
    cfg = config(mode)
    cfg.optimization.bs = 4
    train = torch.randn(4, 22, 3)
    val = torch.randn(8, 22, 3)
    marker = object()
    monkeypatch.setattr(ts_data, "load_peptide", lambda *a, **kw: (iter([train]), 0.25))
    monkeypatch.setattr(ts_systems, "load_split", lambda *a, **kw: val)
    def build(cfg_arg, reference, **kwargs):
        assert reference is val
        return marker
    monkeypatch.setattr(ts_eval, "build_eval_context", build)
    cfg, statics, state = learn.setup_state(cfg, torch.device("cpu"), seed=0)
    assert state.ts_eval_ctx is marker
    assert not hasattr(statics, "distr")
    assert cfg.network.unscale_factor == 0.25


def test_missing_heldout_data_never_falls_back(monkeypatch, tmp_path):
    import huggingface_hub
    system = ts_systems.resolve("ace_a_nme")
    paths = ts_systems.data_paths(system, str(tmp_path))
    Path(paths["base"]).mkdir(parents=True)
    np.save(paths["train"], np.zeros((4, 22, 3)))
    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: seen.append(kw))
    with pytest.raises(FileNotFoundError, match="Incomplete TS dataset"):
        ts_systems.ensure_data(system, str(tmp_path))
    assert seen[0]["repo_id"] == ts_systems.HF_REPO_ID
    assert seen[0]["allow_patterns"] == ["Ace-A-Nme_300K/*"]


class ToyModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.1))
        self.config = SimpleNamespace(divergence_mode=mode)
        self.dit = SimpleNamespace(predict_divergence=mode == "dlogpdt")
        self.velocity_calls = 0
        self.flow_calls = 0

    def calc_b(self, t, x, train=False):
        self.velocity_calls += 1
        return self.rate * x

    def unscale_samples(self, x):
        return x * 0.25

    def sample_forward(self, x, n_steps, return_nfe=True):
        self.flow_calls += 1
        if self.config.divergence_mode == "pretrain_v":
            raise AssertionError("Velocity pretraining must never evaluate finite flow-map jumps")
        x = self.unscale_samples(x + self.rate)
        logp = -x.square().sum((1, 2)) if self.dit.predict_divergence else None
        return x, logp, n_steps


def test_velocity_sampling_integrates_diagonal_and_scales_once():
    net = ToyModel("pretrain_v")
    x0 = torch.randn(5, 22, 3)
    x, logp, nfe = ts_eval.sample_model(net, x0, n_steps=4, batch_size=2,
                                      velocity_solver="rk4")
    torch.testing.assert_close(x, x0 * math.exp(0.1) * 0.25)
    assert logp is None and net.flow_calls == 0
    assert nfe == net.velocity_calls == 3 * 4 * 4


@pytest.mark.parametrize("mode", ["pretrain_v", "pretrain_flowmap"])
@pytest.mark.parametrize("com_aware", [False, True])
def test_pretraining_passes_unweighted_proposals_to_upstream(monkeypatch, mode, com_aware):
    cfg = config(mode)
    cfg.network.unscale_factor = 0.25
    cfg.logging.velocity_solver = "rk4"
    net = ToyModel(mode)
    state = SimpleNamespace(net=net, ema_facs=[0.999], ema_params={0.999: {"rate": torch.tensor(0.2)}})
    norm = torch.tensor(0.25 if com_aware else 1.0)
    physical_seen = []
    def energy(x):
        physical_seen.append(x.clone())
        return x.square().sum((1, 2))
    ctx = SimpleNamespace(target_energy=TargetEnergy(energy, norm), normalization_std=norm)
    captured = {}
    class Evaluator:
        def __init__(self, **kwargs): pass
        def evaluate(self, data, ctx, **kwargs):
            captured.update(data)
            assert set(data) == {"proposal"}
            assert isinstance(data["proposal"], SamplesData)
            assert data["proposal"].logw is None
            return {"val/Ace-A-Nme/proposal/energy-w2": 1.0}
    def forbidden(*args, **kwargs):
        raise AssertionError("SNIS has no role when the model has no likelihood")
    monkeypatch.setattr(ts_eval, "SNISSampler", forbidden)
    monkeypatch.setattr(ts_eval, "PeptideEnsembleEvaluator", Evaluator)
    monkeypatch.setattr(ts_eval, "_wandb_active", lambda: False)
    result = ts_eval.ts_evaluate(cfg, state, torch.device("cpu"), 1, ctx, com_augmentation=com_aware)
    torch.testing.assert_close(physical_seen[0], captured["proposal"].samples * norm)
    assert not any("effective-sample-size" in key or "resampled" in key or "logw-clip" in key for key in result)
    assert float(net.rate) == pytest.approx(0.1) and net.training
    assert (net.velocity_calls > 0) == (mode == "pretrain_v")


def test_sampling_failure_restores_parameters_and_training_mode(monkeypatch):
    cfg = config()
    cfg.network.unscale_factor = 0.25
    net = ToyModel("pretrain_flowmap")
    net.eval()
    state = SimpleNamespace(net=net, ema_facs=[0.999], ema_params={0.999: {"rate": torch.tensor(0.2)}})
    def fail(*args, **kwargs): raise RuntimeError("energy failure")
    ctx = SimpleNamespace(target_energy=TargetEnergy(fail, torch.tensor(1.0)))
    with pytest.raises(RuntimeError, match="energy failure"):
        ts_eval.ts_evaluate(cfg, state, torch.device("cpu"), 1, ctx)
    assert float(net.rate) == pytest.approx(0.1) and not net.training


def test_density_evaluation_is_rank_local_inside_ddp(monkeypatch):
    """Rank-zero evaluation must preserve the full budget and avoid collectives."""
    import transferable_samplers.samplers.snis_sampler as snis_module
    originals = {name: getattr(snis_module, name) for name in
                 ("get_world_size", "get_rank", "all_gather_cat", "broadcast_tensor")}
    cfg = config("dlogpdt")
    cfg.logging.importance_weight_clip = 0.0
    net = ToyModel("dlogpdt")
    state = SimpleNamespace(net=net, ema_facs=[], ema_params={})
    ctx = SimpleNamespace(target_energy=TargetEnergy(lambda x: x.square().sum((1, 2)), torch.tensor(1.0)))
    captured = {}
    class Evaluator:
        def __init__(self, **kwargs): pass
        def evaluate(self, data, ctx, **kwargs):
            captured.update(data)
            return {}
    def forbidden(*args, **kwargs):
        raise AssertionError("Rank-zero evaluation entered a DDP collective")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_gather", forbidden)
    monkeypatch.setattr(torch.distributed, "broadcast", forbidden)
    monkeypatch.setattr(ts_eval, "PeptideEnsembleEvaluator", Evaluator)
    monkeypatch.setattr(ts_eval, "_wandb_active", lambda: False)
    ts_eval.ts_evaluate(cfg, state, torch.device("cpu"), 1, ctx)
    assert len(captured["proposal"]) == len(captured["resampled"]) == cfg.logging.plot_bs
    for name, value in originals.items():
        assert getattr(snis_module, name) is value


@pytest.mark.parametrize('main,visual,expected', [(True, 1, ['train', 'val']),
                                                 (True, 0, ['train']), (False, 1, ['train'])])
def test_training_materializes_only_required_splits(monkeypatch, tmp_path, main, visual, expected):
    from common import distributed
    from launchers import learn
    cfg = config('pretrain_v')
    cfg.optimization.bs = 4
    cfg.logging.visual_freq = visual
    train = np.arange(16*22*3, dtype=np.float32).reshape(16, 22, 3) / 100
    val = np.zeros((8, 22, 3), np.float32)
    paths = {key: str(tmp_path / key) for key in ('train', 'val', 'test', 'pdb')}
    for path in paths.values(): Path(path).touch()
    paths['source'] = 'synthetic'
    checks, reads = [], []
    monkeypatch.setattr(ts_systems, 'ensure_data', lambda *a, **kw: checks.append(True) or paths)
    monkeypatch.setattr(ts_data.np, 'load', lambda path: reads.append(Path(path).name) or {'train': train, 'val': val}[Path(path).name])
    original = ts_data.build_train_loader
    monkeypatch.setattr(ts_data, 'build_train_loader', lambda x, std, bs, com, workers, seed:
                        original(x, std, bs, com, num_workers=0, seed=seed))
    monkeypatch.setattr(distributed, 'is_main', lambda: main)
    evaluated = []
    monkeypatch.setattr(ts_eval, 'build_eval_context', lambda cfg, data, **kw: evaluated.append(data) or object())
    cfg, statics, state = learn.setup_state(cfg, 'cpu', 12)
    assert reads == expected and checks
    centered = torch.from_numpy(train) - torch.from_numpy(train).mean(1, keepdim=True)
    assert cfg.network.unscale_factor == float(centered.std())
    assert next(statics.ds).shape == (4, 22, 3)
    assert len(evaluated) == (1 if main and visual else 0)


@pytest.mark.parametrize('entry', ['eval_peptide_ts', 'eval_ode_int'])
@pytest.mark.parametrize('override', [None, 2.])
def test_evaluator_reads_train_once_and_keeps_measured_scale(monkeypatch, tmp_path, capsys, entry, override):
    from common import state_utils
    launcher = importlib.import_module(f'launchers.{entry}')
    train = torch.arange(8*22*3, dtype=torch.float32).reshape(8, 22, 3) / 100
    expected = float((train - train.mean(1, keepdim=True)).std())
    reads = []
    monkeypatch.setattr(ts_systems, 'ensure_data', lambda *a, **kw: {'source': 'synthetic'})
    monkeypatch.setattr(ts_systems, 'load_split', lambda system, split, data_dir=None: reads.append(split) or train)
    monkeypatch.setattr(ts_systems, 'ParallelEnergy', lambda *a, **kw: SimpleNamespace(close=lambda: None))
    class Checked(Exception): pass
    def check(cfg, *a, **kw):
        assert cfg.network.unscale_factor == (expected if override is None else override)
        raise Checked
    monkeypatch.setattr(state_utils, 'load_evaluation_model', check)
    argv = [entry, '--cfg_path', 'configs.ace_a_nme', '--ckpt_path', 'ace_a_nme.pt',
            '--out_dir', str(tmp_path), '--device', 'cpu']
    if override is not None: argv += ['--unscale_factor', str(override)]
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(Checked): launcher.main()
    assert reads == ['train']
    assert f'ts train std={expected:.6g}' in capsys.readouterr().out
