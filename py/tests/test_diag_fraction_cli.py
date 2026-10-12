"""CLI diagonal proportions must reach the sampler and loss before initialization."""
import pytest
import torch

from common import distributed, loss_args, losses
from launchers import learn


def argv(*extra):
    return ['learn.py', '--cfg_path', 'configs.ace_a_nme', '--device', 'cpu',
            '--mode', 'dlogpdt', *extra]


def resolve(monkeypatch, *extra):
    monkeypatch.setattr('sys.argv', argv(*extra))
    return learn.setup_config_dict(learn.parse_command_line_arguments())


@pytest.mark.parametrize('recipe', [False, True])
@pytest.mark.parametrize('fraction', [None, .25, .75, 1.0])
def test_diagonal_fraction_overrides_config_and_recipe(monkeypatch, recipe, fraction):
    extra = ['--training_recipe', 'branched_muon'] if recipe else []
    if fraction is not None:
        extra += ['--diag_fraction', str(fraction)]
    cfg = resolve(monkeypatch, *extra)
    assert cfg.optimization.diag_fraction == (.5 if fraction is None else fraction)
    assert cfg.training.stopgrad_type == 'stopped'
    assert cfg.optimization.optimizer == 'muon_adamw'


@pytest.mark.parametrize('fraction', ['0', '-0.1', '1.1', 'nan', 'inf', '-inf'])
def test_invalid_fractions_fail_before_training(monkeypatch, fraction):
    with pytest.raises(ValueError, match='--diag_fraction must be finite'):
        resolve(monkeypatch, '--diag_fraction=' + fraction)


@pytest.mark.parametrize('fraction', [None, 1.0, .75])
def test_velocity_pretraining_stays_fully_diagonal(monkeypatch, fraction):
    extra = ['--mode', 'pretrain_v']
    if fraction is not None:
        extra += ['--diag_fraction', str(fraction)]
    if fraction == .75:
        with pytest.raises(ValueError, match='pretrain_v requires --diag_fraction 1'):
            resolve(monkeypatch, *extra)
    else:
        assert resolve(monkeypatch, *extra).optimization.diag_fraction == 1.0


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('fraction,diagonal', [(.75, 768), (1., 1024)])
def test_cli_split_reaches_sampler_and_loss(monkeypatch, world_size, fraction, diagonal):
    cfg = resolve(monkeypatch, '--diag_fraction', str(fraction),
                  '--training_recipe', 'branched_muon')
    monkeypatch.setattr(distributed, 'world_size', lambda: world_size)
    distributed.configure_batch(cfg)
    local_bs, local_diag = 1024 // world_size, diagonal // world_size
    assert cfg.optimization.local_bs == local_bs
    assert cfg.optimization.local_diag_bs == local_diag
    assert losses._get_diag_offdiag_bs(cfg, local_bs) == (local_diag, local_bs - local_diag)
    s, t, _, _ = loss_args.sample_times(cfg, torch.Generator().manual_seed(8))
    assert len(s) == local_bs
    assert torch.equal(s[:local_diag], t[:local_diag])
    assert torch.all(s[:local_diag] < .999)
    assert torch.all(t[local_diag:] > s[local_diag:])


def test_cli_fraction_retains_ddp_split_guard(monkeypatch):
    cfg = resolve(monkeypatch, '--diag_fraction', '.73', '--training_recipe', 'branched_muon')
    monkeypatch.setattr(distributed, 'world_size', lambda: 4)
    with pytest.raises(ValueError, match='divide evenly'):
        distributed.configure_batch(cfg)


@pytest.mark.parametrize('mode', ['dlogpdt', 'pretrain_flowmap', 'hutchinson', 'cond_trace'])
def test_main_applies_fraction_before_batch_and_state_setup(monkeypatch, tmp_path, mode):
    monkeypatch.setattr('sys.argv', argv('--mode', mode, '--batch_size', '1024',
        '--diag_fraction', '.75', '--output_folder', str(tmp_path), '--no_wandb'))

    class StartupChecked(Exception):
        pass

    def inspect(cfg, device, seed):
        assert cfg.optimization.bs == cfg.optimization.local_bs == 1024
        assert cfg.optimization.diag_fraction == .75
        assert cfg.optimization.local_diag_bs == 768
        raise StartupChecked

    monkeypatch.setattr(learn, 'setup_state', inspect)
    with pytest.raises(StartupChecked):
        learn.main()
