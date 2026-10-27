"""The paper's training setup (--training_recipe paper) for the teacher and the distilled map."""
import sys

import pytest
import torch

from common import distributed, state_utils
from common.flow_map import TorchFlowMap
from configs.ace_a_nme import get_config
from launchers import learn
from test_simplification import saved_model, tiny_config


def resolve(argv, monkeypatch):
    monkeypatch.setenv('WANDB_ENTITY', 'RKOY')
    monkeypatch.setattr(sys, 'argv', ['learn.py', '--cfg_path', 'configs.ace_a_nme', '--device', 'cpu', *argv])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    distributed.configure_batch(cfg)
    return cfg


@pytest.mark.parametrize('mode', ['pretrain_v', 'dlogpdt'])
def test_paper_recipe_resolves_to_the_stated_setup(mode, monkeypatch):
    stage = [] if mode == 'pretrain_v' else ['--network_load_path', 't.pt', '--teacher_load_path', 't.pt']
    cfg = resolve(['--mode', mode, '--training_recipe', 'paper', *stage], monkeypatch)
    opt, training, network = cfg.optimization, cfg.training, cfg.network
    assert (opt.optimizer, opt.learning_rate, opt.weight_decay, opt.clip) == ('adamw', 1e-4, 1e-4, 10.)
    assert (opt.schedule_type, opt.warmup_fraction, opt.bs, opt.total_steps) == ('sqrt', .025, 1000, 4_000_000)
    assert opt.diag_fraction == (1. if mode == 'pretrain_v' else .5)
    assert training.tmax == training.diag_tmax == 1 - 1e-5 and training.ema_facs == [.999]
    assert network.use_weight and network.logvar_channels == 128
    assert (network.dit_kwargs.num_shared_layers, network.dit_kwargs.num_div_layers) == (6, 3)
    if mode == 'dlogpdt':
        # The paper's DiT for ALA-2 has about 6.25M parameters (6 shared + 3 log-density blocks).
        params = sum(p.numel() for p in TorchFlowMap(network).parameters())
        assert params == pytest.approx(6.25e6, rel=.005)


def test_paper_recipe_keeps_explicit_overrides_and_rejects_other_modes(monkeypatch):
    cfg = resolve(['--mode', 'dlogpdt', '--training_recipe', 'paper', '--offdiag_time_order', 'square',
                   '--total_steps', '1234', '--teacher_load_path', 't.pt'], monkeypatch)
    assert cfg.training.offdiag_time_order == 'square' and cfg.optimization.total_steps == 1234
    with pytest.raises(ValueError, match='paper recipe'):
        resolve(['--mode', 'pretrain_flowmap', '--training_recipe', 'paper'], monkeypatch)


def test_cycle_flags_resolve_with_their_own_weight_only_when_weights_are_learned(monkeypatch):
    stage = ['--mode', 'dlogpdt', '--teacher_load_path', 't.pt', '--cycle_loss', '--cycle_stopgrad', 'directional']
    cfg = resolve([*stage, '--training_recipe', 'paper'], monkeypatch)
    assert cfg.training.cycle_stopgrad == 'directional' and cfg.network.cycle_logvar
    assert not hasattr(resolve(stage, monkeypatch).network, 'cycle_logvar')
    with pytest.raises(ValueError, match='requires --cycle_loss'):
        resolve(['--mode', 'dlogpdt', '--cycle_stopgrad', 'directional'], monkeypatch)


def test_sqrt_schedule_warms_up_linearly_then_decays():
    cfg = get_config(0)
    cfg.optimization.update(schedule_type='sqrt', warmup_fraction=.025, total_steps=4000,
                            decay_steps=500, learning_rate=1e-4)
    lr = state_utils.setup_schedule(cfg)
    assert [lr(k) for k in (0, 50, 100, 500)] == pytest.approx([0, .5e-4, 1e-4, 1e-4])
    assert lr(2000) == pytest.approx(.5e-4)  # 1e-4 / sqrt(2000 / 500)
    cfg.optimization.warmup_fraction = 0.
    assert state_utils.setup_schedule(cfg)(0) == 1e-4  # unchanged without warm-up


def test_adamw_is_recorded_and_never_restored_into_another_optimizer():
    cfg = tiny_config()
    cfg.optimization.optimizer = 'adamw'
    net = TorchFlowMap(cfg.network)
    optimizer, schedule = state_utils.setup_optimizer(cfg, net)
    assert isinstance(optimizer, torch.optim.AdamW) and optimizer.defaults['weight_decay'] == 1e-4
    saved = state_utils.TrainState(net, optimizer, schedule, [.999]).state_dict()
    assert saved['optimizer_type'] == 'adamw'
    other_cfg = tiny_config()
    other_cfg.optimization.optimizer = 'radam'
    other_net = TorchFlowMap(other_cfg.network)
    other = state_utils.TrainState(other_net, *state_utils.setup_optimizer(other_cfg, other_net), [.999])
    with pytest.raises(ValueError, match='optimizer type'):
        other.load_state_dict(saved)


def test_evaluation_loads_checkpoints_trained_with_learned_weights(tmp_path):
    cfg = tiny_config()
    cfg.network.update(use_weight=True, logvar_channels=8, cycle_logvar=True)
    state = saved_model(cfg, tmp_path / 'weighted.pt')
    net, step, _ = state_utils.load_evaluation_model(tiny_config(), str(tmp_path / 'weighted.pt'), 'cpu',
                                                     require_saved_ema=True, ema_factor=.999)
    assert step == 7 and not hasattr(net, 'logvar_linear')
    for name, value in net.named_parameters():
        torch.testing.assert_close(value, state.ema_params[.999][name])
