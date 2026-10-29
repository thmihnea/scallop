"""Offline end-to-end training-loop smoke test (no JAX, no OpenMM).

The existing ``test_train_smoke.py`` builds a real molecular target and so needs
the TS/OpenMM stack. Here we fabricate the target (mean-free Gaussian data
+ base sampler), which lets us drive the *real* training machinery
(``losses.setup_loss`` -> ``updates.setup_train_step`` -> EMA update) for loss
types and modes the OpenMM-bound smoke never reaches: PSD, ESD, cond_trace, and
the pretrain modes. Assertion: loss stays finite and parameters actually move.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ml_collections
import pytest
import torch

import common.interpolant as interpolant
import common.losses as losses
import common.updates as updates
import common.state_utils as state_utils
import common.loss_args as loss_args_mod
from common.se3_utils import remove_mean
from _equiv_utils import build_network_config

torch.set_default_dtype(torch.float32)

DEVICE = "cpu"
N_PARTICLES = 5
BS = 8


def _make_cfg(loss_type, stopgrad, mode, psd_type=None, vectorized=False):
    predict_div = mode not in ("pretrain_v", "pretrain_flowmap")
    diag_fraction = 1.0 if mode == "pretrain_v" else 0.5

    cfg = ml_collections.ConfigDict()
    cfg.network = build_network_config(
        n_particles=N_PARTICLES,
        predict_divergence=predict_div,
        vectorized_divergence=vectorized,
        divergence_mode=mode,
        d_model=32,
        num_layers=2,
        nhead=4,
        from_scratch=True,
    )

    cfg.problem = ml_collections.ConfigDict()
    cfg.problem.interp_type = "linear"
    cfg.problem.n_particles = N_PARTICLES
    cfg.problem.n_dimension = 3
    cfg.problem.d = N_PARTICLES * 3

    cfg.optimization = ml_collections.ConfigDict()
    cfg.optimization.bs = BS
    cfg.optimization.clip = 1.0
    cfg.optimization.learning_rate = 1e-3
    cfg.optimization.schedule_type = "constant"
    cfg.optimization.decay_steps = 1000
    cfg.optimization.diag_fraction = diag_fraction

    cfg.training = ml_collections.ConfigDict()
    cfg.training.loss_type = loss_type
    cfg.training.stopgrad_type = stopgrad
    cfg.training.psd_type = psd_type
    cfg.training.ema_facs = [0.999]
    cfg.training.tmin = 0.05
    cfg.training.tmax = 0.95
    cfg.training.time_distr = "uniform"
    return cfg


def _statics(seed=0):
    import types

    def sample_rho0(bs, generator, device):
        return remove_mean(torch.randn(bs, N_PARTICLES, 3, generator=generator, device=device))

    def data_stream():
        g = torch.Generator().manual_seed(seed + 7)
        while True:
            yield remove_mean(torch.randn(BS, N_PARTICLES, 3, generator=g))

    return types.SimpleNamespace(sample_rho0=sample_rho0, ds=data_stream())


@pytest.mark.parametrize(
    "loss_type,stopgrad,mode,psd_type,vectorized",
    [
        ("lsd", "convex", "dlogpdt", None, False),
        ("lsd", "convex", "hutchinson", None, False),
        ("lsd", "convex", "cond_trace", None, False),
        ("psd", "convex", "dlogpdt", "uniform", False),
        ("psd", "convex", "dlogpdt", "midpoint", False),
        ("esd", "stopped", "dlogpdt", None, False),
        ("lsd", "convex", "pretrain_v", None, False),
        ("lsd", "convex", "pretrain_flowmap", None, False),
        # Rows added with the PSD/ESD divergence terms (2026-09-06). The
        # pretrain_flowmap ones are the regression for the blind 3-tuple unpack that
        # made psd/esd raise at step 0 whenever the network had no divergence head;
        # the vectorized ones exercise the per-element divergence layout.
        ("psd", "convex", "hutchinson", "uniform", False),
        ("psd", "convex", "cond_trace", "midpoint", False),
        ("psd", "stopped", "dlogpdt", "uniform", False),
        ("psd", "convex", "pretrain_flowmap", "uniform", False),
        ("psd", "convex", "dlogpdt", "uniform", True),
        ("esd", "convex", "dlogpdt", None, False),
        ("esd", "stopped", "hutchinson", None, False),
        ("esd", "stopped", "pretrain_flowmap", None, False),
        ("esd", "stopped", "dlogpdt", None, True),
        # The expanded stopped-target ESD contract (2026-09-08), in the
        # combination the configs actually launch: dlogpdt + vectorized divergence.
        ("esd", "stopped", "dlogpdt", None, True),
    ],
)
def test_offline_training_step(loss_type, stopgrad, mode, psd_type, vectorized):
    torch.manual_seed(0)
    cfg = _make_cfg(loss_type, stopgrad, mode, psd_type=psd_type, vectorized=vectorized)

    ex_input = torch.zeros(N_PARTICLES, 3)
    state, net, _ = state_utils.setup_training_state(cfg, DEVICE)
    interp = interpolant.setup_interpolant(cfg)
    loss = losses.setup_loss(cfg, net, interp)
    train_step = updates.setup_train_step(cfg)
    ema_update = updates.setup_ema_update(cfg)
    statics = _statics()

    gen_cpu = torch.Generator().manual_seed(1)
    gen_dev = torch.Generator().manual_seed(2)

    w_before = torch.cat([p.detach().flatten() for p in net.parameters()]).clone()

    for _ in range(3):
        args = loss_args_mod.get_loss_fn_args(cfg, statics, gen_cpu, DEVICE)
        teacher = state.self_teacher()
        loss_val, metrics = train_step(state, loss, teacher, teacher, args, gen_dev)
        state = ema_update(state)
        assert torch.isfinite(loss_val).all(), f"non-finite loss for {loss_type}/{mode}"

    assert state.step == 3, "three finite steps should advance the counter to 3"
    w_after = torch.cat([p.detach().flatten() for p in net.parameters()])
    assert not torch.allclose(w_before, w_after), "parameters did not move"

    # Per-term diagnostics reach the logger for every family, not just lsd, and the
    # divergence term is present exactly when the network has a divergence head.
    if cfg.optimization.diag_fraction < 1.0:
        div_key = f"{loss_type}/{loss_type}_div_loss"
        assert div_key in metrics, f"{div_key} missing from {sorted(metrics)}"
        if cfg.network.predict_divergence:
            assert float(metrics[div_key]) > 0.0, f"{div_key} is exactly zero"


@pytest.mark.parametrize('start,target,outcomes', [
    (0, 3, [True, True, True]), (1, 3, [True, True]), (3, 3, []),
    (0, 3, [True, False, True, True]), (1, 3, [False, True, False, True]), (4, 3, []),
])
@pytest.mark.parametrize('sources', [('external', 'external'), ('conditional', 'self'), ('ema', 'ema')])
@pytest.mark.parametrize('log_freq', [1, 2])
def test_outer_loop_requires_completed_budget(monkeypatch, start, target, outcomes, sources, log_freq):
    from types import SimpleNamespace as NS
    from launchers import learn
    cfg = _make_cfg('lsd', 'stopped', 'pretrain_v')
    cfg.optimization.total_steps = target
    from unittest.mock import Mock
    cfg.training.diag_teacher_source, cfg.training.offdiag_teacher_source = sources
    cfg.logging = ml_collections.ConfigDict(dict(log_freq=log_freq, save_freq=0, visual_freq=0))
    state = NS(step=start, ema_facs=[.999], self_teacher=Mock(return_value={}), ema_teacher=Mock(return_value={}))
    statics = NS(teacher_params={}, loss=None)
    outcomes_iter = iter(outcomes)
    def update(state, *a):
        state.step += int(next(outcomes_iter))
        return torch.tensor(1.), {}
    saved, logs = [], []
    monkeypatch.setattr(learn.logging, 'register_signal_handlers', lambda *a: None)
    monkeypatch.setattr(learn.loss_args, 'get_loss_fn_args', lambda *a: ())
    monkeypatch.setattr(learn.updates, 'setup_train_step', lambda cfg: update)
    monkeypatch.setattr(learn.updates, 'setup_ema_update', lambda cfg: lambda state: state)
    monkeypatch.setattr(learn.logging, 'save_state', lambda state, cfg: saved.append(state.step))
    monkeypatch.setattr(learn.logging, 'log_metrics', lambda cfg, state, dev, loss, metrics, step, elapsed:
                        logs.append((step, metrics['train/optimizer_step'])))
    actual = start + sum(outcomes)
    if actual == target:
        learn.train_loop(cfg, statics, state, 'cpu', 0)
        assert saved == [target]
    else:
        with pytest.raises(RuntimeError, match=rf'{actual}.*{target}'):
            learn.train_loop(cfg, statics, state, 'cpu', 0)
        assert saved == []
    if outcomes:
        assert logs[-1] == (target, actual)

    assert state.self_teacher.call_count == (len(outcomes) if 'self' in sources else 0)
    assert state.ema_teacher.call_count == (len(outcomes) if 'ema' in sources else 0)


def test_main_reports_failure_and_cleans_process_group(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from launchers import learn
    cfg = _make_cfg('lsd', 'stopped', 'pretrain_v')
    cfg.optimization.local_bs = cfg.optimization.bs
    cfg.training.seed = 42
    cfg.logging = ml_collections.ConfigDict(dict(output_folder=str(tmp_path)))
    monkeypatch.setattr(learn, 'parse_command_line_arguments', lambda: NS(device='cpu'))
    monkeypatch.setattr(learn, 'setup_config_dict', lambda args: cfg)
    monkeypatch.setattr(learn.distributed, 'initialize', lambda *a, **kw: torch.device('cpu'))
    monkeypatch.setattr(learn.distributed, 'configure_batch', lambda cfg: None)
    monkeypatch.setattr(learn.distributed, 'rank', lambda: 0)
    monkeypatch.setattr(learn.distributed, 'world_size', lambda: 1)
    monkeypatch.setattr(learn.distributed, 'active', lambda: True)
    monkeypatch.setattr(learn.distributed, 'barrier', lambda: None)
    monkeypatch.setattr(learn.distributed, 'wrap_loss', lambda *a, **kw: None)
    monkeypatch.setattr(learn, 'setup_state', lambda cfg, *a: (cfg, NS(loss=None), NS(net=None, ema_params={})))
    finish, cleanup = Mock(), Mock()
    monkeypatch.setattr(learn, '_init_wandb', lambda *a: NS(finish=finish))
    monkeypatch.setattr(torch.distributed, 'destroy_process_group', cleanup)
    monkeypatch.setattr(learn, 'train_loop', Mock(side_effect=RuntimeError('Incomplete training: optimizer step 2, requested 3')))
    with pytest.raises(RuntimeError, match='2.*3'):
        learn.main()
    finish.assert_called_once_with(exit_code=1)
    cleanup.assert_called_once_with()
