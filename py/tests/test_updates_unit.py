"""Unit tests for the training-step and EMA update logic (no JAX, no OpenMM).

The step-counter test locks the fix that brings the port in line with the JAX
reference: on a non-finite (NaN) step neither the optimizer update nor the step
counter advances (JAX freezes both inside ``jax.lax.cond(all_finite, ...)``).
The previous port advanced ``state.step`` unconditionally, desynchronising the
LR schedule / EMA timing after any NaN step.
"""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ml_collections
import torch
import torch.nn as nn

import common.updates as updates
from common.state_utils import TrainState

torch.set_default_dtype(torch.float32)


def _cfg(clip=1.0):
    c = ml_collections.ConfigDict()
    c.optimization = ml_collections.ConfigDict()
    c.optimization.clip = clip
    return c


def _fake_state(seed=0):
    torch.manual_seed(seed)
    net = nn.Linear(3, 3, bias=False)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    return types.SimpleNamespace(net=net, optimizer=opt, schedule=lambda step: 0.1, step=0)


# loss_fn signature expected by train_step:
#   loss_fn(teacher_diag, teacher_offdiag, x0, x1, label, s, t, u, h, generator=...)
def _loss_finite(td, to, *args, generator=None):
    net_w = _loss_finite.net.weight
    return net_w.pow(2).sum(), {}


def _loss_nan(td, to, *args, generator=None):
    net_w = _loss_nan.net.weight
    return net_w.sum() * float("nan"), {}


_DUMMY_ARGS = (torch.zeros(1), torch.zeros(1), None, torch.zeros(1), torch.zeros(1), None, None)


def test_step_advances_on_finite_step():
    state = _fake_state()
    _loss_finite.net = state.net
    train_step = updates.setup_train_step(_cfg())
    w0 = state.net.weight.detach().clone()

    loss, metrics = train_step(state, _loss_finite, {}, {}, _DUMMY_ARGS, torch.Generator())

    assert state.step == 1
    assert metrics["train/nan_detected"] is False
    assert not torch.allclose(state.net.weight, w0), "finite step should update weights"


def test_step_frozen_on_nonfinite_step():
    state = _fake_state()
    _loss_nan.net = state.net
    train_step = updates.setup_train_step(_cfg())
    w0 = state.net.weight.detach().clone()

    loss, metrics = train_step(state, _loss_nan, {}, {}, _DUMMY_ARGS, torch.Generator())

    # This is the regression guard: NaN step must NOT advance the counter or the weights.
    assert state.step == 0, "step counter must freeze on a non-finite step"
    assert metrics["train/nan_detected"] is True
    assert torch.allclose(state.net.weight, w0), "weights must not change on a NaN step"


def test_ema_update_matches_formula():
    torch.manual_seed(0)
    net = nn.Linear(2, 2, bias=False)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    fac = 0.9
    state = TrainState(net, opt, schedule=lambda s: 0.1, ema_facs=[fac])

    ema_init = {n: p.detach().clone() for n, p in net.named_parameters()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)  # move the live params away from the EMA
    param_new = {n: p.detach().clone() for n, p in net.named_parameters()}

    state.update_ema()

    for n in ema_init:
        expected = fac * ema_init[n] + (1.0 - fac) * param_new[n]
        assert torch.allclose(state.ema_params[fac][n], expected, atol=1e-6)


def test_ema_teacher_lags_the_live_params():
    """``ema_teacher`` is the slow teacher behind ``--offdiag_teacher_source ema``.

    ``self_teacher`` returns the live params detached, so a self-distillation target
    is rebuilt from weights that moved on the previous step. For ESD that target
    contains the network's spatial Jacobian and the chase diverges (measured on
    ace_a_nme, 2026-09-07; see ``learn._STOPGRAD_DEFAULTS``). This pins that
    ``ema_teacher`` returns the EMA copy instead -- detached, defaulting to the first
    EMA factor, and genuinely lagging the live params after an update.
    """
    torch.manual_seed(0)
    net = nn.Linear(2, 2, bias=False)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    fac = 0.9
    state = TrainState(net, opt, schedule=lambda s: 0.1, ema_facs=[fac])

    # Before any update the EMA is a copy of the init, so the two teachers agree.
    for n, p in state.ema_teacher().items():
        assert torch.allclose(p, dict(net.named_parameters())[n])

    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    state.update_ema()

    live = state.self_teacher()
    slow = state.ema_teacher()
    assert set(slow) == set(live)
    for n in live:
        assert not slow[n].requires_grad, "a teacher must not carry gradient"
        assert not torch.allclose(slow[n], live[n]), "EMA teacher must lag the live params"
        assert torch.allclose(slow[n], state.ema_params[fac][n])

    # An explicit factor selects that EMA copy; the default is ema_facs[0].
    assert torch.allclose(state.ema_teacher(fac)["weight"], state.ema_teacher()["weight"])
