"""Independent value/gradient and RNG contracts for the controlled ESD screen."""
from types import SimpleNamespace

import pytest
import torch

from common import interpolant, loss_args, losses
from _equiv_utils import build_problem_config
from test_stopped_flowmap import PolynomialMap, batch, heads, reference_targets, snapshots


@pytest.mark.parametrize('power', [0.0, 0.5, 1.0])
def test_adaptive_metric_has_frozen_weight_gradient(power):
    net = SimpleNamespace(config=SimpleNamespace(
        velocity_adaptive_power=power, velocity_adaptive_eps=.01,
        predict_divergence=False, use_weight=False, divergence_mode='pretrain_flowmap'))
    error = torch.tensor([[0., .1], [2., -3.]], dtype=torch.float64, requires_grad=True)
    q = error.square().sum(1)
    value, weight = losses._velocity_objective(net, q)
    actual, = torch.autograd.grad(value.sum(), error)
    expected_weight = (q.detach() + .01).pow(-power)
    torch.testing.assert_close(actual, 2 * error.detach() * expected_weight[:, None])
    if power == 0:
        assert value is q and weight is None
    else:
        assert not weight.requires_grad
        assert weight[0] > weight[1]
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('power', [.5, 1.0])
@pytest.mark.parametrize('term', ['diag', 'esd'])
def test_production_adaptive_gradient_matches_explicit_frozen_regression(power, term, monkeypatch):
    net = PolynomialMap(density=False)
    net.config.divergence_mode = 'pretrain_flowmap'
    net.config.velocity_adaptive_power = power
    net.config.velocity_adaptive_eps = .01
    teacher, ema = snapshots(net)
    x0, x1, s, t, h = batch()
    interp = interpolant.setup_interpolant(build_problem_config())
    seen = []
    original_jvp = torch.func.jvp
    def observed(*a, **kw):
        primal, tangent = original_jvp(*a, **kw)
        tangent.register_hook(lambda gradient: seen.append(gradient))
        return primal, tangent
    monkeypatch.setattr(torch.func, 'jvp', observed)
    if term == 'diag':
        x = interp.calc_It(t, x0, x1)
        target = heads(teacher, t, t, x, False)[0].detach()
        pred = net.calc_b(t, x)
        value, metrics = losses.diagonal_term(net, teacher, x0, x1, None, t, interp=interp)
        raw_key = 'diag/velocity_loss'
    else:
        x = interp.calc_It(s, x0, x1)
        target = reference_targets(net, 'esd', 'teacher', teacher, ema, x, x0, x1, s, t, h)[0]
        pred = net.calc_phi(s, t, x)
        value, metrics = losses.esd_term(net, teacher, x0, x1, None, s, t, interp=interp)
        raw_key = 'esd/esd_v_loss'
    q = (pred.detach()-target).square().sum((1, 2))
    w = (q+.01).pow(-power)
    expected, = torch.autograd.grad(pred, net.p,
        grad_outputs=2*w[:,None,None]*(pred.detach()-target))
    actual = torch.autograd.grad(value.sum(), (net.p, *teacher.values()), allow_unused=True)
    torch.testing.assert_close(actual[0], expected)
    assert all(g is None for g in actual[1:])
    assert seen == []
    torch.testing.assert_close(value.detach(), w*q)
    torch.testing.assert_close(metrics[raw_key], q)


def test_forward_order_preserves_diagonal_and_random_stream():
    cfg = SimpleNamespace(training=SimpleNamespace(tmin=0., tmax=1., loss_type='esd'),
                          optimization=SimpleNamespace(bs=1000, diag_fraction=.5))
    g0 = torch.Generator().manual_seed(49)
    g1 = torch.Generator().manual_seed(49)
    s0, t0, _, _ = loss_args.sample_times(cfg, g0)
    cfg.training.offdiag_time_order = 'forward'
    s1, t1, _, _ = loss_args.sample_times(cfg, g1)
    assert torch.equal(s0[:500], s1[:500]) and torch.equal(t0[:500], t1[:500])
    assert torch.equal(s1[500:], torch.minimum(s0[500:], t0[500:]))
    assert torch.equal(t1[500:], torch.maximum(s0[500:], t0[500:]))
    assert (t0[500:] < s0[500:]).any()
    assert (t1 >= s1).all()
    assert torch.equal(g0.get_state(), g1.get_state())


@pytest.mark.parametrize('power,eps', [(-.1,.01), (float('nan'),.01), (.5,0.), (.5,float('inf'))])
def test_invalid_adaptive_settings_rejected(power, eps):
    net = SimpleNamespace(config=SimpleNamespace(velocity_adaptive_power=power, velocity_adaptive_eps=eps))
    with pytest.raises(ValueError):
        losses._velocity_objective(net, torch.ones(2))


@pytest.mark.parametrize('density,weighted', [(True,False), (False,True)])
def test_adaptive_screen_rejects_untested_density_or_logvar_combinations(density, weighted):
    net = SimpleNamespace(config=SimpleNamespace(velocity_adaptive_power=.5,
        predict_divergence=density, use_weight=weighted, divergence_mode='pretrain_flowmap'))
    with pytest.raises(ValueError):
        losses._velocity_objective(net, torch.ones(2))
