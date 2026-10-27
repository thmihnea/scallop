"""LSD cycle losses: live velocity and density round-trip residuals."""
from types import SimpleNamespace

import pytest
import torch

from common import interpolant, losses
from _equiv_utils import build_problem_config
from test_stopped_flowmap import DEVICE, PolynomialMap, batch, heads, snapshots


@pytest.mark.parametrize('density,vectorized', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('stopgrad_type', ['stopped', 'convex'])
@pytest.mark.parametrize('weighted', [False, True])
def test_cycle_adds_weighted_round_trip_values_and_gradients(density, vectorized, stopgrad_type, weighted):
    net = PolynomialMap(vectorized, density, weighted).to(DEVICE)
    teacher, ema = snapshots(net)
    x0, x1, s, t, _ = batch()
    interp = interpolant.setup_interpolant(build_problem_config())
    x = interp.calc_It(s, x0, x1)
    params = dict(net.named_parameters())
    u, d = heads(params, s, t, x, vectorized)
    back_u, back_d = heads(params, t, s, x + (t - s)[:, None, None] * u, vectorized)
    cycle_v = (u - back_u).square().sum((1, 2))
    cycle_d = (d - back_d).square().reshape(4, -1).sum(1) / (6 if vectorized else 1)
    expected = (cycle_v + cycle_d if density else cycle_v) * torch.exp(-torch.as_tensor(net.calc_weight(s, t)))

    kw = dict(interp=interp, stopgrad_type=stopgrad_type, density_ema_params=ema)
    base, _ = losses.lsd_term(net, teacher, x0, x1, None, s, t, **kw)
    value, metrics = losses.lsd_term(net, teacher, x0, x1, None, s, t, cycle_loss=True, **kw)
    torch.testing.assert_close(value - base, expected)
    torch.testing.assert_close(metrics['lsd/cycle_v_loss'], cycle_v.detach())
    torch.testing.assert_close(metrics['lsd/cycle_div_loss'],
                               cycle_d.detach() if density else torch.zeros_like(cycle_v))
    wanted = torch.autograd.grad((base + expected).sum(), (net.p, net.q), allow_unused=True)
    got = torch.autograd.grad(value.sum(), (net.p, net.q), allow_unused=True)
    torch.testing.assert_close(got, wanted)


@pytest.mark.parametrize('density,vectorized', [(False, False), (True, False), (True, True)])
def test_directional_cycle_trains_only_the_backward_in_time_jump(density, vectorized):
    net = PolynomialMap(vectorized, density, False).to(DEVICE)
    x0, x1, s, t, _ = batch()  # diagonal, tiny, forward (t > s) and reverse (t < s) pairs
    x = interpolant.setup_interpolant(build_problem_config()).calc_It(s, x0, x1)
    params = dict(net.named_parameters())
    u, d = heads(params, s, t, x, vectorized)
    back_u, back_d = heads(params, t, s, (x + (t - s)[:, None, None] * u).detach(), vectorized)

    def stop(first, second):
        forward = (t > s).reshape(-1, *[1] * (first.ndim - 1))
        return torch.where(forward, first.detach(), first), torch.where(forward, second, second.detach())
    expected = stop(u, back_u)
    expected = (expected[0] - expected[1]).square().sum((1, 2))
    if density:
        dd = stop(d, back_d)
        expected = expected + (dd[0] - dd[1]).square().reshape(4, -1).sum(1) / (6 if vectorized else 1)

    got_u, got_d = losses._split_div(net.calc_phi(s, t, x, return_div=True))
    got = sum(losses._cycle_loss(net, s, t, x, got_u, got_d, 'directional'))
    torch.testing.assert_close(got, expected)
    wanted = torch.autograd.grad(expected.sum(), (net.p, net.q), allow_unused=True)
    torch.testing.assert_close(torch.autograd.grad(got.sum(), (net.p, net.q), allow_unused=True), wanted)


def test_cycle_term_has_its_own_learned_weight():
    from common.flow_map import TorchFlowMap
    from test_simplification import tiny_config
    cfg = tiny_config()
    cfg.network.update(use_weight=True, logvar_channels=8, cycle_logvar=True)
    torch.manual_seed(0)
    net = TorchFlowMap(cfg.network)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(.05 * torch.randn_like(p))
    frozen = {n: p.detach().clone() for n, p in net.named_parameters()}
    x0, x1 = torch.randn(4, 22, 3), .3 * torch.randn(4, 22, 3)
    s, t = torch.tensor([.1, .2, .7, .5]), torch.tensor([.6, .9, .3, .55])
    kw = dict(interp=interpolant.setup_interpolant(cfg), density_ema_params=frozen)
    base, _ = losses.lsd_term(net, frozen, x0, x1, None, s, t, **kw)
    value, metrics = losses.lsd_term(net, frozen, x0, x1, None, s, t, cycle_loss=True,
                                     cycle_stopgrad='directional', **kw)
    w_c = net.calc_cycle_weight(s, t).reshape(-1)
    cycle = metrics['lsd/cycle_v_loss'] + metrics['lsd/cycle_div_loss']
    torch.testing.assert_close(value - base, torch.exp(-w_c) * cycle + w_c)
    assert not torch.allclose(w_c, net.calc_weight(s, t).reshape(-1))
    grad = torch.autograd.grad(value.sum(), net.logvar_cycle_linear.weight)[0]
    assert grad.abs().sum() > 0


def test_setup_rejects_cycle_loss_outside_lsd():
    cfg = SimpleNamespace(training=SimpleNamespace(
        loss_type='psd', stopgrad_type='stopped', psd_type='uniform', cycle_loss=True,
    ))
    with pytest.raises(ValueError, match='cycle_loss'):
        losses.setup_loss(cfg, PolynomialMap(), interpolant.setup_interpolant(build_problem_config()))
