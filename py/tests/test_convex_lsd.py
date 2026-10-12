"""LSD convex: live velocity/density time corrections with frozen sources."""
import pytest
import torch

from common import interpolant, losses
from _equiv_utils import build_problem_config
from test_stopped_flowmap import DEVICE, PolynomialMap, batch, heads, snapshots


@pytest.mark.parametrize('density,vectorized', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('self_teacher', [False, True])
def test_convex_lsd_matches_analytic_gradients_and_traverses_both_tangents(
    density, vectorized, weighted, self_teacher, monkeypatch,
):
    net = PolynomialMap(vectorized, density, weighted).to(DEVICE)
    teacher, ema = snapshots(net)
    selected_teacher = None if self_teacher else teacher
    x0, x1, s, t, _ = batch()
    interp = interpolant.setup_interpolant(build_problem_config())
    x = interp.calc_It(s, x0, x1)
    ss, tt = s[:, None, None], t[:, None, None]
    gap = tt - ss
    # Freeze the current-student endpoint and both independent sources, but
    # derive the correction analytically from live parameters (no AD oracle).
    with torch.no_grad():
        raw_u, _ = heads(dict(net.named_parameters()), s, t, x, vectorized)
        endpoint = x + gap * raw_u
        b = heads(dict(net.named_parameters()) if self_teacher else teacher,
                  t, t, endpoint, vectorized)[0]
        g = heads(ema, t, t, endpoint, vectorized)[1]
    u, d = heads(dict(net.named_parameters()), s, t, x, vectorized)
    du = (2 * net.p[2] * gap + net.p[3] * ss).expand_as(x)
    dd = (net.q[2] + net.q[3] * (gap + tt)).expand_as(x)
    if not vectorized:
        dd = dd.sum((1, 2))
    v_loss = (u + gap * du - b).square().sum((1, 2))
    d_loss = torch.zeros_like(v_loss)
    if density:
        density_gap = gap if vectorized else t - s
        d_loss = (d + density_gap * dd - g).square().reshape(4, -1).sum(1)
        d_loss = d_loss / (6 if vectorized else 1)
    w = net.calc_weight(s, t)
    expected = torch.exp(-w) * (v_loss + d_loss) + w if weighted else v_loss + d_loss
    wanted = torch.autograd.grad(expected.sum(), tuple(net.parameters()), allow_unused=True)

    seen = {}
    original_jvp = torch.func.jvp
    def watched(*args, **kwargs):
        primal, tangent = original_jvp(*args, **kwargs)
        tangents = tangent if isinstance(tangent, tuple) else (tangent,)
        for name, value in zip(('velocity', 'density'), tangents):
            value.register_hook(lambda grad, name=name: seen.setdefault(name, grad.detach().clone()))
        return primal, tangent
    monkeypatch.setattr(torch.func, 'jvp', watched)
    value, metrics = losses.lsd_term(
        net, selected_teacher, x0, x1, None, s, t, interp=interp,
        stopgrad_type='convex', density_ema_params=ema,
    )
    params = tuple(net.parameters()) + tuple(teacher.values()) + tuple(ema.values())
    actual = torch.autograd.grad(value.sum(), params, allow_unused=True)
    torch.testing.assert_close(value, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(metrics['lsd/lsd_v_loss'], v_loss.detach())
    torch.testing.assert_close(metrics['lsd/lsd_div_loss'], d_loss.detach())
    assert all(grad is None for grad in actual[3:]), 'teacher or EMA received gradients'
    for actual_grad, expected_grad in zip(actual[:3], wanted):
        assert (actual_grad is None) == (expected_grad is None)
        if actual_grad is not None:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)
    assert set(seen) == ({'velocity', 'density'} if density else {'velocity'})
    assert all(torch.isfinite(grad).all() and grad.abs().max() > .01 for grad in seen.values())

    # Same forward objective, different optimization gradients in BOTH heads.
    seen.clear()
    stopped, _ = losses.lsd_term(
        net, selected_teacher, x0, x1, None, s, t, interp=interp,
        stopgrad_type='stopped', density_ema_params=ema,
    )
    stopped_grads = torch.autograd.grad(stopped.sum(), tuple(net.parameters()), allow_unused=True)
    torch.testing.assert_close(value, stopped, rtol=2e-12, atol=2e-12)
    assert not seen
    assert (actual[0] - stopped_grads[0]).abs().max() > .01
    if density:
        assert (actual[1] - stopped_grads[1]).abs().max() > .01
    if weighted:
        torch.testing.assert_close(actual[2], stopped_grads[2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('vectorized', [False, True])
def test_convex_and_stopped_lsd_agree_on_diagonal(vectorized):
    net = PolynomialMap(vectorized, density=True, weighted=True).to(DEVICE)
    teacher, ema = snapshots(net)
    x0, x1, s, _, _ = batch()
    interp = interpolant.setup_interpolant(build_problem_config())
    results = []
    for policy in ('stopped', 'convex'):
        value, _ = losses.lsd_term(net, teacher, x0, x1, None, s, s, interp=interp,
                                  stopgrad_type=policy, density_ema_params=ema)
        grads = torch.autograd.grad(value.sum(), tuple(net.parameters()))
        results.append((value, grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for stopped, convex in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(stopped, convex, rtol=0, atol=0)
