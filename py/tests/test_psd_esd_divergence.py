"""Analytic correctness of the PSD / ESD / LSD *divergence* (likelihood) residuals.

The PSD and ESD divergence terms have no counterpart in the vendored JAX reference
(``jax_impl/common/losses.py`` computes PSD's ``div_teacher`` and then discards it,
and its ``esd_term`` carries no divergence at all), so the JAX<->Torch equivalence
suite cannot cover them directly. The updated JAX oracle and
``test_stopped_flowmap.py`` check the new stopped gradients independently.
They are checked here against the mathematics instead, which is the stronger check
anyway: a second implementation can repeat the same mistake, an exact flow map
cannot.

The exact map
-------------
With ``T(t) = t + t^2/2`` (so ``T'(t) = 1 + t > 0``), define elementwise

    X_{s,t}(x) = asinh( sinh(x) + T(t) - T(s) ).

The shift argument is additive in time, so the semigroup ``X_{u,t} o X_{s,u} =
X_{s,t}`` (paper eq. 10) and ``X_{s,s} = id`` hold exactly, for ``s < t`` and
``s > t`` alike. Differentiating at ``t = s`` gives the generating velocity

    b(t, x) = T'(t) sech(x),                                        (elementwise)

the Jacobian is diagonal with entries ``cosh(x_i) / cosh(X_i)``, hence

    A_{s,t}(x) = log det grad X_{s,t}(x) = sum_i [log cosh(x_i) - log cosh(X_i)],
    div b(s, x) = -T'(s) sum_i sech(x_i) tanh(x_i)  =  d/dt A_{s,t}(x) |_{t=s}.

Feeding ``phi = (X - x)/(t-s)`` and ``D = A/(t-s)`` to the real loss code must drive
every self-distillation residual -- velocity *and* divergence -- to zero. Anything
the implementation gets wrong (a missing source term, a wrong ``(t-s)`` factor, a
swapped ``s``/``t``, the wrong convex weight) leaves a residual that is O(1) here:
``test_residuals_are_not_vacuously_zero`` pins that the terms really are O(1).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

import common.flow_map as fm
import common.interpolant as interpolant
import common.losses as losses
from _equiv_utils import build_network_config, build_problem_config, perturb_params_

torch.set_default_dtype(torch.float32)

TOL = 1e-6  # on the per-sample summed squared residual (values are O(1))


def _T(z):
    return z + 0.5 * z * z


def _Tdot(z):
    return 1.0 + z


class AnalyticFlowMap(fm.TorchFlowMap):
    """A ``TorchFlowMap`` whose ``calc_phi`` is the exact map above.

    Only ``calc_phi`` is overridden, so ``flow_call``, ``partial_s``, ``partial_t``
    and the ``forward`` dispatcher -- i.e. everything the loss terms actually drive
    -- are the real production code paths.
    """

    def calc_phi(self, s, t, x, label=None, train=True, calc_weight=False,
                 return_div=False, return_velocity=True):
        assert not calc_weight, "the analytic stub does not model the logvar head"
        ss, ts, xs, is_single = self.process_input(s, t, x)

        tau = _T(ts) - _T(ss)
        delta = ts - ss
        X = torch.asinh(torch.sinh(xs) + tau)
        A = torch.log(torch.cosh(xs)) - torch.log(torch.cosh(X))

        # t == s limits (calc_b hits this exactly): phi -> b, D -> div b.
        diagonal = delta.abs() < 1e-12
        sech = 1.0 / torch.cosh(xs)
        phi_diag = _Tdot(ss) * sech
        div_diag = -_Tdot(ss) * sech * torch.tanh(xs)
        safe = torch.where(diagonal, torch.ones_like(delta), delta)

        phi = torch.where(diagonal, phi_diag, (X - xs) / safe)
        div = torch.where(diagonal, div_diag, A / safe)
        if not self.config.vectorized_divergence:
            div = div.sum(dim=(1, 2))

        if return_div:
            if is_single:
                return phi[0], div[0]
            return phi, div
        if is_single:
            return phi[0]
        return phi


def _analytic_net(n_particles=5, vectorized=False, predict_divergence=True):
    cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=predict_divergence,
        vectorized_divergence=vectorized,
        divergence_mode="dlogpdt",
        div_loss_weighting_type="uniform",
        divergence_rescale=1.0,
        d_model=16,
        num_layers=1,
        nhead=2,
    )
    torch.manual_seed(0)
    net = AnalyticFlowMap(cfg).to(torch.float32).eval()
    return net


def _interp():
    return interpolant.setup_interpolant(build_problem_config("linear"))


def _inputs(seed=3, B=4, n=5, psd_type="uniform"):
    rng = np.random.default_rng(seed)
    T = lambda a: torch.tensor(a.astype(np.float32))
    x0 = T(rng.standard_normal((B, n, 3)))
    x1 = T(rng.standard_normal((B, n, 3)))
    # Both orderings of (s, t) occur in training (_sample_triangle draws them
    # independently), so cover s < t and s > t in the same batch.
    s = T(np.array([0.15, 0.80, 0.30, 0.62]))
    t = T(np.array([0.75, 0.20, 0.55, 0.10]))
    h = T(rng.uniform(0.15, 0.85, size=(B,)))
    if psd_type == "midpoint":
        u = 0.5 * (s + t)
    else:
        u = h * s + (1 - h) * t  # loss_args.sample_times
    return x0, x1, s, t, u, h


# --------------------------------------------------------------------------- #
# the residuals vanish on an exact likelihood flow map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("stopgrad", ["stopped", "convex"])
@pytest.mark.parametrize("psd_type", ["uniform", "midpoint"])
def test_psd_residuals_vanish(psd_type, stopgrad, vectorized):
    net = _analytic_net(vectorized=vectorized)
    x0, x1, s, t, u, h = _inputs(psd_type=psd_type)

    _, m = losses.psd_term(
        net, None, x0, x1, None, s, t, u, h,
        density_ema_params=net.state_dict(), interp=_interp(), psd_type=psd_type, stopgrad_type=stopgrad,
    )
    v, d = float(m["psd/psd_v_loss"].max()), float(m["psd/psd_div_loss"].max())
    print(f"[psd {psd_type} {stopgrad} vec={vectorized}] v={v:.3e} div={d:.3e}")
    assert v < TOL, f"PSD velocity residual {v} on an exact map"
    assert d < TOL, f"PSD divergence residual {d} on an exact map"


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("stopgrad", ["stopped", "convex"])
def test_esd_residuals_vanish(stopgrad, vectorized):
    net = _analytic_net(vectorized=vectorized)
    x0, x1, s, t, _, _ = _inputs()

    _, m = losses.esd_term(
        net, None, x0, x1, None, s, t, density_ema_params=net.state_dict(), interp=_interp(), stopgrad_type=stopgrad,
    )
    v, d = float(m["esd/esd_v_loss"].max()), float(m["esd/esd_div_loss"].max())
    print(f"[esd {stopgrad} vec={vectorized}] v={v:.3e} div={d:.3e}")
    assert v < TOL, f"ESD velocity residual {v} on an exact map"
    assert d < TOL, f"ESD divergence residual {d} on an exact map"


@pytest.mark.parametrize("vectorized", [False, True])
def test_lsd_residuals_vanish(vectorized):
    """Cross-check: the pre-existing LSD term must vanish on the same exact map."""
    net = _analytic_net(vectorized=vectorized)
    x0, x1, s, t, _, _ = _inputs()

    _, m = losses.lsd_term(
        net, None, x0, x1, None, s, t, density_ema_params=net.state_dict(), interp=_interp(), stopgrad_type="convex",
    )
    v, d = float(m["lsd/lsd_v_loss"].max()), float(m["lsd/lsd_div_loss"].max())
    print(f"[lsd convex vec={vectorized}] v={v:.3e} div={d:.3e}")
    assert v < TOL and d < TOL


def test_esd_divergence_collapses_on_the_diagonal():
    """R_div -> 0 as t -> s when the EMA and current weights agree: the -D_st + D_ss cancellation."""
    net = _real_net()
    x0, x1, s, _, _, _ = _inputs()
    interp = _interp()
    losses_by_gap = []
    for gap in (1e-1, 1e-2, 1e-3):
        t = s + gap
        _, m = losses.esd_term(
            net, None, x0, x1, None, s, t, density_ema_params=net.state_dict(), interp=interp, stopgrad_type="stopped",
        )
        losses_by_gap.append(float(m["esd/esd_div_loss"].max()))
    print(f"[esd collapse] {losses_by_gap}")
    assert losses_by_gap[0] > 0.0, "vacuous: the divergence head output is identically 0"
    # R_div = O(t-s) once the -D_st + D_ss pair cancels, so the squared loss falls
    # by ~100x per decade of gap. Require at least 10x to leave room for the O(gap^2)
    # terms of an untrained net.
    for a, b in zip(losses_by_gap, losses_by_gap[1:]):
        assert a > 10.0 * b, f"divergence residual does not collapse: {losses_by_gap}"


def test_psd_divergence_collapses_on_the_diagonal():
    """Same for PSD: u is squeezed between s and t, so the residual -> 0 with the gap."""
    net = _real_net()
    x0, x1, s, _, _, h = _inputs()
    interp = _interp()
    losses_by_gap = []
    for gap in (1e-1, 1e-2, 1e-3):
        t = s + gap
        u = h * s + (1 - h) * t
        _, m = losses.psd_term(
            net, None, x0, x1, None, s, t, u, h,
            density_ema_params=net.state_dict(), interp=interp, psd_type="uniform", stopgrad_type="stopped",
        )
        losses_by_gap.append(float(m["psd/psd_div_loss"].max()))
    print(f"[psd collapse] {losses_by_gap}")
    assert losses_by_gap[0] > 0.0, "vacuous: the divergence head output is identically 0"
    for a, b in zip(losses_by_gap, losses_by_gap[1:]):
        assert a > 10.0 * b, f"divergence residual does not collapse: {losses_by_gap}"


def test_residuals_are_not_vacuously_zero():
    """The exact-map tests would pass trivially if every term were ~0. They are not."""
    net = _analytic_net()
    x0, x1, s, t, _, _ = _inputs()
    Is = _interp().calc_It(s, x0, x1)

    prim, tang = net.partial_s(s, t, Is, train=False, return_div=True)
    _, D_st = prim
    _, ds_Dst = tang
    _, div_ss = net.calc_b(s, Is, train=False, return_div=True)
    print(f"[scale] |D_st|={float(D_st.abs().max()):.3f} "
          f"|ds_D|={float(ds_Dst.abs().max()):.3f} |D_ss|={float(div_ss.abs().max()):.3f}")
    assert float(D_st.abs().max()) > 0.1
    assert float(ds_Dst.abs().max()) > 0.1
    assert float(div_ss.abs().max()) > 0.1


# --------------------------------------------------------------------------- #
# the new forward-mode plumbing (real network, finite differences)
# --------------------------------------------------------------------------- #
def _real_net(seed=0, n_particles=5, vectorized=False, predict_divergence=True):
    """A real (randomly initialised) DiT flow map, perturbed off its zero-init heads.

    Without ``perturb_params_`` this network returns phi == 0 and D == 0 exactly, and
    every test below would pass on any implementation whatsoever.
    """
    torch.manual_seed(seed)
    cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=predict_divergence,
        vectorized_divergence=vectorized,
        divergence_mode="dlogpdt",
        divergence_rescale=1.0,
        d_model=32,
        num_layers=2,
        nhead=4,
    )
    net = fm.TorchFlowMap(cfg).to(torch.float32).eval()
    return perturb_params_(net, std=0.1, seed=seed + 1)


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).abs().max()) / max(1.0, float(a.abs().max()))


@pytest.mark.parametrize("vectorized", [False, True])
def test_partial_s_divergence_matches_finite_difference(vectorized):
    net = _real_net(vectorized=vectorized)
    x0, _, s, t, _, _ = _inputs()
    eps = 1e-3

    (_, D), (_, dD) = net.partial_s(s, t, x0, train=False, return_div=True)
    fwd = net.flow_call(s + eps, t, x0, train=False, return_div=True)[1]
    bwd = net.flow_call(s - eps, t, x0, train=False, return_div=True)[1]
    fd = (fwd - bwd) / (2 * eps)
    assert float(fd.detach().abs().max()) > 1e-3, "vacuous: the divergence output does not move"
    assert _rel(dD, fd) < 1e-2
    assert torch.isfinite(D).all()




# --------------------------------------------------------------------------- #
# no divergence head: the terms must run, not raise (regression)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("term", ["psd", "esd"])
def test_no_divergence_head_is_not_a_crash(term):
    """``predict_divergence=False`` (pretrain_flowmap) used to raise a ValueError:
    ``flow_call`` returns 2 values, not 3, when the head is absent."""
    net = _real_net(predict_divergence=False)
    x0, x1, s, t, u, h = _inputs()
    interp = _interp()

    if term == "psd":
        lv, m = losses.psd_term(
            net, None, x0, x1, None, s, t, u, h,
            density_ema_params=net.state_dict(), interp=interp, psd_type="uniform", stopgrad_type="convex",
        )
        div_key = "psd/psd_div_loss"
    else:
        lv, m = losses.esd_term(
            net, None, x0, x1, None, s, t, density_ema_params=net.state_dict(), interp=interp, stopgrad_type="stopped",
        )
        div_key = "esd/esd_div_loss"
    assert torch.isfinite(lv).all()
    assert float(m[div_key].abs().max()) == 0.0


@pytest.mark.parametrize("term", ["psd", "esd"])
def test_missing_divergence_head_on_a_divergence_model_raises(term):
    """A likelihood model whose head vanished must fail loudly, not train velocity-only."""
    net = _real_net(predict_divergence=False)
    net.config.predict_divergence = True  # the misconfiguration we want caught
    x0, x1, s, t, u, h = _inputs()
    interp = _interp()

    with pytest.raises(RuntimeError, match="predict_divergence"):
        if term == "psd":
            losses.psd_term(
                net, None, x0, x1, None, s, t, u, h,
                density_ema_params=net.state_dict(), interp=interp, psd_type="uniform", stopgrad_type="convex",
            )
        else:
            losses.esd_term(
                net, None, x0, x1, None, s, t, density_ema_params=net.state_dict(), interp=interp, stopgrad_type="stopped",
            )
