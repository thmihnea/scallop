"""Offline correctness of the flow-map inference math (no JAX, no OpenMM).

A randomly-initialised ``TorchFlowMap`` is enough to check the properties that
must hold for *any* network:

* ``partial_t`` / ``partial_s`` (forward-mode AD) match a central finite
  difference of ``flow_call``;
* the Hutchinson divergence estimator's vjp contraction is exact: summing the
  estimate over an orthonormal basis reproduces the true Jacobian trace
  (computed independently by reverse-mode AD);
* ``sample_forward`` / ``sample_inverse`` run end-to-end and return finite,
  correctly-shaped samples and log-probabilities.

Note: a forward->inverse round trip does NOT recover x0 for a random net (the
one-step Euler integrator evaluates phi(0,1,.) and phi(1,0,.), which only agree
for a *trained* consistent flow map), so we check shape/finiteness there rather
than exact recovery.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from _equiv_utils import build_network_config, perturb_params_

torch.set_default_dtype(torch.float32)


def _rel_max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double()
    b = b.detach().double()
    scale = max(1.0, float(a.abs().max()))
    return float((a - b).abs().max()) / scale


def _net(seed=0, n_particles=5, divergence_mode="hutchinson", vectorized=False):
    torch.manual_seed(seed)
    import common.flow_map as fm

    cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=True,
        vectorized_divergence=vectorized,
        divergence_mode=divergence_mode,
        d_model=32,
        num_layers=2,
        nhead=4,
    )
    net = fm.TorchFlowMap(cfg).to(torch.float32).eval()
    # Both DiT output heads are zero-initialised, so an unperturbed net returns
    # phi == 0 and D == 0 identically -- the finite-difference and Hutchinson checks
    # below then compared 0 against 0 and passed regardless of the implementation.
    perturb_params_(net, std=0.1, seed=seed + 1)
    return net, cfg


def _inputs(seed=1, B=4, n=5):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((B, n, 3)).astype(np.float32))
    s = torch.tensor(rng.uniform(0.05, 0.40, size=(B,)).astype(np.float32))
    t = torch.tensor(rng.uniform(0.60, 0.90, size=(B,)).astype(np.float32))
    return x, s, t


def test_partial_t_matches_finite_difference():
    net, _ = _net()
    x, s, t = _inputs()
    eps = 1e-3

    _, dXdt = net.partial_t(s, t, x, train=False)
    fwd = net.flow_call(s, t + eps, x, train=False)
    bwd = net.flow_call(s, t - eps, x, train=False)
    fd = (fwd - bwd) / (2 * eps)
    assert float(fd.detach().abs().max()) > 1e-3, "vacuous: the flow map output does not move"
    assert _rel_max_diff(dXdt, fd) < 1e-2


def test_partial_s_matches_finite_difference():
    net, _ = _net()
    x, s, t = _inputs()
    eps = 1e-3

    _, dXds = net.partial_s(s, t, x, train=False)
    fwd = net.flow_call(s + eps, t, x, train=False)
    bwd = net.flow_call(s - eps, t, x, train=False)
    fd = (fwd - bwd) / (2 * eps)
    assert float(fd.detach().abs().max()) > 1e-3, "vacuous: the flow map output does not move"
    assert _rel_max_diff(dXds, fd) < 1e-2


def test_hutchinson_vjp_reproduces_exact_trace():
    """Summing the Hutchinson estimate over an orthonormal basis == trace(dphi/dx)."""
    from common.losses import _hutchinson_divergence_with_phi

    net, _ = _net(divergence_mode="hutchinson", vectorized=False)
    x, _, t = _inputs()
    B, n, _ = x.shape
    D = n * 3

    # (a) exact per-sample trace of the velocity Jacobian via reverse-mode AD.
    def phi_of_flat(xf, ti):
        xr = xf.reshape(1, n, 3)
        phi = net.calc_b(ti.reshape(1), xr, train=False)
        return phi.reshape(-1)

    exact = torch.empty(B, dtype=torch.float64)
    for i in range(B):
        J = torch.func.jacrev(lambda xf: phi_of_flat(xf, t[i]))(x[i].reshape(-1))
        exact[i] = torch.diagonal(J).sum().double().detach()

    # (b) sum the estimator over the standard basis (eps = e_k): each gives J[k,k].
    tr_hutch = torch.zeros(B, dtype=torch.float64)
    for k in range(D):
        eps = torch.zeros(B, n, 3)
        eps.view(B, -1)[:, k] = 1.0
        _, div_hat = _hutchinson_divergence_with_phi(net, None, t, x, eps=eps)
        tr_hutch += div_hat.double()

    assert float((tr_hutch - exact).abs().max()) < 1e-3


def test_sample_forward_inverse_shapes_and_finite():
    net, cfg = _net(divergence_mode="dlogpdt")
    B, n = 6, cfg.n_particles
    x0 = torch.randn(B, n, 3)

    x1, logp1, nfe = net.sample_forward(x0, n_steps=2)
    assert x1.shape == (B, n, 3)
    assert logp1.shape == (B,)
    assert torch.isfinite(x1).all() and torch.isfinite(logp1).all()
    assert nfe == 2

    x0b, logp0, logp1b, nfe2 = net.sample_inverse(x1, n_steps=2)
    assert x0b.shape == (B, n, 3)
    assert logp0.shape == (B,) and logp1b.shape == (B,)
    assert torch.isfinite(x0b).all()
    assert torch.isfinite(logp0).all() and torch.isfinite(logp1b).all()
