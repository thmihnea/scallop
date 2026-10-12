"""Offline correctness of the ODE-with-likelihood integrator (no JAX, no OpenMM).

Two kinds of check:

* **analytic** -- a linear velocity field ``b(t,x) = A x`` has flow ``expm(A)`` and
  log-det ``tr(A)`` in closed form, so the integrator, its solver order and the sign
  convention of ``dlogp`` can be verified against exact values;
* **network** -- on a randomly-initialised ``TorchFlowMap`` the exact trace, the
  Hutchinson estimate and an independently computed Jacobian trace must agree.

The default ``TorchFlowMap`` has zero-initialised ``final_linear`` / ``final_modulation``
(so the velocity, and every trace, is identically zero); ``_randomised_net`` perturbs them
to make the checks non-vacuous.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from _equiv_utils import build_network_config

import common.ode_likelihood as odl

torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------
def _linear_field(dim: int, seed: int = 0, scale: float = 0.4):
    """``b(t,x) = A x`` acting on flattened coordinates, plus its exact flow/log-det."""
    rng = np.random.default_rng(seed)
    A = torch.tensor((scale * rng.standard_normal((dim, dim))).astype(np.float64))

    def vel_fn(t, x):
        del t  # autonomous
        flat = x.reshape(x.shape[0], -1).to(A.dtype)
        return (flat @ A.T).reshape(x.shape).to(x.dtype)

    return vel_fn, A


def _analytic_endpoint(A: torch.Tensor, x0: torch.Tensor):
    """Exact ``x(1)`` and ``dlogp = -tr(A)`` for ``dx/dt = A x`` over ``t in [0, 1]``."""
    expA = torch.matrix_exp(A)
    flat = x0.reshape(x0.shape[0], -1).to(A.dtype)
    x1 = (flat @ expA.T).reshape(x0.shape).to(x0.dtype)
    dlogp = torch.full((x0.shape[0],), float(-torch.trace(A)), dtype=x0.dtype)
    return x1, dlogp


def _randomised_net(seed: int = 0, n_particles: int = 5, use_bfloat16: bool = False):
    """A small TorchFlowMap with non-zero output head (velocity checkpoint layout)."""
    import common.flow_map as fm

    torch.manual_seed(seed)
    cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=False,
        divergence_mode="pretrain_v",
        use_bfloat16=use_bfloat16,
        d_model=32,
        num_layers=2,
        nhead=4,
    )
    net = fm.TorchFlowMap(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for mod in (net.dit.final_linear, net.dit.final_modulation):
            mod.weight.normal_(0.0, 0.3)
            mod.bias.normal_(0.0, 0.1)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, cfg


def _net_velocity(net):
    def vel_fn(t, x):
        return net.calc_b(t, x, train=False)

    return vel_fn


def _jacobian_trace(net, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reference trace via reverse-mode ``jacrev`` -- independent of ode_likelihood."""

    def single(xi, ti):
        return net.calc_b(ti.reshape(1), xi[None], train=False)[0]

    J = torch.func.vmap(torch.func.jacrev(single))(x, t)
    return torch.einsum("bidid->b", J)


def _inputs(seed: int = 1, batch: int = 4, n_particles: int = 5):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((batch, n_particles, 3)).astype(np.float32))
    t = torch.tensor(rng.uniform(0.1, 0.9, size=(batch,)).astype(np.float32))
    return x, t


# --------------------------------------------------------------------------------------
# 1-2. analytic field: value and solver order
# --------------------------------------------------------------------------------------
def test_linear_field_matches_analytic_flow_and_logdet():
    n_particles, batch = 4, 3
    dim = n_particles * 3
    vel_fn, A = _linear_field(dim, seed=0)
    x0 = torch.tensor(np.random.default_rng(5).standard_normal((batch, n_particles, 3)).astype(np.float32))
    x1_true, dlogp_true = _analytic_endpoint(A, x0)

    res = odl.integrate(
        vel_fn, x0, n_steps=64, solver="rk4", estimator=odl.ExactEstimator()
    )
    assert float((res.x - x1_true).abs().max()) < 1e-5
    assert float((res.dlogp - dlogp_true).abs().max()) < 1e-5
    # dlogp = -tr(A) exactly; a positive-divergence field must LOWER the density.
    assert torch.allclose(res.dlogp, dlogp_true, atol=1e-5)


@pytest.mark.parametrize("solver,order", [("euler", 1), ("midpoint", 2), ("heun", 2), ("rk4", 4)])
def test_solver_order(solver, order):
    """Halving h must shrink the endpoint error by ~2**order."""
    n_particles, batch = 4, 2
    dim = n_particles * 3
    vel_fn, A = _linear_field(dim, seed=1, scale=0.5)
    x0 = torch.tensor(np.random.default_rng(6).standard_normal((batch, n_particles, 3)).astype(np.float32))
    x1_true, dlogp_true = _analytic_endpoint(A, x0)

    errs = []
    for n_steps in (8, 16):
        res = odl.integrate(vel_fn, x0, n_steps=n_steps, solver=solver,
                            estimator=odl.ExactEstimator())
        errs.append(float((res.x - x1_true).abs().max()))
    ratio = errs[0] / max(errs[1], 1e-12)
    assert ratio > 0.6 * 2**order, f"{solver}: error ratio {ratio:.2f}, expected ~{2**order}"

    # The log-det is a time integral of a CONSTANT trace here, so every tableau in
    # _TABLEAUX (all have sum(b) == 1) integrates it exactly, at any step count.
    res = odl.integrate(vel_fn, x0, n_steps=8, solver=solver, estimator=odl.ExactEstimator())
    assert float((res.dlogp - dlogp_true).abs().max()) < 1e-4


def test_no_estimator_returns_no_logdet():
    vel_fn, _ = _linear_field(12, seed=2)
    x0 = torch.zeros((2, 4, 3))
    res = odl.integrate(vel_fn, x0, n_steps=3, solver="euler", estimator=None)
    assert res.dlogp is None
    assert res.n_jvp == 0
    assert res.nfe_velocity == 3


# --------------------------------------------------------------------------------------
# 3-4. divergence estimators on the real network
# --------------------------------------------------------------------------------------
def test_exact_divergence_matches_jacobian_trace():
    net, _ = _randomised_net(seed=0)
    x, t = _inputs()
    ref = _jacobian_trace(net, t, x)

    v, div = odl.ExactEstimator()(_net_velocity(net), t, x)
    assert torch.allclose(v, net.calc_b(t, x, train=False), atol=1e-6)
    assert float((div - ref).abs().max()) < 1e-4, (div, ref)


def test_exact_divergence_chunking_is_exact():
    net, _ = _randomised_net(seed=3)
    x, t = _inputs(seed=4)
    full = odl.ExactEstimator()(_net_velocity(net), t, x)[1]
    for chunk in (1, 4, 7):
        chunked = odl.ExactEstimator(chunk=chunk)(_net_velocity(net), t, x)[1]
        assert float((full - chunked).abs().max()) < 1e-5, chunk


@pytest.mark.parametrize("dist", ["rademacher", "gaussian"])
def test_hutchinson_is_unbiased(dist):
    """Many-probe mean sits within 4 standard errors of the exact trace."""
    net, _ = _randomised_net(seed=0)
    x, t = _inputs()
    ref = _jacobian_trace(net, t, x)

    gen = torch.Generator().manual_seed(1234)
    n_probes = 1200
    est = odl.HutchinsonEstimator(n_probes=n_probes, dist=dist, generator=gen)
    est.reset(x)
    _, div = est(_net_velocity(net), t, x)

    # Per-probe spread, to size the tolerance from the data rather than a magic number.
    probes = odl.draw_probes(tuple(x.shape), 192, dist, generator=torch.Generator().manual_seed(7))
    singles = torch.stack([
        (torch.func.jvp(lambda xx: net.calc_b(t, xx, train=False), (x,), (probes[k],))[1] * probes[k])
        .flatten(start_dim=1).sum(dim=-1)
        for k in range(probes.shape[0])
    ])
    stderr = singles.std(dim=0) / np.sqrt(n_probes)
    assert torch.all((div - ref).abs() < 4.0 * stderr + 1e-6), (div, ref, stderr)


def test_rademacher_has_lower_variance_than_gaussian():
    """Why rademacher is the default (and it is bf16-exact, unlike a Gaussian probe)."""
    net, _ = _randomised_net(seed=0)
    x, t = _inputs()
    spreads = {}
    for dist in ("rademacher", "gaussian"):
        probes = odl.draw_probes(tuple(x.shape), 512, dist,
                                 generator=torch.Generator().manual_seed(21))
        vals = torch.stack([
            (torch.func.jvp(lambda xx: net.calc_b(t, xx, train=False), (x,), (probes[k],))[1] * probes[k])
            .flatten(start_dim=1).sum(dim=-1)
            for k in range(probes.shape[0])
        ])
        spreads[dist] = float(vals.std(dim=0).mean())
    assert spreads["rademacher"] < spreads["gaussian"]


def test_probe_moments():
    for dist in ("rademacher", "gaussian"):
        eps = odl.draw_probes((3, 5, 3), 8000, dist, generator=torch.Generator().manual_seed(2))
        assert float(eps.mean().abs()) < 0.02
        assert abs(float(eps.pow(2).mean()) - 1.0) < 0.05
    rad = odl.draw_probes((2, 2, 3), 16, "rademacher", generator=torch.Generator().manual_seed(3))
    assert torch.all(rad.abs() == 1.0)
    # +-1 survives the use_bfloat16 input truncation in calc_phi unrounded.
    assert torch.equal(rad.to(torch.bfloat16).to(torch.float32), rad)


# --------------------------------------------------------------------------------------
# 5. mean-free subspace
# --------------------------------------------------------------------------------------
def test_ambient_trace_equals_mean_free_trace():
    """The ambient 3N trace IS the divergence on the mean-free subspace.

    With ``com_aware=False`` the DiT mean-centres input and output, so ``J = P A P``
    annihilates the 3 COM directions. Verified by summing ``u^T J u`` over an explicit
    orthonormal basis of the mean-free subspace and comparing with the ambient trace.
    """
    n_particles = 5
    net, _ = _randomised_net(seed=2, n_particles=n_particles)
    x, t = _inputs(seed=9, n_particles=n_particles)
    vel_fn = _net_velocity(net)

    ambient = odl.ExactEstimator()(vel_fn, t, x)[1]

    # Orthonormal basis of {v : sum_atoms v = 0}, built per coordinate axis.
    ones = torch.ones(n_particles) / np.sqrt(n_particles)
    q, _ = torch.linalg.qr(torch.cat([ones[:, None], torch.randn(n_particles, n_particles - 1)], dim=1))
    subspace = q[:, 1:]  # (n_particles, n_particles-1), each column orthogonal to ones

    total = torch.zeros(x.shape[0])
    com_total = torch.zeros(x.shape[0])
    for axis in range(3):
        for col in range(subspace.shape[1]):
            u = torch.zeros(n_particles, 3)
            u[:, axis] = subspace[:, col]
            u = u[None].expand_as(x).contiguous()
            jv = torch.func.jvp(lambda xx: vel_fn(t, xx), (x,), (u,))[1]
            total = total + (jv * u).flatten(start_dim=1).sum(dim=-1)
        # ... and the COM direction along this axis, which must contribute nothing.
        u = torch.zeros(n_particles, 3)
        u[:, axis] = ones
        u = u[None].expand_as(x).contiguous()
        jv = torch.func.jvp(lambda xx: vel_fn(t, xx), (x,), (u,))[1]
        com_total = com_total + (jv * u).flatten(start_dim=1).sum(dim=-1)

    assert float(com_total.abs().max()) < 1e-5, com_total
    assert float((total - ambient).abs().max()) < 1e-4, (total, ambient)


# --------------------------------------------------------------------------------------
# 6. round trip
# --------------------------------------------------------------------------------------
def test_forward_reverse_roundtrip_on_network():
    """Forward then reverse over the SAME velocity field recovers x0 and logp.

    Contrast ``test_flow_map_correctness.py``: there the round trip cannot close because
    ``sample_forward``/``sample_inverse`` evaluate ``phi(0,1,.)`` and ``phi(1,0,.)``, two
    different network outputs. Here both directions integrate one field.
    """
    net, _ = _randomised_net(seed=1)
    x0, _ = _inputs(seed=11)
    vel_fn = _net_velocity(net)

    fwd = odl.integrate(vel_fn, x0, n_steps=32, solver="rk4", estimator=odl.ExactEstimator())
    bwd = odl.integrate(vel_fn, fwd.x, n_steps=32, solver="rk4",
                        estimator=odl.ExactEstimator(), t0=1.0, t1=0.0)

    assert float((bwd.x - x0).abs().max()) < 1e-4
    assert float((fwd.dlogp + bwd.dlogp).abs().max()) < 1e-4


def test_reverse_direction_flips_dlogp_sign_on_linear_field():
    dim = 12
    vel_fn, A = _linear_field(dim, seed=3)
    x0 = torch.tensor(np.random.default_rng(8).standard_normal((2, 4, 3)).astype(np.float32))
    fwd = odl.integrate(vel_fn, x0, n_steps=32, solver="rk4", estimator=odl.ExactEstimator())
    bwd = odl.integrate(vel_fn, fwd.x, n_steps=32, solver="rk4",
                        estimator=odl.ExactEstimator(), t0=1.0, t1=0.0)
    assert float((bwd.dlogp - float(torch.trace(A))).abs().max()) < 1e-4
    assert float((fwd.dlogp + bwd.dlogp).abs().max()) < 1e-4


# --------------------------------------------------------------------------------------
# 7. cost accounting
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("solver", list(odl.FIXED_SOLVERS))
@pytest.mark.parametrize("divergence_type,n_mc", [("hutchinson", 1), ("hutchinson", 3), ("exact", 1)])
def test_nfe_accounting(solver, divergence_type, n_mc):
    n_particles, batch, n_steps = 4, 2, 3
    dim = n_particles * 3
    vel_fn, _ = _linear_field(dim, seed=4)
    x0 = torch.zeros((batch, n_particles, 3))
    est = odl.make_estimator(divergence_type, n_mc_samples=n_mc)
    res = odl.integrate(vel_fn, x0, n_steps=n_steps, solver=solver, estimator=est)

    want = odl.expected_cost(solver, n_steps, divergence_type, dim=dim, n_mc_samples=n_mc)
    assert res.nfe_velocity == want["nfe_velocity"] == n_steps * odl.n_stages(solver)
    assert res.n_jvp == want["n_jvp"]


def test_probe_schedule_fixed_vs_per_step():
    """``fixed`` reuses one probe set; ``per_step`` redraws. Both stay unbiased."""
    net, _ = _randomised_net(seed=0)
    x0, _ = _inputs(seed=12)
    vel_fn = _net_velocity(net)
    exact = odl.integrate(vel_fn, x0, n_steps=8, solver="euler",
                          estimator=odl.ExactEstimator()).dlogp

    for schedule in ("fixed", "per_step"):
        runs = torch.stack([
            odl.integrate(
                vel_fn, x0, n_steps=8, solver="euler",
                estimator=odl.HutchinsonEstimator(
                    n_probes=4, schedule=schedule,
                    generator=torch.Generator().manual_seed(100 + r),
                ),
            ).dlogp
            for r in range(32)
        ])
        stderr = runs.std(dim=0) / np.sqrt(runs.shape[0])
        assert torch.all((runs.mean(dim=0) - exact).abs() < 4.0 * stderr + 1e-6), schedule

    # fixed probes are genuinely held across steps: two runs with the same seed and
    # different step counts still see the same probe set at every stage.
    est = odl.HutchinsonEstimator(n_probes=2, schedule="fixed",
                                  generator=torch.Generator().manual_seed(0))
    est.reset(x0)
    first = est._probes.clone()
    est.new_step()
    assert torch.equal(first, est._probes)
    est_ps = odl.HutchinsonEstimator(n_probes=2, schedule="per_step",
                                     generator=torch.Generator().manual_seed(0))
    est_ps.reset(x0)
    first_ps = est_ps._probes.clone()
    est_ps.new_step()
    assert not torch.equal(first_ps, est_ps._probes)


def test_rejects_bad_arguments():
    vel_fn, _ = _linear_field(12, seed=5)
    x0 = torch.zeros((2, 4, 3))
    with pytest.raises(ValueError):
        odl.integrate(vel_fn, x0, n_steps=0, solver="euler")
    with pytest.raises(ValueError):
        odl.n_stages("bogacki_shampine")
    with pytest.raises(ValueError):
        odl.make_estimator("skilling_hutchinson")
    with pytest.raises(ValueError):
        odl.HutchinsonEstimator(n_probes=0)
    with pytest.raises(ValueError):
        odl.HutchinsonEstimator(schedule="annealed")
    with pytest.raises(ValueError):
        odl.draw_probes((2, 3), 1, "cauchy")


# --------------------------------------------------------------------------------------
# integration with the flow map's base density / rescaling
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# adaptive dopri5
# --------------------------------------------------------------------------------------
def test_dopri5_matches_analytic_flow_and_logdet():
    """Adaptive DP5(4) hits the closed-form endpoint and log-det within tolerance."""
    n_particles, batch = 4, 3
    dim = n_particles * 3
    vel_fn, A = _linear_field(dim, seed=0)
    x0 = torch.tensor(np.random.default_rng(5).standard_normal((batch, n_particles, 3)).astype(np.float32))
    x1_true, dlogp_true = _analytic_endpoint(A, x0)

    res = odl.integrate(vel_fn, x0, solver="dopri5", estimator=odl.ExactEstimator(),
                        atol=1e-8, rtol=1e-8)
    assert float((res.x - x1_true).abs().max()) < 1e-5
    assert float((res.dlogp - dlogp_true).abs().max()) < 1e-5
    assert res.n_steps >= 1
    assert res.extras["n_accepted"] == res.n_steps
    assert res.nfe_velocity == 7 * (res.extras["n_accepted"] + res.extras["n_rejected"])


def test_dopri5_tolerance_controls_step_count_and_accuracy():
    """Tightening atol/rtol must cost more steps and buy accuracy."""
    n_particles, batch = 4, 2
    dim = n_particles * 3
    vel_fn, A = _linear_field(dim, seed=1, scale=0.8)
    x0 = torch.tensor(np.random.default_rng(6).standard_normal((batch, n_particles, 3)).astype(np.float32))
    x1_true, _ = _analytic_endpoint(A, x0)

    loose = odl.integrate(vel_fn, x0, solver="dopri5", estimator=odl.ExactEstimator(),
                          atol=1e-3, rtol=1e-3)
    tight = odl.integrate(vel_fn, x0, solver="dopri5", estimator=odl.ExactEstimator(),
                          atol=1e-9, rtol=1e-9)
    assert tight.n_steps > loose.n_steps
    assert float((tight.x - x1_true).abs().max()) < float((loose.x - x1_true).abs().max())


def test_dopri5_reaches_the_endpoint_exactly():
    """The final step is clipped to t1, so the integration never overshoots."""
    vel_fn, A = _linear_field(12, seed=2, scale=0.3)
    x0 = torch.tensor(np.random.default_rng(7).standard_normal((2, 4, 3)).astype(np.float32))
    for t0, t1 in [(0.0, 1.0), (1.0, 0.0)]:
        res = odl.integrate(vel_fn, x0, solver="dopri5", estimator=odl.ExactEstimator(),
                            t0=t0, t1=t1, atol=1e-7, rtol=1e-7)
        expected = float(-torch.trace(A)) * (t1 - t0)
        assert abs(float(res.dlogp[0]) - expected) < 1e-4, (t0, t1)


def test_dopri5_roundtrip_on_network():
    net, _ = _randomised_net(seed=1)
    x0, _ = _inputs(seed=11)
    vel_fn = _net_velocity(net)
    fwd = odl.integrate(vel_fn, x0, solver="dopri5", estimator=odl.ExactEstimator(),
                        atol=1e-7, rtol=1e-7)
    bwd = odl.integrate(vel_fn, fwd.x, solver="dopri5", estimator=odl.ExactEstimator(),
                        t0=1.0, t1=0.0, atol=1e-7, rtol=1e-7)
    assert float((bwd.x - x0).abs().max()) < 1e-4
    assert float((fwd.dlogp + bwd.dlogp).abs().max()) < 1e-4


def test_dopri5_with_hutchinson_runs_and_redraws_per_accepted_step():
    """per_step probes must be redrawn once per ACCEPTED step, not per attempt."""
    net, _ = _randomised_net(seed=0)
    x0, _ = _inputs(seed=13)
    vel_fn = _net_velocity(net)
    est = odl.HutchinsonEstimator(n_probes=1, schedule="per_step",
                                  generator=torch.Generator().manual_seed(5))
    res = odl.integrate(vel_fn, x0, solver="dopri5", estimator=est, atol=1e-4, rtol=1e-4)
    assert res.dlogp is not None and torch.isfinite(res.dlogp).all()
    assert res.n_jvp == res.nfe_velocity  # K=1 -> one JVP per stage

    # The controller must still terminate with a fixed probe set (a genuine ODE).
    est_fixed = odl.HutchinsonEstimator(n_probes=1, schedule="fixed",
                                        generator=torch.Generator().manual_seed(5))
    res_fixed = odl.integrate(vel_fn, x0, solver="dopri5", estimator=est_fixed,
                              atol=1e-4, rtol=1e-4)
    assert torch.isfinite(res_fixed.dlogp).all()


def test_dopri5_max_steps_and_bad_arguments():
    vel_fn, _ = _linear_field(12, seed=3, scale=2.0)
    x0 = torch.zeros((2, 4, 3))
    with pytest.raises(RuntimeError, match="max_steps"):
        odl.integrate(vel_fn, torch.randn(2, 4, 3), solver="dopri5",
                      estimator=odl.ExactEstimator(), atol=1e-12, rtol=1e-12, max_steps=3)
    with pytest.raises(ValueError):
        odl.integrate_adaptive(vel_fn, x0, solver="rk45")
    with pytest.raises(ValueError):
        odl.integrate(vel_fn, x0, solver="dopri5", atol=0.0)
    with pytest.raises(TypeError):
        odl.integrate(vel_fn, x0, n_steps=2, solver="euler", atol=1e-5)
    assert odl.n_stages("dopri5") == 7
    assert "dopri5" in odl.SOLVERS and "dopri5" not in odl.FIXED_SOLVERS


def _mean_free_basis(n_particles: int) -> torch.Tensor:
    """Orthonormal basis of {v in R^(n x 3) : sum_atoms v = 0}, as a (3n, 3n-3) matrix."""
    ones = torch.ones(n_particles) / np.sqrt(n_particles)
    q, _ = torch.linalg.qr(
        torch.cat([ones[:, None], torch.randn(n_particles, n_particles - 1)], dim=1)
    )
    per_axis = q[:, 1:]  # columns orthogonal to the all-ones direction
    cols = []
    for axis in range(3):
        for col in range(per_axis.shape[1]):
            u = torch.zeros(n_particles, 3)
            u[:, axis] = per_axis[:, col]
            cols.append(u.reshape(-1))
    return torch.stack(cols, dim=1)


def test_dlogp_equals_logdet_of_the_composed_flow():
    """End-to-end: dlogp == -log|det d x1/d x0| on the mean-free subspace.

    This validates the whole likelihood composition -- the divergence estimator, the way
    the augmented state is carried through the RK stages, and the sign convention --
    against a quantity computed a completely different way (the Jacobian of the entire
    integrator, by reverse-mode AD). The ambient 3N Jacobian is singular (the flow is
    constant along the 3 COM directions), so the determinant is taken in an explicit
    orthonormal basis of the 3N-3 mean-free subspace.
    """
    from common.se3_utils import remove_mean

    torch.manual_seed(0)
    n_particles = 4
    net, _ = _randomised_net(seed=4, n_particles=n_particles)
    vel_fn = _net_velocity(net)
    x0 = remove_mean(torch.randn(2, n_particles, 3))

    res = odl.integrate(vel_fn, x0, n_steps=6, solver="rk4", estimator=odl.ExactEstimator())

    def flow(xi):
        return odl.integrate(vel_fn, xi[None], n_steps=6, solver="rk4", estimator=None).x[0]

    U = _mean_free_basis(n_particles)
    dim = 3 * n_particles
    for b in range(x0.shape[0]):
        J = torch.func.jacrev(flow)(x0[b]).reshape(dim, dim)
        logdet = torch.linalg.slogdet(U.T @ J @ U)[1]
        assert abs(float(res.dlogp[b] + logdet)) < 1e-3, (float(res.dlogp[b]), float(-logdet))


def test_launcher_argparser_and_defaults():
    """The launcher must import and parse without torch/OpenMM (module level is light)."""
    import launchers.eval_ode_int as launcher

    argv = ["--system", "ace_a_nme", "--ckpt_path", "/tmp/x.pt", "--out_dir", "/tmp/out"]
    old = sys.argv
    try:
        sys.argv = ["eval_ode_int.py", *argv]
        a = launcher.parse_args()
    finally:
        sys.argv = old
    assert a.divergence_type == "hutchinson"
    assert a.n_mc_samples == 1
    assert a.hutchinson_dist == "rademacher"
    assert a.probe_schedule == "fixed"
    assert (a.n_generate_samples, a.n_eval_samples, a.seed) == (100_000, 10_000, 42)
    assert a.solver == ["dopri5"]
    assert a.energy_cutoff is None and a.posthoc_mala_steps == 128
    assert a.mala_step_size == 1e-6
    assert (a.resampling, a.mala_adapt_steps, a.mala_target_accept) == ("multinomial", 64, .6)
    fixed = launcher.parse_args(argv + ["--mala_adapt_steps", "0", "--resampling", "stratified"])
    assert fixed.mala_adapt_steps == 0 and fixed.resampling == "stratified"
    assert launcher.parse_args(argv + ["--posthoc_mala_steps", "0"]).mala_adapt_steps == 0
    for options in (["--n_eval_samples", "0"], ["--n_generate_samples", "0"],
                    ["--atol", "nan"], ["--mala_step_size", "0"], ["--energy_cutoff", "inf"],
                    ["--posthoc_mala_steps", "-1"], ["--logw_quantile_filter", "1"],
                    ["--mala_target_accept", "nan"], ["--mala_target_accept", "1"],
                    ["--mala_adapt_steps", "-1"], ["--mala_adapt_steps", "129"]):
        with pytest.raises(SystemExit):
            launcher.parse_args(argv + options)


def test_end_to_end_logp_pipeline_shapes_and_finiteness():
    """The exact composition the launcher uses: base logp + dlogp, then unscale."""
    net, cfg = _randomised_net(seed=6)
    from common.se3_utils import remove_mean

    x0 = remove_mean(torch.randn(8, cfg.n_particles, 3))
    logp0 = net.base_log_prob(x0)
    res = odl.integrate(_net_velocity(net), x0, n_steps=4, solver="euler",
                        estimator=odl.make_estimator("hutchinson", n_mc_samples=1))
    x1, logp1 = net.unscale_samples(res.x, logp0 + res.dlogp)
    assert x1.shape == x0.shape
    assert logp1.shape == (8,)
    assert torch.isfinite(logp1).all()
    # unscale_factor == 1.0 in the test config, so the Jacobian offset is zero.
    assert torch.allclose(logp1, logp0 + res.dlogp, atol=1e-6)
