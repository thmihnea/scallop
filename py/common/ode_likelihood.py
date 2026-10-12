"""Probability-flow ODE integration with an exact / Hutchinson log-density.

This is the reference likelihood path for ``pretrain_v`` (velocity) checkpoints, which
have no divergence head: the instantaneous change-of-variables

    dx/dt = b(t, x),    d log p_t(x_t)/dt = -div_x b(t, x)

is integrated as one augmented ODE, with ``div`` supplied by autodiff. It deliberately
does NOT go through :meth:`common.flow_map.TorchFlowMap.sample_forward`, which evaluates
``calc_phi(t_curr, t_next, .)`` -- a velocity checkpoint is trained only at ``s == t``
(``diag_fraction=1.0``), so every off-diagonal call is out of distribution.

Everything here is written against a generic velocity callable

    vel_fn(t: Tensor(B,), x: Tensor(B, N, D)) -> Tensor(B, N, D)

so the tests can drive it with analytic fields whose flow and log-det are known in closed
form. No JAX, no OpenMM, no config objects.

Mean-free coordinates
---------------------
With ``com_aware=False`` the DiT mean-centres both its input and its output
(``dit.py`` ``FlowMapAtomDiT.forward``), so the Jacobian factorises as ``J = P A P`` with
``P`` the (symmetric, idempotent) mean-removal projector. ``J`` maps into the mean-free
subspace and annihilates its orthogonal complement, therefore

    trace_ambient(J) = trace(J|mean-free subspace) + 0

i.e. the ambient 3N trace computed here IS the divergence that the mean-free base density
in ``TorchFlowMap.base_log_prob`` requires; no projection correction is needed. Asserted
by ``tests/test_ode_likelihood.py::test_ambient_trace_equals_mean_free_trace``.

Probes are drawn over the full ambient 3N space (not mean-centred), matching
``losses._hutchinson_divergence_with_phi``; that is unbiased for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Explicit Runge-Kutta tableaux (c, A, b). ``A[i]`` holds the i-th row's sub-diagonal.
_TABLEAUX: Dict[str, Tuple[Sequence[float], Sequence[Sequence[float]], Sequence[float]]] = {
    "euler": ((0.0,), ((),), (1.0,)),
    "midpoint": ((0.0, 0.5), ((), (0.5,)), (0.0, 1.0)),
    "heun": ((0.0, 1.0), ((), (1.0,)), (0.5, 0.5)),
    "rk4": (
        (0.0, 0.5, 0.5, 1.0),
        ((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
        (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    ),
}

# Dormand-Prince 5(4): 7 stages, 5th-order solution + embedded 4th-order error estimate.
# Matches the ``dopri5`` that transferable-samplers drives through torchdyn (that package is
# not installed in the `flowmaps` env, so the solver is implemented here instead).
_DP5_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_DP5_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_DP5_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_DP5_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)

FIXED_SOLVERS = tuple(_TABLEAUX)
ADAPTIVE_SOLVERS = ("dopri5",)
SOLVERS = FIXED_SOLVERS + ADAPTIVE_SOLVERS
PROBE_DISTS = ("rademacher", "gaussian")
PROBE_SCHEDULES = ("fixed", "per_step")


def n_stages(solver: str) -> int:
    """Velocity evaluations per step for ``solver``."""
    if solver == "dopri5":
        return len(_DP5_C)
    if solver not in _TABLEAUX:
        raise ValueError(f"Unknown solver {solver!r}; known: {SOLVERS}")
    return len(_TABLEAUX[solver][0])


# --------------------------------------------------------------------------------------
# divergence estimators
# --------------------------------------------------------------------------------------
def _jvp(vel_fn: VelocityFn, t: torch.Tensor, x: torch.Tensor, tangent: torch.Tensor):
    """(velocity, J @ tangent) in a single forward-mode pass."""
    return torch.func.jvp(lambda xx: vel_fn(t, xx), (x,), (tangent,))


def draw_probes(
    shape: Sequence[int],
    n_probes: int,
    dist: str,
    *,
    device=None,
    dtype=torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """``(n_probes, *shape)`` Hutchinson probes with identity covariance.

    ``rademacher`` is the default: it has the lower variance of the two, and its +-1
    entries are exactly representable in bfloat16, so the tangent survives the
    ``use_bfloat16`` input truncation in ``TorchFlowMap.calc_phi`` unrounded (a Gaussian
    probe does not).
    """
    if dist not in PROBE_DISTS:
        raise ValueError(f"Unknown probe distribution {dist!r}; known: {PROBE_DISTS}")
    eps = torch.randn((n_probes, *shape), device=device, dtype=dtype, generator=generator)
    if dist == "rademacher":
        # sign() of a Gaussian is +-1 w.p. 1/2 each (P[exactly 0] = 0).
        eps = torch.sign(eps)
    return eps


class DivergenceEstimator:
    """Base: returns ``(velocity, divergence)`` at ``(t, x)`` and counts its own cost."""

    #: Jacobian-vector products consumed by one :meth:`__call__`.
    jvp_per_call: int = 0

    def reset(self, x: torch.Tensor) -> None:
        """Called once at the start of a trajectory."""

    def new_step(self) -> None:
        """Called at the start of every solver step."""

    def __call__(self, vel_fn, t, x) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class HutchinsonEstimator(DivergenceEstimator):
    """``div ~= mean_k <J eps_k, eps_k>`` -- unbiased for any identity-covariance probe.

    ``schedule``:

    * ``fixed`` -- one probe set per trajectory, held across every step and RK stage
      (the FFJORD convention). The augmented system stays a genuine ODE, so the solver's
      order of accuracy is preserved, but the estimator error is perfectly correlated in
      time and therefore does NOT shrink as the step count grows.
    * ``per_step`` -- fresh probes each step. The per-step errors are independent, so the
      variance of the accumulated ``dlogp`` falls like 1/n_steps, at the cost of turning
      a multi-stage solver into a randomised integrator (the RK weights no longer act on
      a smooth field). Still unbiased for ``euler``.
    """

    def __init__(
        self,
        n_probes: int = 1,
        dist: str = "rademacher",
        schedule: str = "fixed",
        generator: Optional[torch.Generator] = None,
    ):
        if n_probes < 1:
            raise ValueError("n_probes must be >= 1")
        if schedule not in PROBE_SCHEDULES:
            raise ValueError(f"Unknown probe schedule {schedule!r}; known: {PROBE_SCHEDULES}")
        self.n_probes = int(n_probes)
        self.dist = dist
        self.schedule = schedule
        self.generator = generator
        self.jvp_per_call = self.n_probes
        self._probes: Optional[torch.Tensor] = None
        self._shape: Optional[Tuple[int, ...]] = None
        self._device = None
        self._dtype = None

    def _draw(self) -> None:
        self._probes = draw_probes(
            self._shape, self.n_probes, self.dist,
            device=self._device, dtype=self._dtype, generator=self.generator,
        )

    def reset(self, x: torch.Tensor) -> None:
        self._shape = tuple(x.shape)
        self._device = x.device
        self._dtype = x.dtype
        self._draw()

    def new_step(self) -> None:
        if self.schedule == "per_step":
            self._draw()

    def __call__(self, vel_fn, t, x):
        if self._probes is None:
            self.reset(x)
        v = None
        div = None
        for k in range(self.n_probes):
            eps = self._probes[k]
            v_k, jv = _jvp(vel_fn, t, x, eps)
            contrib = (jv * eps).flatten(start_dim=1).sum(dim=-1)
            div = contrib if div is None else div + contrib
            if v is None:
                v = v_k
        return v, div / self.n_probes


class ExactEstimator(DivergenceEstimator):
    """Exact Jacobian trace by one JVP per Cartesian basis direction.

    Costs ``D = N * n_dim`` JVPs per call (66 for Ace-A-Nme, 189 for AAAAAA). Forward
    mode, so no activations are stored; ``chunk`` caps how many basis directions are
    vmapped together and hence the peak memory (``chunk * batch`` effective batch).
    """

    def __init__(self, chunk: Optional[int] = None):
        self.chunk = chunk
        self._basis: Optional[torch.Tensor] = None
        self.jvp_per_call = 0

    def reset(self, x: torch.Tensor) -> None:
        dim = int(x[0].numel())
        self._basis = torch.eye(dim, device=x.device, dtype=x.dtype).reshape(dim, *x.shape[1:])
        self.jvp_per_call = dim

    def __call__(self, vel_fn, t, x):
        if self._basis is None or self._basis.shape[0] != x[0].numel():
            self.reset(x)
        basis = self._basis
        dim = basis.shape[0]
        chunk = self.chunk or dim
        batch = x.shape[0]

        v = None
        div = torch.zeros((batch,), device=x.device, dtype=x.dtype)
        for start in range(0, dim, chunk):
            tan = basis[start : start + chunk]  # (c, N, D)
            c = tan.shape[0]
            tan = tan.unsqueeze(1).expand(c, batch, *x.shape[1:])
            prim, jv = torch.func.vmap(lambda e: _jvp(vel_fn, t, x, e))(tan.contiguous())
            if v is None:
                v = prim[0]  # every vmap row shares the same primal
            # jv[k] is J @ e_{start+k}; its (start+k)-th component is J[start+k, start+k].
            cols = jv.flatten(start_dim=2)[:, :, start : start + c]  # (c, batch, c)
            div = div + torch.diagonal(cols, dim1=0, dim2=2).sum(dim=-1)
        return v, div


def make_estimator(
    divergence_type: str,
    *,
    n_mc_samples: int = 1,
    hutchinson_dist: str = "rademacher",
    probe_schedule: str = "fixed",
    exact_chunk: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> DivergenceEstimator:
    if divergence_type == "hutchinson":
        return HutchinsonEstimator(
            n_probes=n_mc_samples, dist=hutchinson_dist,
            schedule=probe_schedule, generator=generator,
        )
    if divergence_type == "exact":
        return ExactEstimator(chunk=exact_chunk)
    raise ValueError(f"Unknown divergence_type {divergence_type!r}; known: hutchinson, exact")


# --------------------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------------------
@dataclass
class ODEResult:
    """Endpoint of the augmented ODE plus the cost actually incurred."""

    x: torch.Tensor
    #: ``int_{t0}^{t1} -div b dt``; ``None`` when no estimator was supplied.
    dlogp: Optional[torch.Tensor]
    n_steps: int
    solver: str
    #: velocity-field evaluations (network forward passes)
    nfe_velocity: int
    #: Jacobian-vector products (0 when ``dlogp`` is None)
    n_jvp: int
    t0: float = 0.0
    t1: float = 1.0
    #: adaptive solvers only: accepted / rejected step counts, final step size, ...
    extras: Dict[str, float] = field(default_factory=dict)


def integrate(
    vel_fn: VelocityFn,
    x0: torch.Tensor,
    *,
    n_steps: Optional[int] = None,
    solver: str = "euler",
    estimator: Optional[DivergenceEstimator] = None,
    t0: float = 0.0,
    t1: float = 1.0,
    **adaptive_kwargs,
) -> ODEResult:
    """Integrate ``(x, l)`` with ``dx/dt = b(t,x)``, ``dl/dt = -div b(t,x)``.

    The log-density is carried through the *same* Runge-Kutta stages as the state, so it
    inherits the solver's order of accuracy. Set ``t0=1, t1=0`` for the reverse direction
    (``dlogp`` then accumulates with the opposite sign, so a forward pass followed by a
    reverse pass sums to zero for an exact solver).

    ``solver="dopri5"`` selects the adaptive path (:func:`integrate_adaptive`); ``n_steps``
    is then ignored and the extra keyword arguments (``atol``, ``rtol``, ...) are forwarded.

    Returns an :class:`ODEResult`; ``result.dlogp`` is ``None`` when ``estimator`` is
    ``None`` (sampling only, no likelihood).
    """
    if solver in ADAPTIVE_SOLVERS:
        return integrate_adaptive(
            vel_fn, x0, solver=solver, estimator=estimator, t0=t0, t1=t1, **adaptive_kwargs
        )
    if adaptive_kwargs:
        raise TypeError(f"{solver!r} is a fixed-step solver; unexpected {sorted(adaptive_kwargs)}")
    if n_steps is None or n_steps < 1:
        raise ValueError("n_steps must be >= 1 for a fixed-step solver")
    c_coef, a_coef, b_coef = _TABLEAUX[solver]
    stages = len(c_coef)

    h = (t1 - t0) / n_steps
    x = x0
    logdet = torch.zeros((x0.shape[0],), device=x0.device, dtype=x0.dtype)
    ones = torch.ones((x0.shape[0],), device=x0.device, dtype=x0.dtype)

    if estimator is not None:
        estimator.reset(x0)

    for step in range(n_steps):
        if estimator is not None:
            estimator.new_step()
        t_step = t0 + step * h
        k_x: List[torch.Tensor] = []
        k_l: List[torch.Tensor] = []
        for i in range(stages):
            xi = x
            for j, a_ij in enumerate(a_coef[i]):
                if a_ij != 0.0:
                    xi = xi + (h * a_ij) * k_x[j]
            ti = ones * (t_step + c_coef[i] * h)
            if estimator is None:
                v = vel_fn(ti, xi)
                div = None
            else:
                v, div = estimator(vel_fn, ti, xi)
            k_x.append(v)
            k_l.append(None if div is None else -div)
        for i, b_i in enumerate(b_coef):
            if b_i == 0.0:
                continue
            x = x + (h * b_i) * k_x[i]
            if k_l[i] is not None:
                logdet = logdet + (h * b_i) * k_l[i]

    nfe = n_steps * stages
    return ODEResult(
        x=x,
        dlogp=None if estimator is None else logdet,
        n_steps=n_steps,
        solver=solver,
        nfe_velocity=nfe,
        n_jvp=0 if estimator is None else nfe * estimator.jvp_per_call,
        t0=t0,
        t1=t1,
    )


def integrate_adaptive(
    vel_fn: VelocityFn,
    x0: torch.Tensor,
    *,
    solver: str = "dopri5",
    estimator: Optional[DivergenceEstimator] = None,
    t0: float = 0.0,
    t1: float = 1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    dlogp_tol_scale: float = 1.0,
    first_step: Optional[float] = None,
    max_steps: int = 10_000,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 10.0,
) -> ODEResult:
    """Adaptive Dormand-Prince 5(4) on the augmented ``(x, l)`` state.

    Step size is shared across the batch (as torchdyn/torchdiffeq do): the error norm is
    the RMS of ``(y5 - y4) / (atol + rtol * max(|y_prev|, |y_new|))`` over *every* element
    of the batch and state, so one rejected sample re-does the step for all of them. The
    log-density channel enters that norm divided by ``dlogp_tol_scale``, matching
    ``transferable_samplers.nn.wrappers.TorchDynWrapper``.

    FSAL is deliberately not used: with ``probe_schedule="per_step"`` the vector field
    changes between steps, so the last stage of one step is not the first stage of the
    next. Every step therefore costs 7 velocity evaluations.

    Probes are redrawn on *accepted* steps only. Within one attempt (including retries at a
    smaller step size) the field is held fixed, which is what makes the embedded error
    estimate meaningful; with ``per_step`` the object being solved is a randomised ODE, and
    the step count the controller lands on reflects that.
    """
    if solver != "dopri5":
        raise ValueError(f"Unknown adaptive solver {solver!r}; known: {ADAPTIVE_SOLVERS}")
    if not (atol > 0 and rtol > 0):
        raise ValueError("atol and rtol must be positive")

    span = t1 - t0
    direction = 1.0 if span >= 0 else -1.0
    h = abs(first_step) if first_step else abs(span) * 0.01
    h = min(h, abs(span))

    x = x0
    logdet = torch.zeros((x0.shape[0],), device=x0.device, dtype=x0.dtype)
    ones = torch.ones((x0.shape[0],), device=x0.device, dtype=x0.dtype)
    if estimator is not None:
        estimator.reset(x0)

    t = t0
    n_accepted = n_rejected = n_stage_evals = 0
    stages = len(_DP5_C)

    while (t1 - t) * direction > 1e-12:
        if n_accepted + n_rejected >= max_steps:
            raise RuntimeError(
                f"dopri5 exceeded max_steps={max_steps} at t={t:.6g} "
                f"({n_accepted} accepted, {n_rejected} rejected, h={h:.3g})"
            )
        h = min(h, abs(t1 - t))
        step = direction * h

        k_x: List[torch.Tensor] = []
        k_l: List[Optional[torch.Tensor]] = []
        for i in range(stages):
            xi = x
            for j, a_ij in enumerate(_DP5_A[i]):
                if a_ij != 0.0:
                    xi = xi + (step * a_ij) * k_x[j]
            ti = ones * (t + _DP5_C[i] * step)
            if estimator is None:
                k_x.append(vel_fn(ti, xi))
                k_l.append(None)
            else:
                v, div = estimator(vel_fn, ti, xi)
                k_x.append(v)
                k_l.append(-div)
            n_stage_evals += 1

        x5 = x
        x_err = torch.zeros_like(x)
        l5 = logdet
        l_err = torch.zeros_like(logdet)
        for i in range(stages):
            if _DP5_B5[i] != 0.0:
                x5 = x5 + (step * _DP5_B5[i]) * k_x[i]
            d = _DP5_B5[i] - _DP5_B4[i]
            if d != 0.0:
                x_err = x_err + (step * d) * k_x[i]
            if k_l[i] is not None:
                if _DP5_B5[i] != 0.0:
                    l5 = l5 + (step * _DP5_B5[i]) * k_l[i]
                if d != 0.0:
                    l_err = l_err + (step * d) * k_l[i]

        # Hairer's scaled RMS error norm, pooled over the whole batch.
        scale_x = atol + rtol * torch.maximum(x.abs(), x5.abs())
        sq_sum = ((x_err / scale_x) ** 2).sum()
        count = x_err.numel()
        if estimator is not None:
            scale_l = atol + rtol * torch.maximum(logdet.abs(), l5.abs()) / dlogp_tol_scale
            sq_sum = sq_sum + (((l_err / dlogp_tol_scale) / scale_l) ** 2).sum()
            count += l_err.numel()
        err = float(torch.sqrt(sq_sum / count))

        if err <= 1.0:
            t = t + step
            x, logdet = x5, l5
            n_accepted += 1
            if estimator is not None:
                estimator.new_step()  # redraw probes only once the step is committed
        else:
            n_rejected += 1

        factor = max_factor if err == 0.0 else safety * err ** (-0.2)
        h = h * min(max_factor, max(min_factor, factor))

    nfe = n_stage_evals
    return ODEResult(
        x=x,
        dlogp=None if estimator is None else logdet,
        n_steps=n_accepted,
        solver=solver,
        nfe_velocity=nfe,
        n_jvp=0 if estimator is None else nfe * estimator.jvp_per_call,
        t0=t0,
        t1=t1,
        extras={
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "accept_rate": n_accepted / max(1, n_accepted + n_rejected),
            "final_step_size": h,
            "atol": atol,
            "rtol": rtol,
            "dlogp_tol_scale": dlogp_tol_scale,
        },
    )


def expected_cost(solver: str, n_steps: int, divergence_type: str, *, dim: int,
                  n_mc_samples: int = 1) -> Dict[str, int]:
    """Analytic NFE / JVP counts for a run -- the formula the tests pin down."""
    stages = n_stages(solver) * n_steps
    per_call = dim if divergence_type == "exact" else n_mc_samples
    return {"nfe_velocity": stages, "n_jvp": stages * per_call}


__all__ = [
    "VelocityFn",
    "SOLVERS",
    "FIXED_SOLVERS",
    "ADAPTIVE_SOLVERS",
    "integrate_adaptive",
    "PROBE_DISTS",
    "PROBE_SCHEDULES",
    "n_stages",
    "draw_probes",
    "DivergenceEstimator",
    "HutchinsonEstimator",
    "ExactEstimator",
    "make_estimator",
    "ODEResult",
    "integrate",
    "expected_cost",
]
