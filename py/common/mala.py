"""Batched MALA for a direct target, e.g. log p(x) = -U(x)/(k_B T).

The fixed-step proposal is x' = x + tau grad log p(x) + sqrt(2 tau) noise.
Its MH ratio includes both forward and reverse Gaussian proposal densities.
No diffusion time, observation, or posterior likelihood is involved.
"""
from __future__ import annotations

import torch


def _geometric_scheduler(omega, accept_prob, iteration, target_accept,
                         lr_up=0.1, lr_down=0.5):
    """Per-chain geometric step on log(ω), biased toward high acceptance.

    ω ← ω · exp(lr · (α − target)) with asymmetric lr: small steps up when
    accept is too high, large steps down when accept is too low. Keeps short
    runs in a high-acceptance regime — a stalled chain is worse than a
    slightly conservative one.
    """
    delta = accept_prob - target_accept
    lr = torch.where(delta > 0, torch.full_like(delta, lr_up), torch.full_like(delta, lr_down))
    return omega * torch.exp(lr * delta)


def _robbins_monro_scheduler(omega, accept_prob, iteration, target_accept,
                             lr=1.0, t0=10.0, kappa=0.75):
    """Robbins-Monro multiplicative update (tau in MALA, omega in MAExL).

    ω ← ω · exp(−η_t · (target − α)), η_t = lr / (iteration + t0)^kappa.
    Diminishing step size; standard 0.5 < kappa ≤ 1.0 keeps variance bounded.
    """
    error = target_accept - accept_prob
    eta = lr / ((iteration + t0) ** kappa)
    return omega * torch.exp(-eta * error)


_SCHEDULERS = {
    "geometric": _geometric_scheduler,
    "robbins_monro": _robbins_monro_scheduler,
}


@torch.no_grad()
def mala_sample(
    x_init, log_prob_and_grad_fn, num_steps, step_size,
    target_accept=0.6, scheduler="robbins_monro", scheduler_kwargs=None,
    checkpoint_steps=None, return_info=True, on_checkpoint=None,
    metropolis=True, adapt=False, project_com=False, iteration_offset=0,
    adapt_steps=None, adapt_per_chain=False,
):
    """Sample a direct target using one chain per batch row.

    The callback returns (logp[B], grad_logp[B, ...]) in x's coordinate units.
    For TS physical coordinates, return -E/kT and the force in kT/nm.
    ``step_size`` is tau (noise variance = 2*tau), scalar or per-chain [B].
    ``project_com`` restricts molecular coordinates, gradients and noise to
    the zero-centroid subspace; the same Gaussian MH ratio holds there.

    Fixed steps (default) preserve the target. With adapt=True, tune for
    adapt_steps (default: half of num_steps), then freeze. Updates use the
    batch mean MH acceptance probability; adapt_per_chain=True instead tunes
    each chain separately. A scalar initial tau stays shared across the batch.
    Warmup counts toward num_steps: no extra target evaluations are made.
    Adaptation is a warmup heuristic, not an exactly invariant finite-run kernel. ULA
    (metropolis=False) is biased and requires adapt=False.
    Snapshots/callbacks are recorded after the specified checkpoint steps.
    """
    if x_init.ndim < 2 or not len(x_init) or not x_init.is_floating_point():
        raise ValueError("x_init must be a nonempty floating-point batch")
    if not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    if project_com and (x_init.ndim != 3 or x_init.shape[-1] != 3):
        raise ValueError("project_com requires (batch, atoms, 3) coordinates")
    if adapt and not metropolis:
        raise ValueError("adapt=True requires metropolis=True")
    if adapt_steps is None:
        adapt_steps = num_steps // 2 if adapt else 0
    if not isinstance(adapt_steps, int) or not 0 <= adapt_steps <= num_steps:
        raise ValueError("adapt_steps must be an integer in [0, num_steps]")
    if adapt_steps and not adapt:
        raise ValueError("Positive adapt_steps requires adapt=True")
    if not 0 < target_accept < 1:
        raise ValueError("target_accept must be in (0, 1)")
    if scheduler not in _SCHEDULERS:
        raise ValueError(f"Unknown scheduler {scheduler!r}; expected {list(_SCHEDULERS)}")
    B = len(x_init)
    shape = (B,) + (1,) * (x_init.ndim - 1)
    axes = tuple(range(1, x_init.ndim))
    tau = torch.as_tensor(step_size, device=x_init.device, dtype=x_init.dtype)
    if tau.ndim == 1 and tau.shape == (B,):
        tau = tau.reshape(shape)
    tau = torch.broadcast_to(tau, shape).clone()
    if not torch.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("step_size must be finite and positive")

    def project(v):
        return v - v.mean(dim=1, keepdim=True) if project_com else v

    def evaluate(x):
        logp, grad = log_prob_and_grad_fn(x)
        if logp.shape != (B,) or grad.shape != x.shape:
            raise ValueError("Target callback must return logp[B] and grad with x's shape")
        return logp.detach(), project(grad.detach())

    x = project(x_init.detach().clone())
    logp, grad = evaluate(x)
    if not (torch.isfinite(x).all() and torch.isfinite(logp).all() and torch.isfinite(grad).all()):
        raise ValueError("Initial states must have finite coordinates, log density and gradient")
    checkpoint_set = set(checkpoint_steps or [])
    if any(not isinstance(k, int) or not 0 <= k <= num_steps for k in checkpoint_set):
        raise ValueError("checkpoint_steps must contain integers in [0, num_steps]")
    checkpoints, checkpoint_logp, checkpoint_acceptance_rate = {}, {}, {}
    accepted_count = x.new_zeros(B)

    def record(k):
        if k in checkpoint_set:
            checkpoints[k], checkpoint_logp[k] = x.clone(), logp.clone()
            checkpoint_acceptance_rate[k] = accepted_count.clone() / max(1, k)
            if on_checkpoint is not None:
                on_checkpoint(k, checkpoints[k], checkpoint_logp[k])

    record(0)
    accept_hist = x.new_empty(num_steps, B) if return_info and metropolis else None
    tau_hist = x.new_empty(num_steps, B) if return_info else None
    for m in range(num_steps):
        if tau_hist is not None:
            tau_hist[m] = tau.reshape(B)
        noise = project(torch.randn_like(x))
        candidate = project(x + tau * grad + torch.sqrt(2 * tau) * noise)
        new_logp, new_grad = evaluate(candidate)
        if metropolis:
            forward_sq = (candidate - x - tau * grad).square().sum(axes)
            reverse_sq = (x - candidate - tau * new_grad).square().sum(axes)
            log_ratio = new_logp - logp + (forward_sq - reverse_sq) / (4 * tau.reshape(B))
            valid = (torch.isfinite(new_logp) & torch.isfinite(new_grad).flatten(1).all(1)
                     & torch.isfinite(candidate).flatten(1).all(1) & ~torch.isnan(log_ratio))
            probability = torch.where(valid, log_ratio.clamp(max=0).exp(), 0.0)
            accept = torch.rand_like(probability) < probability
            # where is essential: multiplying a rejected NaN by zero still gives NaN.
            x = torch.where(accept.reshape(shape), candidate, x)
            grad = torch.where(accept.reshape(shape), new_grad, grad)
            logp = torch.where(accept, new_logp, logp)
            accepted_count += accept
            if accept_hist is not None:
                accept_hist[m] = probability
            if m < adapt_steps:
                tau = _SCHEDULERS[scheduler](
                    tau, probability.reshape(shape) if adapt_per_chain else probability.mean(),
                    m + iteration_offset,
                    target_accept, **(scheduler_kwargs or {}),
                )
                if not torch.isfinite(tau).all() or (tau <= 0).any():
                    raise ValueError("Step-size adaptation produced an invalid step size")
        else:
            if not (torch.isfinite(candidate).all() and torch.isfinite(new_logp).all()
                    and torch.isfinite(new_grad).all()):
                raise ValueError("Unadjusted Langevin produced nonfinite states")
            x, logp, grad = candidate, new_logp, new_grad
        record(m + 1)
    if not return_info:
        return x
    info = {"step_size": tau_hist, "final_step_size": tau.reshape(B), "final_logp": logp,
            "adapt_steps": adapt_steps}
    if metropolis:
        info.update(accept_prob=accept_hist, acceptance_rate=accepted_count / max(1, num_steps))
    if checkpoint_set:
        info.update(checkpoints=checkpoints, checkpoint_logp=checkpoint_logp,
                    checkpoint_acceptance_rate=checkpoint_acceptance_rate)
    return x, info
