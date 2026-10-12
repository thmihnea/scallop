"""transferable-samplers evaluation wired into iF2D2's training loop.

Builds an EvalContext (val reference + OpenMM TargetEnergy + TICA + topology) once,
evaluates raw proposals for pretraining, or uses their SNISSampler for likelihood models, and
drives PeptideEnsembleEvaluator (ESS, energy/torus/tica W2, tica/torus kmeans-JSD,
chirality; rama/energy/atom-dist/com-norm/tica plots), logging metrics + figures to
wandb the way their callback does.
"""
from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch

import torch

from .tsx import (
    PeptideEnsembleEvaluator,
    SNISSampler,
    SamplesData,
    SourceEnergy,
    EvalContext,
)
from .se3_utils import remove_mean
from .com_lift import importance_source_energy

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def _wandb_active() -> bool:
    return wandb is not None and getattr(wandb, "run", None) is not None


def _wandb_safe(fn, *a, **kw):
    """Shared with logging.py: a wandb failure must never abort an eval or a run."""
    from .logging import _wandb_safe as _impl

    return _impl(fn, *a, **kw)


def build_eval_context(cfg, val_physical: torch.Tensor, com_aware: bool = False,
                       num_true: int = 10_000, num_tica: int | None = None,
                       energy_fn=None) -> EvalContext:
    """Build held-out reference, energy, topology and TICA through the TS adapter."""
    from . import ts_systems

    return ts_systems.build_eval_context(
        ts_systems.resolve(cfg), split="val", energy_fn=energy_fn,
        num_eval_samples=num_true,
        data_dir=getattr(cfg.problem, "ts_data_dir", None) or None,
        normalization_std=float(cfg.network.unscale_factor) if com_aware else 1.0,
        num_tica=num_tica, reference_samples=val_physical,
    )


@torch.no_grad()
def sample_model(net, x0, n_steps=1, batch_size=None, velocity_solver="dopri5",
                 velocity_atol=1e-5, velocity_rtol=1e-5, time_discretization_type="linear",
                 output_device=None):
    """Physical-nm proposals, optional log q, and the actual number of field calls.

    Velocity pretraining learns the diagonal b(t,x), so it must be integrated
    along s=t. Flow-map and likelihood models use their learned finite jumps.
    No divergence or importance weights are invented for pretraining models.
    """
    mode = getattr(net.config, "divergence_mode", "pretrain_flowmap")
    bs = len(x0) if batch_size is None else int(batch_size)
    if len(x0) == 0 or bs < 1 or n_steps < 1:
        raise ValueError("sample count, batch_size and n_steps must be positive")
    samples, logps, nfe = [], [], 0
    for batch in x0.split(bs):
        if mode == "pretrain_v":
            from .ode_likelihood import integrate

            kwargs = ({"atol": velocity_atol, "rtol": velocity_rtol}
                      if velocity_solver == "dopri5" else {})
            if time_discretization_type == "linear":
                result = integrate(lambda t, x: net.calc_b(t, x, train=False), batch,
                                   solver=velocity_solver, n_steps=n_steps, **kwargs)
                x, calls = result.x, result.nfe_velocity
            else:
                from .time_discretization import time_grid

                if velocity_solver == "dopri5":
                    raise ValueError("A time discretization requires a fixed-step velocity solver")
                grid = time_grid(n_steps, time_discretization_type)
                x, calls = batch, 0
                for left, right in zip(grid[:-1], grid[1:]):
                    result = integrate(lambda t, x: net.calc_b(t, x, train=False), x,
                                       solver=velocity_solver, n_steps=1, t0=left, t1=right)
                    x, calls = result.x, calls + result.nfe_velocity
            x, logp = net.unscale_samples(x), None
        else:
            schedule = ({"time_discretization_type": time_discretization_type}
                        if time_discretization_type != "linear" else {})
            x, logp, calls = net.sample_forward(batch, n_steps=n_steps, return_nfe=True, **schedule)
        samples.append(x.to(output_device) if output_device is not None else x)
        if logp is not None:
            logps.append(logp.to(output_device) if output_device is not None else logp)
        nfe += calls
    return torch.cat(samples), torch.cat(logps) if logps else None, nfe


@torch.no_grad()
def _get_ema_state_dict(state, ema_fac):
    if ema_fac is not None and ema_fac in state.ema_params:
        return state.ema_params[ema_fac]
    return {n: p.detach() for n, p in state.net.named_parameters()}


def _compute_mean_metrics(metrics: dict, stage: str) -> dict:
    """Per-sequence mean under {stage}/mean/... (drops stage+sequence key segments)."""
    groups: dict[str, list] = defaultdict(list)
    for k, v in metrics.items():
        parts = k.split("/")
        if len(parts) >= 3 and parts[0] == stage and isinstance(v, (int, float)):
            groups[f"{stage}/mean/" + "/".join(parts[2:])].append(v)
    return {k: sum(vs) / len(vs) for k, vs in groups.items()}


@torch.no_grad()
def ts_evaluate(cfg, state, device, step: int, eval_ctx: EvalContext, com_augmentation: bool = False,
                prefix: str | None = None) -> dict:
    """Run the transferable-samplers evaluation on the EMA model; log to wandb.

    ``prefix`` defaults to ``val/{ts sequence}`` for the config's target, so metrics are
    labelled by the system actually being evaluated (``val/Ace-A-Nme`` for ace_a_nme, as
    before; ``val/AAAAAA`` for ala6_implicit, ...).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if prefix is None:
        from . import ts_systems

        prefix = f"val/{ts_systems.resolve(cfg).sequence}"

    net = state.net
    n_part, n_dim = cfg.network.n_particles, cfg.network.n_dimension
    rescale = float(cfg.network.rescale)
    n_steps = int(cfg.logging.n_sampling_steps[0])
    num_samples = int(cfg.logging.plot_bs)
    mode = getattr(cfg.network, "divergence_mode", "")
    has_likelihood = bool(net.dit.predict_divergence) and mode not in ("pretrain_v", "pretrain_flowmap")
    sample_batch_size = int(getattr(cfg.logging, "sample_batch_size", num_samples))
    sampling_nfe = []

    ema_fac = getattr(cfg.logging, "visual_ema_factor", None)
    if ema_fac is None and state.ema_facs:
        ema_fac = state.ema_facs[0]

    gen = torch.Generator(device=device)
    gen.manual_seed(int(getattr(cfg.logging, "evaluation_seed",
                                int(getattr(cfg.training, "seed", 0)) + 4321)))
    uf = float(getattr(cfg.network, "unscale_factor", 1.0))

    # Audit N5: the upstream SNISSampler applies logw_quantile_filter BEFORE forming
    # the logw it attaches to the resampled bucket, so the ESS the evaluator reports is
    # computed after the top `clip_fraction` of importance weights are discarded. That
    # filtered ESS is the project's standard metric; what N5 asked for is only that the
    # discard be VISIBLE. Capture E_source here so the unfiltered logw can be attached
    # to the proposal bucket and reported as a diagnostic alongside it.
    # Same pattern as eval_ode_int.py (ess + ess_after_filter).
    proposal_E_source: list[torch.Tensor] = []

    def sample_fn(n, **_):
        x0 = rescale * torch.randn((n, n_part, n_dim), generator=gen, device=device)
        if not com_augmentation:
            x0 = remove_mean(x0)  # com-aware base keeps the Gaussian COM
        x1, logp, nfe = sample_model(
            net, x0, n_steps=n_steps, batch_size=sample_batch_size,
            velocity_solver=getattr(cfg.logging, "velocity_solver", "dopri5"),
            velocity_atol=float(getattr(cfg.logging, "velocity_atol", 1e-5)),
            velocity_rtol=float(getattr(cfg.logging, "velocity_rtol", 1e-5)),
        )
        sampling_nfe.append(nfe)
        if has_likelihood:
            if logp is None:
                raise RuntimeError("Likelihood model returned no proposal log density")
            # Correct once, at the endpoint, using physical coordinates and log q.
            # Both Hutchinson and dlogpdt heads share this SNIS boundary.
            e_source = importance_source_energy(
                x1, logp, com_augmentation=com_augmentation, unscale_factor=uf,
            ).detach().cpu()
            proposal_E_source.append(e_source)
        else:
            e_source = None
        # The com-aware context expects standardized coordinates and restores nm
        # through normalization_std=uf. A constant density offset cancels in SNIS.
        if com_augmentation:
            x1 = x1 / uf
        return x1.detach().cpu(), e_source

    clip_fraction = float(getattr(cfg.logging, "importance_weight_clip", 0.0)) or 0.0

    backup = {n: p.detach().clone() for n, p in net.named_parameters()}
    was_training = net.training
    try:
        net.eval()
        net.load_state_dict(_get_ema_state_dict(state, ema_fac), strict=False)
        if has_likelihood:
            source = SourceEnergy(
                sample_fn=sample_fn,
                energy_fn=lambda x: torch.zeros(x.shape[0]),  # unused by SNIS.sample
                sample_batch_size=num_samples, energy_batch_size=num_samples,
                # Already corrected above; upstream still uses the radial chi-3 form.
                grad_batch_size=num_samples, use_com_adjustment=False,
            )
            sampler = SNISSampler(num_samples=num_samples, logw_quantile_filter=clip_fraction or None)
            # Training evaluates synchronously on rank zero while peers wait.
            # Keep upstream SNIS filtering/resampling, but make its collectives
            # local; its default DDP path shards samples and gathers CPU tensors.
            import transferable_samplers.samplers.snis_sampler as snis_module
            # ponytail: module-scoped override assumes synchronous evaluation;
            # use upstream process-group support if threaded evaluation is added.
            with patch.multiple(snis_module, get_world_size=lambda: 1, get_rank=lambda: 0,
                                all_gather_cat=lambda x: x, broadcast_tensor=lambda x, src: x):
                samples_dict, _ = sampler.sample(source, eval_ctx.target_energy)
        else:
            proposals, _ = sample_fn(num_samples)
            energy = eval_ctx.target_energy.energy(proposals)
            samples_dict = {"proposal": SamplesData(proposals, energy, logw=None)}
    finally:
        net.load_state_dict(backup, strict=False)
        net.train(was_training)

    # Attach the unfiltered logw to the (pre-filter) proposal bucket. SNISSampler
    # builds proposal_data before it clips, so its samples/E_target line up with the
    # E_source captured above, in order.
    proposal = samples_dict["proposal"]
    if proposal_E_source:
        e_source = torch.cat(proposal_E_source, dim=0)
        if len(e_source) == len(proposal):
            samples_dict["proposal"] = SamplesData(
                proposal.samples, proposal.E_target, logw=e_source - proposal.E_target
            )
        else:  # pragma: no cover - defensive; batching should preserve length
            print(f"[ts_eval] unfiltered logw skipped: captured {len(e_source)} "
                  f"E_source values for {len(proposal)} proposals")

    def log_image_fn(fig, title, title_prefix=""):
        key = f"{title_prefix}/{title}" if title_prefix else title
        if _wandb_active():
            _wandb_safe(wandb.log, {key: wandb.Image(fig)}, step=step)
        plt.close(fig)

    evaluator = PeptideEnsembleEvaluator(num_eval_samples=num_samples,
                                         do_plots=bool(getattr(cfg.logging, "do_plots", True)))
    raw = evaluator.evaluate(samples_dict, eval_ctx, log_image_fn=log_image_fn, prefix=prefix)

    scalars = {}
    for k, v in raw.items():
        try:
            scalars[k] = float(v)
        except (TypeError, ValueError):
            continue
    # PROJECT STANDARD (user decision, 2026-09-04): logw_quantile_filter =
    # cfg.logging.importance_weight_clip = 0.002 applies to every importance-sampling
    # metric. The reported ESS is therefore {prefix}/resampled/effective-sample-size,
    # which the upstream SNISSampler computes from the POST-filter logw. Audit N5 asked
    # for the discard to be visible, not removed, so the unfiltered value is kept under
    # an explicit `-unfiltered` suffix beside the clipped fraction. Nothing selects on
    # the unfiltered key -- see logging.log_metrics.
    unfiltered = scalars.pop(f"{prefix}/proposal/effective-sample-size", None)
    if unfiltered is not None:
        scalars[f"{prefix}/proposal/effective-sample-size-unfiltered"] = unfiltered
    if has_likelihood:
        scalars[f"{prefix}/proposal/logw-clip-fraction"] = clip_fraction
        scalars[f"{prefix}/resampled/n-after-filter"] = float(len(samples_dict["resampled"]))
    scalars[f"{prefix}/sampling/nfe"] = float(sum(sampling_nfe))
    scalars.update(_compute_mean_metrics(scalars, prefix.split("/")[0]))
    if _wandb_active():
        _wandb_safe(wandb.log, scalars, step=step)
    return scalars
