"""Shared proposal filtering, resampling, and physical MALA evaluation stages."""
import numpy as np

# Pinned TS single-system TarFlow SMC evaluation cutoffs, in U/kT.
ENERGY_CUTOFFS_KT = {"ace_a_nme": 10.0, "ace_aaa_nme": 50.0,
                     "ala3_implicit": -120.0, "ala6_implicit": -20.0}


def resampling_indices(logw, n_samples, method="multinomial"):
    """Draw an unweighted ensemble with expected counts n_samples * softmax(logw).

    Stratified draws put one independent uniform in each equal-probability
    stratum, then invert the weighted CDF. The caller controls proposal order.
    Duplicate proposals remain valid; no importance weights are changed.
    """
    import torch

    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if (logw.ndim != 1 or not len(logw) or not logw.is_floating_point()
            or torch.isnan(logw).any() or torch.isposinf(logw).any()
            or not torch.isfinite(logw).any()):
        raise ValueError("logw must be a nonempty vector of finite values or -inf, with positive mass")
    if method == "multinomial":
        return torch.multinomial(torch.softmax(logw, dim=0), n_samples, replacement=True)
    if method != "stratified":
        raise ValueError(f"Unknown resampling method {method!r}")
    # Double precision keeps small CDF intervals and the final stratum accurate.
    cdf = torch.softmax(logw.double(), dim=0).cumsum(0)
    cdf = cdf / cdf[-1]
    positions = (torch.arange(n_samples, dtype=cdf.dtype, device=logw.device)
                 + torch.rand(n_samples, dtype=cdf.dtype, device=logw.device)) / n_samples
    positions.clamp_(max=torch.nextafter(cdf.new_tensor(1.), cdf.new_tensor(0.)))
    # right=True skips zero-mass intervals, including an initial zero CDF.
    return torch.searchsorted(cdf, positions, right=True)


def sample_stages(x, energy_values, logw, *, clip_fraction, energy, mala_steps, mala_step_size,
                  n_eval_samples, energy_cutoff, resampling="multinomial",
                  mala_adapt_steps=None, mala_target_accept=0.6, recipe_v1=False,
                  checkpoint_interval=128, artifacts=None):
    """Compare SNIS/MALA with and without a pre-resampling energy cutoff.

    Both branches share the complete proposal/energy pool. The cutoff precedes
    the TS log-weight filter. ESS uses all retained weights; only n_eval_samples
    weighted draws are materialized for metrics and MALA. Proposal metrics use a
    uniform subset of finite proposals. Relaxation never reapplies the cutoff:
    relaxed_energy_cutoff denotes cutoff-filtered initialization, not a truncated
    Boltzmann target. Identical pools share their resampling and MALA computation.
    Stratification orders the retained pool by energy without changing weights.
    MALA tunes one shared tau across both branches during the existing warmup
    steps, then freezes it; the combined chain batch keeps force costs unchanged.
    """
    import torch
    from common.mala import mala_sample
    from common.se3_utils import remove_mean
    from common.tsx import SamplesData, normalized_ess

    if resampling not in ("stratified", "multinomial"):
        raise ValueError(f"Unknown resampling method {resampling!r}")
    if mala_adapt_steps is None:
        mala_adapt_steps = mala_steps // 2
    if not isinstance(mala_adapt_steps, int) or not 0 <= mala_adapt_steps <= mala_steps:
        raise ValueError("mala_adapt_steps must be an integer in [0, mala_steps]")
    if not 0 < mala_target_accept < 1:
        raise ValueError("mala_target_accept must be in (0, 1)")
    if n_eval_samples < 1 or (not recipe_v1 and not np.isfinite(energy_cutoff)):
        raise ValueError("n_eval_samples must be positive and energy_cutoff finite")
    valid = torch.isfinite(x).flatten(1).all(1) & torch.isfinite(energy_values)
    indices = torch.where(valid)[0]
    if not len(indices):
        raise ValueError("No finite proposals available for TS evaluation")
    if recipe_v1 and len(indices) < n_eval_samples:
        raise ValueError("Fewer finite proposals than the recipe evaluation budget")
    if artifacts is not None:
        artifacts["rng_before_subsampling"] = torch.get_rng_state()
    selected = indices[torch.randperm(len(indices), device=x.device)[:n_eval_samples]]
    if artifacts is not None:
        artifacts["proposal_indices"] = selected.clone()
    stages = {"proposal": SamplesData(x[selected], energy_values[selected])}
    diagnostics = {"proposal": {"n-samples": len(selected), "n-generated": len(x),
                                "n-finite": len(indices), "n-nonfinite": int((~valid).sum())}}
    if logw is None:
        return stages, diagnostics
    eligible = valid & torch.isfinite(logw)
    if not eligible.any():
        raise ValueError("No finite importance weights available for resampling")
    indices = torch.where(eligible)[0]
    unfiltered_ess = float(normalized_ess(logw[indices]))
    diagnostics["proposal"].update({
        "effective-sample-size-unfiltered": unfiltered_ess,
        "ess-unfiltered": unfiltered_ess * len(indices),
        "n-eligible-weights": len(indices),
        "n-invalid-weights": int((valid & ~torch.isfinite(logw)).sum()),
    })
    kept = indices if recipe_v1 else indices[energy_values[indices] < energy_cutoff]
    same_pool = len(kept) == len(indices)
    pools = [("resampled", indices)]
    if not recipe_v1:
        pools.append(("resampled_energy_cutoff", kept))
    for name, pool in pools:
        cutoff = energy_cutoff if name.endswith("energy_cutoff") else None
        stats = {
            "n-eligible-weights": len(indices), "n-after-energy-cutoff": len(pool),
            "n-energy-discarded": len(indices) - len(pool),
            "energy-discard-fraction": 1 - len(pool) / len(indices),
            "energy-cutoff-kT": cutoff, "logw-clip-fraction": clip_fraction,
            "resampling-method": resampling,
            "resampling-order": "energy" if resampling == "stratified" else "input",
        }
        if not len(pool):
            # Keep the other categories usable when a poor model has no survivors.
            stages[name] = SamplesData(x[:0], energy_values[:0])
            diagnostics[name] = dict(stats, **{"n-samples": 0, "n-after-filter": 0,
                "effective-sample-size": None, "ess": None, "status": "empty-after-energy-cutoff"})
            continue
        if cutoff is not None and same_pool:
            stages[name] = stages["resampled"]
            diagnostics[name] = {**diagnostics["resampled"], **stats}
            continue
        weights = logw[pool]
        before_filter_ess = float(normalized_ess(weights))
        threshold = torch.quantile(weights, 1 - clip_fraction) if clip_fraction else None
        if clip_fraction:
            keep = weights <= threshold
            pool, weights = pool[keep], weights[keep]
            threshold = float(threshold)
        if resampling == "stratified":
            order = torch.argsort(energy_values[pool], stable=True)
            pool, weights = pool[order], weights[order]
        if artifacts is not None:
            artifacts["rng_before_resampling"] = torch.get_rng_state()
        chosen = pool[resampling_indices(weights, n_eval_samples, resampling)]
        if artifacts is not None:
            artifacts["resampled_indices"] = chosen.clone()
            artifacts["filtered_indices"] = pool.clone()
        stages[name] = SamplesData(x[chosen], energy_values[chosen])
        filtered_ess = float(normalized_ess(weights))
        diagnostics[name] = {**stats,
            "n-samples": len(chosen), "n-after-filter": len(weights),
            "effective-sample-size": filtered_ess, "ess": filtered_ess * len(weights),
            "effective-sample-size-unfiltered": before_filter_ess,
            "logw-filter-threshold": threshold,
            "n-quantile-discarded": stats["n-after-energy-cutoff"] - len(weights),
        }

    if mala_steps:
        def log_prob_and_grad(coords):
            e, grad = energy.energy_and_grad(coords)
            return -e, -grad

        # One batched MALA call for the two evaluation-sized branches. Duplicated
        # draws remain independent chains; deduplicating them would change that.
        names = ["resampled"] + ([] if same_pool else ["resampled_energy_cutoff"])
        sizes = [len(stages[name]) for name in names]
        initial = torch.cat([stages[name].samples for name in names])
        if artifacts is not None:
            artifacts["rng_before_mala"] = torch.get_rng_state()
        checkpoint_steps = list(range(checkpoint_interval, mala_steps + 1, checkpoint_interval)) if recipe_v1 else None
        relaxed, info = mala_sample(
            remove_mean(initial), log_prob_and_grad, mala_steps, mala_step_size,
            project_com=True, adapt=mala_adapt_steps > 0, adapt_steps=mala_adapt_steps,
            target_accept=mala_target_accept, scheduler="robbins_monro", adapt_per_chain=False,
            checkpoint_steps=checkpoint_steps,
        )
        if recipe_v1:
            for k in checkpoint_steps:
                name = f"relaxed-{k}"
                stages[name] = SamplesData(info["checkpoints"][k], -info["checkpoint_logp"][k])
                diagnostics[name] = {
                    "n-samples": n_eval_samples, "mala-steps": k,
                    "mala-adapt-steps": mala_adapt_steps, "mala-includes-adaptation": k <= mala_adapt_steps,
                    "mala-target-accept": mala_target_accept,
                    "mala-step-size-nm2": float(info["step_size"][k - 1].mean()),
                    "mala-next-step-size-nm2": float((info["step_size"][k] if k < mala_steps else info["final_step_size"]).mean()),
                    "mala-acceptance-rate": float(info["checkpoint_acceptance_rate"][k].mean()),
                    "mala-mean-accept-prob": float(info["accept_prob"][:k].mean()),
                    "mala-interval-mean-accept-prob": float(info["accept_prob"][k-checkpoint_interval:k].mean()),
                }
            if artifacts is not None:
                artifacts["mala"] = info
                artifacts["rng_after_mala"] = torch.get_rng_state()
            return stages, diagnostics
        offset = 0
        for name, size in zip(names, sizes):
            target_name = name.replace("resampled", "relaxed", 1)
            span = slice(offset, offset + size)
            stages[target_name] = SamplesData(relaxed[span], -info["final_logp"][span])
            diagnostics[target_name] = {
                "n-samples": size, "mala-steps": mala_steps,
                "initial-energy-cutoff-kT": diagnostics[name]["energy-cutoff-kT"],
                "mala-step-size-nm2": mala_step_size,
                "mala-adapt-steps": mala_adapt_steps,
                "mala-target-accept": mala_target_accept,
                "mala-final-step-size-nm2": float(info["final_step_size"][span].mean()) if size else None,
                "mala-post-adapt-mean-accept-prob": (
                    float(info["accept_prob"][mala_adapt_steps:, span].mean())
                    if size and mala_adapt_steps < mala_steps else None),
                "mala-acceptance-rate": float(info["acceptance_rate"][span].mean()) if size else None,
                "mala-mean-accept-prob": float(info["accept_prob"][:, span].mean()) if size else None,
            }
            if not size:
                diagnostics[target_name]["status"] = "empty-after-energy-cutoff"
            offset += size
        if same_pool:
            stages["relaxed_energy_cutoff"] = stages["relaxed"]
            diagnostics["relaxed_energy_cutoff"] = {
                **diagnostics["relaxed"], "initial-energy-cutoff-kT": energy_cutoff,
            }
    return stages, diagnostics
