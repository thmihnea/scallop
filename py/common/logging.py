"""Minimal logging / checkpointing for the torch training loop.

The JAX ``logging.py`` is ~1.7k lines of plotting/FID/sampling. For this
molecule branch we keep the essential pieces: scalar logging (optionally to
wandb), periodic checkpointing, signal-handler saves, and an optional periodic
sample-based evaluation against the target distribution.
"""

import json
import math
import os
import signal
from typing import Dict, Optional

import torch

from . import distributed

try:
    import wandb  # noqa: WPS433
except Exception:  # pragma: no cover
    wandb = None


def _wandb_active() -> bool:
    return wandb is not None and getattr(wandb, "run", None) is not None


def _wandb_safe(fn, *a, **kw):
    """Run a wandb call; on any failure warn once and carry on.

    Logging must never be able to kill a multi-day training run. EXPERIMENT_LOG
    2026-08-09 records a completed 72 h integration lost to a wandb
    ServiceStartTimeoutError (the /tmp reaper deleted wandb's port file mid-run), and
    2026-09-03 a chain launch killed at step 0 by the same error. ``eval_ode_int.py``
    grew this guard then; the training path did not. Same pattern, same reason.
    """
    try:
        return fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001 - logging is best effort, always
        if not getattr(_wandb_safe, "_warned", False):
            print(f"[wandb] WARNING: logging failed ({type(exc).__name__}: {exc}); "
                  f"continuing without it. Checkpoints and stdout are unaffected.")
            _wandb_safe._warned = True
        return None


def save_state(state, cfg, name: Optional[str] = None) -> str:
    from . import state_utils

    out_dir = cfg.logging.output_folder or "."
    os.makedirs(out_dir, exist_ok=True)
    name = name or getattr(cfg.logging, "output_name", "model")
    path = os.path.join(out_dir, f"{name}.pt")
    state_utils.save_state(state, path)
    print(f"Saved checkpoint to {path}")
    return path


def register_signal_handlers(cfg, state):
    def _handler(signum, frame):
        try:
            if distributed.is_main():
                save_state(state, cfg, name=f"{getattr(cfg.logging, 'output_name', 'model')}_signal")
        finally:
            # Interrupted stages must not look successful to a sequential launcher.
            raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


@torch.no_grad()
def evaluate_samples(cfg, state, device, step: int = 0) -> Dict[str, float]:
    """Evaluate every training mode through the upstream peptide evaluator."""
    from . import ts_eval

    return ts_eval.ts_evaluate(
        cfg, state, device, step, state.ts_eval_ctx,
        com_augmentation=getattr(state, "ts_com_aug", False),
    )


def _find_metric(scalars, *substrs):
    """Return the scalar whose key contains all substrs (e.g. the resampled ESS).

    Deterministic: matches are sorted, and keys under a ``/mean/`` aggregate are only
    used if nothing else matches. The old version returned the first dict-insertion
    match, which was fine while exactly one ESS key existed and is not any more.
    """
    matches = [
        (k, float(v)) for k, v in scalars.items()
        if all(sub in k for sub in substrs)
        and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not matches:
        return None
    matches.sort(key=lambda kv: ("/mean/" in kv[0], kv[0]))
    return matches[0][1]


def log_metrics(cfg, state, device, loss_value, metrics, step, elapsed):
    scalars = {"train/loss": float(loss_value), "train/step_time": float(elapsed)}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu().item()) if v.numel() == 1 else float(v.detach().cpu().mean())
        scalars[k if "/" in k else f"train/{k}"] = v

    out_name = getattr(cfg.logging, "output_name", "model")

    eval_scalars = {}
    visual_freq = getattr(cfg.logging, "visual_freq", 0)
    if getattr(state, "ts_eval_ctx", None) is not None and visual_freq and step > 0 and step % visual_freq == 0:
        eval_scalars = evaluate_samples(cfg, state, device, step=step)
        scalars.update(eval_scalars)

    # Best-checkpoint tracking on the val eval (ESS / resampled energy-w2).
    if eval_scalars:
        best = state.best_metrics
        candidates = [
            ("ess", "best/ess", "ess", _find_metric(eval_scalars, "resampled", "effective-sample-size"), True),
            ("energy_w2", "best/resampled_energy_w2", "energy_w2", _find_metric(eval_scalars, "resampled", "energy-w2"), False),
        ]
        network_cfg = getattr(cfg, "network", None)
        if network_cfg is not None and not getattr(network_cfg, "predict_divergence", True):
            candidates.append(("proposal_energy_w2", "best/proposal_energy_w2", "proposal_energy_w2",
                               _find_metric(eval_scalars, "proposal", "energy-w2"), False))
        improved = []
        for key, metric, suffix, value, maximize in candidates:
            if value is None or not math.isfinite(value):
                continue
            previous = best.get(key)
            if (previous is None or not math.isfinite(previous)
                    or (value > previous if maximize else value < previous)):
                best[key] = value
                improved.append(suffix)
            scalars[metric] = best[key]
        # Every best file must contain all thresholds from this evaluation.
        for suffix in improved:
            save_state(state, cfg, name=f"{out_name}_best_{suffix}")
            print(f"[best] new best {suffix}={best[suffix]:.5g} @ step {step}")

    # Durable local scalars remain available if the W&B service drops mid-run.
    out_dir = getattr(cfg.logging, "output_folder", None)
    if out_dir is not None:
        os.makedirs(out_dir or ".", exist_ok=True)
        with open(os.path.join(out_dir or ".", "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, **scalars}) + "\n")
    if _wandb_active():
        _wandb_safe(wandb.log, scalars, step=step)

    save_freq = getattr(cfg.logging, "save_freq", 0)
    if save_freq and step > 0 and step % save_freq == 0:
        save_state(state, cfg, name=f"{out_name}_step{step}")  # step-tagged, kept
        save_state(state, cfg)                                  # latest (overwritten)
