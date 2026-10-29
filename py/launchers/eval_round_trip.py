"""Invertibility diagnostics for a likelihood flow map (the cycle-loss ablation).

Recipe v1 scores only forward jumps. This script measures the backward jumps that
the cycle loss trains, with the recipe's saved EMA 0.999 weights and step grid:

* model round trip: x0 -> forward -> inverse -> x0', in position (base units)
  and in log q (forward log q of x1 against the inverse log q of the same x1);
* data round trip: held-out frames -> inverse -> forward, in position (nm);
* inverse-map NLL of the held-out frames, -log q(x) with log q from the learned
  backward jumps. It is meaningful only when those jumps are trained, and it is
  an estimate from the learned density head, not an exact likelihood.

Example:
    python py/launchers/eval_round_trip.py --cfg_path configs.ace_a_nme \
        --ckpt_path density.pt --out_dir outputs/eval --ts_data_dir datasets/ts --steps 4
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..")))

import numpy as np
import torch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-trip (invertibility) evaluation of a density flow map.")
    p.add_argument("--cfg_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ts_data_dir", type=str, default=None)
    p.add_argument("--steps", type=int, default=4, help="Flow-map jumps in each direction (linear grid)")
    p.add_argument("--n_samples", type=int, default=10_000, help="Model samples and held-out frames used")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=10_000)
    p.add_argument("--mode", choices=["dlogpdt", "hutchinson"], default="dlogpdt")
    p.add_argument("--divergence_rescale", type=float, default=None)
    p.add_argument("--vectorized_divergence", action=argparse.BooleanOptionalAction, default=True)
    from common.com_lift import add_com_cli_arguments
    add_com_cli_arguments(p)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    for name in ("steps", "n_samples", "batch_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    return args


def summarize(values: torch.Tensor) -> dict:
    v = values.detach().double().cpu()
    return {"mean": float(v.mean()), "median": float(v.median()),
            "p99": float(torch.quantile(v, .99)), "max": float(v.max())}


def per_sample_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().reshape(len(a), -1).mean(dim=1).sqrt()


@torch.no_grad()
def round_trip(net, x0, frames, n_steps, batch_size, unscale_factor, com_aware):
    """Forward/inverse residuals for base draws ``x0`` and physical-nm ``frames``."""
    from common.com_lift import com_log_prob

    def chunked(fn, x):
        outs = [fn(chunk) for chunk in x.split(batch_size)]
        return [torch.cat(parts) for parts in zip(*outs)]

    def forward(x):
        x1, logq, _ = net.sample_forward(x, n_steps=n_steps)
        return x1, logq

    def inverse(x):
        z, _, logq, _ = net.sample_inverse(x, n_steps=n_steps)
        return z, logq

    x1, logq_fwd = chunked(forward, x0)
    x0_back, logq_inv = chunked(inverse, x1)
    z, logq_data = chunked(inverse, frames)
    frames_back, _ = chunked(forward, z)
    # A COM-augmented model's density is lifted by g(COM); held-out frames are mean-free.
    lift = com_log_prob(frames, unscale_factor) if com_aware else torch.zeros_like(logq_data)
    nll = -(logq_data - lift)
    finite = torch.isfinite(nll)
    dof = frames.shape[1] * frames.shape[2] - 3
    return {
        "model_x0_rms_error": summarize(per_sample_rms(x0_back, x0)),
        "model_logq_abs_gap_nats": summarize((logq_inv - logq_fwd).abs()),
        "data_rms_error_nm": summarize(per_sample_rms(frames_back, frames)),
        "data_inverse_nll_nats": float(nll[finite].mean()) if finite.any() else float("nan"),
        "data_inverse_nll_nats_per_dof": float(nll[finite].mean() / dof) if finite.any() else float("nan"),
        "data_inverse_nll_nonfinite": int((~finite).sum()),
    }


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out_dir, exist_ok=True)

    import common.datasets as datasets
    import common.state_utils as state_utils
    from common import ts_systems
    from common.se3_utils import remove_mean
    from launchers.eval_peptide_ts import configure_density_scale

    cfg = importlib.import_module(args.cfg_path).get_config(0, "", args.out_dir)
    cfg.network.divergence_mode = args.mode
    cfg.network.predict_divergence = True
    configure_density_scale(cfg, args)
    cfg.problem.com_augmentation = args.com_augmentation
    cfg.network.com_aware = args.com_augmentation
    cfg.network.load_path = args.ckpt_path
    cfg.teacher.load_path = ""
    cfg.training.ndevices = 1

    device = torch.device(args.device)
    system = ts_systems.resolve(cfg)
    cfg, _ = datasets.setup_target(cfg, data_dir=args.ts_data_dir)
    uf = float(cfg.network.unscale_factor)
    net, ckpt_step, ema_fac = state_utils.load_evaluation_model(
        cfg, args.ckpt_path, device, require_saved_ema=True, ema_factor=.999,
    )

    gen = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.n_samples, cfg.network.n_particles, cfg.network.n_dimension)
    x0 = float(cfg.network.rescale) * torch.randn(shape, generator=gen, device=device)
    if not cfg.network.com_aware:
        x0 = remove_mean(x0)
    frames = ts_systems.load_split(system, args.split, args.ts_data_dir)
    order = torch.randperm(len(frames), generator=torch.Generator().manual_seed(args.seed))
    frames = remove_mean(frames[order[:args.n_samples]]).to(device)

    t0 = time.time()
    metrics = round_trip(net, x0, frames, args.steps, args.batch_size, uf, cfg.network.com_aware)
    with open(args.ckpt_path, "rb") as fh:
        checkpoint_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    results = {
        "ckpt_path": args.ckpt_path, "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": ckpt_step, "params": f"EMA {ema_fac}", "steps": args.steps,
        "n_model_samples": len(x0), "n_frames": len(frames), "split": args.split,
        "seed": args.seed, "unscale_factor": uf, "com_augmentation": args.com_augmentation,
        "units": {"model_x0_rms_error": "base (standardized) coordinates",
                  "data_rms_error_nm": "nm", "nll": "nats, physical nm, mean-free frames"},
        "seconds": time.time() - t0, **metrics,
    }
    out = os.path.join(args.out_dir, "round_trip.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[round_trip] x0 rms {metrics['model_x0_rms_error']['mean']:.4g}, "
          f"|dlogq| {metrics['model_logq_abs_gap_nats']['mean']:.4g} nats, "
          f"data rms {metrics['data_rms_error_nm']['mean']:.4g} nm, "
          f"inverse NLL {metrics['data_inverse_nll_nats']:.4g} nats")
    print("[round_trip] wrote", out)


if __name__ == "__main__":
    main()
