"""TS five-category evaluation using integrated velocity and divergence.

Matches eval_peptide_ts.py: seed 42, 100,000 generated proposals, 10,000
samples per evaluated category, TS energy-cutoff comparison and 128 direct
Boltzmann MALA updates. Resampling and relaxation are shared across reference
splits. COM flags must match the velocity checkpoint's training convention.

Dopri5 jointly integrates dx/dt = b(t, x) and dlogq/dt = -div b(t, x),
using Hutchinson (default: one fixed Rademacher probe) or exact JVP traces.
The adaptive solver chooses its own steps from --atol/--rtol; --steps only
applies when explicitly selecting a fixed-step solver. NFE, JVP counts and
accepted/rejected steps are recorded. Exact runs always save generated samples.

Example:
    python py/launchers/eval_ode_int.py --system ace_a_nme \
        --ckpt_path ace_a_nme_velocity.pt --out_dir outputs/ode \
        --divergence_type hutchinson --n_generate_samples 100000 \
        --n_eval_samples 10000 --posthoc_mala_steps 128 --seed 42
"""

import argparse
import importlib
import json
import os
import random
import sys
import time

# Keep optional Flax checkpoint imports on CPU; torch owns the GPU.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..")))

import numpy as np

from common.sample_stages import ENERGY_CUTOFFS_KT, sample_stages

STAGES = ("proposal", "resampled", "resampled_energy_cutoff", "relaxed", "relaxed_energy_cutoff")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ODE-integration + likelihood + resampling benchmark (velocity checkpoints).",
    )
    # --- what to run -------------------------------------------------------------
    p.add_argument("--system", type=str, default=None,
                   choices=["ace_a_nme", "ace_aaa_nme", "aaa", "aaaaaa"],
                   help="peptide system; resolves to configs.<system> unless --cfg_path is given")
    p.add_argument("--cfg_path", type=str, default=None, help="overrides --system")
    p.add_argument("--training_config", type=str, default=None,
                   help="saved training config.json; restores architecture and prior settings")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default=None,
                   help="basename for the output JSON (default: derived from the settings)")

    # --- ODE / likelihood hyperparameters ----------------------------------------
    p.add_argument("--steps", type=int, nargs="+", default=[4],
                   help="ODE step counts to sweep (ignored by the adaptive solver dopri5)")
    p.add_argument("--solver", type=str, nargs="+", default=["dopri5"],
                   choices=["euler", "midpoint", "heun", "rk4", "dopri5"],
                   help="solvers to sweep on the augmented (x, logp) system. dopri5 is "
                        "adaptive: it ignores --steps and reports the step count it chose")
    p.add_argument("--atol", type=float, default=1e-5, help="dopri5 absolute tolerance")
    p.add_argument("--rtol", type=float, default=1e-5, help="dopri5 relative tolerance")
    p.add_argument("--dlogp_tol_scale", type=float, default=1.0,
                   help="dopri5: divisor on the log-density channel in the error norm "
                        "(ts TorchDynWrapper convention)")
    p.add_argument("--first_step", type=float, default=None, help="dopri5 initial step size")
    p.add_argument("--max_steps", type=int, default=10_000, help="dopri5 step-count cap")
    p.add_argument("--divergence_type", type=str, default="hutchinson",
                   choices=["hutchinson", "exact"],
                   help="exact = one JVP per Cartesian direction (D=3N per stage)")
    p.add_argument("--hutchinson_dist", type=str, default="rademacher",
                   choices=["rademacher", "gaussian"])
    p.add_argument("--n_mc_samples", type=int, default=1,
                   help="Hutchinson probes per divergence evaluation")
    p.add_argument("--probe_schedule", type=str, default="fixed", choices=["fixed", "per_step"],
                   help="fixed = one probe set per trajectory (FFJORD); per_step = redraw each step")
    p.add_argument("--exact_chunk", type=int, default=None,
                   help="basis directions vmapped together for --divergence_type exact")

    # --- sampling -----------------------------------------------------------------
    p.add_argument("--n_generate_samples", "--n_samples", "--num_samples", type=int, default=100_000,
                   help="Number of proposals to integrate and score before filtering")
    p.add_argument("--n_eval_samples", "--num_eval_samples", type=int, default=10_000,
                   help="Evaluation/reference budget; MALA runs only on selected draws")
    p.add_argument("--energy_cutoff", type=float, default=None,
                   help="Cutoff-branch U/kT threshold; default matches the TS peptide config")
    p.add_argument("--posthoc_mala_steps", "--mala_steps", type=int, default=128,
                   help="Boltzmann MALA updates after resampling; 0 disables relaxation")
    p.add_argument("--mala_step_size", type=float, default=1e-6,
                   help="Initial MALA tau in nm^2; proposal noise std=sqrt(2*tau)")
    p.add_argument("--mala_adapt_steps", type=int, default=None,
                   help="Tune tau during this many existing MALA steps, then freeze; "
                        "default: half of --posthoc_mala_steps. Use 0 for fixed tau.")
    p.add_argument("--mala_target_accept", type=float, default=0.6,
                   help="Target batch-mean MALA acceptance probability during warmup")
    p.add_argument("--batch_size", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_ema", action="store_true", help="use raw params instead of EMA")
    from common.com_lift import add_com_cli_arguments
    add_com_cli_arguments(p)
    p.add_argument("--allow_ckpt_name_mismatch", action="store_true",
                   help="skip the guard that --ckpt_path mentions the target system")
    p.add_argument("--fp32_input", action="store_true",
                   help="disable the config's bf16 input truncation in calc_phi (the primal is "
                        "bf16-rounded otherwise; rademacher tangents are bf16-exact, gaussian "
                        "ones are not). Off by default to stay faithful to training.")
    p.add_argument("--mixed_precision", choices=["off", "bf16_trunk"], default="off",
                   help="run the DiT trunk's matmuls in bfloat16 (see learn.py). Defaults to "
                        "off REGARDLESS of what the checkpoint was trained with, so a "
                        "head-to-head evaluation of an fp32 and a mixed-precision chain is "
                        "run in identical arithmetic. Pass bf16_trunk to reproduce training.")

    # --- resampling ---------------------------------------------------------------
    p.add_argument("--resampling", type=str, default="multinomial",
                   choices=["stratified", "multinomial"],
                   help="Multinomial SNIS (default), or stratified draws ordered by energy")
    p.add_argument("--logw_quantile_filter", type=float, default=None,
                   help="drop the top fraction of proposals by logw before resampling "
                        "(ts SNISSampler). Default: cfg.logging.importance_weight_clip "
                        "(0.002), the PROJECT STANDARD for importance-sampling metrics. "
                        "Pass 0 to disable; full-pool ESS is saved per category.")

    # --- reference / metrics --------------------------------------------------------
    p.add_argument("--reference_splits", type=str, nargs="+", default=["val", "test"],
                   choices=["val", "test"])
    p.add_argument("--num_tica", type=int, default=None)
    p.add_argument("--ts_data_dir", type=str, default=None)
    p.add_argument("--unscale_factor", type=float, default=None,
                   help="override the model's input scale (default: the ts train std)")
    p.add_argument("--n_workers", type=int, default=64)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--save_energies", action="store_true")
    p.add_argument("--save_samples", action="store_true")

    # --- diagnostics ----------------------------------------------------------------
    p.add_argument("--roundtrip", type=int, default=0,
                   help="reverse-integrate this many generated samples and report the state / "
                        "log-density recovery error (0 = skip)")
    p.add_argument("--device", type=str, default="cuda")

    # --- wandb --------------------------------------------------------------------
    p.add_argument("--wandb", action="store_true",
                   help="log all five TS categories and integration cost to wandb")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="default: cfg.logging.wandb_project")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    args = p.parse_args(argv)
    for name in ("n_generate_samples", "n_eval_samples", "batch_size", "n_workers", "n_mc_samples", "max_steps"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if any(n < 1 for n in args.steps):
        p.error("--steps must be positive")
    if args.posthoc_mala_steps < 0 or args.roundtrip < 0:
        p.error("--posthoc_mala_steps and --roundtrip must be nonnegative")
    for name in ("mala_step_size", "atol", "rtol", "dlogp_tol_scale", "first_step", "unscale_factor", "exact_chunk", "num_tica"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            p.error(f"--{name} must be finite and positive")
    if args.mala_adapt_steps is None:
        args.mala_adapt_steps = args.posthoc_mala_steps // 2
    if not 0 <= args.mala_adapt_steps <= args.posthoc_mala_steps:
        p.error("--mala_adapt_steps must be in [0, posthoc_mala_steps]")
    if not 0 < args.mala_target_accept < 1:
        p.error("--mala_target_accept must be finite and in (0, 1)")
    if args.energy_cutoff is not None and not np.isfinite(args.energy_cutoff):
        p.error("--energy_cutoff must be finite (negative values are allowed)")
    if args.logw_quantile_filter is not None and not 0 <= args.logw_quantile_filter < 1:
        p.error("--logw_quantile_filter must be in [0, 1)")
    return args


def _run_tag(args) -> str:
    """Filename stem identifying this configuration (JSON + npz share it)."""
    return args.tag or (
        f"{args.divergence_type}"
        + (f"-{args.hutchinson_dist}-K{args.n_mc_samples}-{args.probe_schedule}"
           if args.divergence_type == "hutchinson" else "")
        + f"_{'-'.join(args.solver)}_n{args.n_generate_samples}"
    )


def _wandb_log_row(split: str, key: str, run: dict, entry: dict, args) -> None:
    """Log all category metrics and diagnostics against integration cost."""
    import wandb

    cost = run["cost"]
    payload = {
        "ode/nfe_velocity": cost["nfe_velocity_per_sample"],
        "ode/n_jvp": cost["n_jvp_per_sample"],
        "ode/t_sample_s": cost["t_sample_s"],
        "ode/samples_per_s": cost["samples_per_s"],
        f"{split}/{key}/logq_std": entry["logq"]["std"],
        f"{split}/{key}/logw_std": entry["logw"]["std"],
    }
    if run["n_steps"] is not None:
        payload["ode/n_steps"] = run["n_steps"]
    adaptive = cost.get("adaptive")
    if adaptive:
        # The whole point of the adaptive run: what step count did the controller pick?
        payload.update({
            "ode/adaptive_n_accepted_mean": adaptive["n_accepted_mean"],
            "ode/adaptive_n_accepted_min": adaptive["n_accepted_min"],
            "ode/adaptive_n_accepted_max": adaptive["n_accepted_max"],
            "ode/adaptive_n_rejected_mean": adaptive["n_rejected_mean"],
            "ode/adaptive_accept_rate": adaptive["accept_rate"],
        })
    for bucket in STAGES:
        for name, value in entry.get(bucket, {}).items():
            payload[f"{split}/{key}/{bucket}/{name}"] = value
    wandb.log(payload, commit=True)


def _wandb_table(split_res: dict, args):
    """One row per run/category, including cutoff ESS and MALA acceptance."""
    import wandb

    metrics = ("n-samples", "effective-sample-size", "ess", "energy-w2", "torus-w2",
               "tica-w2", "tica-k-jsd", "mala-acceptance-rate")
    table = wandb.Table(columns=["run", "category", "nfe_velocity", "n_jvp", *metrics])
    for key, entry in split_res["runs"].items():
        for name in STAGES:
            if name in entry:
                table.add_data(key, name, entry["cost"]["nfe_velocity_per_sample"],
                               entry["cost"]["n_jvp_per_sample"],
                               *(entry[name].get(metric) for metric in metrics))
    return table


def _init_wandb(args, cfg, system, dim, ckpt_step, uf, params_desc):
    """Best-effort wandb init. Returns the run, or None if wandb is off or unavailable."""
    if not args.wandb:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project or getattr(cfg.logging, "wandb_project", "iF2D2_ode"),
            entity=args.wandb_entity or getattr(cfg.logging, "wandb_entity", None),
            name=args.wandb_name or f"ode_{system.sequence}_{_run_tag(args)}",
            group=args.wandb_group,
            tags=args.wandb_tags,
            config={**vars(args), "system": system.trajectory_name, "dim": dim,
                    "ckpt_step": ckpt_step, "unscale_factor": uf, "params": params_desc},
            dir=args.out_dir,
        )
        # Charts plot against cost rather than a meaningless global step, so the fixed-step
        # sweep and the adaptive solver land on one comparable x-axis.
        wandb.define_metric("ode/nfe_velocity")
        wandb.define_metric("*", step_metric="ode/nfe_velocity")
        print(f"[ode] wandb run {run.name} ({run.url})")
        return run
    except Exception as exc:  # noqa: BLE001 - logging must never be fatal
        print(f"[ode] WARNING: wandb disabled for this run ({type(exc).__name__}: {exc}). "
              f"Metrics still go to the JSON in --out_dir.")
        return None


def _summary(v: np.ndarray) -> dict:
    """Finite-aware summary of a 1-D quantity."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    finite = np.isfinite(v)
    if not finite.any():
        return {"n_nonfinite": int((~finite).sum()), "mean": None, "std": None,
                "min": None, "median": None, "max": None}
    f = v[finite]
    return {
        "n_nonfinite": int((~finite).sum()),
        "mean": float(f.mean()), "std": float(f.std()),
        "min": float(f.min()), "median": float(np.median(f)), "max": float(f.max()),
    }


def main():
    args = parse_args()

    # Audit N9 (2026-09-03): pin matmul precision explicitly. The upstream
    # transferable-samplers entry points set exactly these three
    # (transferable_samplers/train.py:23-28, "issues with invertibility"),
    # but iF2D2 never imports those modules, so the values were torch's defaults
    # by accident. Same values, now chosen -- and stable across torch upgrades.
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out_dir, exist_ok=True)
    # A short final batch is fine (the generation loop slices), so --n_generate_samples need not be
    # a multiple of --batch_size.

    # An exact run costs hours to days, and its coordinates are not reconstructible from
    # anything else, so always persist them. (A 72 h AAAAAA run was reduced to energies-only
    # because --save_samples had been left off; the energy/logq/logw npz survived but the
    # conformations, and hence torus/tica/JSD, did not.)
    if args.divergence_type == "exact" and not args.save_samples:
        args.save_samples = True
        print("[ode] --divergence_type exact: forcing --save_samples (expensive run; "
              "coordinates cannot be regenerated cheaply)")
    if args.cfg_path is None and args.system is None:
        raise SystemExit("pass --system (or --cfg_path)")
    cfg_path = args.cfg_path or f"configs.{args.system}"

    from common import ts_systems

    cfg = importlib.import_module(cfg_path).get_config(0, "", args.out_dir)
    if args.training_config:
        from ml_collections import ConfigDict
        with open(args.training_config) as fh:
            cfg = ConfigDict(json.load(fh))
        if cfg.network.divergence_mode != "pretrain_v":
            raise SystemExit("--training_config must describe pretrain_v")
        if bool(cfg.network.com_aware) != args.com_augmentation or bool(
            cfg.problem.get("com_augmentation", False)
        ) != args.com_augmentation:
            raise SystemExit("CoM flag must match --training_config")
        if args.system:
            expected_target = importlib.import_module(f"configs.{args.system}").get_config(0).problem.target
            if cfg.problem.target != expected_target:
                raise SystemExit("--system must match --training_config")
        cfg.logging.output_folder = args.out_dir
    # PROJECT STANDARD (2026-09-04): importance-sampling metrics use the config's
    # logw_quantile_filter. `None` means "unset", not "off" -- 0 turns it off.
    if args.logw_quantile_filter is None:
        args.logw_quantile_filter = float(
            getattr(cfg.logging, "importance_weight_clip", 0.0)
        )
        print(f"[eval] logw_quantile_filter defaulted to cfg.logging."
              f"importance_weight_clip = {args.logw_quantile_filter}")
    if not 0 <= args.logw_quantile_filter < 1:
        raise ValueError("The configured importance-weight clip must be in [0, 1)")
    # Velocity checkpoints only: no divergence head, diagonal (s == t) evaluation.
    cfg.network.divergence_mode = "pretrain_v"
    cfg.network.predict_divergence = False
    cfg.network.load_path = args.ckpt_path
    cfg.teacher.load_path = ""
    cfg.training.ndevices = 1
    if args.fp32_input:
        cfg.network.use_bfloat16 = False
    cfg.network.mixed_precision = args.mixed_precision
    cfg.problem.com_augmentation = args.com_augmentation
    cfg.network.com_aware = args.com_augmentation

    system = ts_systems.resolve(cfg)
    if args.energy_cutoff is None:
        args.energy_cutoff = ENERGY_CUTOFFS_KT[system.target]
    print(f"[ode] generation={args.n_generate_samples}, evaluation={args.n_eval_samples}, "
          f"seed={args.seed}, cutoff branch E < {args.energy_cutoff:g} kT")

    # The DiT's parameter shapes do not depend on the atom count (the positional
    # embedding is a buffer), so a checkpoint for a DIFFERENT peptide loads without any
    # error and silently produces garbage. Nothing in the checkpoint identifies its
    # system, so guard on the path instead.
    if not args.allow_ckpt_name_mismatch and cfg.problem.target not in args.ckpt_path:
        raise SystemExit(
            f"--ckpt_path does not mention the target {cfg.problem.target!r}:\n  "
            f"{args.ckpt_path}\nA checkpoint trained on a different peptide would load "
            f"silently and give meaningless results. Pass --allow_ckpt_name_mismatch if "
            f"the name is simply unconventional."
        )

    paths = ts_systems.ensure_data(system, args.ts_data_dir)
    dim = int(cfg.network.n_particles * cfg.network.n_dimension)
    print(f"[ode] system={system.trajectory_name} ({system.num_atoms} atoms, "
          f"{system.temperature:g} K, D={dim}), data source={paths['source']}")

    # Fork the energy pool BEFORE torch initialises CUDA.
    t0 = time.time()
    energy = ts_systems.ParallelEnergy(system, n_workers=args.n_workers, data_dir=args.ts_data_dir)
    print(f"[ode] ts OpenMMEnergy pool: {args.n_workers} workers in {time.time()-t0:.1f}s")

    try:
        _run(args, cfg, cfg_path, system, paths, energy, dim)
    finally:
        energy.close()


def _load_net(args, cfg, device):
    """Load a velocity checkpoint under this CLI's saved-config EMA policy."""
    from common.state_utils import load_evaluation_model

    net, step, ema_fac = load_evaluation_model(
        cfg, args.ckpt_path, device, use_ema=not args.no_ema,
        require_saved_ema=bool(args.training_config), velocity_only=True,
    )
    params_desc = "raw" if args.no_ema else f"EMA {ema_fac}"
    print(f"[ode] params: {params_desc}, step {step}, use_bfloat16={cfg.network.use_bfloat16}")
    return net, params_desc, step


def _run(args, cfg, cfg_path, system, paths, energy, dim):
    from common.logging import _wandb_safe
    import torch

    import common.datasets as datasets
    import common.ode_likelihood as odl
    from common import ts_systems
    from common.se3_utils import remove_mean
    from common.com_lift import importance_source_energy
    from common.tsx import PeptideEnsembleEvaluator

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    random.seed(args.seed)
    device = torch.device(args.device)

    # Initialize model shape and scale from the official TS training split.
    cfg, ex_input = datasets.setup_target(cfg, data_dir=args.ts_data_dir)
    ts_train_std = float(cfg.network.unscale_factor)
    if args.unscale_factor is not None:
        cfg.network.unscale_factor = args.unscale_factor
    uf = float(cfg.network.unscale_factor)
    print(f"[ode] unscale_factor={uf:.6g} (ts train std={ts_train_std:.6g})")

    net, params_desc, ckpt_step = _load_net(args, cfg, device)

    # ---------------- one shared base draw, so every config is paired ----------------
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    rescale = float(cfg.network.rescale)
    n_part, n_dim = cfg.network.n_particles, cfg.network.n_dimension
    x0_all = rescale * torch.randn((args.n_generate_samples, n_part, n_dim), generator=gen, device=device)
    if not cfg.network.com_aware:
        x0_all = remove_mean(x0_all)
    logp0_all = net.base_log_prob(x0_all)

    def vel_fn(t, x):
        return net.calc_b(t, x, train=False)

    is_adaptive = lambda s: s in odl.ADAPTIVE_SOLVERS  # noqa: E731

    def _solver_kwargs(solver: str, n_steps):
        """dopri5 chooses its own steps; the fixed-step solvers take --steps."""
        if is_adaptive(solver):
            return dict(
                solver=solver, atol=args.atol, rtol=args.rtol,
                dlogp_tol_scale=args.dlogp_tol_scale,
                first_step=args.first_step, max_steps=args.max_steps,
            )
        return dict(solver=solver, n_steps=n_steps)

    def _estimator(seed_offset: int):
        return odl.make_estimator(
            args.divergence_type,
            n_mc_samples=args.n_mc_samples,
            hutchinson_dist=args.hutchinson_dist,
            probe_schedule=args.probe_schedule,
            exact_chunk=args.exact_chunk,
            generator=torch.Generator(device=device).manual_seed(args.seed + seed_offset),
        )

    @torch.no_grad()
    def generate(solver: str, n_steps):
        """Integrate every batch; returns (x1_phys, logq, cost dict)."""
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        xs, logps = [], []
        nfe = njvp = 0
        accepted, rejected = [], []
        t_start = time.time()
        for i in range(0, args.n_generate_samples, args.batch_size):
            x0 = x0_all[i : i + args.batch_size]
            batch_start = time.time()
            print(f"[ode] {solver}: batch {i // args.batch_size + 1}/"
                  f"{(args.n_generate_samples + args.batch_size - 1) // args.batch_size} "
                  f"({x0.shape[0]} samples) starting", flush=True)
            res = odl.integrate(vel_fn, x0, estimator=_estimator(991 + i),
                                **_solver_kwargs(solver, n_steps))
            x1, logp1 = net.unscale_samples(res.x, logp0_all[i : i + args.batch_size] + res.dlogp)
            xs.append(x1.detach().float().cpu())
            logps.append(logp1.detach().float().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"[ode] {solver}: {i + x0.shape[0]}/{args.n_generate_samples} samples completed, "
                  f"batch {time.time() - batch_start:.1f}s, NFE {res.nfe_velocity}", flush=True)
            # NFE is per trajectory, so weight each batch's count by how many samples it
            # carries (the final batch may be short).
            nfe += res.nfe_velocity * x0.shape[0]
            njvp += res.n_jvp * x0.shape[0]
            if is_adaptive(solver):
                accepted.append(int(res.extras["n_accepted"]))
                rejected.append(int(res.extras["n_rejected"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cost = {
            "t_sample_s": time.time() - t_start,
            "nfe_velocity_per_sample": nfe / args.n_generate_samples,
            "n_jvp_per_sample": njvp / args.n_generate_samples,
            "peak_cuda_mem_GiB": (
                float(torch.cuda.max_memory_allocated(device)) / 2**30
                if device.type == "cuda" else None
            ),
        }
        cost["samples_per_s"] = args.n_generate_samples / max(cost["t_sample_s"], 1e-9)
        if is_adaptive(solver):
            # The step size is shared within a batch, so each batch lands on its own count.
            cost["adaptive"] = {
                "atol": args.atol, "rtol": args.rtol,
                "dlogp_tol_scale": args.dlogp_tol_scale,
                "n_accepted_per_batch": accepted,
                "n_rejected_per_batch": rejected,
                "n_accepted_mean": sum(accepted) / len(accepted),
                "n_accepted_min": min(accepted),
                "n_accepted_max": max(accepted),
                "n_rejected_mean": sum(rejected) / len(rejected),
                "accept_rate": sum(accepted) / max(1, sum(accepted) + sum(rejected)),
                "stages_per_step": odl.n_stages(solver),
            }
        return torch.cat(xs).float().cpu(), torch.cat(logps).float().cpu(), cost

    @torch.no_grad()
    def roundtrip(solver: str, n_steps, n: int):
        """Reverse-integrate generated samples: recovery of x0 and of the log-density."""
        x0 = x0_all[:n]
        fwd = odl.integrate(vel_fn, x0, estimator=_estimator(7),
                            **_solver_kwargs(solver, n_steps))
        bwd = odl.integrate(vel_fn, fwd.x, estimator=_estimator(7), t0=1.0, t1=0.0,
                            **_solver_kwargs(solver, n_steps))
        return {
            "n": int(n),
            "max_abs_x_error": float((bwd.x - x0).abs().max()),
            "rms_x_error": float(((bwd.x - x0) ** 2).mean().sqrt()),
            "max_abs_dlogp_residual": float((fwd.dlogp + bwd.dlogp).abs().max()),
            "mean_abs_dlogp_residual": float((fwd.dlogp + bwd.dlogp).abs().mean()),
        }

    # ---------------- generate once per (solver, n_steps) ----------------
    # An adaptive solver picks its own step count, so --steps does not apply to it: it
    # contributes exactly one row per run.
    grid = [
        (solver, None) if is_adaptive(solver) else (solver, n)
        for solver in args.solver
        for n in ([None] if is_adaptive(solver) else args.steps)
    ]
    runs = {}
    for solver, n_steps in grid:
        key = f"{solver}/adaptive" if n_steps is None else f"{solver}/{n_steps}"
        x1, logq_c, cost = generate(solver, n_steps)

        t0 = time.time()
        E = energy.numpy(x1.numpy())
        cost["t_energy_s"] = time.time() - t0

        E_t = torch.from_numpy(np.asarray(E, dtype=np.float64))
        # Correct at the endpoint, leaving integrated log q and roundtrips intact.
        logw = importance_source_energy(
            x1, logq_c, com_augmentation=cfg.network.com_aware, unscale_factor=uf,
        ) - E_t
        runs[key] = {
            "solver": solver, "n_steps": n_steps,
            "x1": x1, "E": E_t, "logq": logq_c, "logw": logw, "cost": cost,
        }
        finite = np.isfinite(E)
        adaptive = cost.get("adaptive")
        chose = (f", CHOSE {adaptive['n_accepted_mean']:.1f} steps "
                 f"(min {adaptive['n_accepted_min']}, max {adaptive['n_accepted_max']}, "
                 f"{adaptive['n_rejected_mean']:.1f} rejected, "
                 f"accept {100*adaptive['accept_rate']:.0f}%)" if adaptive else "")
        print(f"[ode] {key:>16}: {cost['t_sample_s']:7.1f}s sample "
              f"({cost['samples_per_s']:7.1f} samp/s, nfe {cost['nfe_velocity_per_sample']:.0f}, "
              f"jvp {cost['n_jvp_per_sample']}), {cost['t_energy_s']:5.1f}s energy, "
              f"median {np.median(E[finite]):.4g} kT, nonfinite {int((~finite).sum())}, "
              f"logq mean {float(logq_c.mean()):.1f}{chose}")

    tag = _run_tag(args)

    if args.save_energies or args.save_samples:
        payload = {f"energies_{k.replace('/', '_')}": v["E"].numpy() for k, v in runs.items()}
        payload.update({f"logq_{k.replace('/', '_')}": v["logq"].numpy() for k, v in runs.items()})
        payload.update({f"logw_{k.replace('/', '_')}": v["logw"].numpy() for k, v in runs.items()})
        if args.save_samples:
            payload.update({f"samples_{k.replace('/', '_')}": v["x1"].numpy() for k, v in runs.items()})
        # tagged, so a second configuration in the same --out_dir cannot clobber the first
        out_npz = os.path.join(args.out_dir, f"ode_samples_{tag}.npz")
        np.savez_compressed(out_npz, **payload)
        print("[ode] wrote", out_npz)

    # Shared TS pipeline: resample and relax once, then reuse for every split.
    # Raw generated samples above are persisted before the potentially long MALA run.
    torch.manual_seed(args.seed)
    for key, run in runs.items():
        t0 = time.time()
        run["stages"], run["diagnostics"] = sample_stages(
            run["x1"], run["E"], run["logw"], clip_fraction=args.logw_quantile_filter,
            energy=energy, mala_steps=args.posthoc_mala_steps, mala_step_size=args.mala_step_size,
            n_eval_samples=args.n_eval_samples, energy_cutoff=args.energy_cutoff,
            resampling=args.resampling, mala_adapt_steps=args.mala_adapt_steps,
            mala_target_accept=args.mala_target_accept,
        )
        run["cost"]["t_resample_relax_s"] = time.time() - t0
        if args.save_samples:
            np.savez_compressed(
                os.path.join(args.out_dir, f"ode_stages_{tag}_{key.replace('/', '_')}.npz"),
                **{f"samples_{name}": data.samples.numpy() for name, data in run["stages"].items()},
                **{f"energies_{name}": data.E_target.numpy() for name, data in run["stages"].items()},
            )
        print(f"[ode] {key}: stages={list(run['stages'])}, "
              f"resample/MALA {run['cost']['t_resample_relax_s']:.1f}s", flush=True)

    # ---------------- ts metrics ----------------
    evaluator = PeptideEnsembleEvaluator(
        num_eval_samples=args.n_eval_samples, do_plots=not args.no_plots
    )
    results = {
        "system": system.trajectory_name,
        "cfg_path": cfg_path,
        "ckpt_path": args.ckpt_path,
        "ckpt_step": ckpt_step,
        "params": params_desc,
        "mode": "pretrain_v (velocity field, ODE integrated with calc_b at s == t)",
        "dim": dim,
        "n_generate_samples": args.n_generate_samples,
        "batch_size": args.batch_size,
        "n_eval_samples": args.n_eval_samples,
        "energy_cutoff_kT": args.energy_cutoff,
        "energy_cutoff_order": "before logw filtering and evaluation draws; only cutoff branches",
        "posthoc_mala_steps": args.posthoc_mala_steps,
        "mala_step_size_nm2": args.mala_step_size,
        "mala_adapt": args.mala_adapt_steps > 0,
        "mala_adapt_steps": args.mala_adapt_steps,
        "mala_target_accept": args.mala_target_accept,
        "mala_adaptation": "batch-mean Robbins-Monro on log(tau); freeze after warmup",
        "relaxation_target": "exp(-U/kT), mean-free physical nm coordinates",
        "ess_convention": "full eligible pool before resampling; effective-sample-size is normalized, ess is count",
        "com_augmentation": args.com_augmentation,
        "com_weight_correction": "gaussian" if args.com_augmentation else None,
        "unscale_factor": uf,
        "ts_train_std": ts_train_std,
        "data_source": paths["source"],
        "use_bfloat16": bool(cfg.network.use_bfloat16),
        "mixed_precision": str(cfg.network.mixed_precision),
        "energy_units": "kT (ts OpenMMEnergy, MOLAR_GAS_CONSTANT_R), unclamped",
        "likelihood": {
            "divergence_type": args.divergence_type,
            "hutchinson_dist": args.hutchinson_dist,
            "n_mc_samples": args.n_mc_samples,
            "probe_schedule": args.probe_schedule,
            "exact_chunk": args.exact_chunk,
            "com_augmentation": cfg.network.com_aware,
            "com_weight_correction": "gaussian" if cfg.network.com_aware else None,
            "logw": ("logw = -E_target - log q + log N(COM; 0, unscale_factor^2/N I_3)"
                     if cfg.network.com_aware else "logw = -E_target - log q"),
            "modeled_dimension": net.modeled_dimension,
            "constant_offsets": "D=3N for lifted models; D=3N-3 for mean-free models",
        },
        "resampling": {
            "method": args.resampling,
            "logw_quantile_filter": args.logw_quantile_filter,
            "cfg_importance_weight_clip": float(
                getattr(cfg.logging, "importance_weight_clip", 0.0)
            ),
            "order": "energy" if args.resampling == "stratified" else "input",
            "note": "same sample_stages pipeline as eval_peptide_ts; evaluation-sized draws",
        },
        "seed": args.seed,
        "argv": vars(args),
        "cost": {k: v["cost"] for k, v in runs.items()},
        "splits": {},
    }

    if args.roundtrip:
        results["roundtrip"] = {
            (f"{solver}/adaptive" if n_steps is None else f"{solver}/{n_steps}"):
                roundtrip(solver, n_steps, min(args.roundtrip, args.n_generate_samples))
            for solver, n_steps in grid
        }
        for k, v in results["roundtrip"].items():
            print(f"[ode] roundtrip {k:>16}: max|dx| {v['max_abs_x_error']:.3e}, "
                  f"max|dlogp residual| {v['max_abs_dlogp_residual']:.3e}")

    # ---------------- wandb ----------------
    # NEVER let a logging failure destroy the run: a wandb service timeout once killed a
    # completed 72 h exact integration at exactly this point (the /tmp reaper had removed
    # wandb's port file mid-run). Logging is best effort; the JSON on disk is the result.
    run_wandb = _init_wandb(args, cfg, system, dim, ckpt_step, uf, params_desc)

    def make_log_image_fn(split: str, key: str):
        """ts evaluator figure sink: save next to the JSON and (if enabled) log to wandb.

        Same ``(fig, title, title_prefix)`` contract the ts callback uses via
        ``common.ts_eval.ts_evaluate``; the evaluator emits one set per sample set, so
        all five category figures come through here.
        """
        if args.no_plots:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(args.out_dir, "figs", split, key.replace("/", "_"))
        os.makedirs(fig_dir, exist_ok=True)

        def log_image_fn(fig, title, title_prefix=""):
            # ts already builds the prefix as "{split}/{sequence}/{run key}", so use its
            # name verbatim (as common.ts_eval.ts_evaluate does) rather than prepending
            # split/key again.
            name = f"{title_prefix}/{title}" if title_prefix else str(title)
            fig.savefig(os.path.join(fig_dir, name.replace("/", "_") + ".png"),
                        dpi=110, bbox_inches="tight")
            if run_wandb is not None:
                import wandb

                _wandb_safe(wandb.log, {name: wandb.Image(fig)}, commit=False)
            plt.close(fig)

        return log_image_fn

    ref_energies = {}
    for split in args.reference_splits:
        t0 = time.time()
        eval_ctx = ts_systems.build_eval_context(
            system, split=split, energy_fn=energy,
            num_eval_samples=args.n_eval_samples, data_dir=args.ts_data_dir,
            normalization_std=1.0,  # samples handed over in physical nm
            num_tica=args.num_tica,
        )
        true_E = eval_ctx.true_data.E_target
        if args.save_energies:
            ref_energies[f"reference_{split}"] = true_E.numpy()
        print(f"[ode] split={split}: {len(eval_ctx.true_data)} reference frames in "
              f"{time.time()-t0:.0f}s, mean {true_E.mean():.2f} kT")

        ref_E_np = true_E.numpy().astype(np.float64)
        split_res = {
            "n_reference_used": int(len(eval_ctx.true_data)),
            "reference_energy_mean_kT": float(true_E.mean()),
            "reference_energy_std_kT": float(true_E.std()),
            "reference_energy_median_kT": float(true_E.median()),
            "reference_energy_min_kT": float(ref_E_np.min()),
            "reference_energy_max_kT": float(ref_E_np.max()),
            "runs": {},
        }

        for key, run in runs.items():
            E_t, logq_c, logw = run["E"], run["logq"], run["logw"]

            stages = run["stages"]
            prefix = f"{split}/{system.sequence}/{key}"
            t0 = time.time()
            raw = evaluator.evaluate(
                {name: data for name, data in stages.items() if len(data)}, eval_ctx,
                log_image_fn=make_log_image_fn(split, key), prefix=prefix,
            )
            entry = {name: dict(run["diagnostics"][name]) for name in stages}
            for metric_key, value in raw.items():
                suffix = metric_key.removeprefix(prefix + "/")
                name, _, metric = suffix.partition("/")
                if name in entry:
                    try:
                        entry[name][metric] = float(value)
                    except (TypeError, ValueError):
                        pass
            entry["t_metric_s"] = time.time() - t0
            entry["logq"] = _summary(logq_c.numpy())
            entry["logw"] = _summary(np.where(np.isfinite(logw.numpy()), logw.numpy(), np.nan))
            entry["n_energy_nonfinite"] = int((~np.isfinite(E_t.numpy())).sum())
            entry["cost"] = run["cost"]

            split_res["runs"][key] = entry
            if run_wandb is not None:
                _wandb_safe(_wandb_log_row, split, key, run, entry, args)
            for name in stages:
                metrics = entry[name]
                ess = metrics.get("effective-sample-size", metrics.get("effective-sample-size-unfiltered"))
                print(f"[ode] {split} {key} {name}: "
                      f"energy-w2={metrics.get('energy-w2', float('nan')):.4g} "
                      f"torus-w2={metrics.get('torus-w2', float('nan')):.4g} "
                      f"tica-w2={metrics.get('tica-w2', float('nan')):.4g} "
                      f"ESS={float('nan') if ess is None else ess:.4g}")

        results["splits"][split] = split_res

    if args.save_energies:
        np.savez_compressed(
            os.path.join(args.out_dir, f"ode_energies_{tag}.npz"),
            **{f"{key.replace('/', '_')}_{name}": data.E_target.numpy()
               for key, run in runs.items() for name, data in run["stages"].items()},
            **ref_energies,
        )

    out = os.path.join(args.out_dir, f"ode_metrics_{tag}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("[ode] wrote", out)

    if run_wandb is not None:
        import wandb

        for split, split_res in results["splits"].items():
            _wandb_safe(wandb.log,
                        {f"{split}/summary": _wandb_table(split_res, args)}, commit=True)
        _wandb_safe(wandb.save, out, base_path=args.out_dir, policy="now")
        _wandb_safe(run_wandb.finish)


if __name__ == "__main__":
    main()
