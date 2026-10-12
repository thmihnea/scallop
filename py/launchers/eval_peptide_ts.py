"""TS proposal -> filtered SNIS resampling -> Boltzmann MALA evaluation.

Uses the installed transferable-samplers reference splits, OpenMM energies,
filtering and peptide metrics. Defaults: 100,000 proposals, 10,000 evaluation
samples per category, seed 42, four linear
flow-map steps, multinomial resampling, and 128 MALA updates
(first 64 adapt tau to 60% acceptance, then freeze). Pass the checkpoint's density
mode (dlogpdt or hutchinson) and its COM convention explicitly when needed.
Pretraining checkpoints remain usable via --mode pretrain_v/pretrain_flowmap,
but cannot supply resampling or ESS without a learned density head.

Example:
    python py/launchers/eval_peptide_ts.py --cfg_path configs.ace_a_nme \
        --ckpt_path density.pt --out_dir outputs/eval --mode dlogpdt \
        --n_generate_samples 100000 --n_eval_samples 10000 --seed 42 \
        --steps 4 --time_discretization_type edm \
        --posthoc_mala_steps 128 --com_augmentation
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

from common.time_discretization import TIME_DISCRETIZATION_TYPES, time_grid

from common.sample_stages import ENERGY_CUTOFFS_KT, sample_stages

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ts-repo evaluation of an iF2D2 peptide flow map.")
    p.add_argument("--cfg_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--eval_recipe", choices=["v1"], default=None,
                   help="Recipe v1: strict EMA, one SNIS pool, saved MALA snapshots and evidence")
    p.add_argument("--mala_checkpoint_interval", type=int, default=128)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default="RKOY")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_id", default=None)
    p.add_argument("--steps", type=int, nargs="+", default=[4],
                   help="Flow-map jumps, or RK4 integration steps for --mode pretrain_v")
    p.add_argument("--n_generate_samples", "--n_samples", "--num_samples", type=int, default=100_000,
                   help="Number of proposals to generate and score before filtering")
    p.add_argument("--n_eval_samples", "--num_eval_samples", type=int, default=10_000,
                   help="Evaluation/reference sample budget; MALA runs only on selected draws")
    p.add_argument("--energy_cutoff", type=float, default=None,
                   help="Cutoff-branch U/kT threshold; defaults: Ace-A-Nme 10, Ace-AAA-Nme 50, "
                        "AAA -120, AAAAAA -20. The no-cutoff branch is also evaluated.")
    p.add_argument("--time_discretization_type", choices=TIME_DISCRETIZATION_TYPES, default="linear")
    p.add_argument("--posthoc_mala_steps", "--mala_steps", type=int, default=128,
                   help="Boltzmann MALA updates after resampling; 0 disables relaxation")
    p.add_argument("--mala_step_size", type=float, default=1e-6,
                   help="Initial MALA tau in nm^2; proposal noise std=sqrt(2*tau)")
    p.add_argument("--mala_adapt_steps", type=int, default=None,
                   help="Tune tau during this many existing MALA steps, then freeze; "
                        "default: half of --posthoc_mala_steps. Use 0 for fixed tau.")
    p.add_argument("--mala_target_accept", type=float, default=0.6,
                   help="Target batch-mean MALA acceptance probability during warmup")
    p.add_argument("--batch_size", type=int, default=10_000)
    p.add_argument("--reference_splits", type=str, nargs="+", default=["val", "test"],
                   choices=["val", "test"],
                   help="official ts splits used as reference (ts eval.py reports on test)")
    p.add_argument("--ts_data_dir", type=str, default=None)
    p.add_argument("--num_tica", type=int, default=None,
                   help="cap on frames used to fit TICA (default: the full split, as ts does)")
    p.add_argument("--unscale_factor", type=float, default=None,
                   help="override the model's input scale (default: the ts train std, "
                        "which is what training standardizes with)")
    p.add_argument("--mode", choices=["dlogpdt", "hutchinson", "pretrain_v", "pretrain_flowmap"],
                   default="dlogpdt",
                   help="divergence_mode the checkpoint was trained with")
    p.add_argument("--divergence_rescale", type=float, default=None,
                   help="Custom training density-head scale; default resolves the config's training sentinel")
    p.add_argument("--vectorized_divergence", action=argparse.BooleanOptionalAction, default=True,
                   help="Default matches current vectorized training; use --no-vectorized_divergence "
                        "for historical scalar heads (also changes the default density scale)")
    from common.com_lift import add_com_cli_arguments
    add_com_cli_arguments(p)
    p.add_argument("--resampling", choices=["stratified", "multinomial"], default="multinomial",
                   help="Multinomial SNIS (default), or stratified draws ordered by energy")
    p.add_argument("--logw_quantile_filter", type=float, default=None,
                   help="drop the top fraction of proposals by logw before resampling/ESS. "
                        "Default: cfg.logging.importance_weight_clip (0.002), the PROJECT "
                        "STANDARD for importance-sampling metrics. Pass 0 to disable; the "
                        "unfiltered value is reported as effective-sample-size-unfiltered. "
                        "Proposal metrics retain the unfiltered sample set; resampled and "
                        "relaxed metrics use the filtered SNIS draws.")
    p.add_argument("--save_energies", action="store_true",
                   help="write per-step generated energies (kT) to energies.npz")
    p.add_argument("--n_workers", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", choices=["off", "bf16_trunk"], default="off",
                   help="run the DiT trunk's matmuls in bfloat16 (see learn.py). Defaults to "
                        "off REGARDLESS of what the checkpoint was trained with, so a "
                        "head-to-head evaluation of an fp32 and a mixed-precision chain is "
                        "run in identical arithmetic. Pass bf16_trunk to reproduce training.")
    p.add_argument("--no_ema", action="store_true", help="use raw params instead of EMA")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args(argv)
    if args.eval_recipe == "v1":
        recipe_steps = {"configs.ace_a_nme": 4, "configs.aaa": 4,
                        "configs.ace_aaa_nme": 8, "configs.aaaaaa": 16,
                        "configs.chignolin": 16}
        if args.cfg_path not in recipe_steps:
            p.error("Recipe v1 requires a supported peptide system")
        p.set_defaults(n_generate_samples=1_000_000, posthoc_mala_steps=1024,
                       logw_quantile_filter=.002, steps=[recipe_steps[args.cfg_path]])
        args = p.parse_args(argv)
        if args.no_ema or args.mode not in ("dlogpdt", "hutchinson"):
            p.error("Recipe v1 requires a density checkpoint and EMA 0.999")
        if len(args.steps) != 1 or args.mixed_precision != "off":
            p.error("Recipe v1 requires one step count and FP32")
        if args.energy_cutoff is not None:
            p.error("Recipe v1 has no energy-cutoff comparison branch")
        if (args.mala_checkpoint_interval < 1 or args.posthoc_mala_steps < 1
                or args.posthoc_mala_steps % args.mala_checkpoint_interval):
            p.error("Recipe v1 MALA budget must be a positive multiple of the snapshot interval")
    for name in ("n_generate_samples", "n_eval_samples", "batch_size", "n_workers"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if any(n < 1 for n in args.steps):
        p.error("--steps must be positive")
    if args.posthoc_mala_steps < 0:
        p.error("--posthoc_mala_steps must be nonnegative")
    if not np.isfinite(args.mala_step_size) or args.mala_step_size <= 0:
        p.error("--mala_step_size must be finite and positive")
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
    if args.divergence_rescale is not None and (
            not np.isfinite(args.divergence_rescale) or args.divergence_rescale <= 0):
        p.error("--divergence_rescale must be finite and positive")
    return args


def configure_density_scale(cfg, args):
    """Resolve the same scalar/vector density units as learn.setup_config_dict."""
    if args.vectorized_divergence is not None:
        cfg.network.vectorized_divergence = args.vectorized_divergence
    scale = args.divergence_rescale
    if scale is None:
        scale = float(cfg.network.divergence_rescale)
        if scale <= 0:
            scale = 1.0 if cfg.network.vectorized_divergence else float(cfg.problem.d)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The density-head scale must be finite and positive")
    cfg.network.divergence_rescale = scale


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

    from common import ts_systems

    cfg = importlib.import_module(args.cfg_path).get_config(0, "", args.out_dir)
    cfg.network.divergence_mode = args.mode
    cfg.network.predict_divergence = args.mode in ("dlogpdt", "hutchinson")
    configure_density_scale(cfg, args)
    cfg.problem.com_augmentation = args.com_augmentation
    cfg.network.com_aware = args.com_augmentation
    cfg.network.load_path = args.ckpt_path
    cfg.network.mixed_precision = args.mixed_precision
    cfg.teacher.load_path = ""
    cfg.training.ndevices = 1

    # PROJECT STANDARD (2026-09-04): importance-sampling metrics use the config's
    # logw_quantile_filter. `None` means "unset", not "off" -- pass 0 to turn it off.
    clip_frac = args.logw_quantile_filter
    if clip_frac is None:
        clip_frac = float(getattr(cfg.logging, "importance_weight_clip", 0.0))
        print(f"[eval] logw_quantile_filter defaulted to cfg.logging."
              f"importance_weight_clip = {clip_frac}")

    if not 0 <= clip_frac < 1:
        raise ValueError("The configured importance-weight clip must be in [0, 1)")
    for n in args.steps:
        time_grid(n, args.time_discretization_type)

    system = ts_systems.resolve(cfg)
    energy_cutoff = (None if args.eval_recipe else
                     ENERGY_CUTOFFS_KT[system.target] if args.energy_cutoff is None else args.energy_cutoff)
    print(f"[eval] generation={args.n_generate_samples}, evaluation={args.n_eval_samples}, "
          f"seed={args.seed}, recipe={args.eval_recipe}, cutoff={energy_cutoff}", flush=True)
    paths = ts_systems.ensure_data(system, args.ts_data_dir)
    print(f"[eval] system={system.trajectory_name} ({system.num_atoms} atoms, "
          f"{system.temperature:g} K), data source={paths['source']}")

    # Fork the energy pool BEFORE torch initialises CUDA.
    t0 = time.time()
    energy = ts_systems.ParallelEnergy(system, n_workers=args.n_workers, data_dir=args.ts_data_dir)
    print(f"[eval] ts OpenMMEnergy pool: {args.n_workers} workers in {time.time()-t0:.1f}s")

    wb_run = None
    wb_errors = []
    completed = False
    def wb_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:
            wb_errors.append(type(exc).__name__)
            print(f"[wandb] {type(exc).__name__}; retaining local evaluation evidence", flush=True)
            return None

    try:
        import torch

        import common.datasets as datasets
        import common.state_utils as state_utils
        from common.se3_utils import remove_mean
        from common.com_lift import importance_source_energy
        from common.ts_eval import sample_model
        from common.tsx import PeptideEnsembleEvaluator

        torch.manual_seed(args.seed)
        np.random.seed(args.seed % (2**32))
        random.seed(args.seed)
        device = torch.device(args.device)

        # Initialize model shape and scale from the official TS training split.
        cfg, ex_input = datasets.setup_target(cfg, data_dir=args.ts_data_dir)
        ts_std = float(cfg.network.unscale_factor)
        if args.unscale_factor is not None:
            cfg.network.unscale_factor = args.unscale_factor
        uf = float(cfg.network.unscale_factor)
        print(f"[eval] unscale_factor={uf:.6g} (ts train std={ts_std:.6g})")

        net, ckpt_step, ema_fac = state_utils.load_evaluation_model(
            cfg, args.ckpt_path, device, use_ema=not args.no_ema,
            require_saved_ema=bool(args.eval_recipe),
            ema_factor=.999 if args.eval_recipe else None,
        )
        print(f"[eval] params: {'raw' if args.no_ema else f'EMA {ema_fac}'}", flush=True)
        provenance = {}
        if args.eval_recipe:
            import hashlib
            from pathlib import Path
            with open(args.ckpt_path, "rb") as fh:
                checkpoint_hash = hashlib.file_digest(fh, "sha256").hexdigest()
            prior = Path(args.out_dir) / "provenance.json"
            if prior.exists():
                provenance.update(json.loads(prior.read_text()))
            provenance.update(checkpoint_sha256=checkpoint_hash, checkpoint_step=ckpt_step,
                              ema=.999, resolved_config=cfg.to_dict(), arguments=vars(args),
                              symmetry={"fix_symmetry": True, "drop_unfixable_symmetry": False})
            prior.write_text(json.dumps(provenance, indent=2) + "\n")
        if args.wandb_project:
            import wandb
            wb_run = wb_call(wandb.init, project=args.wandb_project, entity=args.wandb_entity,
                             name=args.wandb_name, id=args.wandb_id, resume="never",
                             dir=args.out_dir, config={**vars(args), **provenance}, job_type="evaluation")
            if wb_run is not None:
                print(f"[wandb] evaluation run: {wb_run.url}", flush=True)
                with open(os.path.join(args.out_dir, "wandb-status.json"), "w") as fh:
                    json.dump({"url": wb_run.url, "id": wb_run.id, "status": "started"}, fh)
        # Model construction and logger initialization must not affect sampling RNGs.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed % (2**32))
        random.seed(args.seed)

        # ---------------- generate once per step count ----------------
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)
        rescale = float(cfg.network.rescale)
        n_part, n_dim = cfg.network.n_particles, cfg.network.n_dimension
        x0 = rescale * torch.randn((args.n_generate_samples, n_part, n_dim), generator=gen, device=device)
        if not getattr(cfg.network, "com_aware", False):
            x0 = remove_mean(x0)

        torch.manual_seed(args.seed)
        generated, stage_diagnostics = {}, {}
        sampling_nfe = {}
        for n_steps in args.steps:
            t0 = time.time()
            with torch.no_grad():
                out = sample_model(net, x0, n_steps=n_steps, batch_size=args.batch_size,
                                   velocity_solver="rk4", time_discretization_type=args.time_discretization_type,
                                   output_device="cpu")
            t_sample = time.time() - t0
            sampling_nfe[n_steps] = int(out[2])
            x1 = out[0].detach().cpu().float()
            logp = out[1].detach().cpu().float() if out[1] is not None else None
            del out

            t0 = time.time()
            E = energy.numpy(x1.numpy())
            finite = np.isfinite(E)
            if cfg.network.predict_divergence and logp is None:
                raise RuntimeError("Density checkpoint returned no log density")
            logw = None
            if logp is not None:
                logw = importance_source_energy(
                    x1, logp, com_augmentation=args.com_augmentation, unscale_factor=uf,
                ) - torch.from_numpy(E)
            print(f"[eval] n_steps={n_steps:>3}: {t_sample:5.1f}s sample, "
                  f"{time.time()-t0:5.1f}s energy, median {np.median(E[finite]):.4g} kT, "
                  f"non-finite {int((~finite).sum())}")
            artifacts = {} if args.eval_recipe else None
            if args.eval_recipe:
                # Save the expensive pool before MALA or metric computation can fail.
                torch.save({"x": x1, "energy": torch.from_numpy(E), "logq": logp, "logw": logw,
                            "generation_seed": args.seed, "generation_rng_state": gen.get_state(),
                            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate()}, os.path.join(args.out_dir, "proposal-pool.pt"))
            generated[n_steps], stage_diagnostics[n_steps] = sample_stages(
                x1, torch.from_numpy(E), logw, clip_fraction=clip_frac, energy=energy,
                mala_steps=args.posthoc_mala_steps, mala_step_size=args.mala_step_size,
                n_eval_samples=args.n_eval_samples, energy_cutoff=energy_cutoff,
                resampling=args.resampling, mala_adapt_steps=args.mala_adapt_steps,
                mala_target_accept=args.mala_target_accept, recipe_v1=bool(args.eval_recipe),
                checkpoint_interval=args.mala_checkpoint_interval, artifacts=artifacts,
            )
            if args.eval_recipe:
                torch.save(artifacts, os.path.join(args.out_dir, "resampling-mala.pt"))
            print(f"[eval] n_steps={n_steps}: stages={list(generated[n_steps])}", flush=True)

        evaluator = PeptideEnsembleEvaluator(
            num_eval_samples=args.n_eval_samples, do_plots=not args.no_plots,
            **({"fix_symmetry": True, "drop_unfixable_symmetry": False} if args.eval_recipe else {})
        )

        results = {
            "eval_recipe": args.eval_recipe,
            "provenance": provenance,
            "wandb_url": wb_run.url if wb_run is not None else None,
            "system": system.trajectory_name,
            "cfg_path": args.cfg_path,
            "mode": args.mode,
            "vectorized_divergence": bool(cfg.network.vectorized_divergence),
            "divergence_rescale": float(cfg.network.divergence_rescale),
            "sampler": "rk4" if args.mode == "pretrain_v" else "flowmap",
            "ckpt_path": args.ckpt_path,
            "params": "raw" if args.no_ema else f"EMA {ema_fac}",
            "n_generate_samples": args.n_generate_samples,
            "n_eval_samples": args.n_eval_samples,
            "energy_cutoff_kT": energy_cutoff,
            "energy_cutoff_order": "before logw filtering and evaluation draws; only cutoff branches",
            "seed": args.seed,
            "steps": args.steps,
            "time_discretization_type": args.time_discretization_type,
            "time_grids": {str(n): time_grid(n, args.time_discretization_type).tolist() for n in args.steps},
            "posthoc_mala_steps": args.posthoc_mala_steps,
            "mala_step_size_nm2": args.mala_step_size,
            "mala_adapt": args.mala_adapt_steps > 0,
            "mala_adapt_steps": args.mala_adapt_steps,
            "mala_target_accept": args.mala_target_accept,
            "mala_adaptation": "batch-mean Robbins-Monro on log(tau); freeze after warmup",
            "resampling": args.resampling,
            "resampling_order": "energy" if args.resampling == "stratified" else "input",
            "relaxation_target": "exp(-U/kT), mean-free physical nm coordinates",
            "logw_quantile_filter": clip_frac,
            "ess_convention": "full eligible pool before resampling; effective-sample-size is normalized, ess is count",
            "unscale_factor": uf,
            "ts_train_std": ts_std,
            "com_augmentation": args.com_augmentation,
            "com_weight_correction": "gaussian" if args.com_augmentation and cfg.network.predict_divergence else None,
            "data_source": paths["source"],
            "energy_units": "kT (ts OpenMMEnergy, MOLAR_GAS_CONSTANT_R), unclamped",
            "importance_weighting": (
                ("logw = -E_target - logp_model + log N(COM; 0, unscale_factor^2/N I_3)"
                 if args.com_augmentation else "logw = -E_target - logp_model") if cfg.network.predict_divergence
                else "none (no divergence head -> no logw; ESS undefined)"
            ),
            "splits": {},
        }

        ref_energies = {}
        for split in args.reference_splits:
            t0 = time.time()
            eval_ctx = ts_systems.build_eval_context(
                system,
                split=split,
                energy_fn=energy,
                num_eval_samples=args.n_eval_samples,
                data_dir=args.ts_data_dir,
                normalization_std=1.0,  # samples handed over in physical nm
                num_tica=args.num_tica,
            )
            if args.eval_recipe and len(eval_ctx.true_data) != args.n_eval_samples:
                raise ValueError("Reference split does not satisfy the exact recipe budget")
            true_E = eval_ctx.true_data.E_target
            if args.save_energies:
                ref_energies[f"reference_{split}"] = true_E.numpy()
            print(f"[eval] split={split}: {len(eval_ctx.true_data)} reference frames in "
                  f"{time.time()-t0:.0f}s, mean {true_E.mean():.2f} kT")

            ref_E_np = true_E.numpy().astype(np.float64)
            split_res = {
                "n_reference_used": int(len(eval_ctx.true_data)),
                "reference_energy_mean_kT": float(true_E.mean()),
                "reference_energy_std_kT": float(true_E.std()),
                "reference_energy_median_kT": float(true_E.median()),
                "reference_energy_min_kT": float(ref_E_np.min()),
                "reference_energy_max_kT": float(ref_E_np.max()),
                "steps": {},
            }

            for n_steps in args.steps:
                stages = generated[n_steps]
                prefix = f"{split}/{system.sequence}/{n_steps}steps"
                t0 = time.time()

                def log_image(fig, title, title_prefix=""):
                    import matplotlib.pyplot as plt
                    directory = os.path.join(args.out_dir, "plots", split, str(n_steps))
                    os.makedirs(directory, exist_ok=True)
                    name = f"{title_prefix}_{title}".replace("/", "_")
                    fig.savefig(os.path.join(directory, name + ".png"), bbox_inches="tight")
                    plt.close(fig)

                if args.eval_recipe:
                    from common.tsx import SamplesData
                    raw = {}
                    for stage_name, data in stages.items():
                        def recipe_image(fig, title, title_prefix=""):
                            import matplotlib.pyplot as plt
                            directory = os.path.join(args.out_dir, "plots", stage_name, split)
                            os.makedirs(directory, exist_ok=True)
                            plot_name = f"{title_prefix}_{title}".replace("/", "_")
                            fig.savefig(os.path.join(directory, plot_name + ".png"), bbox_inches="tight")
                            if wb_run is not None:
                                wb_call(wb_run.log, {f"{stage_name}/{split}/plots/{plot_name}": wandb.Image(fig)})
                            plt.close(fig)
                        # Symmetry fixing and metric randomness operate only on saved copies,
                        # after the continuous MALA trajectory has finished.
                        metric_copy = SamplesData(data.samples.clone(), data.E_target.clone())
                        metrics = evaluator.evaluate({stage_name: metric_copy}, eval_ctx,
                            log_image_fn=None if args.no_plots else recipe_image, prefix=prefix)
                        for key, value in metrics.items():
                            suffix = key.removeprefix(prefix + "/")
                            if not suffix.startswith(stage_name + "/"):
                                suffix = stage_name + "/" + suffix
                            raw[prefix + "/" + suffix] = value
                        logged = {f"{stage_name}/{split}/{key.removeprefix(prefix + '/' + stage_name + '/')}": float(value)
                                  for key, value in raw.items() if key.startswith(prefix + "/" + stage_name + "/")}
                        logged.update({f"{stage_name}/{split}/{key}": value
                                       for key, value in stage_diagnostics[n_steps][stage_name].items()
                                       if isinstance(value, (int, float))})
                        if wb_run is not None:
                            wb_call(wb_run.log, logged)
                        with open(os.path.join(args.out_dir, f"metrics-{stage_name}-{split}.json"), "w") as fh:
                            json.dump(logged, fh, indent=2)
                        print(f"[eval] metrics complete: {stage_name}/{split}", flush=True)
                else:
                    raw = evaluator.evaluate({name: data for name, data in stages.items() if len(data)}, eval_ctx,
                                             log_image_fn=None if args.no_plots else log_image,
                                             prefix=prefix)
                entry = {name: dict(stage_diagnostics[n_steps][name]) for name in stages}
                for key, value in raw.items():
                    suffix = key.removeprefix(prefix + "/")
                    name, _, metric = suffix.partition("/")
                    if name in entry:
                        try:
                            entry[name][metric] = float(value)
                        except (TypeError, ValueError):
                            pass
                for name in stages:
                    metrics = entry[name]
                    ess = metrics.get("effective-sample-size", metrics.get("effective-sample-size-unfiltered"))
                    print(f"[eval] {split} n_steps={n_steps} {name}: "
                          f"energy-w2={metrics.get('energy-w2', float('nan')):.4g} "
                          f"torus-w2={metrics.get('torus-w2', float('nan')):.4g} "
                          f"tica-w2={metrics.get('tica-w2', float('nan')):.4g} "
                          f"ESS={float('nan') if ess is None else ess:.4g}")
                entry["nfe"] = sampling_nfe[n_steps]
                entry["t_metric_s"] = time.time() - t0
                split_res["steps"][str(n_steps)] = entry

            results["splits"][split] = split_res

        if args.save_energies:
            np.savez(
                os.path.join(args.out_dir, "energies.npz"),
                **{f"steps_{k}_{name}": data.E_target.numpy()
                   for k in args.steps for name, data in generated[k].items()},
                **ref_energies,
            )
            print("[eval] wrote", os.path.join(args.out_dir, "energies.npz"))

        out = os.path.join(args.out_dir, "ts_metrics.json")
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print("[eval] wrote", out)
        completed = True
    finally:
        energy.close()
        if wb_run is not None:
            wb_call(wb_run.finish, exit_code=0 if completed else 1)
        if args.wandb_project:
            with open(os.path.join(args.out_dir, "wandb-status.json"), "w") as fh:
                json.dump({"url": wb_run.url if wb_run is not None else None,
                           "id": args.wandb_id, "evaluation_complete": completed,
                           "finish_succeeded": wb_run is not None and not wb_errors,
                           "errors": wb_errors}, fh, indent=2)


if __name__ == "__main__":
    main()
