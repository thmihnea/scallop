"""
Nicholas M. Boffi
10/5/25

Code for basic wandb visualization and logging.
"""

import functools
import json
import shutil
import signal
import sys
import tempfile
from types import NoneType
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import os
import seaborn as sns
import wandb
from flax.serialization import to_bytes
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt
from ml_collections import config_dict

from . import datasets, dist_utils, fid_utils, flow_map, state_utils
from .losses import _hutchinson_divergence_with_phi
from .free_energy_utils import bar

Parameters = Dict[str, Dict]


def get_params_for_sampling(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    param_type: str = "visual",
) -> jnp.ndarray:
    """
    Get the appropriate parameters for sampling (visualization or FID).

    Args:
        cfg: Configuration
        train_state: Current training state
        param_type: Either "visual" or "fid" to select the right config parameter

    Returns:
        Parameters to use for sampling (unreplicated for single-device use)
    """
    # Determine which config parameter to check based on param_type
    if param_type == "visual":
        config_param = "visual_ema_factor"
    elif param_type == "fid":
        config_param = "fid_ema_factor"
    elif param_type == "bpd":
        config_param = "bpd_ema_factor"
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

    # Select which parameters to use
    if (
        hasattr(cfg.logging, config_param)
        and getattr(cfg.logging, config_param) is not None
    ):
        ema_factor = getattr(cfg.logging, config_param)
        # Use EMA parameters with specified factor
        if ema_factor in train_state.ema_params:
            params = train_state.ema_params[ema_factor]
        else:
            print(
                f"Warning: EMA factor {ema_factor} not found in ema_params. Using instantaneous params."
            )
            params = train_state.params
    else:
        # Use instantaneous parameters (default)
        params = train_state.params

    # Visual/BPD uses unreplicated params, FID uses replicated params for pmap
    if param_type in ("visual", "bpd"):
        return dist_utils.safe_unreplicate(cfg, params)
    else:
        return params


def compute_fid_on_the_fly(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    n_samples: int = 10000,
    batch_size: int = 256,
    n_steps_flow: int = 8,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute FID on the fly during training using distributed sampling.

    Args:
        cfg: Configuration
        statics: Static arguments containing dataset stats
        train_state: Current training state
        prng_key: Random key
        n_samples: Number of samples to generate (default 10,000)
        batch_size: Batch size for sampling (will be split across devices)
        n_steps_flow: Number of steps for flow map models

    Returns:
        Tuple of FID score and updated PRNG key
    """
    # Check if FID reference statistics are available
    if not hasattr(cfg.logging, "fid_stats_path"):
        print(
            "Warning: No FID reference statistics path configured. Set cfg.logging.fid_stats_path"
        )
        return jnp.nan, prng_key

    # Load reference statistics
    try:
        fid_stats = np.load(cfg.logging.fid_stats_path)
        mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
    except FileNotFoundError:
        print(f"Warning: FID stats file not found at {cfg.logging.fid_stats_path}")
        return jnp.nan, prng_key
    except Exception as e:
        print(f"Warning: Error loading FID stats: {e}")
        return jnp.nan, prng_key

    # Get number of devices and adjust batch size
    per_device_batch_size = batch_size // cfg.training.ndevices
    if batch_size % cfg.training.ndevices != 0:
        per_device_batch_size += 1
        batch_size = per_device_batch_size * cfg.training.ndevices

    # Use flow map steps
    n_steps = n_steps_flow

    # Get pre-initialized FID network from statics
    if statics.inception_fn is None:
        print("Warning: Inception network not initialized. FID computation disabled.")
        return jnp.nan, prng_key
    inception_fn = statics.inception_fn

    # Get pmap sampler function based on number of devices
    if cfg.training.ndevices == 1:
        sampler = flow_map.batch_sample
    else:
        sampler = flow_map.pmap_batch_sample

    # Initialize statistics for Welford's online algorithm
    n_seen = 0
    mu_gen = None
    M2_gen = None

    # Generate samples in batches
    n_full_batches = n_samples // batch_size
    remainder = n_samples % batch_size

    for batch_idx in range(n_full_batches + (1 if remainder > 0 else 0)):
        # Determine current batch size
        if batch_idx == n_full_batches and remainder > 0:
            current_batch_size = remainder
            current_per_device_batch = (
                remainder + cfg.training.ndevices - 1
            ) // cfg.training.ndevices
            padded_batch_size = current_per_device_batch * cfg.training.ndevices
        else:
            current_batch_size = batch_size
            current_per_device_batch = per_device_batch_size
            padded_batch_size = batch_size

        # Generate noise and reshape for pmap
        prng_key, sample_key = jax.random.split(prng_key)
        x0_full = statics.sample_rho0(padded_batch_size, sample_key)

        if cfg.training.ndevices > 1:
            x0_batched = x0_full.reshape(
                cfg.training.ndevices, current_per_device_batch, *cfg.problem.image_dims
            )
        else:
            x0_batched = x0_full

        # Handle labels for conditional generation
        if cfg.training.conditional:
            if cfg.training.class_dropout > 0:
                labels = jnp.array(
                    np.random.choice(cfg.problem.num_classes + 1, padded_batch_size)
                ).reshape(cfg.training.ndevices, current_per_device_batch)
            else:
                labels = jnp.array(
                    np.random.choice(cfg.problem.num_classes, padded_batch_size)
                ).reshape(cfg.training.ndevices, current_per_device_batch)
        else:
            labels = None

        # Get parameters for FID sampling
        params_for_fid = get_params_for_sampling(cfg, train_state, param_type="fid")

        # Sample images across devices
        imgs_batched = sampler(
            train_state.apply_fn,
            params_for_fid,
            x0_batched,
            n_steps,
            labels,
        )

        # Flatten from devices and clip
        imgs = imgs_batched.reshape(padded_batch_size, *cfg.problem.image_dims)
        imgs = jnp.clip(imgs, -1, 1)

        # Only keep the actual samples we need
        imgs = imgs[:current_batch_size]

        # Convert from NCHW to NHWC for Inception
        imgs = imgs.transpose(0, 2, 3, 1)

        # Extract Inception features (no need for pmap here since we're back to single batch)
        features = fid_utils.resize_and_incept(imgs, inception_fn)
        features = np.asarray(np.squeeze(features))

        # Update running statistics using Welford's method
        batch_mean = features.mean(0)
        batch_cov = (
            np.cov(features, rowvar=False)
            if features.shape[0] > 1
            else np.zeros((features.shape[1], features.shape[1]))
        )

        n_seen += current_batch_size

        if mu_gen is None:
            mu_gen = batch_mean
            M2_gen = (
                batch_cov * (current_batch_size - 1)
                if current_batch_size > 1
                else np.zeros_like(batch_cov)
            )
        else:
            delta = batch_mean - mu_gen
            mu_gen += delta * current_batch_size / n_seen
            M2_gen += (
                batch_cov * (current_batch_size - 1)
                + np.outer(delta, delta)
                * (n_seen - current_batch_size)
                * current_batch_size
                / n_seen
            )

    # Compute final covariance and FID
    sigma_gen = M2_gen / (n_seen - 1)
    fid_score = fid_utils.fid_from_stats(mu_gen, sigma_gen, mu_real, sigma_real)

    return float(fid_score), prng_key


def _save_ckpt_on_signal(
    cfg: config_dict.ConfigDict, train_state: state_utils.EMATrainState
) -> None:
    save_state(train_state, cfg)
    sys.exit(0)


def register_signal_handlers(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
) -> None:
    """Drop a checkpoint on SIGTERM or SIGINT."""
    handler = functools.partial(_save_ckpt_on_signal, cfg, train_state)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def save_state(
    train_state: state_utils.EMATrainState,
    cfg: config_dict.ConfigDict,
) -> None:
    """Save flax training state."""

    with open(
        f"{cfg.logging.output_folder}/{cfg.logging.output_name}_{dist_utils.safe_index(cfg, train_state.step)//cfg.logging.save_freq}.pkl",
        "wb",
    ) as f:
        state = jax.device_get(dist_utils.safe_unreplicate(cfg, train_state))
        f.write(to_bytes(state))


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return jnp.linalg.norm(flat_params)

def _save_ax(
    ax,
    savepath: str,
    *,
    dpi: int = 200,
    pad_inches: float = 0.02,
) -> None:
    """Save one axes' region to a PNG file.

    Uses bbox cropping on ``fig.savefig`` only for this call; the figure and other
    subplots are unchanged, so later full-figure saves still include every panel.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Use axes window extent (not tightbbox) and slightly expand it so
    # border strokes (spines) are never clipped in per-axes exports.
    bbox = ax.get_window_extent(renderer=renderer).expanded(1.03, 1.06).transformed(
        fig.dpi_scale_trans.inverted()
    )
    dirname = os.path.dirname(savepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    fig.savefig(savepath, bbox_inches=bbox, pad_inches=pad_inches, dpi=dpi)


def _metrics_to_jsonable(obj):
    """Recursively convert nested dicts/lists with NumPy/JAX scalars to JSON-safe Python."""
    if isinstance(obj, dict):
        return {str(k): _metrics_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_metrics_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _metrics_to_jsonable(obj.item())
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (float, int)):
        return obj
    try:
        arr = np.asarray(obj)
    except (TypeError, ValueError):
        return float(obj)
    if arr.ndim == 0:
        return _metrics_to_jsonable(arr.item())
    return arr.tolist()


def log_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    grads: jnp.ndarray,
    loss_value: float,
    loss_metrics: Dict[str, jnp.ndarray],
    loss_fn_args: Tuple,
    prng_key: jnp.ndarray,
    step_time: float,
) -> jnp.ndarray:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""

    grads = dist_utils.safe_unreplicate(cfg, grads)
    loss_value = dist_utils.safe_index(cfg, jnp.array(loss_value))
    loss_metrics = dist_utils.safe_unreplicate(cfg, loss_metrics)
    step = dist_utils.safe_index(cfg, train_state.step)
    wandb_step = int(np.asarray(step).item())
    learning_rate = statics.schedule(step)

    # Standard metrics
    metrics = {
        f"loss": loss_value,
        f"grad": compute_grad_norm(grads),
        f"learning_rate": learning_rate,
        f"step_time": step_time,
    }
    metrics.update(loss_metrics)
    
    # Compute FID on-the-fly if enabled and at the right frequency
    if (
        hasattr(cfg.logging, "fid_freq")
        and cfg.logging.fid_freq > 0
        and (step % cfg.logging.fid_freq) == 0
        and step > 0
    ):
        try:
            # Get step counts configuration - can be a list or single value
            steps_config = getattr(cfg.logging, "fid_n_steps_flow", 8)

            # Convert to list if single value
            if isinstance(steps_config, (list, tuple)):
                n_steps_list = list(steps_config)
            else:
                n_steps_list = [steps_config]

            # Compute FID for each step count
            for n_steps in n_steps_list:
                fid_score, prng_key = compute_fid_on_the_fly(
                    cfg,
                    statics,
                    train_state,
                    prng_key,
                    n_samples=getattr(cfg.logging, "fid_n_samples", 10000),
                    batch_size=getattr(cfg.logging, "fid_batch_size", 256),
                    n_steps_flow=n_steps,
                )
                # Log with step-specific key
                metrics[f"fid_{n_steps}_steps"] = fid_score
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")

    # Compute CelebA BPD on a single batch if enabled
    if (
        cfg.problem.target == "celeb_a"
        and hasattr(cfg.logging, "bpd_freq")
        and cfg.logging.bpd_freq > 0
        and (step % cfg.logging.bpd_freq) == 0
        and step > 0
    ):
        try:
            steps_config = getattr(cfg.logging, "bpd_n_steps", [1, 2, 4, 8])
            if isinstance(steps_config, (list, tuple)):
                n_steps_list = tuple(steps_config)
            else:
                n_steps_list = (int(steps_config),)
            bpd_metrics = compute_celeba_bpd_on_batch(
                cfg, train_state, loss_fn_args, n_steps_list
            )
            metrics.update(bpd_metrics)
        except Exception as e:
            print(f"Warning: BPD computation failed: {e}")

    if (dist_utils.safe_index(cfg, train_state.step) % cfg.logging.visual_freq) == 0:
        make_loss_fn_args_plot(
            cfg, statics, train_state, loss_fn_args, wandb_step=wandb_step
        )
        metrics, prng_key = log_metrics_eval(
            metrics, cfg, statics, train_state, prng_key, wandb_step=wandb_step
        )

    wandb.log(metrics, step=wandb_step)
    if (dist_utils.safe_index(cfg, train_state.step) % cfg.logging.save_freq) == 0:
        save_state(train_state, cfg)

    return prng_key

def log_metrics_eval(
    metrics,
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    wandb_step: Optional[int] = None,
    log_stratified_div_error: bool = True,
) -> Tuple[Dict, jnp.ndarray]:
    """Log metrics to wandb."""
    if wandb_step is None:
        wandb_step = int(
            np.asarray(dist_utils.safe_index(cfg, train_state.step)).item()
        )
    is_pretraining = cfg.network.divergence_mode in ["pretrain_flowmap", "pretrain_v"]
    if not is_pretraining and log_stratified_div_error:
        div_metrics = make_stratified_div_loss(cfg, statics, train_state, prng_key)
        metrics.update(div_metrics)

    is_image_target = cfg.problem.target in ["cifar10", "celeb_a"]
    # Apply resampling and free-energy calculation only for non-pretraining cases
    if is_image_target:
        target = None
    else:
        target = statics.distr
    is_resample_and_calc_ess = not is_image_target and not is_pretraining and cfg.problem.target != "checker"

    
    if not is_image_target:
        prng_key = make_lowd_plot(
            cfg,
            statics,
            train_state,
            prng_key,
            target,
            is_resample_and_calc_ess,
            wandb_step=wandb_step,
        )
        if not is_pretraining:
            if cfg.problem.d == 2:
                prng_key = make_likelihood_heatmap_plot(
                    cfg, statics, train_state, prng_key, wandb_step=wandb_step
                )
    else:
        prng_key = make_image_plot(
            cfg, statics, train_state, prng_key, wandb_step=wandb_step
        )

    return metrics, prng_key


def _resample_and_calc_ess(key, xhat, logp_model, logp_target, clip_max=0.998, clip_min=0.00, return_weights=False):
    """
    Reweight the samples based on the target log probability and calculate the ESS.
    Clip the highest <clip> of the weights
    """
    log_weights = logp_target - logp_model
    upper_bound = jnp.quantile(log_weights, clip_max)
    lower_bound = jnp.quantile(log_weights, clip_min)
    unmask = jnp.logical_and(log_weights >= lower_bound, log_weights <= upper_bound)
    xhat_ = xhat[unmask]
    logp_target_ = logp_target[unmask]
    log_weights_ = log_weights[unmask]
    weights = jax.nn.softmax(log_weights_)
    indices = jax.random.choice(key, jnp.arange(xhat_.shape[0]), shape=(xhat.shape[0],), p=weights, replace=True, axis=0)
    xhat_resampled = xhat_[indices]
    xhat_resampled_energy = -logp_target_[indices]
    ess = 1.0 / jnp.sum(weights**2)
    ess /= xhat_.shape[0]
    if return_weights:
        return xhat_resampled, xhat_resampled_energy, ess, weights
    return xhat_resampled, xhat_resampled_energy, ess

    # clipped_log_weights = jnp.clip(log_weights, a_min=lower_bound, a_max=upper_bound)
    # weights = jax.nn.softmax(clipped_log_weights)
    # indices = jax.random.choice(key, jnp.arange(xhat.shape[0]), shape=(xhat.shape[0],), p=weights, replace=True, axis=0)
    # xhat_resampled = xhat[indices]
    # xhat_resampled_energy = -logp_target[indices]
    # ess = 1.0 / jnp.sum(weights**2)
    # ess /= xhat.shape[0]
    # if return_weights:
    #     return xhat_resampled, xhat_resampled_energy, ess, weights
    # return xhat_resampled, xhat_resampled_energy, ess

def _clip_work_values(work, clip_max=0.90, clip_min=0.10):
    """Clips work values to prevent exponential overflow in Jarzynski/BAR."""
    upper_bound = jnp.quantile(work, clip_max)
    lower_bound = jnp.quantile(work, clip_min)
    return jnp.clip(work, a_min=lower_bound, a_max=upper_bound)

def _make_r2_plot(ax, logp_model, logp_target, title, fontsize):
    """Makes a plot of the R^2 of the model's log probability and the target log probability."""
    q02_m = jnp.quantile(logp_model, 0.02)
    q98_m = jnp.quantile(logp_model, 0.98)
    q02_t = jnp.quantile(logp_target, 0.02)
    q98_t = jnp.quantile(logp_target, 0.98)

    outliers = jnp.logical_or(
        jnp.logical_or(logp_model < q02_m, logp_model > q98_m),
        jnp.logical_or(logp_target < q02_t, logp_target > q98_t)
    )
    outlier_frac = jnp.mean(outliers)

    valid_idx = jnp.logical_not(outliers)
    logp_model_f = logp_model[valid_idx]
    logp_target_f = logp_target[valid_idx]
    
    b = jnp.mean(logp_target_f - logp_model_f)
    y_pred = b + logp_model_f
    mse = jnp.mean((logp_target_f - y_pred)**2)
    var_target = jnp.var(logp_target_f)

    eps = jnp.finfo(logp_target_f.dtype).eps
    r2 = 1. - (mse / jnp.maximum(var_target, eps))

    ax.scatter(logp_model_f, logp_target_f, alpha=0.5)
    
    # xlims = [float(jnp.min(logp_model_f)), float(jnp.max(logp_model_f))]
    # ylims = [x + float(b) for x in xlims]
    # ax.plot(xlims, ylims, 'r--', alpha=0.5, zorder=0)
    ax.set_title(title, fontsize=fontsize)
    return ax, r2
    

def make_lowd_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    target,
    is_resample_and_calc_ess: bool = False,
    wandb_step: int = 0,
) -> jnp.ndarray:
    # Use flow map batch sampler for single-device visualization
    batch_sample = flow_map.batch_sample

    # Get parameters for visualization
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    steps = list(
        getattr(
            cfg.logging,
            "n_sampling_steps",
            [1, 2, 5, 10, 25]
            if cfg.network.divergence_mode not in ["pretrain_flowmap", "pretrain_v"]
            else [1, 10, 100],
        )
    )
    if getattr(cfg.logging, "with_dorpi5", True):
        steps.append(None)
    # Overwrite steps if pretraining
    if cfg.network.divergence_mode in ["pretrain_flowmap", "pretrain_v"]:
        steps = [1, 10, 100]
    titles = ["base_and_target"] + [f"{step}_steps" for step in steps]
    all_metrics = {}

    ## extract target samples
    n_samples = cfg.logging.n_samples if hasattr(cfg.logging, "n_samples") else cfg.logging.plot_bs
    x1s_list = []
    gathered = 0
    while gathered < n_samples:
        batch = next(statics.ds)
        x1s_list.append(batch * cfg.network.unscale_factor)
        gathered += batch.shape[0]
    x1s = np.concatenate(x1s_list, axis=0)[:n_samples]
    x1s_energy = target.energy(x1s)
    # target.log_prob(x) == -target.energy(x); cache once to avoid repeated expensive target calls.
    logp_x1s_target = -jnp.asarray(x1s_energy)

    ## draw multi-step samples from the model
    x0s = statics.sample_rho0(n_samples, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), n_samples, cfg.problem.d))
    xhats_energy = np.zeros((len(steps), n_samples))
    xhats_resampled = np.zeros((len(steps), n_samples, cfg.problem.d))
    xhats_resampled_energy = np.zeros((len(steps), n_samples))
    logps = np.zeros((len(steps), n_samples))

    xhats_inv = np.zeros((len(steps), n_samples, cfg.problem.d))
    logps_inv = np.zeros((len(steps), n_samples))

    labels = -jnp.ones(n_samples)
    is_pretraining = cfg.network.divergence_mode in ["pretrain_flowmap", "pretrain_v"]
    clip_max = 1.0 - cfg.logging.importance_weight_clip
    clip_min = 0.0
    res_dir = getattr(cfg.logging, "res_dir", None)
    if res_dir is not None:
        os.makedirs(res_dir, exist_ok=True)
    # Panel renders always write PNGs; avoid f"{None}/..." (literal None/) when res_dir unset.
    panel_dir = res_dir if res_dir is not None else tempfile.mkdtemp(prefix="iF2D2_lowd_plot_")
    _cleanup_panel_dir = res_dir is None
    for kk, step in enumerate(steps):
        use_dorpi5 = step is None
        if is_pretraining:
            # x_out = _sample_model_nsteps(
            #     train_state.apply_fn,
            #     params_for_visual,
            #     x0s,
            #     step,
            #     labels,
            #     batch_size=cfg.logging.plot_bs,
            #     cfg=cfg,
            # )
            x_out, _, nfe = train_state.apply_fn(
                params_for_visual,
                x0s,
                labels,
                batch_size=cfg.logging.plot_bs,
                n_steps=step,
                return_nfe=True,
                method="sample_forward",
            )
        else:

            x_out, logps[kk], nfe = train_state.apply_fn(
                params_for_visual,
                x0s,
                labels,
                batch_size=cfg.logging.plot_bs,
                n_steps=step,
                return_nfe=True,
                method="sample_forward",
            )
            x0s_hat, _, logps_inv[kk], nfe = train_state.apply_fn(
                params_for_visual,
                x1s,
                labels,
                batch_size=cfg.logging.plot_bs,
                n_steps=step,
                return_nfe=True,
                method="sample_inverse",
            )
            xhats_inv[kk] = x0s_hat.reshape(x0s_hat.shape[0], cfg.problem.d)
        if use_dorpi5:
            steps[kk] = nfe
            titles[-1] = rf"{nfe}_steps_Dopri5"
        xhats[kk] = x_out.reshape(x_out.shape[0], cfg.problem.d)

    ## construct the figure
    nrows = 1 if not is_resample_and_calc_ess else 5
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    axs_grid = np.atleast_2d(axs)

    def _render_panel_to_file(
        savepath: str,
        draw_fn: Callable,
        *,
        panel_title: Optional[str] = None,
    ) -> np.ndarray:
        panel_fig, panel_ax = plt.subplots(
            figsize=(fw, fh),
            constrained_layout=True,
        )
        panel_ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        panel_ax.tick_params(axis="both", labelsize=fontsize)
        draw_fn(panel_ax)
        if panel_title is not None:
            panel_ax.set_title(panel_title, fontsize=fontsize)
        dirname = os.path.dirname(savepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        panel_fig.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(panel_fig)
        return plt.imread(savepath)

    def _show_panel_image(ax, panel_img: np.ndarray, *, title: Optional[str] = None) -> None:
        ax.imshow(panel_img)
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title, fontsize=fontsize)

    plot_true = cfg.problem.target in ["checker", "gmm"]
    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        if nrows > 1:
            ax1 = axs_grid[0, jj]  # sample plots
            ax2 = axs_grid[1, jj]  # reweighted sample plots
            ax3 = axs_grid[2, jj]  # fwd R^2 plot
            ax4 = axs_grid[3, jj]  # bwd R^2 plot
            ax5 = axs_grid[4, jj]  # energy plot
        else:
            ax1 = axs_grid[0, jj]
            ax2 = None
        if jj == 0:
            sample_path = f"{panel_dir}/samples_{title}.png"

            def _draw_samples_base(panel_ax):
                if plot_true:
                    target.plot_samples(
                        panel_ax,
                        x0s,
                        fontsize=fontsize,
                        title=None,
                        color="black",
                    )
                target.plot_samples(
                    panel_ax,
                    x1s,
                    fontsize=fontsize,
                    title=None,
                    color="C0",
                )

            sample_img = _render_panel_to_file(sample_path, _draw_samples_base)
            _show_panel_image(ax1, sample_img, title=title)
        else:
            sample_path = f"{panel_dir}/samples_{title}.png"

            def _draw_samples_step(panel_ax):
                if plot_true:
                    target.plot_samples(
                        panel_ax,
                        x1s,
                        fontsize=fontsize,
                        title=None,
                        color="C0",
                    )
                target.plot_samples(
                    panel_ax,
                    xhats[jj - 1],
                    fontsize=fontsize,
                    title=None,
                    color="black",
                )

            sample_img = _render_panel_to_file(sample_path, _draw_samples_step)
            _show_panel_image(ax1, sample_img, title=title)

        if ax2 is not None:
            if jj == 0:
                xhat_ = x0s
                if cfg.problem.base == "mean-free-gaussian":
                    logps_ = _base_log_prob_mean_free(cfg, xhat_)
                elif cfg.problem.base == "gaussian":
                    logps_ = _base_log_prob(cfg, xhat_)
                else:
                    raise ValueError(f"Invalid base distribution: {cfg.problem.base}")
            else:
                xhat_ = xhats[jj - 1]
                logps_ = logps[jj - 1]

            # target.log_prob = -target.energy
            logp_target = target.log_prob(xhat_)
            xhats_energy[jj - 1] = -logp_target
            
            subkey = jax.random.split(prng_key)[0]
            xhat_resampled, xhats_resampled_energy[jj - 1], ess = _resample_and_calc_ess(
                subkey, xhat_, logps_, logp_target, clip_max=clip_max, clip_min=clip_min
            )
            
            xhats_resampled[jj - 1] = xhat_resampled.reshape(xhat_resampled.shape[0], -1)
            
            # calculate FWD Jarzynski, from x0 to x1
            work_fwd = -logp_target + logps_

            # calculate BWD Jarzynski, from x1 to x0
            work_bwd = logp_x1s_target - logps_inv[jj - 1]

            work_fwd = _clip_work_values(work_fwd, clip_max=clip_max, clip_min=clip_min)
            work_bwd = _clip_work_values(work_bwd, clip_max=clip_max, clip_min=clip_min)
            free_energy_bar, free_energy_fwd, free_energy_bwd = bar(work_fwd, work_bwd)

            resampled_path = f"{panel_dir}/samples_{title}_resampled.png"

            def _draw_resampled(panel_ax):
                target.plot_samples(
                    panel_ax,
                    xhat_resampled,
                    fontsize=fontsize,
                    title=None,
                    color="black",
                )

            resampled_img = _render_panel_to_file(resampled_path, _draw_resampled)
            _show_panel_image(ax2, resampled_img)
            ax2.text(
                0.05,
                0.95,
                f"ESS: {ess:.5f}",
                transform=ax2.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
            )
            ax2.text(
                0.05,
                0.90,
                f"FWD Free-Energy: {free_energy_fwd:{'.0e' if abs(float(free_energy_fwd)) > 1000 else '.2f'}}",
                transform=ax2.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
            )
            ax2.text(
                0.05,
                0.85,
                f"BWD Free-Energy: {free_energy_bwd:{'.0e' if abs(float(free_energy_bwd)) > 1000 else '.2f'}}",
                transform=ax2.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
            )
            ax2.text(
                0.05,
                0.80,
                f"BAR Free-Energy: {free_energy_bar:{'.0e' if abs(float(free_energy_bar)) > 1000 else '.2f'}}",
                transform=ax2.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
            )

            r2_fwd_path = f"{panel_dir}/r2_{title}_fwd.png"
            r2_tmp_fig, r2_tmp_ax = plt.subplots()
            r2_tmp_ax, r2_fwd = _make_r2_plot(r2_tmp_ax, logps_, logp_target, "FWD R2", fontsize)
            plt.close(r2_tmp_fig)

            def _draw_r2_fwd(panel_ax):
                _make_r2_plot(panel_ax, logps_, logp_target, "FWD R2", fontsize)

            r2_fwd_img = _render_panel_to_file(r2_fwd_path, _draw_r2_fwd)
            _show_panel_image(ax3, r2_fwd_img)
            ax3.text(
                0.05,
                0.95,
                f"R^2: {r2_fwd:.5f}",
                transform=ax3.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
            )

            r2_bwd_path = f"{panel_dir}/r2_{title}_bwd.png"
            r2_tmp_fig, r2_tmp_ax = plt.subplots()
            r2_tmp_ax, r2_bwd = _make_r2_plot(r2_tmp_ax, logp_target, logps_, "BWD R2", fontsize)
            plt.close(r2_tmp_fig)

            def _draw_r2_bwd(panel_ax):
                _make_r2_plot(panel_ax, logp_target, logps_, "BWD R2", fontsize)

            r2_bwd_img = _render_panel_to_file(r2_bwd_path, _draw_r2_bwd)
            _show_panel_image(ax4, r2_bwd_img)
            ax4.text(
                0.05,
                0.95,
                f"R^2: {r2_bwd:.5f}",
                transform=ax4.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
            )

            flare_colors = sns.color_palette("flare_r", 3)
            energy_path = f"{panel_dir}/energy_{title}.png"

            def _draw_energy(panel_ax):
                target.plot_energy_hist(
                    panel_ax,
                    x1s_energy,
                    fontsize=fontsize,
                    color=flare_colors[0],
                    label="True data",
                )
                target.plot_energy_hist(
                    panel_ax,
                    xhats_energy[jj - 1],
                    fontsize=fontsize,
                    color=flare_colors[1],
                    label="Proposal",
                )
                target.plot_energy_hist(
                    panel_ax,
                    xhats_resampled_energy[jj - 1],
                    fontsize=fontsize,
                    color=flare_colors[2],
                    label="Reweighted proposal",
                )
                panel_ax.legend(fontsize=fontsize)

            energy_img = _render_panel_to_file(energy_path, _draw_energy)
            _show_panel_image(ax5, energy_img)

            if jj > 0:
                all_metrics.setdefault(title, {})
                all_metrics[title]["ess"] = ess
                all_metrics[title]["free_energy_fwd"] = free_energy_fwd
                all_metrics[title]["free_energy_bwd"] = free_energy_bwd
                all_metrics[title]["free_energy_bar"] = free_energy_bar
                all_metrics[title]["r2_fwd"] = r2_fwd
                all_metrics[title]["r2_bwd"] = r2_bwd

    wandb.log({"samples": wandb.Image(fig)}, step=wandb_step)

    # if hasattr(target, "compute_metrics"):
    #     for j in range(len(steps)):
    #         step_title = titles[j + 1]
    #         all_metrics.setdefault(step_title, {})
    #         metrics = target.compute_metrics(
    #             xhats[j], 
    #             x1s, 
    #             energy_samples=xhats_energy[j], 
    #             energy_ref=x1s_energy
    #         )
    #         all_metrics[step_title]["metrics"] = metrics
    #         for key, value in metrics.items():
    #             wandb.log(
    #                 {f"metrics/{key}/{steps[j]}-steps": value}, step=wandb_step
    #             )
    #         metrics = target.compute_metrics(
    #             xhats_resampled[j], 
    #             x1s, 
    #             energy_samples=xhats_resampled_energy[j], 
    #             energy_ref=x1s_energy,
    #         )
    #         all_metrics[step_title]["metrics_resampled"] = metrics
    #         for key, value in metrics.items():
    #             wandb.log(
    #                 {f"metrics/{key}-resampled/{steps[j]}-steps": value},
    #                 step=wandb_step,
    #             )
    saved_samples = {
        "x0s": x0s,
        "x1s": x1s,
        "x1s_energy": x1s_energy,
        "xhats": xhats,
        "xhats_energy": xhats_energy,
        "logps": logps,
        "xhats_resampled": xhats_resampled,
        "xhats_resampled_energy": xhats_resampled_energy,
        "xhats_inv": xhats_inv,
        "logps_inv": logps_inv,
        "steps": np.asarray(steps),
        "titles": np.asarray(titles),
        "meta": {
            "target": cfg.problem.target,
            "base": cfg.problem.base,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "plot_true": plot_true,
            "fw": fw,
            "fh": fh,
            "fontsize": fontsize,
        },
    }

    if _cleanup_panel_dir:
        shutil.rmtree(panel_dir, ignore_errors=True)

    if res_dir is not None:
        os.makedirs(res_dir, exist_ok=True)

        samples_path = os.path.join(res_dir, "saved_samples.npz")
        np.savez(
            samples_path,
            x0s=np.asarray(saved_samples["x0s"]),
            x1s=np.asarray(saved_samples["x1s"]),
            x1s_energy=np.asarray(saved_samples["x1s_energy"]),
            xhats=np.asarray(saved_samples["xhats"]),
            xhats_energy=np.asarray(saved_samples["xhats_energy"]),
            logps=np.asarray(saved_samples["logps"]),
            xhats_resampled=np.asarray(saved_samples["xhats_resampled"]),
            xhats_resampled_energy=np.asarray(saved_samples["xhats_resampled_energy"]),
            xhats_inv=np.asarray(saved_samples["xhats_inv"]),
            logps_inv=np.asarray(saved_samples["logps_inv"]),
            steps=np.asarray(saved_samples["steps"]),
            titles=np.asarray(saved_samples["titles"], dtype=str),
        )

        samples_meta_path = os.path.join(res_dir, "saved_samples_meta.json")
        with open(samples_meta_path, "w", encoding="utf-8") as f:
            json.dump(_metrics_to_jsonable(saved_samples["meta"]), f, indent=2)

    if all_metrics and res_dir is not None:
        os.makedirs(res_dir, exist_ok=True)
        metrics_path = os.path.join(res_dir, f"all_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_metrics_to_jsonable(all_metrics), f, indent=2)

    return prng_key


def make_image_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    wandb_step: int = 0,
) -> jnp.ndarray:
    """Make a plot of the generated images."""
    # Use flow map batch sampler for single-device visualization
    batch_sample = flow_map.batch_sample

    # Get parameters for visualization (already unreplicated)
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 1, 1
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 4, 8, 16]

    titles = [rf"{step}-step" for step in steps]

    ## draw multi-step samples from the model
    n_images = 16
    x0s = statics.sample_rho0(n_images, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), n_images, *cfg.problem.image_dims))

    ## set up conditioning information
    if cfg.training.conditional:
        if cfg.training.class_dropout > 0:
            assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
            labels = jnp.array(np.random.choice(cfg.problem.num_classes + 1, n_images))
        else:
            labels = jnp.array(np.random.choice(cfg.problem.num_classes, n_images))
        prng_key = jax.random.split(prng_key)[0]
    else:
        labels = None

    for kk, step in enumerate(steps):
        xhats[kk] = batch_sample(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            labels,
        )

    # transpose (S, N, C, H, W) -> (S, N, H, W, C)
    xhats = xhats.transpose(0, 1, 3, 4, 2)

    ## make the image grids
    nrows = 2 if n_images > 8 else 1
    ncols = n_images // nrows

    for kk, title in enumerate(titles):
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fw * ncols, fh * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = axs.reshape((nrows, ncols))

        fig.suptitle(title, fontsize=fontsize)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal")

        ## visualize the generated images
        for ii in range(nrows):
            for jj in range(ncols):
                index = ii * ncols + jj
                image = datasets.unnormalize_image(xhats[kk, index])
                axs[ii, jj].imshow(image)

        wandb.log({titles[kk]: wandb.Image(fig)}, step=wandb_step)

    return prng_key


def make_loss_fn_args_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
    wandb_step: int = 0,
) -> None:
    """Make a plot of the loss function arguments."""
    # unpack the full loss arguments
    data_args = loss_fn_args[2:]
    (x0batch, x1batch, _, sbatch, tbatch, _, _, _) = (
        dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    )

    # remove pmap reshaping
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    sbatch = jnp.squeeze(sbatch)
    tbatch = jnp.squeeze(tbatch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    # compute xts
    xtbatch = statics.interp.batch_calc_It(tbatch, x0batch, x1batch)

    ## set up plot array
    if cfg.problem.target in ["checker", "gmm"]:
        titles = [r"$x_0$", r"$x_1$", r"$x_t$", r"$(s, t)$"]
    else:
        titles = [r"$(s, t)$"]

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    if cfg.problem.target in ["checker", "gmm"]:
        all_x = np.concatenate(
            [np.asarray(x0batch), np.asarray(x1batch), np.asarray(xtbatch)],
            axis=0,
        )
        margin = 0.5
        xmin = float(all_x[:, 0].min()) - margin
        xmax = float(all_x[:, 0].max()) + margin
        ymin = float(all_x[:, 1].min()) - margin
        ymax = float(all_x[:, 1].max()) + margin

    for kk, ax in enumerate(axs.ravel()):
        if kk == (len(titles) - 1):
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
        else:
            if cfg.problem.target in ["checker", "gmm"]:
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[0, jj]
        ax.set_title(title, fontsize=fontsize)

        if cfg.problem.target in ["checker", "gmm"]:
            if jj == 0:
                ax.scatter(x0batch[:, 0], x0batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 1:
                ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 2:
                ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 3:
                ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")
        else:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)}, step=wandb_step)


def _base_log_prob(cfg: config_dict.ConfigDict, x: jnp.ndarray) -> jnp.ndarray:
    """Log prob of N(0, diag(rescale^2)) for base distribution."""
    sigma = jnp.asarray(cfg.network.rescale)
    if sigma.ndim == 0:
        sigma = jnp.ones((x.shape[-1],)) * sigma
    quad = jnp.sum((x / sigma) ** 2, axis=-1)
    log_det = jnp.sum(jnp.log(sigma**2))
    log_norm = 0.5 * (x.shape[-1] * jnp.log(2 * jnp.pi) + log_det)
    return -0.5 * quad - log_norm


def _base_log_prob_image(x: jnp.ndarray, rescale: float) -> jnp.ndarray:
    """Log prob of N(0, diag(rescale^2)) for image-shaped tensors."""
    sigma = jnp.asarray(rescale, dtype=x.dtype)
    dims = tuple(range(1, x.ndim))
    quad = jnp.sum((x / sigma) ** 2, axis=dims)
    d = x[0].size
    log_det = d * jnp.log(sigma**2)
    log_norm = 0.5 * (d * jnp.log(2 * jnp.pi) + log_det)
    return -0.5 * quad - log_norm

def _base_log_prob_mean_free(cfg: config_dict.ConfigDict, x: jnp.ndarray) -> jnp.ndarray:
    """Log prob of mean-free N(0, diag(rescale^2)) for base distribution."""
    assert x.ndim == 3 and x.shape[2] == 3, "x must be of shape (batch, n_atoms, 3)"
    sigma = jnp.asarray(cfg.network.rescale)
    if sigma.ndim == 0:
        sigma = jnp.ones(x.shape[1:]) * sigma
    dims = tuple(range(1, x.ndim))
    quad = jnp.sum((x / sigma) ** 2, axis=dims)
    total_dim = float(x.shape[1] * x.shape[2])
    log_det = jnp.sum(jnp.log(sigma**2))
    log_norm = 0.5 * (total_dim * jnp.log(2 * jnp.pi) + log_det)
    unconstrained_log_prob = -0.5 * quad - log_norm    
    var_center_of_mass = jnp.mean(sigma**2, axis=0)  # Shape: (3,)
    correction = 0.5 * (3 * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(var_center_of_mass)))
    return unconstrained_log_prob + correction

# def _sample_model_nsteps(
#     apply_fn: Callable,
#     params: Dict,
#     x0: jnp.ndarray,
#     n_steps: int,
#     label: jnp.ndarray | None = None,
#     batch_size: int | None = None,
#     *,
#     cfg: config_dict.ConfigDict,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Sample with Euler steps and track logp using model divergence."""
#     if batch_size is None:
#         batch_size = x0.shape[0]
#     assert x0.shape[0] % batch_size == 0, "x0.shape[0] must be divisible by batch_size"
#     n_batches = x0.shape[0] // batch_size
#     dt = 1.0 / n_steps
#     x = x0.reshape((n_batches, batch_size, *x0.shape[1:]))
#     if label is not None:
#         label = label.reshape((n_batches, batch_size, *label.shape[1:]))
#     out_list = []
#     for batch_idx in range(n_batches):
#         t_curr = jnp.zeros((batch_size,), dtype=jnp.float32)
#         x_batch = x[batch_idx]
#         for _ in range(n_steps):
#             t_next = t_curr + dt
#             out = apply_fn(
#                 params,
#                 t_curr,
#                 t_next,
#                 x_batch,
#                 label[batch_idx] if label is not None else None,
#                 train=False,
#                 method="calc_phi",
#                 return_div=True,
#             )
#             if isinstance(out, tuple):
#                 phi = out[0]
#             else:
#                 phi = out
#             x_batch = x_batch + dt * phi
#             t_curr = t_next
#         out_list.append(x_batch)
#     xs = jnp.concatenate(out_list, axis=0)

#     return np.asarray(xs) * cfg.network.unscale_factor

# def _sample_model_nsteps_with_logp(
#     apply_fn: Callable,
#     params: Dict,
#     x0: jnp.ndarray,
#     n_steps: int,
#     label: jnp.ndarray | None = None,
#     batch_size: int | None = None,
#     *,
#     cfg: config_dict.ConfigDict,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Sample with Euler steps and track logp using model divergence."""
#     dt = 1.0 / n_steps
#     if batch_size is None:
#         batch_size = x0.shape[0]
#     assert x0.shape[0] % batch_size == 0, "x0.shape[0] must be divisible by batch_size"
#     n_batches = x0.shape[0] // batch_size
#     x = x0.reshape((n_batches, batch_size, *x0.shape[1:]))
#     if label is not None:
#         label = label.reshape((n_batches, batch_size, *label.shape[1:]))
#     # TODO: CHECK SHAPE
#     if cfg.problem.base == "mean-free-gaussian":
#         logp = _base_log_prob_mean_free(cfg, x0)
#     elif cfg.problem.base == "gaussian":
#         logp = _base_log_prob(cfg, x0)
#     else:
#         raise ValueError(f"Invalid base distribution: {cfg.problem.base}")

#     divergence_rescale = getattr(cfg.network, "divergence_rescale")
#     x_list = []
#     logp_list = []
#     for batch_idx in range(n_batches):
#         t_curr = jnp.zeros((batch_size,), dtype=jnp.float32)
#         x_batch = x[batch_idx]
#         logp_batch = jnp.zeros((batch_size,))
#         if label is not None:
#             label_batch = label[batch_idx]
#         else:
#             label_batch = None
#         for _ in range(n_steps):
#             t_next = t_curr + dt
#             phi, div = apply_fn(
#                 params,
#                 t_curr,
#                 t_next,
#                 x_batch,
#                 label_batch,
#                 train=False,
#                 method="calc_phi",
#                 return_div=True,
#             )
#             div = div * divergence_rescale
#             x_batch = x_batch + dt * phi
#             div_batch = div.reshape((div.shape[0], -1)).sum(axis=-1)
#             logp_batch = logp_batch - dt * div_batch
#             t_curr = t_next
#         x_list.append(x_batch)
#         logp_list.append(logp_batch)
#     x = jnp.concatenate(x_list, axis=0)
#     logp_ = jnp.concatenate(logp_list, axis=0)
#     logp = logp + logp_


#     return np.asarray(x) * cfg.network.unscale_factor, np.asarray(logp) - cfg.problem.d * jnp.log(cfg.network.unscale_factor)


def _make_mean_logp_heatmap(xs, logp, xlim, ylim, bins=100):
    """Mean logp per bin, matching try_ll.py."""
    x = xs[:, 0]
    y = xs[:, 1]
    hist_sum, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim], weights=logp
    )
    hist_count, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    mean_logp = hist_sum / (hist_count + 1e-6)
    mean_logp[hist_count == 0] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return mean_logp, extent


def _make_log_density_heatmap(xs, xlim, ylim, bins=100):
    """Numerical log-density estimate from samples."""
    x = xs[:, 0]
    y = xs[:, 1]
    hist_count, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim]
    )
    bin_area = (
        (xedges[-1] - xedges[0]) / bins * (yedges[-1] - yedges[0]) / bins
    )
    density = hist_count / (hist_count.sum() * bin_area + 1e-12)
    logp = np.full_like(density, np.nan, dtype=np.float64)
    mask = hist_count > 0
    logp[mask] = np.log(density[mask])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return logp, extent


def _make_uniform_logp_heatmap(xs, xlim, ylim, bins=100, logp_value=-np.log(32.0)):
    """Uniform logp over occupied bins for checkerboard."""
    x = xs[:, 0]
    y = xs[:, 1]
    hist_count, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim]
    )
    logp = np.full_like(hist_count, np.nan, dtype=np.float64)
    logp[hist_count > 0] = logp_value
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return logp, extent


def _make_logp_heatmap(cfg, distr, xlim, ylim, bins=100):
    """Analytical logp heatmap for the target."""
    assert cfg.problem.d == 2, "Logp heatmap only implemented for d=2."
    xedges = np.linspace(xlim[0], xlim[1], bins + 1)
    yedges = np.linspace(ylim[0], ylim[1], bins + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")

    # Stack into (N, 2) array
    pts = np.stack([Xc.flatten(), Yc.flatten()], axis=1)

    logp_flat = distr.log_prob(jnp.array(pts))
    logp = np.array(logp_flat).reshape(Xc.shape)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return logp, extent


def _make_grid_points(xlim, ylim, grid_size=200):
    xs = jnp.linspace(xlim[0], xlim[1], grid_size)
    ys = jnp.linspace(ylim[0], ylim[1], grid_size)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    pts = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return pts, X, Y


# def _inverse_logp_points_with_divhead(
#     apply_fn: Callable,
#     params: Dict,
#     x_t: jnp.ndarray,
#     n_steps: int,
#     label: jnp.ndarray,
#     batch_size: int | None = None,
#     *,
#     cfg: config_dict.ConfigDict,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Inverse logp on points using divergence head."""
#     x_t = x_t / cfg.network.unscale_factor
#     dt = -1.0 / n_steps
#     if batch_size is None:
#         batch_size = x_t.shape[0]
#     assert x_t.shape[0] % batch_size == 0, "x_t.shape[0] must be divisible by batch_size"
#     n_batches = x_t.shape[0] // batch_size
#     x = x_t.reshape((n_batches, batch_size, *x_t.shape[1:]))
#     if label is not None:
#         label = label.reshape((n_batches, batch_size, *label.shape[1:]))

#     divergence_rescale = getattr(cfg.network, "divergence_rescale")
#     x_list = []
#     delta_logp_list = []
#     for batch_idx in range(n_batches):
#         x_batch = x[batch_idx]
#         t_curr = jnp.ones((batch_size,), dtype=jnp.float32)
#         delta_logp_batch = jnp.zeros((batch_size,), dtype=jnp.float32)
#         if label is not None:
#             label_batch = label[batch_idx]
#         else:
#             label_batch = None
#         for _ in range(n_steps):
#             t_next = t_curr + dt
#             phi, div = apply_fn(
#                 params,
#                 t_curr,
#                 t_next,
#                 x_batch,
#                 label_batch,
#                 train=False,
#                 method="calc_phi",
#                 return_div=True,
#             )
#             print(
#                 "t_curr:", np.asarray(t_curr[0]),
#                 "t_next:", np.asarray(t_next[0]),
#                 "div:", np.asarray(div.mean()),
#             )
#             div = div * divergence_rescale
#             div_batch = div.reshape((div.shape[0], -1)).sum(axis=-1)
#             delta_logp_batch = delta_logp_batch - dt * div_batch
#             x_batch = x_batch + dt * phi
#             t_curr = t_next
#         x_list.append(x_batch)
#         delta_logp_list.append(delta_logp_batch)
#     x = jnp.concatenate(x_list, axis=0)
#     delta_logp = jnp.concatenate(delta_logp_list, axis=0)

#     if cfg.problem.base == "mean-free-gaussian":
#         logp0 = _base_log_prob_mean_free(cfg, x)
#     elif cfg.problem.base == "gaussian":
#         logp0 = _base_log_prob(cfg, x)
#     else:
#         raise ValueError(f"Invalid base distribution: {cfg.problem.base}")
#     logp1 = logp0 - delta_logp - cfg.problem.d * jnp.log(cfg.network.unscale_factor)
#     return np.asarray(x), np.asarray(logp1)


def _clip_heatmaps(heatmaps):
    """Replace NaNs and compute shared vmin/vmax."""
    vals = []
    heatmaps = [np.asarray(h) for h in heatmaps]  # Ensure numpy arrays
    for h in heatmaps:
        finite = np.isfinite(h)
        if finite.any():
            vals.append(h[finite])
    if not vals:
        return heatmaps, 0.0, 1.0
    all_vals = np.concatenate(vals)
    vmin = all_vals.min()
    vmax = all_vals.max()
    clipped = []
    for h in heatmaps:
        h2 = h.copy()
        h2[~np.isfinite(h2)] = vmin
        clipped.append(np.clip(h2, vmin, vmax))
    return clipped, vmin, vmax


def make_likelihood_heatmap_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    wandb_step: int = 0,
) -> jnp.ndarray:
    """Plot likelihood heatmaps for 1/2/4/8 steps and target checkerboard."""
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")
    steps = [2, 4, 8]
    n_samples = cfg.logging.plot_bs
    if cfg.problem.target == "checker":
        n_samples = getattr(cfg.logging, "likelihood_n_samples", max(n_samples, 500_000))
        # Analytical target heatmap for checkerboard (uniform over support)
        margin = 0.0
        base_range = 4.0
        xlim = (-base_range - margin, base_range + margin)
        ylim = (-base_range - margin, base_range + margin)
        target_title = "Target log p(x) = -ln(32)"
        vmin_fixed, vmax_fixed = -4.0, -3.0
    elif cfg.problem.target == "gmm":
        n_samples = getattr(cfg.logging, "likelihood_n_samples", max(n_samples, 500_000))
        margin = 0.0
        base_range = 2.0
        xlim = (-base_range - margin, base_range + margin)
        ylim = (-base_range - margin, base_range + margin)
        target_title = "Target p(x)"
        vmin_fixed, vmax_fixed = None, None
    else:
        raise ValueError(f"Target {cfg.problem.target} not implemented for likelihood heatmap plot.")
    h_target, extent = _make_logp_heatmap(cfg, statics.distr, xlim, ylim, bins=100)

    # Model samples + logp for each step
    prng_key, sample_key = jax.random.split(prng_key)
    x0s = statics.sample_rho0(n_samples, sample_key)
    # labels = -jnp.ones((n_samples,))
    labels = None

    is_p = cfg.problem.target != "checker"  # plot logp for checkerboard, and p for others

    heatmaps = [h_target if not is_p else jnp.exp(h_target)]
    titles = [target_title]
    for step in steps:
        xs, logp = train_state.apply_fn(
            params_for_visual,
            x0s,
            labels,
            batch_size=cfg.logging.plot_bs,
            n_steps=step,
            method="sample_forward",
        )
        h_step, _ = _make_mean_logp_heatmap(xs, logp, xlim, ylim, bins=100)            
        # heatmaps.append(h_step if not is_p else jax.nn.softmax(h_step))
        heatmaps.append(h_step if not is_p else jnp.exp(h_step))
        titles.append(f"Mean {'log p(x)' if not is_p else 'p(x)'}, {step} steps")

        grid_pts, X, Y = _make_grid_points(xlim, ylim, grid_size=200)
        # grid_labels = -jnp.ones((grid_pts.shape[0],))
        grid_labels = None
        _, _, logp_inv = train_state.apply_fn(
            params_for_visual,
            grid_pts,
            grid_labels,
            n_steps=step,
            method="sample_inverse",
        )
        h_inv = logp_inv.reshape(X.shape)
        # heatmaps.append(np.asarray(h_inv) if not is_p else jax.nn.softmax(np.asarray(h_inv)))
        heatmaps.append(np.asarray(h_inv) if not is_p else jnp.exp(np.asarray(h_inv)))
        titles.append(f"Inverse {'log p(x)' if not is_p else 'p(x)'}, {step} steps")

    heatmaps, vmin_auto, vmax_auto = _clip_heatmaps(heatmaps)
    if vmin_fixed is not None:
        vmin, vmax = vmin_fixed, vmax_fixed
        heatmaps = [np.clip(h, vmin, vmax) for h in heatmaps]
    else:
        vmin, vmax = vmin_auto, vmax_auto

    plt.close("all")
    ncols = len(heatmaps)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(4 * ncols, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axs = [axs]
    for ax, h, title in zip(axs, heatmaps, titles):
        im = ax.imshow(
            h.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    wandb.log({"likelihood_heatmap": wandb.Image(fig)}, step=wandb_step)
    return prng_key


def make_stratified_div_loss(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Dict[str, float]:
    prng_key, sample_key = jax.random.split(prng_key)
    tmin = 0.0
    tmax = 0.999
    n_steps = 10
    time_steps = np.linspace(tmin, tmax, n_steps)
    x1s = next(statics.ds)
    n_samples = x1s.shape[0]
    x0s = statics.sample_rho0(n_samples, sample_key)
    # labels = -jnp.ones((n_samples,))
    # Handle labels for conditional generation
    if cfg.training.conditional:
        raise NotImplementedError("Conditional generation not implemented for stratified div loss.")
        # if cfg.training.class_dropout > 0:
        #     labels = jnp.array(
        #         np.random.choice(cfg.problem.num_classes + 1, n_samples)
        #     ).reshape(cfg.training.ndevices, current_per_device_batch)
        # else:
        #     labels = jnp.array(
        #         np.random.choice(cfg.problem.num_classes, n_samples)
        #     ).reshape(cfg.training.ndevices, current_per_device_batch)
    else:
        labels = None

    use_true_div_loss = getattr(cfg.logging, "use_true_stratified_div_loss", False)

    @jax.jit
    def compute_step(t_curr, step_key):
        xt = statics.interp.calc_It(t_curr, x0s, x1s)

        if use_true_div_loss:
            def div_wrapper(x_in, label_in, rng_in):
                def phi_fn(x):
                    return statics.net.apply(
                        train_state.params,
                        t_curr,
                        x,
                        label_in,
                        train=False,
                        method="calc_b",
                        rngs={"dropout": rng_in},
                    )
                # jac has shape (22, 3, 22, 3) for ALDP
                jac = jax.jacrev(phi_fn)(x_in)
                if x_in.ndim == 2:
                    return jnp.einsum("ijij->", jac)
                elif x_in.ndim == 1:
                    return jnp.einsum("ii->", jac)
                elif x_in.ndim == 3:
                    return jnp.einsum("ijkijk->", jac)
                else:
                    # fallback for arbitrary shape, though slow
                    jac_flat = jac.reshape(-1, jac.size // jac.shape[0])
                    return jnp.trace(jac_flat)
        else:
            def div_wrapper(x_in, label_in, rng_in):
                _, div = _hutchinson_divergence_with_phi(
                    train_state.params,
                    t_curr,
                    x_in,
                    label_in,
                    rng_in,
                    X=statics.net,
                    method="calc_b",
                )
                return div

        step_keys = jax.random.split(step_key, xt.shape[0])
        div_hat = jax.vmap(div_wrapper)(xt, labels, step_keys)
        
        _, div_tt = statics.net.apply(
            train_state.params,
            t_curr,
            xt,
            labels,
            train=False,
            method="calc_b",
            rngs=step_key,
            return_div=True,
        )
        div_hat_unscaled = div_hat.reshape((div_hat.shape[0], -1)).sum(axis=-1)
        div_tt_unscaled = div_tt.reshape((div_tt.shape[0], -1)).sum(axis=-1) * statics.net.config.divergence_rescale
        
        l2_error = jnp.linalg.norm(div_hat_unscaled - div_tt_unscaled, ord=2)
        l2_magnitude = jnp.linalg.norm(div_tt_unscaled, ord=2)
        
        return l2_error, l2_magnitude

    metrics = {}
    for t_curr in time_steps:
        prng_key, step_key = jax.random.split(prng_key)
        l2_error, l2_magnitude = compute_step(t_curr, step_key)
        metrics[f"eval/stratified_div/l2_error/t={t_curr:.2f}"] = float(l2_error)
        metrics[f"eval/stratified_div/l2_magnitude/t={t_curr:.2f}"] = float(l2_magnitude)
        metrics[f"eval/stratified_div/l2_relative_error/t={t_curr:.2f}"] = float(l2_error / l2_magnitude)
    return metrics