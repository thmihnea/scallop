"""PyTorch port of common/loss_args.py (time / batch sampling).

Randomness comes from a ``torch.Generator`` instead of JAX PRNG keys, so the
*distributions* are reproduced but exact values are not bit-identical to JAX.
"""

import math
from typing import Tuple

import torch


def _sample_diagonal(generator, bs, tmin, tmax, distr="uniform"):
    if not (0 <= tmin < tmax <= 1):
        raise ValueError("Diagonal time bounds must satisfy 0 <= tmin < tmax <= 1")
    u = torch.rand(bs, generator=generator)
    if distr == "uniform":
        s = u * (tmax - tmin) + tmin
    elif distr in ("linear", "square"):
        if tmax >= 1:
            raise ValueError("Singular importance time densities require tmax < 1")
        if distr == "linear":
            logs = (1 - u) * math.log1p(-tmin) + u * math.log1p(-tmax)
            s = 1 - torch.exp(logs)
        else:
            inverse_gap = (1 - u) / (1 - tmin) + u / (1 - tmax)
            s = 1 - 1 / inverse_gap
    else:
        raise ValueError(f"Unknown time distribution: {distr}")
    return s, s


def _sample_triangle(generator, bs, tmin, tmax, order="square"):
    s = torch.rand(bs, generator=generator) * (tmax - tmin) + tmin
    t = torch.rand(bs, generator=generator) * (tmax - tmin) + tmin
    if order == "forward":
        return torch.minimum(s, t), torch.maximum(s, t)
    if order != "square":
        raise ValueError(f"Unknown off-diagonal time order: {order}")
    return s, t


def _get_diag_offdiag_bs(cfg, bs: int) -> Tuple[int, int]:
    if hasattr(cfg.optimization, "local_diag_bs"):
        diag_bs = cfg.optimization.local_diag_bs
    elif hasattr(cfg.optimization, "diag_fraction"):
        diag_bs = max(1, int(bs * cfg.optimization.diag_fraction))
    elif hasattr(cfg.optimization, "diag_bs"):
        diag_bs = cfg.optimization.diag_bs
    else:
        raise ValueError("Either diag_fraction or diag_bs must be specified")
    return diag_bs, bs - diag_bs


def sample_times(cfg, generator):
    """Sample (s, t, u, h) for the batch following the JAX scheme."""
    bs = getattr(cfg.optimization, "local_bs", cfg.optimization.bs)
    tmin = cfg.training.tmin
    tmax = cfg.training.tmax
    diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, bs)
    time_distr = getattr(cfg.training, "time_distr", "uniform")
    time_order = getattr(cfg.training, "offdiag_time_order", "square")
    if time_order not in ("square", "forward"):
        raise ValueError(f"Unknown off-diagonal time order: {time_order}")
    singular_anchor = getattr(getattr(cfg, "network", None), "divergence_mode", None) in ("dlogpdt", "cond_trace")
    diag_tmax = min(tmax, getattr(cfg.training, "diag_tmax", 0.999 if singular_anchor else tmax))
    if singular_anchor and diag_tmax >= 1:
        raise ValueError("dlogpdt/cond_trace require training.diag_tmax < 1")

    if offdiag_bs == 0:
        sbatch, tbatch = _sample_diagonal(generator, bs, tmin, diag_tmax, distr=time_distr)
    else:
        if diag_bs > 0:
            s_diag, t_diag = _sample_diagonal(generator, diag_bs, tmin, diag_tmax, distr=time_distr)
        else:
            s_diag = t_diag = torch.empty(0)
        s_off, t_off = _sample_triangle(generator, offdiag_bs, tmin, tmax, order=time_order)
        sbatch = torch.cat([s_diag, s_off])
        tbatch = torch.cat([t_diag, t_off])

    # psd_type only means anything for loss_type="psd". A config that leaves a stale
    # psd_type behind (they are copied around a lot) used to silently pay for u/h it
    # never uses; a config that asks for psd without one used to die deep inside
    # psd_term on ``1 - None``.
    loss_type = getattr(cfg.training, "loss_type", None)
    psd_type = getattr(cfg.training, "psd_type", None)
    if loss_type == "psd":
        if psd_type not in ("uniform", "midpoint"):
            raise ValueError(
                "training.loss_type='psd' requires training.psd_type in "
                f"('uniform', 'midpoint'); got {psd_type!r}"
            )
    elif loss_type is not None:
        psd_type = None

    if psd_type == "midpoint":
        ubatch = 0.5 * (sbatch + tbatch)
        hbatch = None
    elif psd_type == "uniform":
        hbatch = torch.rand(bs, generator=generator)
        ubatch = hbatch * sbatch + (1 - hbatch) * tbatch
    elif psd_type is None:
        ubatch = None
        hbatch = None
    else:
        raise ValueError(f"Unknown psd_type: {psd_type}")

    return sbatch, tbatch, ubatch, hbatch


def get_loss_fn_args(cfg, statics, generator, device):
    """Return (x0, x1, label, s, t, u, h) for one training step."""
    sbatch, tbatch, ubatch, hbatch = sample_times(cfg, generator)
    x0 = statics.sample_rho0(getattr(cfg.optimization, "local_bs", cfg.optimization.bs), generator, device)
    x1 = next(statics.ds)

    label = None  # molecule training is unconditional

    def _to(x):
        return None if x is None else x.to(device)

    return (
        x0,
        x1,
        label,
        _to(sbatch),
        _to(tbatch),
        _to(ubatch),
        _to(hbatch),
    )
