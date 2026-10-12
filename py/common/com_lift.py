"""Gaussian translation lift: correction at the importance-weight boundary.

The TS transform adds c ~ N(0, I_3/N) in standardized coordinates. Use the
Cartesian Gaussian density, not the chi-3 density of ||c|| (SBG Eq. 5).
See arXiv:2602.03729, Appendix C.1, Eq. 15, and experimental_logs/com-lift.
"""
import math

import torch


def add_com_cli_arguments(parser):
    """Use the same default and explicit mean-free override in every launcher."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--com_augmentation", "--com_lift", action="store_true", default=True,
        help="use Gaussian COM augmentation, a full Gaussian prior and Cartesian "
             "COM correction for importance weights (default: on; must match training)",
    )
    group.add_argument(
        "--no_com_augmentation", "--no_com_lift", dest="com_augmentation", action="store_false",
        help="disable COM augmentation for mean-free training or checkpoints",
    )


def com_log_prob(x: torch.Tensor, unscale_factor: float = 1.0) -> torch.Tensor:
    """Log N(mean(x); 0, unscale_factor**2/N I_3), one value per sample.

    Pass physical coordinates with their training unscale factor, or
    standardized coordinates with factor 1. The centroid is the arithmetic
    atom mean, matching TS; it is not mass weighted. This is finite at c=0.
    """
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != 3:
        raise ValueError("x must have shape (batch, n_atoms, 3)")
    scale = float(unscale_factor)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("unscale_factor must be finite and positive")
    variance = scale**2 / x.shape[1]
    c = x.mean(dim=1)
    return -0.5 * c.square().sum(dim=-1) / variance - 1.5 * math.log(2 * math.pi * variance)


def importance_source_energy(x: torch.Tensor, logp: torch.Tensor, *,
                             com_augmentation: bool = False,
                             unscale_factor: float = 1.0) -> torch.Tensor:
    """Return -log q + log g(c) for lifted SNIS, or -log q without a lift.

    x and logp must use the same coordinate units. Final log weights are this
    value minus E/kT. Keep the model's raw logp unchanged: the lift correction
    belongs only in importance sampling, after the complete trajectory/map.
    The constant Jacobian N**(3/2) cancels in self-normalized weights.
    """
    energy = -logp
    if com_augmentation:
        energy = energy + com_log_prob(x, unscale_factor)
    return energy
