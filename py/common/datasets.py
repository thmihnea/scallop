"""Base sampling and shape/scale setup for the TS peptide data pipeline."""
from typing import Callable

import numpy as np
import torch

from .se3_utils import remove_mean


def setup_base(cfg, ex_input_shape) -> Callable:
    """Return ``sample_rho0(bs, generator, device)`` for the base density."""
    n_shape = tuple(ex_input_shape)

    if cfg.problem.base == "gaussian":
        def sample_rho0(bs, generator, device):
            rescale = float(cfg.network.rescale)
            samples = rescale * torch.randn((bs, *n_shape), generator=generator)
            return samples.to(device)
    elif cfg.problem.base == "mean-free-gaussian":
        def sample_rho0(bs, generator, device):
            rescale = float(cfg.network.rescale)
            samples = rescale * torch.randn((bs, *n_shape), generator=generator)
            if getattr(cfg.network, "com_aware", False):
                return samples.to(device)  # keep the Gaussian COM for the com-aware flow
            return remove_mean(samples).to(device)
    else:
        raise ValueError("Specified base density is not implemented.")

    return sample_rho0


def setup_target(cfg, data_dir=None, train_std=None):
    """Return ``(cfg, ex_input)`` using the TS registry and training scale.

    DiT uses coordinates and atom-index positional embeddings; it needs no
    topology, chemical labels, bonds or molecular simulation during construction.
    ``train_std`` lets the training loader supply its already computed scale.
    """
    from . import ts_systems

    system = ts_systems.resolve(cfg)
    expected = (system.num_atoms, 3)
    for name, section in (("problem", cfg.problem), ("network", cfg.network)):
        shape = (int(section.n_particles), int(section.n_dimension))
        if shape != expected:
            raise ValueError(f"{name} shape {shape} does not match {system.sequence}: {expected}")
    if int(cfg.problem.d) != system.num_atoms * 3:
        raise ValueError("problem.d does not match the TS system")
    if data_dir is None:
        data_dir = getattr(cfg.problem, "ts_data_dir", None) or None
    std = ts_systems.train_std(system, data_dir) if train_std is None else float(train_std)
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid TS training standard deviation: {std}")
    cfg.network.sigma_data = 1.0
    cfg.network.rescale = 1.0
    cfg.network.unscale_factor = std
    return cfg, np.zeros(expected, dtype=np.float32)
