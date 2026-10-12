"""FALCON Appendix D.2.2 schedules, converted from noise time to flow time."""
import math

import numpy as np


TIME_DISCRETIZATION_TYPES = ("linear", "geometric", "cosine", "chebyshev", "edm")


def time_grid(n_steps: int, kind: str = "linear") -> np.ndarray:
    """Return N+1 increasing nodes from noise (0) to data (1).

    Uses Eqs. 16--21 of https://arxiv.org/pdf/2512.09914: alpha=2;
    EDM rho=7, sigma_min=1e-3, sigma_max=1. Chebyshev's interior nodes
    follow Eq. 19, with both endpoints pinned to cover the full trajectory.
    The paper writes decreasing noise times r; our flow time is 1-r.
    """
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    i = np.arange(n_steps + 1, dtype=np.float64)
    u = i / n_steps
    if kind == "linear":
        grid = u
    elif kind == "geometric":
        # Algebraically (2**(N-i)-1)/(2**N-1), without overflow.
        tail = math.exp2(-n_steps)
        grid = 1 - (np.exp2(-i) - tail) / (1 - tail)
    elif kind == "cosine":
        grid = 1 - np.cos(np.pi * u / 2) ** 2
    elif kind == "chebyshev":
        grid = (1 - np.cos(np.pi * (i + 0.5) / (n_steps + 1))) / 2
    elif kind == "edm":
        sigma = (1 + u * (1e-3 ** (1 / 7) - 1)) ** 7
        grid = 1 - (sigma - 1e-3) / (1 - 1e-3)
    else:
        raise ValueError(f"Unknown time discretization {kind!r}; choose {TIME_DISCRETIZATION_TYPES}")
    grid[0], grid[-1] = 0.0, 1.0
    if not np.all(np.diff(grid) > 0):
        raise ValueError(f"{kind} with {n_steps} steps has indistinguishable time nodes")
    return grid
