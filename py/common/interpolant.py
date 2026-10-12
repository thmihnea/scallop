"""PyTorch port of common/interpolant.py.

Stochastic interpolant: I_t = alpha(t) * x0 + beta(t) * x1.

``t`` may be a scalar or a tensor broadcastable against ``x0``/``x1``
(shape ``(B,)`` is broadcast to ``(B, 1, 1)`` for molecule data).
"""

import dataclasses
import math
from typing import Callable

import torch


def _broadcast_t(t, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    t = t.to(dtype=x.dtype, device=x.device)
    # broadcast (B,) -> (B, 1, ..., 1) to match x's trailing dims
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


@dataclasses.dataclass
class Interpolant:
    alpha: Callable
    beta: Callable
    alpha_dot: Callable
    beta_dot: Callable
    kind: str | None = None

    def calc_It(self, t, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        tb = _broadcast_t(t, x0)
        return self.alpha(tb) * x0 + self.beta(tb) * x1

    def calc_It_dot(self, t, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        tb = _broadcast_t(t, x0)
        return self.alpha_dot(tb) * x0 + self.beta_dot(tb) * x1


def setup_interpolant(cfg) -> Interpolant:
    if cfg.problem.interp_type == "linear":
        return Interpolant(
            kind="linear",
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda t: -torch.ones_like(t),
            beta_dot=lambda t: torch.ones_like(t),
        )
    elif cfg.problem.interp_type == "trig":
        return Interpolant(
            kind="trig",
            alpha=lambda t: torch.cos(math.pi * t / 2),
            beta=lambda t: torch.sin(math.pi * t / 2),
            alpha_dot=lambda t: -0.5 * math.pi * torch.sin(math.pi * t / 2),
            beta_dot=lambda t: 0.5 * math.pi * torch.cos(math.pi * t / 2),
        )
    raise NotImplementedError("Interpolant type not implemented.")
