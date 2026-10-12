"""PyTorch port of common/se3_utils.py.

Only the routines required for molecule training / evaluation are kept
(image-specific helpers were never present here).
"""


import torch


def _assert_shape(x: torch.Tensor) -> None:
    assert (
        x.ndim == 3 and x.shape[-1] == 3
    ), f"Shape of x must be (N, n_atoms, 3), but got {tuple(x.shape)}"


def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """Remove the center of mass (assuming equal masses)."""
    _assert_shape(x)
    return x - x.mean(dim=1, keepdim=True)
