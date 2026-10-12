"""Mean-removal equivalence against the independent JAX reference."""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _samples(seed=0, B=4, n=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, n, 3)).astype(np.float32)


def test_remove_mean_equiv():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    import jax_impl.common.se3_utils as js
    import common.se3_utils as tsu

    x = _samples()
    j = js.remove_mean(jnp.asarray(x))
    t = tsu.remove_mean(torch.tensor(x))
    assert _max_abs_diff(j, t.numpy()) < 1e-6
    # And the result is genuinely mean-free.
    assert _max_abs_diff(t.mean(dim=1).numpy(), 0.0) < 1e-6
