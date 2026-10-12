"""Interpolant: JAX<->Torch equivalence + standalone analytic identities.

The equivalence tests need JAX; the identity/finite-difference tests do not.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import math

import numpy as np
import pytest
import torch

from _equiv_utils import build_problem_config

torch.set_default_dtype(torch.float32)


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _inputs(seed=0, B=5, n=6):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((B, n, 3)).astype(np.float32)
    x1 = rng.standard_normal((B, n, 3)).astype(np.float32)
    t = rng.uniform(0.05, 0.95, size=(B,)).astype(np.float32)
    return x0, x1, t


@pytest.mark.parametrize("interp_type", ["linear", "trig"])
def test_interpolant_equiv(interp_type):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import common.interpolant as ti

    cfg = build_problem_config(interp_type)
    jinterp = ji.setup_interpolant(cfg)
    tinterp = ti.setup_interpolant(cfg)
    x0, x1, t = _inputs()

    j_It = jax.vmap(jinterp.calc_It)(jnp.asarray(t), jnp.asarray(x0), jnp.asarray(x1))
    j_Itd = jax.vmap(jinterp.calc_It_dot)(jnp.asarray(t), jnp.asarray(x0), jnp.asarray(x1))
    t_It = tinterp.calc_It(torch.tensor(t), torch.tensor(x0), torch.tensor(x1))
    t_Itd = tinterp.calc_It_dot(torch.tensor(t), torch.tensor(x0), torch.tensor(x1))

    assert _max_abs_diff(j_It, t_It.numpy()) < 1e-6
    assert _max_abs_diff(j_Itd, t_Itd.numpy()) < 1e-6


@pytest.mark.parametrize("interp_type", ["linear", "trig"])
def test_interpolant_endpoints(interp_type):
    """I_0 == x0, I_1 == x1 (both interpolants pin the endpoints)."""
    cfg = build_problem_config(interp_type)
    import common.interpolant as ti

    interp = ti.setup_interpolant(cfg)
    x0, x1, _ = _inputs()
    x0t, x1t = torch.tensor(x0), torch.tensor(x1)

    I0 = interp.calc_It(torch.zeros(x0.shape[0]), x0t, x1t)
    I1 = interp.calc_It(torch.ones(x0.shape[0]), x0t, x1t)
    assert _max_abs_diff(I0.numpy(), x0) < 1e-6
    assert _max_abs_diff(I1.numpy(), x1) < 1e-6


@pytest.mark.parametrize("interp_type", ["linear", "trig"])
def test_interpolant_dot_finite_difference(interp_type):
    """calc_It_dot matches a central finite difference of calc_It in t."""
    cfg = build_problem_config(interp_type)
    import common.interpolant as ti

    interp = ti.setup_interpolant(cfg)
    x0, x1, t = _inputs()
    x0t, x1t = torch.tensor(x0), torch.tensor(x1)
    tt = torch.tensor(t)
    eps = 1e-3

    fd = (interp.calc_It(tt + eps, x0t, x1t) - interp.calc_It(tt - eps, x0t, x1t)) / (2 * eps)
    ana = interp.calc_It_dot(tt, x0t, x1t)
    assert _max_abs_diff(fd.numpy(), ana.numpy()) < 1e-3
