"""Equivalence test: JAX FlowMap vs Torch TorchFlowMap forward / derivatives."""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import numpy as np
import torch

from _equiv_utils import build_network_config, init_jax_flow_map

torch.set_default_dtype(torch.float32)


def _max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def _setup(seed=0, B=5, n_particles=6, predict_divergence=True, vectorized_divergence=False):
    import jax.numpy as jnp
    import common.flow_map as torch_flow_map
    from common.weight_port import load_flax_params_into_torch

    cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=predict_divergence,
        vectorized_divergence=vectorized_divergence,
        use_bfloat16=False,
        d_model=32,
        num_layers=2,
        nhead=4,
    )
    ex = np.zeros((n_particles, 3), np.float32)
    jax_net, params = init_jax_flow_map(cfg, ex, seed=seed)

    import jax
    rng_params = np.random.default_rng(seed + 103)
    # Nonzero heads/conditioning prevent vacuous zero-output parity.
    params = jax.tree_util.tree_map(
        lambda value: value + jnp.asarray(rng_params.normal(0, .02, value.shape), dtype=value.dtype), params)

    torch_net = torch_flow_map.TorchFlowMap(cfg).to(torch.float32)
    torch_net.eval()
    load_flax_params_into_torch(torch_net, params)

    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    s = rng.uniform(0.05, 0.95, size=(B,)).astype(np.float32)
    t = rng.uniform(0.05, 0.95, size=(B,)).astype(np.float32)
    return cfg, jax_net, params, torch_net, x, s, t


def test_calc_b_equiv():
    import jax.numpy as jnp

    cfg, jax_net, params, torch_net, x, s, t = _setup()
    jb, jdiv = jax_net.apply(
        params, jnp.asarray(t), jnp.asarray(x), train=False, method="calc_b", return_div=True
    )
    with torch.no_grad():
        tb, tdiv = torch_net.calc_b(torch.tensor(t), torch.tensor(x), train=False, return_div=True)
    assert np.max(np.abs(jb)) > 1e-5 and np.max(np.abs(jdiv)) > 1e-5
    assert _max_abs_diff(jb, tb.numpy()) < 1e-4
    assert _max_abs_diff(jdiv, tdiv.numpy()) < 1e-4


def test_calc_phi_equiv():
    import jax.numpy as jnp

    cfg, jax_net, params, torch_net, x, s, t = _setup()
    jphi, jdiv = jax_net.apply(
        params, jnp.asarray(s), jnp.asarray(t), jnp.asarray(x),
        train=False, method="calc_phi", return_div=True,
    )
    with torch.no_grad():
        tphi, tdiv = torch_net.calc_phi(
            torch.tensor(s), torch.tensor(t), torch.tensor(x), train=False, return_div=True
        )
    assert np.max(np.abs(jphi)) > 1e-5 and np.max(np.abs(jdiv)) > 1e-5
    assert _max_abs_diff(jphi, tphi.numpy()) < 1e-4
    assert _max_abs_diff(jdiv, tdiv.numpy()) < 1e-4


def test_flow_call_equiv():
    import jax
    import jax.numpy as jnp

    cfg, jax_net, params, torch_net, x, s, t = _setup()
    fn = lambda ss, tt, xx: jax_net.apply(params, ss, tt, xx, train=False, return_div=True)
    jX, jdiv = jax.vmap(fn)(jnp.asarray(s), jnp.asarray(t), jnp.asarray(x))
    with torch.no_grad():
        tX, tdiv = torch_net.flow_call(
            torch.tensor(s), torch.tensor(t), torch.tensor(x), train=False, return_div=True
        )
    assert _max_abs_diff(jX, tX.numpy()) < 1e-4
    assert _max_abs_diff(jdiv, tdiv.numpy()) < 1e-4


def test_partial_t_equiv():
    import jax
    import jax.numpy as jnp

    cfg, jax_net, params, torch_net, x, s, t = _setup()
    fn = lambda ss, tt, xx: jax_net.apply(
        params, ss, tt, xx, train=False, method="partial_t", return_div=True
    )
    (jX, jdiv), (jdtX, jdtdiv) = jax.vmap(fn)(jnp.asarray(s), jnp.asarray(t), jnp.asarray(x))
    tout, ttan = torch_net.partial_t(
        torch.tensor(s), torch.tensor(t), torch.tensor(x), train=False, return_div=True
    )
    tX, tdiv = tout
    tdtX, tdtdiv = ttan
    assert _max_abs_diff(jX, tX.detach().numpy()) < 1e-4
    assert _max_abs_diff(jdtX, tdtX.detach().numpy()) < 1e-3
    assert _max_abs_diff(jdiv, tdiv.detach().numpy()) < 1e-4
    assert _max_abs_diff(jdtdiv, tdtdiv.detach().numpy()) < 1e-3


def test_partial_s_equiv():
    import jax
    import jax.numpy as jnp

    cfg, jax_net, params, torch_net, x, s, t = _setup()
    fn = lambda ss, tt, xx: jax_net.apply(
        params, ss, tt, xx, train=False, method="partial_s"
    )
    jX, jdsX = jax.vmap(fn)(jnp.asarray(s), jnp.asarray(t), jnp.asarray(x))
    tout, ttan = torch_net.partial_s(
        torch.tensor(s), torch.tensor(t), torch.tensor(x), train=False
    )
    assert _max_abs_diff(jX, tout.detach().numpy()) < 1e-4
    assert _max_abs_diff(jdsX, ttan.detach().numpy()) < 1e-3


if __name__ == "__main__":
    test_calc_b_equiv()
    print("calc_b OK")
    test_calc_phi_equiv()
    print("calc_phi OK")
    test_flow_call_equiv()
    print("flow_call OK")
    test_partial_t_equiv()
    print("partial_t OK")
    test_partial_s_equiv()
    print("partial_s OK")
    print("ALL NETWORK EQUIV TESTS PASSED")
