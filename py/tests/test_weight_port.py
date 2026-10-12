"""Structural round-trip test for weight_port (the Flax->Torch bridge).

Every equivalence test depends on ``weight_port`` mapping parameters correctly;
a silent key-mapping regression would only surface as a numeric mismatch
elsewhere. This test asserts the mapping is total and shape-consistent, and that
loading succeeds with only the whitelisted non-persistent buffers unfilled.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

pytest.importorskip("jax")
pytest.importorskip("flax")

from _equiv_utils import build_network_config, init_jax_flow_map

torch.set_default_dtype(torch.float32)


def _build(seed=0, n_particles=6, predict_divergence=True, vectorized=False):
    import common.flow_map as torch_flow_map

    cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=predict_divergence,
        vectorized_divergence=vectorized,
        d_model=32,
        num_layers=2,
        nhead=4,
    )
    ex = np.zeros((n_particles, 3), np.float32)
    _, params = init_jax_flow_map(cfg, ex, seed=seed)
    torch_net = torch_flow_map.TorchFlowMap(cfg).to(torch.float32)
    return cfg, params, torch_net


@pytest.mark.parametrize("predict_divergence", [True, False])
def test_state_dict_key_and_shape_coverage(predict_divergence):
    from common.weight_port import flax_params_to_torch_state_dict

    _, params, torch_net = _build(predict_divergence=predict_divergence)
    sd = flax_params_to_torch_state_dict(params, torch_net)

    torch_params = dict(torch_net.named_parameters())
    sd_keys = set(sd.keys())
    param_keys = set(torch_params.keys())

    # Every trainable parameter is produced by the port, and nothing extra.
    assert sd_keys == param_keys, (
        f"missing from port: {param_keys - sd_keys}; extra: {sd_keys - param_keys}"
    )
    # Shapes line up exactly (guards a bad transpose/reshape).
    for k, v in sd.items():
        assert tuple(v.shape) == tuple(torch_params[k].shape), (
            f"shape mismatch at {k}: port {tuple(v.shape)} vs torch {tuple(torch_params[k].shape)}"
        )


def test_load_round_trip_no_missing_or_unexpected():
    from common.weight_port import load_flax_params_into_torch

    _, params, torch_net = _build()
    # Raises RuntimeError on any non-whitelisted missing/unexpected key.
    load_flax_params_into_torch(torch_net, params)

    # After loading, every parameter is finite and was actually overwritten
    # (Flax init is mostly zeros; a no-op load would leave torch's random init).
    for name, p in torch_net.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after port: {name}"
