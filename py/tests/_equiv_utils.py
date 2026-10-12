"""Shared helpers for JAX<->PyTorch equivalence tests."""

import os
import sys

import numpy as np

# Make the PyTorch ``common`` package (under ``py/``) importable. The reference
# JAX implementation (``jax_impl.common``) is put on ``sys.path`` by conftest.py.
_PY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

import ml_collections  # noqa: E402


def perturb_params_(net, std: float = 0.1, seed: int = 0):
    """Move a freshly built flow map off its zero-initialised output heads.

    ``dit.FlowMapAtomDiT`` zero-initialises BOTH final linears -- velocity
    (``final_linear``) and divergence (``div_final_linear``) -- so a fresh
    ``TorchFlowMap`` returns ``phi == 0`` and ``D == 0`` identically. Any test that
    compares a derivative of the output against a finite difference of the output
    then compares 0 with 0 and passes no matter what the code does. Call this first.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in net.parameters():
            prm.add_(torch.randn(prm.shape, generator=g) * std)
    return net


def build_network_config(
    n_particles: int = 8,
    n_dimension: int = 3,
    predict_divergence: bool = True,
    vectorized_divergence: bool = False,
    use_bfloat16: bool = False,
    d_model: int = 32,
    num_layers: int = 2,
    nhead: int = 4,
    divergence_mode: str = "hutchinson",
    div_loss_weighting_type: str = "uniform",
    divergence_rescale: float = 1.0,
    from_scratch: bool = False,
    sigma_data: float = 1.0,
) -> ml_collections.ConfigDict:
    """A small network config shared by the JAX and Torch flow maps."""
    c = ml_collections.ConfigDict()
    c.network_type = "dit"
    c.n_particles = n_particles
    c.n_dimension = n_dimension
    c.atom_type_labels = tuple(range(n_particles))
    c.bonds = tuple()
    c.input_dims = (n_particles, n_dimension)
    c.predict_divergence = predict_divergence
    c.vectorized_divergence = vectorized_divergence
    c.divergence_mode = divergence_mode
    c.div_loss_weighting_type = div_loss_weighting_type
    c.divergence_rescale = divergence_rescale
    c.rescale = 1.0
    c.unscale_factor = 1.0
    c.sigma_data = sigma_data
    c.use_weight = False
    c.logvar_channels = None
    c.use_bfloat16 = use_bfloat16
    c.from_scratch = from_scratch
    c.dit_kwargs = {
        "d_model": d_model,
        "num_layers": num_layers,
        "nhead": nhead,
        "mlp_ratio": 4.0,
        "time_embedding_dim": 16,
    }
    return c


def build_problem_config(interp_type: str = "linear") -> ml_collections.ConfigDict:
    c = ml_collections.ConfigDict()
    c.problem = ml_collections.ConfigDict()
    c.problem.interp_type = interp_type
    return c


def init_jax_flow_map(net_cfg, ex_input_np, seed: int = 0):
    """Initialise the JAX FlowMap and return (net, params)."""
    import jax
    import jax.numpy as jnp
    import jax_impl.common.flow_map as jax_flow_map

    ex_input = jnp.asarray(ex_input_np)
    key = jax.random.PRNGKey(seed)
    net, params, _ = jax_flow_map.initialize_flow_map(net_cfg, ex_input, key)
    return net, params


def jax_to_numpy(x):
    return np.asarray(x, dtype=np.float64)
