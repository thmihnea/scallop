"""Port Flax ``FlowMap`` parameters into a ``TorchFlowMap`` state dict.

Used by the equivalence tests (and for converting trained JAX checkpoints).
The mapping mirrors the module hierarchy:

    Flax:  params/flow_map/dit/<sub>      Torch: dit.<sub>

with the usual Dense ``kernel`` (in, out) -> Linear ``weight`` (out, in)
transpose and the attention head reshape.
"""

import re
from typing import Dict

import numpy as np
import torch


def _t(arr) -> torch.Tensor:
    return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))


def _dense(flax_sub: Dict, out_key: str, sd: Dict) -> None:
    """Map a Flax Dense {kernel,bias} to a Torch Linear {weight,bias}."""
    sd[f"{out_key}.weight"] = _t(flax_sub["kernel"]).t().contiguous()
    sd[f"{out_key}.bias"] = _t(flax_sub["bias"])


def _attention(flax_attn: Dict, out_prefix: str, num_heads: int, head_dim: int, sd: Dict) -> None:
    d_model = num_heads * head_dim
    for src, dst in (("query", "q"), ("key", "k"), ("value", "v")):
        kernel = _t(flax_attn[src]["kernel"]).reshape(d_model, num_heads * head_dim)
        bias = _t(flax_attn[src]["bias"]).reshape(num_heads * head_dim)
        sd[f"{out_prefix}.{dst}.weight"] = kernel.t().contiguous()
        sd[f"{out_prefix}.{dst}.bias"] = bias
    out_kernel = _t(flax_attn["out"]["kernel"]).reshape(num_heads * head_dim, d_model)
    sd[f"{out_prefix}.out.weight"] = out_kernel.t().contiguous()
    sd[f"{out_prefix}.out.bias"] = _t(flax_attn["out"]["bias"])


def _block(flax_block: Dict, out_prefix: str, num_heads: int, head_dim: int, sd: Dict) -> None:
    _dense(flax_block["Dense_0"], f"{out_prefix}.modulation", sd)
    _dense(flax_block["Dense_1"], f"{out_prefix}.mlp_fc1", sd)
    _dense(flax_block["Dense_2"], f"{out_prefix}.mlp_fc2", sd)
    _attention(
        flax_block["MultiHeadDotProductAttention_0"], f"{out_prefix}.attn", num_heads, head_dim, sd
    )


def flax_params_to_torch_state_dict(flax_params: Dict, torch_module) -> Dict[str, torch.Tensor]:
    # Unwrap {'params': ...} collection.
    params = flax_params.get("params", flax_params)
    dit = params["flow_map"]["dit"]

    num_heads = torch_module.dit.nhead
    head_dim = torch_module.dit.d_model // num_heads

    sd: Dict[str, torch.Tensor] = {}

    # time embedders
    for emb in ("emb_s", "emb_t"):
        _dense(dit[emb]["Dense_0"], f"dit.{emb}.linear1", sd)
        _dense(dit[emb]["Dense_1"], f"dit.{emb}.linear2", sd)

    _dense(dit["x_embedder"], "dit.x_embedder", sd)

    block_re = re.compile(r"^blocks_(\d+)$")
    div_block_re = re.compile(r"^div_blocks_(\d+)$")
    for key, sub in dit.items():
        m = block_re.match(key)
        if m:
            _block(sub, f"dit.blocks.{int(m.group(1))}", num_heads, head_dim, sd)
            continue
        m = div_block_re.match(key)
        if m:
            _block(sub, f"dit.div_blocks.{int(m.group(1))}", num_heads, head_dim, sd)

    _dense(dit["final_modulation"], "dit.final_modulation", sd)
    _dense(dit["final_linear"], "dit.final_linear", sd)

    if "div_final_modulation" in dit:
        _dense(dit["div_final_modulation"], "dit.div_final_modulation", sd)
        _dense(dit["div_final_linear"], "dit.div_final_linear", sd)

    return sd


def load_flax_params_into_torch(torch_module, flax_params: Dict) -> None:
    sd = flax_params_to_torch_state_dict(flax_params, torch_module)
    missing, unexpected = torch_module.load_state_dict(sd, strict=False)
    # Only non-persistent buffers (freqs, pos_emb) may be missing.
    real_missing = [k for k in missing if not (
        k.endswith("freqs") or k.endswith("pos_emb")
    )]
    if real_missing:
        raise RuntimeError(f"Missing torch params not filled from flax: {real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when porting: {unexpected}")
