import math
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

def remove_mean(x: jnp.ndarray) -> jnp.ndarray:
    """Removes the center of mass (assuming equal masses for simplicity)."""
    mean = jnp.mean(x, axis=1, keepdims=True)
    return x - mean

class PositionalEmbedding(nn.Module):
    """Sinusoidal embeddings used for scalar time inputs."""

    num_channels: int
    max_positions: int = 10000

    def setup(self):
        half = self.num_channels // 2
        if half == 0:
            self.freqs = jnp.zeros((0,), dtype=jnp.float32)
            return
        freqs = np.arange(half, dtype=np.float32) / half
        self.freqs = jnp.asarray((1.0 / self.max_positions) ** freqs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.asarray(t, dtype=jnp.float32).reshape(-1)
        args = t[:, None] * self.freqs[None, :]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if self.num_channels % 2:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb


class TimestepEmbedder(nn.Module):
    hidden_dim: int
    frequency_embedding_dim: int = 256

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_freq = PositionalEmbedding(self.frequency_embedding_dim)(t)
        t_emb = nn.Dense(self.hidden_dim)(t_freq)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.hidden_dim)(t_emb)
        return t_emb


def get_pos_embedding(indices: jnp.ndarray, emb_dim: int, max_len: int = 2048) -> jnp.ndarray:
    """Creates fixed sine/cosine token positional embeddings."""

    half = emb_dim // 2
    if half == 0:
        return jnp.zeros(indices.shape + (0,), dtype=jnp.float32)

    k = jnp.arange(half, dtype=jnp.float32)
    denom = max_len ** (2 * k / emb_dim)
    args = indices[..., None].astype(jnp.float32) * math.pi / denom
    pos_emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if emb_dim % 2:
        pos_emb = jnp.pad(pos_emb, ((0, 0),) * indices.ndim + ((0, 1),))
    return pos_emb

# ---------------------------------------------------------------------------- #
# DiT Blocks
# ---------------------------------------------------------------------------- #
def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Applies adaLN modulation."""
    return x * (1 + jnp.expand_dims(scale, 1)) + jnp.expand_dims(shift, 1)

class DiTBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        modulation = nn.Dense(
            6 * self.hidden_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(nn.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        x_mod = modulate(
            nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x), shift_msa, scale_msa
        )
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=0.0,
        )(x_mod)
        x = x + jnp.expand_dims(gate_msa, 1) * attn_out

        x_mod_mlp = modulate(
            nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x), shift_mlp, scale_mlp
        )
        mlp_hidden = int(self.hidden_dim * self.mlp_ratio)
        mlp_out = nn.Dense(mlp_hidden)(x_mod_mlp)
        mlp_out = nn.gelu(mlp_out, approximate=True)
        mlp_out = nn.Dense(self.hidden_dim)(mlp_out)
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp_out

        return x

# ---------------------------------------------------------------------------- #
# Main DiT Model
# ---------------------------------------------------------------------------- #
class FlowMapAtomDiT(nn.Module):
    n_particles: int
    n_dimension: int
    atom_type_labels: Tuple[int, ...]
    bonds: Tuple[Tuple[int, int], ...]
    d_model: int = 192
    num_layers: int = 6
    nhead: int = 6
    mlp_ratio: float = 4.0
    time_embedding_dim: int = 64
    predict_divergence: bool = False
    vectorized_divergence: bool = False

    def setup(self):
        self.atom_type_labels_arr = np.array(self.atom_type_labels, dtype=np.int32)
        self.bonds_arr = np.array(self.bonds, dtype=np.int32)

        self.emb_s = TimestepEmbedder(self.d_model, self.time_embedding_dim)
        self.emb_t = TimestepEmbedder(self.d_model, self.time_embedding_dim)
        self.x_embedder = nn.Dense(self.d_model)
        self.blocks = [DiTBlock(self.d_model, self.nhead, self.mlp_ratio) for _ in range(self.num_layers)]
        self.final_norm = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)
        self.final_modulation = nn.Dense(
            2 * self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
        self.final_linear = nn.Dense(
            self.n_dimension,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )

        if self.predict_divergence:
            self.div_blocks = [DiTBlock(self.d_model, self.nhead, self.mlp_ratio) for _ in range(3)]
            self.div_final_norm = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)
            self.div_final_modulation = nn.Dense(
                2 * self.d_model,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )
            self.div_final_linear = nn.Dense(
                self.n_dimension,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )

    def _assert_input_shape(self, x, ss, ts):
        assert x.ndim == ss.ndim == ts.ndim == 3, "Input must have 3 dimensions"
        assert x.shape[1] == self.n_particles, "Particle dimension mismatch"

    def __call__(self, x: jnp.ndarray, ss: jnp.ndarray, ts: jnp.ndarray, train: bool = False):
        self._assert_input_shape(x, ss, ts)

        x = remove_mean(x)

        del train

        batch_size = x.shape[0]
        s = ss.reshape(batch_size)
        t = ts.reshape(batch_size)
        c = self.emb_s(s) + self.emb_t(t)

        token_index = jnp.arange(self.n_particles, dtype=jnp.int32)[None, :]
        pos_emb = get_pos_embedding(token_index, self.d_model)

        x_sc = jnp.zeros_like(x)
        x_tokens = self.x_embedder(jnp.concatenate([x, x_sc], axis=-1)) + pos_emb

        for block in self.blocks:
            x_tokens = block(x_tokens, c)

        shift, scale = jnp.split(self.final_modulation(nn.silu(c)), 2, axis=-1)
        x_out = modulate(self.final_norm(x_tokens), shift, scale)
        vel = self.final_linear(x_out)
        vel = remove_mean(vel)

        if self.predict_divergence:
            for block in self.div_blocks:
                x_tokens = block(x_tokens, c)
            div_shift, div_scale = jnp.split(self.div_final_modulation(nn.silu(c)), 2, axis=-1)
            div = self.div_final_linear(modulate(self.div_final_norm(x_tokens), div_shift, div_scale))
            if not self.vectorized_divergence:
                div = div.sum(axis=(1, 2))
        else:
            div = None

        return vel, div