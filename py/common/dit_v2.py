"""Canonical DiT backbone adapted to molecular flow maps.

Reference: https://github.com/facebookresearch/DiT/blob/main/models.py
Copyright (c) Meta Platforms, Inc. and affiliates; reference code is licensed
under CC-BY-NC 4.0: https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt

Adaptations: atom tokens, separate source/gap embeddings, a branched raw-density
head, and primitive normalization/attention for differentiable flow-map JVPs.
The latter implement the canonical equations without fused autodiff kernels.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import LayerNorm, TimestepEmbedder, _no_autocast, modulate, remove_mean


def get_pos_embedding(indices: torch.Tensor, emb_dim: int, max_len: int = 10000) -> torch.Tensor:
    """Canonical DiT 1D sine/cosine positions for the sequence of atom tokens."""
    half = emb_dim // 2
    if half == 0:
        return torch.zeros(indices.shape + (0,), dtype=torch.float32, device=indices.device)
    k = torch.arange(half, dtype=torch.float64, device=indices.device)
    denom = max_len ** (2 * k / emb_dim)
    args = indices[..., None].to(torch.float64) / denom
    pos_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb_dim % 2:
        pos_emb = F.pad(pos_emb, (0, 1))
    return pos_emb.float()


class Attention(nn.Module):
    """Canonical packed-QKV attention, with the explicit (non-fused) path."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (self.attn_drop(attention) @ v).transpose(1, 2).reshape(batch, tokens, width)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """DiT's two-layer, tanh-GELU feed-forward network (dropout disabled)."""

    def __init__(self, hidden_dim: int, mlp_ratio: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio))
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DiTBlock(nn.Module):
    """DiT adaLN-Zero block; residual arithmetic stays in FP32 under autocast."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(hidden_dim, num_heads)
        self.norm2 = LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = Mlp(hidden_dim, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x.float() + (gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))).float()
        x = x.float() + (gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))).float()
        return x


class FinalLayer(nn.Module):
    """Canonical DiT final layer, producing one vector per atom."""

    def __init__(self, hidden_dim: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_dim, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with _no_autocast(x):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            return self.linear(modulate(self.norm_final(x), shift, scale))


class FlowMapAtomDiTV2(nn.Module):
    # Preserve compatibility with the archived canonical DiT checkpoints.
    architecture_version = "canonical_atom_dit_v2"

    def __init__(
        self, n_particles: int, n_dimension: int, d_model: int = 192,
        num_layers: int = 6, nhead: int = 6, mlp_ratio: float = 4.0,
        time_embedding_dim: int = 64, predict_divergence: bool = False,
        vectorized_divergence: bool = False, com_aware: bool = False,
        num_shared_layers: Optional[int] = None, num_div_layers: int = 3,
    ):
        super().__init__()
        self.com_aware = com_aware
        self.n_particles = n_particles
        self.n_dimension = n_dimension
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_shared_layers = num_layers if num_shared_layers is None else num_shared_layers
        self.num_div_layers = num_div_layers
        if not 0 <= self.num_shared_layers <= num_layers:
            raise ValueError("num_shared_layers must lie between 0 and num_layers")
        if num_div_layers < 0:
            raise ValueError("num_div_layers must be nonnegative")
        self.nhead = nhead
        self.mlp_ratio = mlp_ratio
        self.predict_divergence = predict_divergence
        self.vectorized_divergence = vectorized_divergence

        # Preserve the project's independent source-time and signed-gap MLPs.
        self.emb_s = TimestepEmbedder(d_model, time_embedding_dim)
        self.emb_t = TimestepEmbedder(d_model, time_embedding_dim)
        # An atom is a single token, analogous to a patch of size one.
        self.x_embedder = nn.Linear(n_dimension, d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.final_layer = FinalLayer(d_model, n_dimension)
        token_index = torch.arange(n_particles, dtype=torch.int64)[None, :]
        self.register_buffer("pos_emb", get_pos_embedding(token_index, d_model), persistent=False)
        if predict_divergence:
            self.div_blocks = nn.ModuleList([DiTBlock(d_model, nhead, mlp_ratio) for _ in range(num_div_layers)])
            self.div_final_layer = FinalLayer(d_model, n_dimension)
        self.initialize_weights()

    def initialize_weights(self):
        """Canonical DiT initialization, applied independently to both heads."""
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(basic_init)
        # Canonical DiT explicitly reinitializes the token projection as Linear.
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        for embedding in (self.emb_s, self.emb_t):
            nn.init.normal_(embedding.linear1.weight, std=0.02)
            nn.init.normal_(embedding.linear2.weight, std=0.02)
        for block in list(self.blocks) + list(getattr(self, "div_blocks", [])):
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for head in [self.final_layer] + ([self.div_final_layer] if self.predict_divergence else []):
            nn.init.zeros_(head.adaLN_modulation[-1].weight)
            nn.init.zeros_(head.adaLN_modulation[-1].bias)
            nn.init.zeros_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)

    def forward(
        self, x: torch.Tensor, ss: torch.Tensor, ts: torch.Tensor, train: bool = False,
        *, output: str = "both",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return selected (velocity, raw density); ``ts`` is the signed gap t-s."""
        if output not in ("velocity", "divergence", "both"):
            raise ValueError(f"Unknown DiT output: {output!r}")
        if output == "divergence" and not self.predict_divergence:
            raise ValueError("Divergence output requires a density head")
        want_velocity = output != "divergence"
        want_divergence = output != "velocity" and self.predict_divergence
        assert x.ndim == ss.ndim == ts.ndim == 3, "Input must have 3 dimensions"
        assert x.shape[1] == self.n_particles, "Particle dimension mismatch"
        if not self.com_aware:
            x = remove_mean(x)
        del train
        with _no_autocast(x):
            c = self.emb_s(ss.reshape(x.shape[0])) + self.emb_t(ts.reshape(x.shape[0]))
        tokens = self.x_embedder(x) + self.pos_emb.to(dtype=x.dtype)
        for i in range(self.num_shared_layers):
            tokens = self.blocks[i](tokens, c)
        velocity = density = None
        if want_velocity:
            velocity_tokens = tokens
            for i in range(self.num_shared_layers, self.num_layers):
                velocity_tokens = self.blocks[i](velocity_tokens, c)
            velocity = self.final_layer(velocity_tokens, c)
            if not self.com_aware:
                velocity = remove_mean(velocity)
        if want_divergence:
            for block in self.div_blocks:
                tokens = block(tokens, c)
            density = self.div_final_layer(tokens, c)
            if not self.vectorized_divergence:
                density = density.sum(dim=(1, 2))
        return velocity, density
