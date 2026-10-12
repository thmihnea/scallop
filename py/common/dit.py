"""PyTorch port of common/dit.py (the molecule DiT network).

The legacy layout matches Flax. New peptide configs branch after four shared
blocks into two velocity and two density blocks. Velocity blocks keep their
original ``blocks.0`` through ``blocks.5`` names for pretrained checkpoint reuse.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """Removes the center of mass (assuming equal masses)."""
    return x - x.mean(dim=1, keepdim=True)


def _no_autocast(x: torch.Tensor):
    """Force fp32 for a region, even inside a bf16 autocast context.

    A no-op when autocast is not active, so the fp32 path is bitwise unchanged.
    Note this only stops FURTHER downcasting -- tensors that are already bf16 stay
    bf16, so callers must hand it fp32 inputs.
    """
    return torch.amp.autocast(x.device.type, enabled=False)


class PositionalEmbedding(nn.Module):
    """Sinusoidal embeddings for scalar time inputs (matches Flax order: cos|sin)."""

    def __init__(self, num_channels: int, max_positions: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        half = num_channels // 2
        if half == 0:
            freqs = np.zeros((0,), dtype=np.float32)
        else:
            exps = np.arange(half, dtype=np.float32) / half
            freqs = (1.0 / max_positions) ** exps
        self.register_buffer("freqs", torch.from_numpy(np.asarray(freqs, dtype=np.float32)), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32).reshape(-1)
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.num_channels % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.pos = PositionalEmbedding(frequency_embedding_dim)
        self.linear1 = nn.Linear(frequency_embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.pos(t)
        t_emb = self.linear1(t_freq)
        t_emb = F.silu(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


def get_pos_embedding(indices: torch.Tensor, emb_dim: int, max_len: int = 2048) -> torch.Tensor:
    """Fixed sine/cosine token positional embeddings (matches Flax: sin|cos)."""
    half = emb_dim // 2
    if half == 0:
        return torch.zeros(indices.shape + (0,), dtype=torch.float32, device=indices.device)
    k = torch.arange(half, dtype=torch.float32, device=indices.device)
    denom = max_len ** (2 * k / emb_dim)
    args = indices[..., None].to(torch.float32) * math.pi / denom
    pos_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb_dim % 2:
        pos_emb = F.pad(pos_emb, (0, 1))
    return pos_emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation (shift/scale have shape (B, d_model))."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm(nn.Module):
    """Affine-free LayerNorm composed from primitive ops.

    NOTE: this deliberately does *not* use ``torch.nn.LayerNorm`` /
    ``F.layer_norm``. The fused kernel has a hand-written double-backward whose
    second-order (reverse-over-forward) gradient is numerically inaccurate -- it
    disagrees with JAX/Flax by several percent. The LSD and ESD losses
    differentiate through ``partial_t`` / ``partial_s`` (a ``jvp``) and then take
    a reverse-mode gradient, so they depend on exactly that second-order path.
    Composing the normalisation from primitive ops (mean / var / rsqrt) makes the
    forward, first- and second-order derivatives all match ``flax.linen.LayerNorm``
    (with ``use_bias=False, use_scale=False``) to float precision.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 interior, always. Under bf16 autocast the OUTPUT of this module is
        # float32 anyway (torch.rsqrt is fp32-promoted and bf16*fp32 -> fp32), which
        # hides the fact that ``x - mu`` and ``var`` would otherwise run in bf16.
        # Both are cancellations, and eps=1e-6 is far below bf16's resolution at
        # var ~ 1, so the guard would silently stop existing. A no-op when x is
        # already float32, so the fp32 path is bitwise unchanged.
        xf = x.float()
        mu = xf.mean(dim=-1, keepdim=True)
        xc = xf - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention matching Flax ``nn.MultiHeadDotProductAttention`` defaults."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, hd = self.num_heads, self.head_dim

        def split_heads(t):
            return t.reshape(B, N, H, hd).permute(0, 2, 1, 3)  # (B, H, N, hd)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))

        scale = 1.0 / math.sqrt(hd)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, N, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * hd)
        return self.out(out)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modulation = nn.Linear(hidden_dim, 6 * hidden_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.norm1 = LayerNorm(hidden_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.norm2 = LayerNorm(hidden_dim, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(hidden_dim, mlp_hidden)
        self.mlp_fc2 = nn.Linear(mlp_hidden, hidden_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = self.modulation(F.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # The residual stream is kept in fp32 across all blocks. Under bf16 autocast
        # the Linears return bf16 and type promotion would already give fp32 here
        # (x_tokens enters fp32 via the fp32 pos_emb), but that is correct by
        # accident: it depends on a promotion chain that a later edit could break.
        # The .float() calls are no-ops on the fp32 path.
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x_mod)
        x = x.float() + (gate_msa.unsqueeze(1) * attn_out).float()

        x_mod_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp_fc1(x_mod_mlp)
        mlp_out = F.gelu(mlp_out, approximate="tanh")
        mlp_out = self.mlp_fc2(mlp_out)
        x = x.float() + (gate_mlp.unsqueeze(1) * mlp_out).float()
        return x


class FlowMapAtomDiT(nn.Module):
    architecture_version = "original_atom_dit_v1"

    def __init__(
        self,
        n_particles: int,
        n_dimension: int,
        d_model: int = 192,
        num_layers: int = 6,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        time_embedding_dim: int = 64,
        predict_divergence: bool = False,
        vectorized_divergence: bool = False,
        com_aware: bool = False,
        num_shared_layers: Optional[int] = None,
        num_div_layers: int = 3,
    ):
        super().__init__()
        self.com_aware = com_aware
        self.n_particles = n_particles
        self.n_dimension = n_dimension
        self.d_model = d_model
        self.num_layers = num_layers
        # Missing fields retain the historical all-shared + three-block density
        # layout for saved configs and Flax checkpoints. New configs set 4 and 2.
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

        self.emb_s = TimestepEmbedder(d_model, time_embedding_dim)
        self.emb_t = TimestepEmbedder(d_model, time_embedding_dim)
        self.x_embedder = nn.Linear(2 * n_dimension, d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.final_norm = LayerNorm(d_model, eps=1e-6)
        self.final_modulation = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        self.final_linear = nn.Linear(d_model, n_dimension)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        # Fixed token positional embedding.
        token_index = torch.arange(n_particles, dtype=torch.int64)[None, :]
        self.register_buffer("pos_emb", get_pos_embedding(token_index, d_model), persistent=False)

        if predict_divergence:
            self.div_blocks = nn.ModuleList([DiTBlock(d_model, nhead, mlp_ratio) for _ in range(num_div_layers)])
            self.div_final_norm = LayerNorm(d_model, eps=1e-6)
            self.div_final_modulation = nn.Linear(d_model, 2 * d_model)
            nn.init.zeros_(self.div_final_modulation.weight)
            nn.init.zeros_(self.div_final_modulation.bias)
            self.div_final_linear = nn.Linear(d_model, n_dimension)
            nn.init.zeros_(self.div_final_linear.weight)
            nn.init.zeros_(self.div_final_linear.bias)

    def forward(
        self, x: torch.Tensor, ss: torch.Tensor, ts: torch.Tensor, train: bool = False,
        *, output: str = "both",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluate requested branches, returning (velocity_or_None, density_or_None).

        ``output`` is "velocity", "divergence", or "both". The last option keeps
        the historical (velocity, None) result for velocity-only models.
        """
        if output not in ("velocity", "divergence", "both"):
            raise ValueError(f"Unknown DiT output: {output!r}")
        if output == "divergence" and not self.predict_divergence:
            raise ValueError("Divergence output requires a density head")
        want_velocity = output != "divergence"
        want_divergence = output != "velocity" and self.predict_divergence
        # x: (B, n_particles, n_dim); ss, ts: (B, 1, 1). ``ts`` carries (t - s).
        assert x.ndim == ss.ndim == ts.ndim == 3, "Input must have 3 dimensions"
        assert x.shape[1] == self.n_particles, "Particle dimension mismatch"

        # COM-invariant by default; with com_aware (translation augmentation) the
        # network keeps the center of mass so the flow can model the COM DOF.
        if not self.com_aware:
            x = remove_mean(x)
        del train

        batch_size = x.shape[0]
        s = ss.reshape(batch_size)
        t = ts.reshape(batch_size)
        # The conditioning vector is computed once and reused by every block's
        # modulation, so it is a negligible share of the FLOPs and a poor place to
        # spend precision. PositionalEmbedding already pins t to fp32 (:37), but the
        # TimestepEmbedder Linears after it do not.
        with _no_autocast(x):
            c = self.emb_s(s) + self.emb_t(t)

        pos_emb = self.pos_emb.to(dtype=x.dtype)
        x_sc = torch.zeros_like(x)
        x_tokens = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb

        for i in range(self.num_shared_layers):
            x_tokens = self.blocks[i](x_tokens, c)

        vel = None
        if want_velocity:
            velocity_tokens = x_tokens
            for i in range(self.num_shared_layers, self.num_layers):
                velocity_tokens = self.blocks[i](velocity_tokens, c)
            with _no_autocast(x):
                shift, scale = self.final_modulation(F.silu(c)).chunk(2, dim=-1)
                x_out = modulate(self.final_norm(velocity_tokens), shift, scale)
                vel = self.final_linear(x_out)
                if not self.com_aware:
                    vel = remove_mean(vel)

        if want_divergence:
            # Branch from the shared representation, before velocity-only blocks.
            for block in self.div_blocks:
                x_tokens = block(x_tokens, c)
            # Same for the divergence head: div feeds a loss whose target carries a
            # 1/(1-t) amplification of up to 1000x (losses.py:102,105), and the
            # non-vectorized branch sums 3N elements.
            with _no_autocast(x):
                div_shift, div_scale = self.div_final_modulation(F.silu(c)).chunk(2, dim=-1)
                div = self.div_final_linear(modulate(self.div_final_norm(x_tokens), div_shift, div_scale))
                if not self.vectorized_divergence:
                    div = div.sum(dim=(1, 2))
        else:
            div = None

        return vel, div
