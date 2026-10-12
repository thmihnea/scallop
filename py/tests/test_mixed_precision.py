"""Coverage for ``network.mixed_precision = "bf16_trunk"`` -- real mixed precision.

This is a DIFFERENT thing from ``use_bfloat16`` (see ``test_bf16_input.py``). That flag
rounds the coordinate input to bf16 and casts straight back, leaving every weight and
every matmul in fp32; measured against the two Ace-A-Nme chains it bought no throughput
at all (28.12 vs 28.61 it/s on pretrain_v). ``bf16_trunk`` instead puts the DiT trunk's
matmuls -- ``x_embedder``, the per-block modulation, the four attention projections,
``q@kT`` / ``attn@v``, and the MLP pair -- on bf16 tensor cores, while pinning to fp32
every operation the objective actually depends on:

* the ``LayerNorm`` interior (``dit.py``): ``x - mu`` and ``var`` are cancellations, and
  ``eps=1e-6`` is below bf16's resolution at ``var ~ 1``, so the guard would stop
  existing. This module's *second-order* path is what LSD/ESD differentiate through;
* the residual stream across all 9 blocks;
* the conditioning vector ``c``, computed once and reused by every block;
* both output heads -- ``remove_mean(vel)`` is exactly the op the JAX/torch equivalence
  tests are sensitive to, and ``div`` feeds a loss whose target carries a ``1/(1-t)``
  amplification of up to 1000x (``losses.py``);
* everything outside the DiT, which follows for free because the autocast region is
  scoped to the single ``self.dit(...)`` call in ``flow_map.calc_phi``.

Same vacuity trap as ``test_bf16_input.py``: a fresh ``TorchFlowMap`` returns exactly
zero because the DiT zero-initialises its output and divergence heads, so every test
here perturbs the weights first and asserts the output is non-degenerate.

Two of these tests exist specifically because a *weaker* version of them would pass
while the scheme was wrong:

* ``test_layernorm_interior_is_fp32_not_just_its_output`` -- the LayerNorm's output
  dtype is fp32 under autocast even when its interior ran in bf16 (``torch.rsqrt`` is
  fp32-promoted and ``bf16 * fp32 -> fp32``), so an output-dtype check is vacuous;
* ``test_the_trunk_really_does_run_in_bf16`` -- a scheme that pinned *everything* to
  fp32 would satisfy every accuracy test here and buy nothing.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from _equiv_utils import build_network_config

torch.set_default_dtype(torch.float32)

# bfloat16 has an 8-bit mantissa. 50x is the same headroom test_bf16_input.py allows for
# the JAX/torch bf16 gap: one bf16 rounding's worth, times a factor for amplification
# through the trunk. It is a CHOSEN bound, not a measured gap -- it is here to catch a
# regression that makes the departure larger, not to claim a size.
_BF16_EPS = 2.0 ** -8
_TRUNK_TOL = 50 * _BF16_EPS


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a)
    assert denom > 0.0, "reference output is exactly zero -- the comparison would be vacuous"
    return float(np.linalg.norm(a - b) / denom)


@torch.no_grad()
def _perturb_(net, seed=17, scale=0.1):
    """Randomise every parameter so the zero-initialised heads actually emit signal."""
    gen = torch.Generator().manual_seed(seed)
    for p in net.parameters():
        p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _net(mixed_precision="off", use_bfloat16=False, seed=0, n_particles=6,
         vectorized_divergence=True):
    import common.flow_map as torch_flow_map

    cfg = build_network_config(
        n_particles=n_particles,
        use_bfloat16=use_bfloat16,
        vectorized_divergence=vectorized_divergence,
    )
    cfg.mixed_precision = mixed_precision
    torch.manual_seed(seed)
    net = torch_flow_map.TorchFlowMap(cfg).to(torch.float32).eval()
    _perturb_(net)
    return net


def _pair(**kw):
    """An fp32 net and a bf16_trunk net carrying bitwise-identical weights."""
    a = _net(mixed_precision="off", **kw)
    b = _net(mixed_precision="bf16_trunk", **kw)
    b.load_state_dict(a.state_dict())
    return a, b


def _inputs(seed=1, B=8, n_particles=6):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    t = rng.uniform(0.05, 0.95, size=(B,)).astype(np.float32)
    return torch.tensor(x), torch.tensor(t)


def _dtypes_by_module(net, x, t):
    """Forward-hook every submodule and record the dtype it emits."""
    seen, handles = {}, []

    def hook(name):
        def f(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o):
                seen.setdefault(name, o.dtype)
        return f

    for name, mod in net.named_modules():
        if name:
            handles.append(mod.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            net.calc_b(t, x, train=False, return_div=True)
    finally:
        for h in handles:
            h.remove()
    return seen


# --------------------------------------------------------------------------- #
# the flag itself
# --------------------------------------------------------------------------- #
def test_default_is_off_and_unknown_values_are_rejected():
    """A typo must not silently degrade to fp32 -- that is audit N17's failure mode."""
    import common.flow_map as torch_flow_map

    cfg = build_network_config(n_particles=6)
    assert not hasattr(cfg, "mixed_precision") or cfg.mixed_precision == "off"
    assert torch_flow_map.TorchFlowMap(cfg).mixed_precision == "off", "default must be off"

    cfg.mixed_precision = "bf16"  # plausible typo for bf16_trunk
    with pytest.raises(ValueError, match="mixed_precision"):
        torch_flow_map.TorchFlowMap(cfg)


def test_off_is_bitwise_identical_to_the_fp32_path():
    """Adding the flag must not perturb the configuration everything else is tested in."""
    net_a = _net(mixed_precision="off")
    net_b = _net(mixed_precision="off")
    x, t = _inputs()
    with torch.no_grad():
        a = net_a.calc_b(t, x, train=False, return_div=False)
        b = net_b.calc_b(t, x, train=False, return_div=False)
    assert float(a.abs().max()) > 0.0, "output is zero -- the comparison would be vacuous"
    assert torch.equal(a, b)


def test_the_two_precision_flags_are_independent():
    """use_bfloat16 rounds the input; mixed_precision changes the kernels. Neither implies
    the other, and all four combinations must construct and run."""
    x, t = _inputs()
    outs = {}
    for ub in (False, True):
        for mp in ("off", "bf16_trunk"):
            net = _net(mixed_precision=mp, use_bfloat16=ub)
            assert net.use_bfloat16 is ub
            assert net.mixed_precision == mp
            with torch.no_grad():
                outs[(ub, mp)] = net.calc_b(t, x, train=False, return_div=False)
            assert torch.isfinite(outs[(ub, mp)]).all()
    # each flag must move the output on its own
    assert not torch.equal(outs[(False, "off")], outs[(True, "off")])
    assert not torch.equal(outs[(False, "off")], outs[(False, "bf16_trunk")])


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def test_bf16_trunk_changes_the_field_but_within_the_stated_bound():
    net_fp32, net_mp = _pair()
    x, t = _inputs()
    with torch.no_grad():
        a = net_fp32.calc_b(t, x, train=False, return_div=False)
        b = net_mp.calc_b(t, x, train=False, return_div=False)
    rel = _rel_l2(a.numpy(), b.numpy())
    assert rel > 0.0, "bf16_trunk changed nothing -- the flag is not reaching the kernels"
    assert rel < _TRUNK_TOL, f"bf16_trunk departure {rel:.3e} exceeds {_TRUNK_TOL:.3e}"


def test_layernorm_interior_is_fp32_not_just_its_output():
    """The output dtype is fp32 either way -- assert on the VALUE, not the dtype.

    Under autocast, ``torch.rsqrt`` is fp32-promoted and ``bf16 * fp32 -> fp32``, so a
    LayerNorm whose ``x - mu`` and ``var`` ran entirely in bf16 still *returns* fp32. The
    only honest check is that feeding it a bf16 tensor gives bitwise the same answer as
    feeding it that tensor upcast to fp32.
    """
    import common.dit as dit_mod

    ln = dit_mod.LayerNorm(32, eps=1e-6)
    torch.manual_seed(3)
    x = torch.randn(4, 6, 32)
    x_bf16 = x.to(torch.bfloat16)
    with torch.no_grad():
        out_bf16_in = ln(x_bf16)
        out_upcast_in = ln(x_bf16.float())
    assert out_bf16_in.dtype is torch.float32
    assert torch.equal(out_bf16_in, out_upcast_in), (
        "LayerNorm is normalising in bf16: its mean/var/rsqrt must run in fp32"
    )


# --------------------------------------------------------------------------- #
# the precision map -- both halves of it
# --------------------------------------------------------------------------- #
def test_the_trunk_really_does_run_in_bf16():
    """Without this, a scheme that pinned everything to fp32 would pass every other test
    in this file and buy exactly nothing."""
    net = _net(mixed_precision="bf16_trunk")
    x, t = _inputs()
    seen = _dtypes_by_module(net, x, t)
    expected_bf16 = [
        "dit.x_embedder",
        "dit.blocks.0.modulation",
        "dit.blocks.0.attn.q",
        "dit.blocks.0.attn.out",
        "dit.blocks.0.mlp_fc1",
        "dit.blocks.0.mlp_fc2",
    ]
    for name in expected_bf16:
        assert name in seen, f"{name} never ran -- hook names are stale"
        assert seen[name] is torch.bfloat16, f"{name} ran in {seen[name]}, expected bfloat16"


def test_the_fp32_islands_hold():
    """Everything the objective actually depends on must stay fp32."""
    net = _net(mixed_precision="bf16_trunk")
    x, t = _inputs()
    seen = _dtypes_by_module(net, x, t)
    expected_fp32 = [
        "dit.emb_s",                    # conditioning, reused by all blocks
        "dit.emb_t",
        "dit.blocks.0.norm1",           # LayerNorm (see the value test above too)
        "dit.final_norm",
        "dit.final_modulation",
        "dit.final_linear",             # -> remove_mean(vel)
        "dit.div_final_norm",
        "dit.div_final_modulation",
        "dit.div_final_linear",         # -> the 1/(1-t)-amplified divergence loss
    ]
    for name in expected_fp32:
        assert name in seen, f"{name} never ran -- hook names are stale"
        assert seen[name] is torch.float32, f"{name} ran in {seen[name]}, expected float32"


def test_residual_stream_stays_fp32():
    """Nine blocks of bf16 residual accumulation is the classic silent-drift failure."""
    import common.dit as dit_mod

    block = dit_mod.DiTBlock(32, 4, 4.0)
    torch.manual_seed(5)
    x = torch.randn(4, 6, 32)
    c = torch.randn(4, 32)
    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = block(x, c)
    assert out.dtype is torch.float32, f"residual stream is {out.dtype}"


def test_nothing_outside_the_dit_sees_bf16():
    """The autocast region is scoped to the single self.dit(...) call, so calc_phi's
    outputs -- and therefore div_hat, the LSD residuals and the dlogp accumulator --
    are fp32 by construction. This pins that scoping."""
    net = _net(mixed_precision="bf16_trunk")
    x, t = _inputs()
    with torch.no_grad():
        phi, div = net.calc_b(t, x, train=False, return_div=True)
    assert phi.dtype is torch.float32
    assert div is not None and div.dtype is torch.float32
    assert torch.isfinite(phi).all() and torch.isfinite(div).all()

    # and the sampler's log-density accumulator, which is where a bf16 ULP would be
    # worth ~0.5-1.0 nat against a logp of magnitude 100-250
    with torch.no_grad():
        out = net.sample_forward(x, n_steps=2)
    x1, logp = out[0], out[1]
    assert x1.dtype is torch.float32 and logp.dtype is torch.float32


# --------------------------------------------------------------------------- #
# the AD stack -- the reason this scheme could have been impossible
# --------------------------------------------------------------------------- #
def test_autocast_composes_with_the_forward_mode_ad_used_by_lsd():
    """LSD calls partial_t (a torch.func.jvp) and then backprops through it; hutchinson
    takes a reverse-mode VJP. Both are sharp edges under autocast."""
    net = _net(mixed_precision="bf16_trunk")
    x, t = _inputs(B=4)
    s = torch.zeros_like(t)

    prim, tang = net.partial_t(s, t, x, train=False, return_div=True)
    Xst, D_st = prim
    dt_Xst, dt_Dst = tang
    for name, v in [("Xst", Xst), ("D_st", D_st), ("dt_Xst", dt_Xst), ("dt_Dst", dt_Dst)]:
        assert v.dtype is torch.float32, f"{name} came back {v.dtype}"
        assert torch.isfinite(v).all(), f"{name} is not finite"

    loss = (dt_Xst ** 2).sum()
    grads = torch.autograd.grad(loss, [p for p in net.parameters() if p.requires_grad],
                                allow_unused=True)
    nonzero = [g for g in grads if g is not None]
    assert nonzero, "reverse-over-forward produced no gradients"
    assert all(g.dtype is torch.float32 for g in nonzero), "parameter grads must be fp32"
    assert any(float(g.abs().max()) > 0 for g in nonzero), "all gradients are zero"


def test_autocast_composes_with_the_reverse_mode_vjp_used_by_hutchinson():
    net = _net(mixed_precision="bf16_trunk")
    x, t = _inputs(B=4)
    x_in = x.detach().requires_grad_(True)
    eps = torch.randn_like(x_in)
    with torch.enable_grad():
        phi = net.calc_b(t, x_in, train=False, return_div=False)
        (vjp_eps,) = torch.autograd.grad(phi, x_in, grad_outputs=eps)
    assert vjp_eps.dtype is torch.float32, f"VJP came back {vjp_eps.dtype}"
    assert torch.isfinite(vjp_eps).all()
    assert float(vjp_eps.abs().max()) > 0.0
