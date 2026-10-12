"""Coverage for ``use_bfloat16=True`` -- the configuration everything was trained in.

Audit N1 (reports/2026-09-03-torch-branch-numerical-audit.md): all four canonical
configs shipped ``config.network.use_bfloat16 = True`` while ``DECISIONS.md`` recorded
it as off, and *every* equivalence and smoke test forced it ``False``
(``_equiv_utils.py``, ``test_network_equiv.py``, ``test_losses_equiv.py``,
``test_psd_esd_equiv.py``, ``test_full_loss_equiv.py``, ``test_ode_likelihood.py``,
``test_train_smoke.py``). The default is now ``False`` and ``learn.py --use_bfloat16``
re-enables it, so this file exists to keep the bf16 path exercised at all.

IMPORTANT, and the reason every test here perturbs the weights first: a freshly
constructed ``TorchFlowMap`` returns **exactly zero** for any input, because the DiT
zero-initialises its output and divergence heads (``dit.py:211-212``, ``:225-226``).
Comparing bf16-on against bf16-off on an untrained net therefore compares 0.0 with 0.0
and passes vacuously. ``_perturbed_params`` makes the network a generic nonlinear map,
and each test additionally asserts the output is non-degenerate so this trap cannot
silently reappear.

What is asserted, and the distinction matters:

1. the truncation is live, is exactly an *input* rounding, and reaches derivatives
   (torch only -- no JAX needed);
2. torch and JAX still agree at ``use_bfloat16=True`` **to a loosened, stated
   tolerance**. That second bound is NOT an equality claim: JAX hands the bf16 array
   itself to the module (``jax_impl/common/network_utils.py:94,102``) so ``remove_mean``
   runs in bf16 there, while torch casts back to fp32 at ``flow_map.py:140`` before
   ``dit.forward``. That is the real, documented divergence deferred in ``DECISIONS.md``.
   The test pins its size so a regression making it *larger* is caught, rather than
   pretending it is zero.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from _equiv_utils import build_network_config

torch.set_default_dtype(torch.float32)

# bfloat16 has an 8-bit mantissa: 2^-8 = 3.9e-3 worst-case relative rounding on the
# input (reports/probes/output/probe1_2.txt measures 1.68e-3 rms on real coordinates).
_BF16_EPS = 2.0 ** -8


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


def _torch_nets(seed=0, n_particles=6):
    """Two torch flow maps with identical, non-trivial weights, differing only in bf16."""
    import common.flow_map as torch_flow_map

    cfg_fp32 = build_network_config(n_particles=n_particles, use_bfloat16=False)
    cfg_bf16 = build_network_config(n_particles=n_particles, use_bfloat16=True)

    torch.manual_seed(seed)
    net_fp32 = torch_flow_map.TorchFlowMap(cfg_fp32).to(torch.float32).eval()
    _perturb_(net_fp32)
    net_bf16 = torch_flow_map.TorchFlowMap(cfg_bf16).to(torch.float32).eval()
    net_bf16.load_state_dict(net_fp32.state_dict())
    return net_fp32, net_bf16


def _inputs(seed=1, B=8, n_particles=6):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    t = rng.uniform(0.05, 0.95, size=(B,)).astype(np.float32)
    return torch.tensor(x), torch.tensor(t)


def test_bfloat16_flag_is_wired_through():
    """The config flag must actually reach the truncation, in both directions."""
    net_fp32, net_bf16 = _torch_nets()
    assert net_fp32.use_bfloat16 is False
    assert net_bf16.use_bfloat16 is True


def test_untrained_flow_map_would_make_this_file_vacuous():
    """Guard the guard: document that a fresh net emits exactly zero.

    If the DiT ever stops zero-initialising its heads this test fails loudly, which is
    the signal to simplify the rest of this file rather than a bug.
    """
    import common.flow_map as torch_flow_map

    torch.manual_seed(0)
    fresh = torch_flow_map.TorchFlowMap(
        build_network_config(n_particles=6, use_bfloat16=False)
    ).to(torch.float32).eval()
    x, t = _inputs()
    with torch.no_grad():
        b = fresh.calc_b(t, x, train=False)
    assert float(b.abs().max()) == 0.0, (
        "a freshly initialised flow map no longer returns exactly zero; the _perturb_ "
        "workaround in this file may no longer be necessary"
    )


def test_bfloat16_truncation_changes_the_velocity_field():
    """bf16 on must differ from bf16 off, or the flag is silently dead."""
    net_fp32, net_bf16 = _torch_nets()
    x, t = _inputs()
    with torch.no_grad():
        b_fp32 = net_fp32.calc_b(t, x, train=False)
        b_bf16 = net_bf16.calc_b(t, x, train=False)

    assert float(b_fp32.abs().max()) > 0.0, "degenerate network: comparison would be vacuous"
    rel = _rel_l2(b_fp32.numpy(), b_bf16.numpy())
    # A dead flag gives exactly 0.0 ...
    assert rel > 0.0
    # ... and the perturbation must stay a perturbation. No tight ceiling is asserted:
    # the audit measured up to 20x amplification of the 0.17% input rounding on the
    # TRAINED Ace-A-Nme net (report section 2.1), and an untrained randomised net has no
    # predictable amplification factor. What must hold is that truncating the input does
    # not destroy the output.
    assert rel < 1.0, f"bf16 perturbation {rel:.3e} is not a perturbation"


def test_bfloat16_rounds_the_input_and_nothing_else():
    """The truncation is an input round-trip only; kernels stay fp32 (flow_map.py:136-142)."""
    x, t = _inputs()
    rounded = x.to(torch.bfloat16).to(torch.float32)
    assert not torch.equal(x, rounded)
    assert _rel_l2(x.numpy(), rounded.numpy()) < 2 * _BF16_EPS

    net_fp32, net_bf16 = _torch_nets()
    with torch.no_grad():
        b_bf16 = net_bf16.calc_b(t, x, train=False)
        b_fp32_on_rounded = net_fp32.calc_b(t, rounded, train=False)
        b_fp32_on_raw = net_fp32.calc_b(t, x, train=False)

    assert float(b_bf16.abs().max()) > 0.0, "degenerate network: comparison would be vacuous"
    # The bf16 net on raw input is BITWISE the fp32 net on pre-rounded input: the only
    # thing the flag does is round x. bf16(bf16(x)) == bf16(x), so it cannot be anything
    # else -- if weights or activations were also being truncated this would fail.
    assert torch.equal(b_bf16, b_fp32_on_rounded)
    # ... and that is genuinely different from not rounding at all.
    assert not torch.equal(b_bf16, b_fp32_on_raw)


def test_bfloat16_quantizes_tangents_except_rademacher():
    """Audit N1(c): the cast quantizes derivative vectors too -- but +-1 is bf16-exact.

    This is why ``ode_likelihood`` picks Rademacher probes while training's Hutchinson
    estimator (``losses.py:68``, Gaussian) gets no such protection.
    """
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.standard_normal((4, 6, 3)).astype(np.float32))

    def cast(v):
        return v.to(torch.bfloat16).to(torch.float32)

    gauss = torch.tensor(rng.standard_normal(x.shape).astype(np.float32))
    _, jvp_gauss = torch.func.jvp(cast, (x,), (gauss,))
    assert torch.equal(jvp_gauss, cast(gauss))
    assert not torch.equal(jvp_gauss, gauss)  # Gaussian tangents ARE quantized

    rade = torch.tensor(rng.integers(0, 2, size=x.shape).astype(np.float32) * 2.0 - 1.0)
    _, jvp_rade = torch.func.jvp(cast, (x,), (rade,))
    assert torch.equal(jvp_rade, rade)  # Rademacher tangents pass through untouched


def test_jax_torch_agree_at_bfloat16_within_the_known_remove_mean_gap():
    """Equivalence at use_bfloat16=True, to a LOOSENED and deliberately stated bound.

    The fp32 tests assert 1e-4 (``test_network_equiv.py``). Here the two stacks are
    genuinely not equal -- JAX runs ``remove_mean`` in bf16, torch does not -- so the
    bound says "the disagreement stays at the scale of the bf16 input rounding", which is
    what that divergence predicts. It does not say the disagreement is absent.
    """
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    import jax
    import jax.numpy as jnp

    import common.flow_map as torch_flow_map
    from common.weight_port import load_flax_params_into_torch
    from _equiv_utils import init_jax_flow_map

    n_particles = 6
    cfg = build_network_config(n_particles=n_particles, use_bfloat16=True)
    ex = np.zeros((n_particles, 3), np.float32)
    jax_net, params = init_jax_flow_map(cfg, ex, seed=0)

    # Same zero-init problem as above (jax_impl/common/dit.py mirrors the zero-init
    # heads), so perturb the JAX params and port THOSE into torch: both stacks then
    # evaluate the identical non-trivial network.
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(123), len(leaves))
    noisy = [
        leaf + 0.1 * jax.random.normal(k, leaf.shape, leaf.dtype)
        if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
        for leaf, k in zip(leaves, keys)
    ]
    params = jax.tree_util.tree_unflatten(treedef, noisy)

    torch_net = torch_flow_map.TorchFlowMap(cfg).to(torch.float32).eval()
    load_flax_params_into_torch(torch_net, params)

    x, t = _inputs(seed=7, n_particles=n_particles)
    jb, jdiv = jax_net.apply(
        params, jnp.asarray(t.numpy()), jnp.asarray(x.numpy()),
        train=False, method="calc_b", return_div=True,
    )
    with torch.no_grad():
        tb, tdiv = torch_net.calc_b(t, x, train=False, return_div=True)

    jb_np = np.asarray(jb, dtype=np.float64)
    jdiv_np = np.asarray(jdiv, dtype=np.float64)
    assert np.max(np.abs(jb_np)) > 0.0, "degenerate network: comparison would be vacuous"

    rel_b = _rel_l2(jb_np, tb.numpy())
    rel_div = _rel_l2(jdiv_np, tdiv.numpy())

    # One bf16 input rounding's worth of headroom, with a factor for the amplification
    # through two DiT blocks. Tight enough to catch a real regression; honest that this
    # is not the 1e-4 the fp32 path holds to.
    assert rel_b < 50 * _BF16_EPS, f"calc_b bf16 gap {rel_b:.3e}"
    assert rel_div < 50 * _BF16_EPS, f"divergence bf16 gap {rel_div:.3e}"
