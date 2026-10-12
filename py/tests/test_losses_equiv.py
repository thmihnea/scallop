"""JAX-network parity for the selected correction-gradient objectives.

The unchanged vendored JAX losses are historical and differentiate correction
JVPs. _stopped_jax_losses is an independent oracle for the corrected equations;
production value/gradient and teacher-isolation tests also run without JAX.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import numpy as np
import torch

from _equiv_utils import build_network_config, build_problem_config, init_jax_flow_map

torch.set_default_dtype(torch.float32)


def _max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def _setup(seed=0, B=4, n_particles=6, divergence_mode="dlogpdt"):
    import jax.numpy as jnp
    import common.flow_map as torch_flow_map
    from common.weight_port import load_flax_params_into_torch

    net_cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=True,
        vectorized_divergence=False,
        use_bfloat16=False,
        d_model=32,
        num_layers=2,
        nhead=4,
        divergence_mode=divergence_mode,
        div_loss_weighting_type="uniform",
        divergence_rescale=1.0,
    )
    ex = np.zeros((n_particles, 3), np.float32)
    jax_net, params = init_jax_flow_map(net_cfg, ex, seed=seed)

    # Perturb the (mostly zero-initialised) params so the flow map is non-trivial
    # and LSD exercises real gradients. Same perturbation feeds both frameworks.
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(params)
    prng = np.random.default_rng(seed + 5)
    new_leaves = []
    for leaf in leaves:
        arr = np.asarray(leaf)
        new_leaves.append(jnp.asarray((arr + 0.15 * prng.standard_normal(arr.shape)).astype(np.float32)))
    params = jax.tree_util.tree_unflatten(treedef, new_leaves)

    torch_net = torch_flow_map.TorchFlowMap(net_cfg).to(torch.float32)
    torch_net.train()
    load_flax_params_into_torch(torch_net, params)

    rng = np.random.default_rng(seed + 1)
    x0 = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    x1 = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    s = rng.uniform(0.05, 0.9, size=(B,)).astype(np.float32)
    t = rng.uniform(0.05, 0.9, size=(B,)).astype(np.float32)

    interp_cfg = build_problem_config("linear")
    return net_cfg, jax_net, params, torch_net, x0, x1, s, t, interp_cfg


def _compare_grads(jax_grad_tree, torch_net, tol=6e-3):
    """Compare grads (normalised per-tensor). float32 reverse-over-forward through
    the transformer accumulates ~1e-3 relative reduction-order noise, hence tol."""
    from common.weight_port import flax_params_to_torch_state_dict

    ported = flax_params_to_torch_state_dict(jax_grad_tree, torch_net)
    torch_grads = {n: p.grad for n, p in torch_net.named_parameters()}
    worst = 0.0
    worst_key = None
    diffs = []
    for key, jg in ported.items():
        tg = torch_grads[key]
        assert tg is not None, f"missing torch grad for {key}"
        scale = max(1.0, float(jg.abs().max()))
        d = _max_abs_diff(jg.numpy(), tg.detach().cpu().numpy()) / scale
        diffs.append(d)
        if d > worst:
            worst, worst_key = d, key
    assert worst < tol, f"grad mismatch at {worst_key}: rel {worst}"
    return worst, float(np.mean(diffs))


def test_lsd_convex_value_and_grad():
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    net_cfg, jax_net, params, torch_net, x0, x1, s, t, interp_cfg = _setup()
    jinterp = ji.setup_interpolant(interp_cfg)
    tinterp = ti.setup_interpolant(interp_cfg)

    # ---- JAX scalar loss + grad ----
    def jax_loss(p):
        def fn(a0, a1, ss, tt):
            lv, _ = jl.lsd_term(
                p, p, a0, a1, None, ss, tt, {"dropout": jax.random.PRNGKey(0)},
                interp=jinterp, X=jax_net, stopgrad_type="convex",
            )
            return lv
        return jnp.mean(jax.vmap(fn)(jnp.asarray(x0), jnp.asarray(x1), jnp.asarray(s), jnp.asarray(t)))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(params)

    # ---- Torch scalar loss + grad ----
    torch_net.zero_grad(set_to_none=True)
    lv, _ = tl.lsd_term(
        torch_net, None,
        torch.tensor(x0), torch.tensor(x1), None, torch.tensor(s), torch.tensor(t),
        interp=tinterp, stopgrad_type="convex", density_ema_params=torch_net.state_dict(),
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"LSD loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, torch_net)
    print(f"[lsd convex] L_jax={float(L_jax):.6f} L_torch={float(L_torch):.6f} "
          f"rel_L={rel_L:.2e} worst_grad_rel={worst:.2e} mean_grad_rel={mean:.2e}")


def test_diagonal_dlogpdt_value_and_grad():
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    net_cfg, jax_net, params, torch_net, x0, x1, s, t, interp_cfg = _setup()
    jinterp = ji.setup_interpolant(interp_cfg)
    tinterp = ti.setup_interpolant(interp_cfg)

    def jax_loss(p):
        def fn(a0, a1, tt):
            lv, _ = jl.diagonal_term(
                p, p, a0, a1, None, tt, {"dropout": jax.random.PRNGKey(0)},
                interp=jinterp, X=jax_net,
            )
            return lv
        return jnp.mean(jax.vmap(fn)(jnp.asarray(x0), jnp.asarray(x1), jnp.asarray(t)))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(params)

    teacher = {n: p.detach().clone() for n, p in torch_net.named_parameters()}
    torch_net.zero_grad(set_to_none=True)
    lv, _ = tl.diagonal_term(
        torch_net, teacher,
        torch.tensor(x0), torch.tensor(x1), None, torch.tensor(t),
        interp=tinterp,
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"diagonal loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, torch_net)
    print(f"[diag dlogpdt] L_jax={float(L_jax):.6f} L_torch={float(L_torch):.6f} "
          f"rel_L={rel_L:.2e} worst_grad_rel={worst:.2e} mean_grad_rel={mean:.2e}")


def test_hutchinson_vjp_equiv():
    """Same eps -> identical Hutchinson divergence estimate (vjp equivalence)."""
    import jax
    import jax.numpy as jnp
    import common.losses as tl

    net_cfg, jax_net, params, torch_net, x0, x1, s, t, interp_cfg = _setup(
        divergence_mode="hutchinson"
    )
    B, n = x0.shape[0], x0.shape[1]
    rng = np.random.default_rng(123)
    eps = rng.standard_normal((B, n, 3)).astype(np.float32)
    xt = x0  # arbitrary evaluation point

    # JAX: vjp per sample with provided eps
    def jax_div(a_t, a_x, a_eps):
        def phi_fn(z):
            return jax_net.apply(params, a_t, z, train=False, method="calc_b")
        phi, vjp = jax.vjp(phi_fn, a_x)
        vjp_eps = vjp(a_eps)[0]
        return jnp.sum(vjp_eps * a_eps)

    jdiv = jax.vmap(jax_div)(jnp.asarray(t), jnp.asarray(xt), jnp.asarray(eps))

    _, tdiv = tl._hutchinson_divergence_with_phi(
        torch_net, None, torch.tensor(t), torch.tensor(xt), eps=torch.tensor(eps)
    )
    assert _max_abs_diff(jdiv, tdiv.numpy()) < 1e-4
    print("[hutchinson vjp] OK")


if __name__ == "__main__":
    test_lsd_convex_value_and_grad()
    test_diagonal_dlogpdt_value_and_grad()
    test_hutchinson_vjp_equiv()
    print("ALL LOSS EQUIV TESTS PASSED")
