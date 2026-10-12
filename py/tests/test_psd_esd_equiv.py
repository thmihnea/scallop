"""JAX-network parity for the corrected stopped-target objectives.

The unchanged vendored JAX losses are historical and differentiate correction
JVPs. _stopped_jax_losses is an independent oracle for the corrected equations;
production value/gradient and teacher-isolation tests also run without JAX.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

pytest.importorskip("jax")
pytest.importorskip("flax")

from _equiv_utils import build_network_config, build_problem_config, init_jax_flow_map

torch.set_default_dtype(torch.float32)


def _max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def _compare_grads(jax_grad_tree, torch_net, tol):
    """Per-tensor normalised grad comparison (see test_losses_equiv._compare_grads)."""
    from common.weight_port import flax_params_to_torch_state_dict

    ported = flax_params_to_torch_state_dict(jax_grad_tree, torch_net)
    torch_grads = {n: p.grad for n, p in torch_net.named_parameters()}
    worst, worst_key = 0.0, None
    diffs = []
    for key, jg in ported.items():
        tg = torch_grads[key]
        if tg is None:
            # The param did not participate in the Torch loss, so its grad is 0.
            # Equivalence still holds iff the JAX grad is ~0 too (e.g. the div-head
            # params, which PSD/ESD do not use). A non-zero JAX grad here would
            # (correctly) fail the tolerance check below and flag a real mismatch.
            tg = torch.zeros_like(jg)
        scale = max(1.0, float(jg.abs().max()))
        d = _max_abs_diff(jg.numpy(), tg.detach().cpu().numpy()) / scale
        diffs.append(d)
        if d > worst:
            worst, worst_key = d, key
    assert worst < tol, f"grad mismatch at {worst_key}: rel {worst} (tol {tol})"
    return worst, float(np.mean(diffs))


def _setup(seed=0, B=4, n_particles=6, predict_divergence=True):
    """Init JAX flow map + perturbed params, port to Torch, build ordered inputs."""
    import jax
    import jax.numpy as jnp
    import common.flow_map as torch_flow_map
    from common.weight_port import load_flax_params_into_torch

    net_cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=predict_divergence,
        vectorized_divergence=False,
        use_bfloat16=False,
        d_model=32,
        num_layers=2,
        nhead=4,
        divergence_mode="dlogpdt",
        div_loss_weighting_type="uniform",
        divergence_rescale=1.0,
    )
    ex = np.zeros((n_particles, 3), np.float32)
    jax_net, params = init_jax_flow_map(net_cfg, ex, seed=seed)

    leaves, treedef = jax.tree_util.tree_flatten(params)
    prng = np.random.default_rng(seed + 5)
    new_leaves = [
        jnp.asarray((np.asarray(l) + 0.15 * prng.standard_normal(np.asarray(l).shape)).astype(np.float32))
        for l in leaves
    ]
    params = jax.tree_util.tree_unflatten(treedef, new_leaves)

    torch_net = torch_flow_map.TorchFlowMap(net_cfg).to(torch.float32)
    torch_net.train()
    load_flax_params_into_torch(torch_net, params)

    rng = np.random.default_rng(seed + 1)
    x0 = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    x1 = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    # Ordered times s < u < t so the two-step PSD composition s->u->t is sensible.
    s = rng.uniform(0.05, 0.40, size=(B,)).astype(np.float32)
    t = rng.uniform(0.60, 0.90, size=(B,)).astype(np.float32)
    u = (s + (t - s) * rng.uniform(0.2, 0.8, size=(B,))).astype(np.float32)
    h = rng.uniform(0.1, 0.9, size=(B,)).astype(np.float32)

    interp_cfg = build_problem_config("linear")
    return net_cfg, jax_net, params, torch_net, x0, x1, s, t, u, h, interp_cfg


@pytest.mark.parametrize("psd_type", ["uniform", "midpoint"])
@pytest.mark.parametrize("stopgrad", ["stopped", "convex"])
def test_psd_value_and_grad(psd_type, stopgrad):
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    # velocity objective only: see the module docstring.
    net_cfg, jax_net, params, torch_net, x0, x1, s, t, u, h, interp_cfg = _setup(
        predict_divergence=False
    )
    jinterp = ji.setup_interpolant(interp_cfg)
    tinterp = ti.setup_interpolant(interp_cfg)
    u = .5*(s+t) if psd_type == "midpoint" else h*s+(1-h)*t

    def jax_loss(p):
        def fn(a0, a1, ss, tt, uu, hh):
            return jl.psd_term(
                p, p, a0, a1, None, ss, tt, uu, hh,
                {"dropout": jax.random.PRNGKey(0)},
                interp=jinterp, X=jax_net, psd_type=psd_type, stopgrad_type=stopgrad,
            )
        return jnp.mean(jax.vmap(fn)(
            jnp.asarray(x0), jnp.asarray(x1), jnp.asarray(s),
            jnp.asarray(t), jnp.asarray(u), jnp.asarray(h),
        ))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(params)

    torch_net.zero_grad(set_to_none=True)
    lv, _ = tl.psd_term(
        torch_net, None,
        torch.tensor(x0), torch.tensor(x1), None,
        torch.tensor(s), torch.tensor(t), torch.tensor(u), torch.tensor(h),
        interp=tinterp, psd_type=psd_type, stopgrad_type=stopgrad,
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"PSD[{psd_type},{stopgrad}] loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, torch_net, tol=6e-3)
    print(f"[psd {psd_type} {stopgrad}] rel_L={rel_L:.2e} worst_grad={worst:.2e} mean_grad={mean:.2e}")


@pytest.mark.parametrize("stopgrad", ["stopped", "convex"])
def test_esd_value_and_grad(stopgrad):
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    # velocity objective only: see the module docstring.
    net_cfg, jax_net, params, torch_net, x0, x1, s, t, u, h, interp_cfg = _setup(
        predict_divergence=False
    )
    jinterp = ji.setup_interpolant(interp_cfg)
    tinterp = ti.setup_interpolant(interp_cfg)

    def jax_loss(p):
        def fn(a0, a1, ss, tt):
            return jl.esd_term(
                p, p, a0, a1, None, ss, tt,
                {"dropout": jax.random.PRNGKey(0)},
                interp=jinterp, X=jax_net, stopgrad_type=stopgrad,
            )
        return jnp.mean(jax.vmap(fn)(
            jnp.asarray(x0), jnp.asarray(x1), jnp.asarray(s), jnp.asarray(t)
        ))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(params)

    torch_net.zero_grad(set_to_none=True)
    lv, _ = tl.esd_term(
        torch_net, None,
        torch.tensor(x0), torch.tensor(x1), None,
        torch.tensor(s), torch.tensor(t),
        interp=tinterp, stopgrad_type=stopgrad,
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"ESD[{stopgrad}] loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, torch_net, tol=8e-3)
    print(f"[esd {stopgrad}] rel_L={rel_L:.2e} worst_grad={worst:.2e} mean_grad={mean:.2e}")


def test_lsd_stopped_value_and_grad():
    """The stopgrad_type='none' LSD branch (only 'convex' was previously covered)."""
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    net_cfg, jax_net, params, torch_net, x0, x1, s, t, u, h, interp_cfg = _setup()
    jinterp = ji.setup_interpolant(interp_cfg)
    tinterp = ti.setup_interpolant(interp_cfg)

    def jax_loss(p):
        def fn(a0, a1, ss, tt):
            lv, _ = jl.lsd_term(
                p, p, a0, a1, None, ss, tt, {"dropout": jax.random.PRNGKey(0)},
                interp=jinterp, X=jax_net, stopgrad_type="stopped",
            )
            return lv
        return jnp.mean(jax.vmap(fn)(
            jnp.asarray(x0), jnp.asarray(x1), jnp.asarray(s), jnp.asarray(t)
        ))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(params)

    torch_net.zero_grad(set_to_none=True)
    lv, _ = tl.lsd_term(
        torch_net, None,
        torch.tensor(x0), torch.tensor(x1), None, torch.tensor(s), torch.tensor(t),
        interp=tinterp, stopgrad_type="stopped", density_ema_params=torch_net.state_dict(),
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"LSD[none] loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, torch_net, tol=6e-3)
    print(f"[lsd none] rel_L={rel_L:.2e} worst_grad={worst:.2e} mean_grad={mean:.2e}")


if __name__ == "__main__":
    for pt in ("uniform", "midpoint"):
        for sg in ("convex", "none"):
            test_psd_value_and_grad(pt, sg)
    for sg in ("full", "convex", "none"):
        test_esd_value_and_grad(sg)
    test_lsd_stopped_value_and_grad()
    print("ALL PSD/ESD/LSD-none EQUIV TESTS PASSED")
