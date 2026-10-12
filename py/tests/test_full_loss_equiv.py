"""JAX-network parity for the selected correction-gradient objectives.

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

import ml_collections

from _equiv_utils import build_network_config, build_problem_config, init_jax_flow_map

torch.set_default_dtype(torch.float32)


def _max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def _build_loss_cfg(diag_fraction, loss_type="lsd", stopgrad_type="convex", psd_type=None):
    cfg = ml_collections.ConfigDict()
    cfg.training = ml_collections.ConfigDict()
    cfg.training.loss_type = loss_type
    cfg.training.stopgrad_type = stopgrad_type
    cfg.training.psd_type = psd_type
    cfg.optimization = ml_collections.ConfigDict()
    cfg.optimization.diag_fraction = diag_fraction
    return cfg


def _setup(
    seed=0,
    B=4,
    n_particles=6,
    divergence_mode="dlogpdt",
    vectorized_divergence=True,
    div_loss_weighting_type="uniform",
    predict_divergence=True,
    perturb_scale=0.15,
):
    """Build matched JAX + Torch flow maps (perturbed) and a *separate* teacher."""
    import jax
    import jax.numpy as jnp
    import common.flow_map as torch_flow_map
    from common.weight_port import (
        load_flax_params_into_torch,
        flax_params_to_torch_state_dict,
    )

    net_cfg = build_network_config(
        n_particles=n_particles,
        predict_divergence=predict_divergence,
        vectorized_divergence=vectorized_divergence,
        use_bfloat16=False,
        d_model=32,
        num_layers=2,
        nhead=4,
        divergence_mode=divergence_mode,
        div_loss_weighting_type=div_loss_weighting_type,
        divergence_rescale=float(n_particles * 3) if not vectorized_divergence else 1.0,
    )
    ex = np.zeros((n_particles, 3), np.float32)
    jax_net, params = init_jax_flow_map(net_cfg, ex, seed=seed)

    def _perturb(tree, offset):
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        prng = np.random.default_rng(seed + offset)
        new = []
        for leaf in leaves:
            arr = np.asarray(leaf)
            new.append(jnp.asarray((arr + perturb_scale * prng.standard_normal(arr.shape)).astype(np.float32)))
        return jax.tree_util.tree_unflatten(treedef, new)

    # student and a distinct teacher (distinct so the diagonal term genuinely
    # depends on the teacher params).
    params = _perturb(params, 5)
    teacher_params = _perturb(params, 99)

    torch_net = torch_flow_map.TorchFlowMap(net_cfg).to(torch.float32)
    torch_net.train()
    load_flax_params_into_torch(torch_net, params)
    teacher_torch = {
        k: v.detach() for k, v in flax_params_to_torch_state_dict(teacher_params, torch_net).items()
    }

    rng = np.random.default_rng(seed + 1)
    x0 = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    x1 = rng.standard_normal((B, n_particles, 3)).astype(np.float32)
    s = rng.uniform(0.05, 0.9, size=(B,)).astype(np.float32)
    t = rng.uniform(0.05, 0.9, size=(B,)).astype(np.float32)

    interp_cfg = build_problem_config("linear")
    return dict(
        net_cfg=net_cfg,
        jax_net=jax_net,
        params=params,
        teacher_params=teacher_params,
        torch_net=torch_net,
        teacher_torch=teacher_torch,
        x0=x0,
        x1=x1,
        s=s,
        t=t,
        interp_cfg=interp_cfg,
    )


def _compare_grads(jax_grad_tree, torch_net, tol=1e-3):
    from common.weight_port import flax_params_to_torch_state_dict

    ported = flax_params_to_torch_state_dict(jax_grad_tree, torch_net)
    torch_grads = {n: p.grad for n, p in torch_net.named_parameters()}
    worst, worst_key = 0.0, None
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


def _run_assembled_lsd(divergence_mode, vectorized, div_weight="uniform", diag_fraction=0.5,
                       stopgrad_type="convex"):
    """Compare assembled LSD loss value + grad under the selected policy."""
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    d = _setup(
        divergence_mode=divergence_mode,
        vectorized_divergence=vectorized,
        div_loss_weighting_type=div_weight,
    )
    cfg = _build_loss_cfg(diag_fraction, stopgrad_type=stopgrad_type)
    jinterp = ji.setup_interpolant(d["interp_cfg"])
    tinterp = ti.setup_interpolant(d["interp_cfg"])

    jax_loss = jl.setup_loss(cfg, d["jax_net"], jinterp)
    torch_loss = tl.setup_loss(cfg, d["torch_net"], tinterp)

    B = d["x0"].shape[0]
    # dummy dropout keys (unused: LSD-convex + analytic divergence is deterministic)
    dropout_keys = jax.random.split(jax.random.PRNGKey(0), B)

    x0 = jnp.asarray(d["x0"]); x1 = jnp.asarray(d["x1"])
    s = jnp.asarray(d["s"]); t = jnp.asarray(d["t"])

    def loss_of_params(p):
        total, _ = jax_loss(
            p, d["teacher_params"], d["teacher_params"],
            x0, x1, None, s, t, None, None, dropout_keys,
        )
        return total

    L_jax, grad_jax = jax.value_and_grad(loss_of_params)(d["params"])

    d["torch_net"].zero_grad(set_to_none=True)
    L_torch, _ = torch_loss(
        d["teacher_torch"], d["teacher_torch"],
        torch.tensor(d["x0"]), torch.tensor(d["x1"]), None,
        torch.tensor(d["s"]), torch.tensor(d["t"]), None, None,
        density_ema_params=d["torch_net"].state_dict(),
    )
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    worst, mean = _compare_grads(grad_jax, d["torch_net"])
    return rel_L, worst, mean, float(L_jax), float(L_torch.item())


# --------------------------------------------------------------------------- #
# assembled LSD objective (the real training loss)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("stopgrad", ["stopped", "convex"])
def test_assembled_lsd_dlogpdt(vectorized, stopgrad):
    rel_L, worst, mean, Lj, Lt = _run_assembled_lsd("dlogpdt", vectorized, stopgrad_type=stopgrad)
    assert rel_L < 1e-4, f"loss mismatch rel {rel_L}"
    assert worst < 1e-3, f"grad mismatch rel {worst}"
    print(f"[assembled lsd dlogpdt vec={vectorized}] L_jax={Lj:.6f} L_torch={Lt:.6f} "
          f"rel_L={rel_L:.2e} worst_grad={worst:.2e} mean_grad={mean:.2e}")


def test_assembled_lsd_cond_trace():
    rel_L, worst, mean, Lj, Lt = _run_assembled_lsd("cond_trace", vectorized=True)
    assert rel_L < 1e-4, f"loss mismatch rel {rel_L}"
    assert worst < 1e-3, f"grad mismatch rel {worst}"
    print(f"[assembled lsd cond_trace] rel_L={rel_L:.2e} worst_grad={worst:.2e}")


@pytest.mark.parametrize("div_weight", ["uniform", "square", "linear"])
def test_assembled_lsd_div_weighting(div_weight):
    # scalar divergence so div_loss_weighting actually varies the objective
    rel_L, worst, mean, Lj, Lt = _run_assembled_lsd(
        "dlogpdt", vectorized=False, div_weight=div_weight
    )
    assert rel_L < 1e-4, f"loss mismatch rel {rel_L} ({div_weight})"
    assert worst < 1e-3, f"grad mismatch rel {worst} ({div_weight})"
    print(f"[assembled lsd div_weight={div_weight}] rel_L={rel_L:.2e} worst_grad={worst:.2e}")


@pytest.mark.parametrize("diag_fraction", [0.25, 1.0])
def test_assembled_lsd_diag_fraction(diag_fraction):
    rel_L, worst, mean, Lj, Lt = _run_assembled_lsd(
        "dlogpdt", vectorized=True, diag_fraction=diag_fraction
    )
    assert rel_L < 1e-4, f"loss mismatch rel {rel_L}"
    assert worst < 1e-3, f"grad mismatch rel {worst}"
    print(f"[assembled lsd diag_fraction={diag_fraction}] rel_L={rel_L:.2e} worst_grad={worst:.2e}")


# --------------------------------------------------------------------------- #
# vectorised LSD term, value + grad (complements the scalar test in
# test_losses_equiv.py)
# --------------------------------------------------------------------------- #
def test_lsd_convex_vectorized_value_and_grad():
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    d = _setup(divergence_mode="dlogpdt", vectorized_divergence=True)
    jinterp = ji.setup_interpolant(d["interp_cfg"])
    tinterp = ti.setup_interpolant(d["interp_cfg"])

    def jax_loss(p):
        def fn(a0, a1, ss, tt):
            lv, _ = jl.lsd_term(
                p, p, a0, a1, None, ss, tt, {"dropout": jax.random.PRNGKey(0)},
                interp=jinterp, X=d["jax_net"], stopgrad_type="convex",
            )
            return lv
        return jnp.mean(jax.vmap(fn)(
            jnp.asarray(d["x0"]), jnp.asarray(d["x1"]), jnp.asarray(d["s"]), jnp.asarray(d["t"])
        ))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(d["params"])

    d["torch_net"].zero_grad(set_to_none=True)
    lv, _ = tl.lsd_term(
        d["torch_net"], None,
        torch.tensor(d["x0"]), torch.tensor(d["x1"]), None,
        torch.tensor(d["s"]), torch.tensor(d["t"]),
        interp=tinterp, stopgrad_type="convex", density_ema_params=d["torch_net"].state_dict(),
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"vectorized LSD loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, d["torch_net"])
    print(f"[lsd convex vectorized] rel_L={rel_L:.2e} worst_grad={worst:.2e} mean_grad={mean:.2e}")


# --------------------------------------------------------------------------- #
# analytic divergence target equivalence (dlogpdt / cond_trace) -- not yet
# covered by the hutchinson-only vjp test.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["dlogpdt", "cond_trace"])
@pytest.mark.parametrize("vectorized", [True, False])
def test_calc_divergence_with_phi_equiv(mode, vectorized):
    import jax
    import jax.numpy as jnp
    import _stopped_jax_losses as jl
    import common.losses as tl

    d = _setup(divergence_mode=mode, vectorized_divergence=vectorized)
    B = d["x0"].shape[0]
    xt = (1-d["t"][:,None,None])*d["x0"]+d["t"][:,None,None]*d["x1"]  # original sampled state

    def jax_div(a_t, a_xt, a_x0, a_x1):
        phi, div = jl._calc_divergence_with_phi(
            d["params"], a_t, a_xt, a_x0, a_x1, None, {"dropout": jax.random.PRNGKey(0)},
            X=d["jax_net"], method="calc_b",
        )
        return phi, div

    jphi, jdiv = jax.vmap(jax_div)(
        jnp.asarray(d["t"]), jnp.asarray(xt), jnp.asarray(d["x0"]), jnp.asarray(d["x1"])
    )

    with torch.no_grad():
        tphi, tdiv = tl._calc_divergence_with_phi(
            d["torch_net"], None,
            torch.tensor(d["t"]), torch.tensor(xt),
            torch.tensor(d["x0"]), torch.tensor(d["x1"]),
        )
    assert _max_abs_diff(jphi, tphi.numpy()) < 1e-4
    assert _max_abs_diff(jdiv, tdiv.numpy()) < 1e-4
    print(f"[calc_div mode={mode} vec={vectorized}] OK")


# --------------------------------------------------------------------------- #
# hutchinson diagonal-term equivalence with *identical* eps.
# --------------------------------------------------------------------------- #
def test_diagonal_hutchinson_term_equiv():
    import jax
    import jax.numpy as jnp
    import jax_impl.common.interpolant as ji
    import _stopped_jax_losses as jl
    import common.interpolant as ti
    import common.losses as tl

    d = _setup(divergence_mode="hutchinson", vectorized_divergence=False)
    jinterp = ji.setup_interpolant(d["interp_cfg"])
    tinterp = ti.setup_interpolant(d["interp_cfg"])
    B, n = d["x0"].shape[0], d["x0"].shape[1]

    # Build per-sample dropout keys and reproduce the exact eps JAX will draw.
    keys = jax.random.split(jax.random.PRNGKey(7), B)
    eps_np = np.stack([
        np.asarray(jax.random.normal(keys[i], shape=(n, 3))) for i in range(B)
    ]).astype(np.float32)

    def jax_loss(p):
        def fn(a0, a1, tt, key):
            lv, _ = jl.diagonal_term(
                p, d["teacher_params"], a0, a1, None, tt, {"dropout": key},
                interp=jinterp, X=d["jax_net"],
            )
            return lv
        return jnp.mean(jax.vmap(fn)(
            jnp.asarray(d["x0"]), jnp.asarray(d["x1"]), jnp.asarray(d["t"]), keys
        ))

    L_jax, grad_jax = jax.value_and_grad(jax_loss)(d["params"])

    d["torch_net"].zero_grad(set_to_none=True)
    lv, _ = tl.diagonal_term(
        d["torch_net"], d["teacher_torch"],
        torch.tensor(d["x0"]), torch.tensor(d["x1"]), None, torch.tensor(d["t"]),
        interp=tinterp, eps=torch.tensor(eps_np),
    )
    L_torch = lv.mean()
    L_torch.backward()

    rel_L = abs(float(L_jax) - float(L_torch.item())) / max(1.0, abs(float(L_jax)))
    assert rel_L < 1e-4, f"hutchinson diagonal loss mismatch rel {rel_L}"
    worst, mean = _compare_grads(grad_jax, d["torch_net"])
    print(f"[diag hutchinson] rel_L={rel_L:.2e} worst_grad={worst:.2e} mean_grad={mean:.2e}")


if __name__ == "__main__":
    for vec in (True, False):
        print(test_assembled_lsd_dlogpdt.__name__, vec, _run_assembled_lsd("dlogpdt", vec)[:3])
    print("cond_trace", _run_assembled_lsd("cond_trace", True)[:3])
    test_lsd_convex_vectorized_value_and_grad()
    test_diagonal_hutchinson_term_equiv()
    print("DONE")
