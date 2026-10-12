"""Tests for the com-aware flow map (translation-augmentation mode).

When com_aware is set (via --com_augmentation) the DiT drops its COM-invariance
(no remove_mean on input or output velocity), the base keeps a Gaussian COM, and
base_log_prob is the full 3N Gaussian. This makes the flow model the COM DOF so the
transferable-samplers com_energy_adjustment is non-degenerate.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

pytest.importorskip("mdtraj")  # common.tsx (SourceEnergy) transitively needs it

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import common.flow_map as fm  # noqa: E402
from _equiv_utils import build_network_config  # noqa: E402

torch.set_default_dtype(torch.float32)


def _net(com_aware, n=5, seed=0):
    torch.manual_seed(seed)
    cfg = build_network_config(
        n_particles=n, predict_divergence=True, vectorized_divergence=False,
        divergence_mode="dlogpdt", d_model=32, num_layers=2, nhead=4,
    )
    cfg.com_aware = com_aware
    net = fm.TorchFlowMap(cfg).eval()
    # final_linear is zero-init; perturb so the velocity is non-trivial.
    torch.nn.init.normal_(net.dit.final_linear.weight, std=0.1)
    torch.nn.init.normal_(net.dit.final_linear.bias, std=0.1)
    return net, cfg


def test_velocity_is_mean_free_when_not_com_aware():
    net, _ = _net(False)
    phi = net.calc_b(torch.full((4,), 0.5), torch.randn(4, 5, 3), train=False)
    assert float(phi.mean(dim=1).abs().max()) < 1e-5  # COM-invariant: mean-free velocity


def test_velocity_carries_com_when_com_aware():
    net, _ = _net(True)
    phi = net.calc_b(torch.full((4,), 0.5), torch.randn(4, 5, 3), train=False)
    assert float(phi.mean(dim=1).abs().max()) > 1e-4  # com-aware: velocity has a COM


def test_base_log_prob_differs_between_modes():
    net_mf, _ = _net(False)
    net_ca, _ = _net(True)
    x = torch.randn(4, 5, 3)
    # com-aware drops the mean-free (3-COM-DOF) correction constant
    assert not torch.allclose(net_mf.base_log_prob(x), net_ca.base_log_prob(x))


def test_com_energy_adjustment_finite_for_com_aware_samples():
    from common.tsx import SourceEnergy

    net, _ = _net(True)
    x0 = torch.randn(32, 5, 3)  # non-mean-free base (com-aware)
    x1, _, _ = net.sample_forward(x0, n_steps=2)
    adj = SourceEnergy.com_energy_adjustment(x1)
    assert torch.isfinite(adj).all()                       # non-degenerate (COM present)
    assert float(x1.mean(dim=1).norm(dim=-1).mean()) > 1e-3  # samples genuinely carry a COM
