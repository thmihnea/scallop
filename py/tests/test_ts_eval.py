"""Tests for the installed transferable-samplers evaluation pieces (ESS, energy
Wasserstein, SNIS) and the iF2D2 eval glue."""
import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

pytest.importorskip("transferable_samplers")
pytest.importorskip("mdtraj")

import torch

pytest.importorskip("ot")      # POT, used by energy_wasserstein

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from common import tsx  # noqa: E402
from common.ts_eval import _compute_mean_metrics  # noqa: E402

torch.set_default_dtype(torch.float32)


def test_normalized_ess_matches_kish():
    torch.manual_seed(0)
    lw = torch.randn(1000)
    got = float(tsx.normalized_ess(lw))
    kish = float(torch.exp(2 * torch.logsumexp(lw, 0) - torch.logsumexp(2 * lw, 0)) / len(lw))
    assert abs(got - kish) < 1e-6
    # uniform weights -> full ESS
    assert abs(float(tsx.normalized_ess(torch.zeros(500))) - 1.0) < 1e-6


def test_energy_wasserstein_finite_and_symmetric():
    from transferable_samplers.evaluation.metrics.wasserstein_distances import energy_wasserstein

    a = torch.randn(500)
    b = torch.randn(500) + 0.5
    m1 = energy_wasserstein(a, b)
    m2 = energy_wasserstein(b, a)
    key = [k for k in m1 if k.endswith("energy-w2")][0]
    assert np.isfinite(float(m1[key]))
    assert abs(float(m1[key]) - float(m2[key])) < 1e-4  # 1-D OT is symmetric


def test_snis_shapes_and_ess_range():
    n_atoms = 5

    def sample_fn(n, **_):
        s = torch.randn(n, n_atoms, 3)
        return s, s.flatten(1).pow(2).sum(-1)  # (samples, E_source)

    source = tsx.SourceEnergy(
        sample_fn=sample_fn,
        energy_fn=lambda x: torch.zeros(x.shape[0]),
        sample_batch_size=200, energy_batch_size=200, grad_batch_size=200,
        use_com_adjustment=False,
    )
    target = tsx.TargetEnergy(
        energy_fn=lambda x: x.flatten(1).pow(2).sum(-1) * 1.1,
        normalization_std=torch.tensor(1.0),
    )
    samples_dict, diag = tsx.SNISSampler(num_samples=200, logw_quantile_filter=None).sample(source, target)
    assert set(samples_dict.keys()) == {"proposal", "resampled"}
    res = samples_dict["resampled"]
    assert res.logw is not None and len(res) == 200
    assert res.samples.shape == (200, n_atoms, 3)
    ess = float(tsx.normalized_ess(res.logw))
    assert 0.0 < ess <= 1.0 + 1e-6


def test_com_adjustment_applied_in_source_energy():
    # with use_com_adjustment, E_source gets the CoM term added
    n_atoms = 6

    def sample_fn(n, **_):
        s = torch.randn(n, n_atoms, 3) * 0.3  # non-mean-free -> finite CoM term
        return s, torch.zeros(n)

    src_off = tsx.SourceEnergy(sample_fn, lambda x: torch.zeros(x.shape[0]), 100, 100, 100, use_com_adjustment=False)
    src_on = tsx.SourceEnergy(sample_fn, lambda x: torch.zeros(x.shape[0]), 100, 100, 100, use_com_adjustment=True)
    torch.manual_seed(0)
    _, e_off = src_off.sample(100)
    torch.manual_seed(0)
    _, e_on = src_on.sample(100)
    assert not torch.allclose(e_off, e_on)  # adjustment changed E_source


def test_compute_mean_metrics():
    m = {
        "val/Ace-A-Nme/resampled/energy-w2": 2.0,
        "val/Ace-A-Nme/proposal/energy-w2": 4.0,
        "val/Ace-A-Nme/resampled/effective-sample-size": 0.01,
    }
    out = _compute_mean_metrics(m, "val")
    assert abs(out["val/mean/resampled/energy-w2"] - 2.0) < 1e-9
    assert abs(out["val/mean/proposal/energy-w2"] - 4.0) < 1e-9
