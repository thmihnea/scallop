"""Tests for the vendored transferable-samplers dataloader (Ace-A-Nme).

Verifies the default pipeline (StandardizeTransform + Random3DRotationTransform)
and the optional CenterOfMassTransform + its density correction.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

pytest.importorskip("transferable_samplers")
pytest.importorskip("mdtraj")

import torch

pytest.importorskip("scipy")
pytest.importorskip("mdtraj")  # common.tsx transitively imports the evaluator subtree

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from common import tsx, ts_data  # noqa: E402

torch.set_default_dtype(torch.float32)

from common import ts_systems
TRAIN = ts_systems.data_paths(ts_systems.resolve("ace_a_nme"))["train"]


def _data(n=1000):
    if not os.path.exists(TRAIN):
        pytest.skip("local Ace-A-Nme train.npy not present")
    return torch.from_numpy(np.load(TRAIN)[:n]).float()


def test_standardize_mean_free_and_scaled():
    x = _data()
    std = ts_systems.compute_std(x)
    out = tsx.StandardizeTransform(std)({"x": x.clone()})["x"]
    assert out.shape == x.shape
    assert float(out.mean(dim=1).abs().max()) < 1e-5              # per-frame mean-free
    assert abs(float(out.std()) - 1.0) < 0.05                     # ÷ scalar std -> ~unit


def test_rotation_is_so3_and_distance_preserving():
    x = _data(64)
    out = tsx.Random3DRotationTransform()({"x": x.clone()})["x"]
    d0 = torch.cdist(x, x)
    d1 = torch.cdist(out, out)
    assert float((d0 - d1).abs().max()) < 1e-4                    # rigid rotation
    assert float((out - x).abs().max()) > 1e-3                    # actually rotated


def test_com_transform_adds_rigid_translation():
    x = _data(64)
    xs = tsx.StandardizeTransform(ts_systems.compute_std(x))({"x": x.clone()})["x"]
    out = tsx.CenterOfMassTransform()({"x": xs.clone()})["x"]
    offset = out - xs
    # same offset for every atom within a sample (a rigid translation)
    assert float((offset - offset.mean(dim=1, keepdim=True)).abs().max()) < 1e-5
    assert float(out.mean(dim=1).abs().max()) > 1e-3             # COM moved off zero


def test_loader_yields_normalized_meanfree_batches():
    x = _data(2000)
    std = ts_systems.compute_std(x)
    loader = ts_data.build_train_loader(x, std, bs=128, com_augmentation=False, num_workers=0)
    b = next(iter(loader))["x"]
    assert b.shape == (128, x.shape[1], 3) and b.dtype == torch.float32
    assert float(b.mean(dim=1).abs().max()) < 1e-4               # mean-free
    assert abs(float(b.std()) - 1.0) < 0.1                       # normalized


def test_com_energy_adjustment_matches_formula():
    import math
    import scipy.special

    torch.manual_seed(0)
    x = torch.randn(5, 22, 3) * 0.2
    got = tsx.SourceEnergy.com_energy_adjustment(x)
    com_std = 1.0 / math.sqrt(22)
    com_norms = x.mean(dim=1).norm(dim=-1)
    expect = -(com_norms**2) / (2 * com_std**2) + torch.log(
        com_norms**2 / (math.sqrt(2) * com_std**3 * scipy.special.gamma(1.5))
    )
    assert torch.allclose(got, expect, atol=1e-5)
