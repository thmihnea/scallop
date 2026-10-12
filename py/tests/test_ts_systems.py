"""Tests for the iF2D2 <-> transferable-samplers system registry.

Structural checks run anywhere. Data-backed checks need the ts splits cached under
``datasets/ts`` (they skip otherwise, no network access in tests). Energy checks
additionally need a working OpenMM (skipped on hosts where it cannot import).
"""
import importlib
import os
import sys

import numpy as np
import pytest

pytest.importorskip("transferable_samplers")
pytest.importorskip("mdtraj")

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import ts_systems  # noqa: E402


def _have_data(system) -> bool:
    paths = ts_systems.data_paths(system)
    return all(os.path.exists(paths[k]) for k in ("train", "val", "test", "pdb"))


def _have_openmm() -> bool:
    try:
        import openmm  # noqa: F401

        return True
    except Exception:
        return False


ALL_SYSTEMS = sorted(ts_systems.TS_SYSTEMS)


def test_registry_covers_the_molecular_configs():
    """Every molecular config in py/configs maps to a ts system."""
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    names = {n[:-3] for n in os.listdir(cfg_dir) if n.endswith(".py") and not n.startswith("_")}
    targets = {importlib.import_module(f"configs.{name}").get_config(0).problem.target for name in names}
    assert targets <= set(ts_systems.TS_SYSTEMS), targets - set(ts_systems.TS_SYSTEMS)


@pytest.mark.parametrize("target", ALL_SYSTEMS)
def test_trajectory_naming_matches_ts(target):
    """ts names trajectories f"{sequence}_{int(temperature)}K"."""
    s = ts_systems.TS_SYSTEMS[target]
    assert s.trajectory_name == f"{s.sequence}_{int(s.temperature)}K"
    assert ts_systems.resolve(target) is s


def test_resolve_accepts_cfg_and_rejects_unknown():
    import types

    cfg = types.SimpleNamespace(problem=types.SimpleNamespace(target="ala6_implicit"))
    assert ts_systems.resolve(cfg).sequence == "AAAAAA"
    with pytest.raises(KeyError):
        ts_systems.resolve("not_a_system")


@pytest.mark.parametrize("target", ALL_SYSTEMS)
def test_splits_shapes_and_disjointness(target):
    s = ts_systems.TS_SYSTEMS[target]
    if not _have_data(s):
        pytest.skip(f"ts data not cached for {s.trajectory_name}")
    train = ts_systems.load_split(s, "train")
    val = ts_systems.load_split(s, "val")
    for x in (train, val):
        assert x.ndim == 3 and x.shape[1] == s.num_atoms and x.shape[2] == 3
    # held-out: no val frame appears in train
    tr_keys = {r.tobytes() for r in train[:5000].reshape(5000, -1).numpy()}
    va_keys = {r.tobytes() for r in val[:200].reshape(200, -1).numpy()}
    assert not (tr_keys & va_keys)


@pytest.mark.parametrize("target", ALL_SYSTEMS)
def test_train_std_matches_ts_convention(target):
    """ts std = std of per-frame-centred train coords."""
    s = ts_systems.TS_SYSTEMS[target]
    if not _have_data(s):
        pytest.skip(f"ts data not cached for {s.trajectory_name}")
    train = ts_systems.load_split(s, "train")
    expected = float((train - train.mean(dim=1, keepdim=True)).std())
    assert ts_systems.train_std(s) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("target", ALL_SYSTEMS)
def test_topology_atom_count(target):
    s = ts_systems.TS_SYSTEMS[target]
    if not _have_data(s):
        pytest.skip(f"ts data not cached for {s.trajectory_name}")
    assert ts_systems.topology(s).n_atoms == s.num_atoms


@pytest.mark.parametrize("target", ALL_SYSTEMS)
def test_reference_energies_are_physical(target):
    """Reference conformations must have finite, non-explosive energies."""
    s = ts_systems.TS_SYSTEMS[target]
    if not _have_data(s):
        pytest.skip(f"ts data not cached for {s.trajectory_name}")
    if not _have_openmm():
        pytest.skip("openmm unavailable")
    pot = ts_systems.build_potential(s)
    e = pot(ts_systems.load_split(s, "val")[:32]).numpy()
    assert np.isfinite(e).all()
    assert np.abs(e).max() < 1e4, e  # kT, MD frames sit in the hundreds at most


def test_parallel_energy_matches_serial():
    s = ts_systems.TS_SYSTEMS["ala6_implicit"]
    if not _have_data(s):
        pytest.skip("ts data not cached")
    if not _have_openmm():
        pytest.skip("openmm unavailable")
    val = ts_systems.load_split(s, "val")[:64]
    serial = ts_systems.build_potential(s)(val).numpy()
    par = ts_systems.ParallelEnergy(s, n_workers=4, chunk=16)
    try:
        parallel = par.numpy(val)
    finally:
        par.close()
    assert np.allclose(serial, parallel, rtol=1e-5, atol=1e-3)


def test_build_eval_context_follows_prepare_eval():
    s = ts_systems.TS_SYSTEMS["ala6_implicit"]
    if not _have_data(s):
        pytest.skip("ts data not cached")
    if not _have_openmm():
        pytest.skip("openmm unavailable")
    n = 200
    ctx = ts_systems.build_eval_context(s, split="val", num_eval_samples=n, num_tica=2000)
    assert len(ctx.true_data) == n
    assert ctx.topology.n_atoms == s.num_atoms
    assert float(ctx.normalization_std) == 1.0
    # strided subsample of the full split, exactly as ts does
    full = ts_systems.load_split(s, "val")
    expected = full[:: max(1, len(full) // n)][:n]
    assert torch.equal(ctx.true_data.samples, expected)


@pytest.mark.parametrize("target", ALL_SYSTEMS)
def test_generic_loader_batches(target):
    from common import ts_data

    s = ts_systems.TS_SYSTEMS[target]
    if not _have_data(s):
        pytest.skip(f"ts data not cached for {s.trajectory_name}")
    it, std = ts_data.load_peptide(
        target, bs=4, device=torch.device("cpu"), num_workers=0, seed=0
    )
    batch = next(it)
    assert batch.shape == (4, s.num_atoms, 3)
    assert std == pytest.approx(ts_systems.train_std(s), rel=1e-6)

