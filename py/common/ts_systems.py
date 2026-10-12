"""Registry, official TS data cache, OpenMMEnergy and evaluation contexts.

The installed transferable-samplers package supplies transforms, energies, TICA,
metrics and plots. This adapter follows its SinglePeptideDataModule conventions
without importing the unused Lightning training framework.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .tsx import EvalContext, SamplesData, TargetEnergy, get_tica_model

HF_REPO_ID = "transferable-samplers/sequential-boltzmann-generators-data"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "datasets", "ts")

# Sequences ts simulates in explicit-ish Amber99 rather than amber14 implicit solvent.
_ACE_SEQUENCES = ("Ace-A-Nme", "Ace-AAA-Nme")


@dataclass(frozen=True)
class TSSystem:
    """A ts single-peptide system, keyed by the iF2D2 ``cfg.problem.target``."""

    target: str
    sequence: str
    temperature: float
    num_atoms: int

    @property
    def trajectory_name(self) -> str:
        return f"{self.sequence}_{int(self.temperature)}K"


TS_SYSTEMS = {
    s.target: s
    for s in (
        TSSystem("ace_a_nme", "Ace-A-Nme", 300.0, 22),
        TSSystem("ace_aaa_nme", "Ace-AAA-Nme", 300.0, 42),
        TSSystem("ala3_implicit", "AAA", 310.0, 33),
        TSSystem("ala6_implicit", "AAAAAA", 310.0, 63),
        TSSystem("chignolin", "GYDPETGTWG", 310.0, 138),
    )
}


def resolve(target_or_cfg) -> TSSystem:
    """Accept an iF2D2 config, a ``cfg.problem``, or a target string."""
    target = target_or_cfg
    if not isinstance(target, str):
        problem = getattr(target_or_cfg, "problem", target_or_cfg)
        target = getattr(problem, "target")
    if target not in TS_SYSTEMS:
        raise KeyError(
            f"target {target!r} has no transferable-samplers system; known: "
            f"{sorted(TS_SYSTEMS)}"
        )
    return TS_SYSTEMS[target]


def data_paths(system: TSSystem, data_dir: Optional[str] = None) -> dict:
    """Paths in the ts cache layout (``{data_dir}/{repo}/{traj}/{traj}_{split}.npy``)."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    base = os.path.join(data_dir, HF_REPO_ID.split("/")[-1], system.trajectory_name)
    traj = system.trajectory_name
    return {
        "train": os.path.join(base, f"{traj}_train.npy"),
        "val": os.path.join(base, f"{traj}_val.npy"),
        "test": os.path.join(base, f"{traj}_test.npy"),
        "pdb": os.path.join(base, f"{traj}.pdb"),
        "base": base,
    }


def ensure_data(system: TSSystem, data_dir: Optional[str] = None) -> dict:
    """Use the official TS train/val/test/PDB cache; download missing files.

    A missing held-out split is an error. Training frames are never silently
    substituted for validation or test data.
    """
    from huggingface_hub import snapshot_download

    data_dir = data_dir or DEFAULT_DATA_DIR
    paths = data_paths(system, data_dir)
    required = ("train", "val", "test", "pdb")
    if all(os.path.exists(paths[k]) for k in required):
        return {**paths, "source": "cache"}
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{system.trajectory_name}/*"],
        local_dir=os.path.join(data_dir, HF_REPO_ID.split("/")[-1]),
    )
    missing = [paths[k] for k in required if not os.path.exists(paths[k])]
    if missing:
        raise FileNotFoundError(f"Incomplete TS dataset for {system.sequence}: {missing}")
    return {**paths, "source": "huggingface"}


def load_split(system: TSSystem, split: str, data_dir: Optional[str] = None) -> torch.Tensor:
    paths = ensure_data(system, data_dir)
    path = paths[split]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split} split missing for {system.trajectory_name}: {path}")
    return torch.from_numpy(np.load(path)).float()


def compute_std(train: torch.Tensor) -> float:
    """Scalar std of per-frame mean-free coords (matches ts SinglePeptideDataModule)."""
    return float((train - train.mean(dim=1, keepdim=True)).std())


def train_std(system: TSSystem, data_dir: Optional[str] = None) -> float:
    """ts ``SinglePeptideDataModule.std``: std of per-frame-centred train coords."""
    return compute_std(load_split(system, "train", data_dir))


def topology(system: TSSystem, data_dir: Optional[str] = None):
    import mdtraj as md

    return md.load_topology(ensure_data(system, data_dir)["pdb"])


@contextlib.contextmanager
def _openmm_threads(threads: Optional[int]):
    """Pin the CPU thread count ts's ``OpenMMEnergy`` will choose.

    ts sizes its pool as ``os.cpu_count() // LOCAL_WORLD_SIZE``, assuming one rank per
    GPU; a single process on a 256-core host therefore asks for 256 threads. These
    peptides are far too small to use that: measured on AAAAAA, thread count makes
    essentially no difference when cores are free (~1900-2100 conf/s for 1/4/16/256),
    but under heavy core contention the 256-thread context collapsed to 69.7 conf/s
    against 1702 conf/s at one thread. Pinning to a small count is therefore a
    robustness guard against oversubscription, NOT a general speedup. Energies are
    unchanged up to CPU reduction order. ``threads=None`` restores their default.
    """
    if threads is None:
        yield
        return
    prev = os.environ.get("LOCAL_WORLD_SIZE")
    os.environ["LOCAL_WORLD_SIZE"] = str(max(1, (os.cpu_count() or 1) // max(1, threads)))
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("LOCAL_WORLD_SIZE", None)
        else:
            os.environ["LOCAL_WORLD_SIZE"] = prev


def build_potential(
    system: TSSystem,
    data_dir: Optional[str] = None,
    platform_name: str = "CPU",
    device_index: Optional[int] = None,
    threads: Optional[int] = 4,
):
    """ts ``OpenMMEnergy`` built exactly as ``_setup_potential`` does.

    Returns unclamped energies in kT (TS's ``1 / (T * MOLAR_GAS_CONSTANT_R)`` convention).

    ``threads`` overrides ts's CPU thread heuristic (see :func:`_openmm_threads`); pass
    ``None`` for their default.
    """
    import openmm
    import openmm.app
    import openmm.unit

    from transferable_samplers.data.energy.openmm_energy import OpenMMEnergy

    paths = ensure_data(system, data_dir)
    pdb = openmm.app.PDBFile(paths["pdb"])

    if system.sequence in _ACE_SEQUENCES:
        forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")
        omm_system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=openmm.app.NoCutoff,
            nonbondedCutoff=0.9 * openmm.unit.nanometer,
            constraints=None,
        )
        temperature = 300
        friction = 0.3 if system.sequence == "Ace-AAA-Nme" else 1.0
    else:
        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        omm_system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=openmm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        temperature = 310
        friction = 0.3

    integrator = openmm.LangevinMiddleIntegrator(
        temperature * openmm.unit.kelvin,
        friction / openmm.unit.picosecond,
        1.0 * openmm.unit.femtosecond,
    )
    with _openmm_threads(threads if platform_name == "CPU" else None):
        return OpenMMEnergy(
            omm_system, integrator, platform_name=platform_name, device_index=device_index
        )


# --------------------------------------------------------------------------------------
# Parallel energy evaluation
# --------------------------------------------------------------------------------------

_WORKER_POT = None


def _init_energy_worker(system: TSSystem, data_dir: Optional[str]):
    global _WORKER_POT
    # One thread per worker: the parallelism comes from the process pool, not from
    # OpenMM's own threads (which would oversubscribe the box n_workers-fold).
    _WORKER_POT = build_potential(system, data_dir, platform_name="CPU", threads=1)


def _energy_worker(coords: np.ndarray, with_grad=False):
    """TS energies/forces; isolate invalid proposals so MALA can reject them."""
    from openmm import OpenMMException

    try:
        e, forces = _WORKER_POT._compute(torch.from_numpy(coords))
    except (ValueError, OpenMMException):
        e = np.full(len(coords), np.inf)
        forces = np.full_like(coords, np.nan)
        for i, frame in enumerate(coords):
            try:
                ei, fi = _WORKER_POT._compute(torch.from_numpy(frame[None]))
                e[i], forces[i] = ei[0], fi[0]
            except (ValueError, OpenMMException):
                pass
    return (e, -forces) if with_grad else e


class ParallelEnergy:
    """Process pool of ts ``OpenMMEnergy`` evaluators; drop-in for their ``energy_fn``.

    ts's evaluator is serial over frames, which is far too slow for a step sweep
    (~12 conformations/s for AAAAAA). This forks ``n_workers`` single-threaded copies of
    *their* evaluator, so the energy call is theirs while the throughput is usable.

    Construct BEFORE CUDA is initialised in the parent process (fork safety); the
    workers only ever touch OpenMM on CPU.
    """

    def __init__(self, system: TSSystem, n_workers: int = 64, data_dir: Optional[str] = None,
                 chunk: int = 250):
        self.system = system
        self.chunk = chunk
        self._pool = mp.Pool(
            n_workers, initializer=_init_energy_worker, initargs=(system, data_dir)
        )

    def numpy(self, coords) -> np.ndarray:
        """Energies in kT for ``(B, n_atoms, 3)`` positions in nm."""
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, self.system.num_atoms, 3)
        parts = self._pool.map(
            _energy_worker, [coords[i : i + self.chunk] for i in range(0, len(coords), self.chunk)]
        )
        return np.concatenate(parts) if parts else np.empty(0)

    def energy_and_grad(self, positions: torch.Tensor):
        """Return E/kT and d(E/kT)/dx in physical nm, using TS OpenMM forces.

        This is a detached evaluation, not an autograd-enabled energy function.
        """
        coords = positions.detach().cpu().numpy().astype(np.float64)
        parts = self._pool.starmap(
            _energy_worker,
            [(coords[i:i + self.chunk], True) for i in range(0, len(coords), self.chunk)],
        )
        e, grad = zip(*parts)
        return (torch.as_tensor(np.concatenate(e), device=positions.device, dtype=positions.dtype),
                torch.as_tensor(np.concatenate(grad), device=positions.device, dtype=positions.dtype))

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.numpy(positions)).float()

    def close(self):
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------------------
# Evaluation context
# --------------------------------------------------------------------------------------


def build_eval_context(
    system: TSSystem,
    split: str = "val",
    energy_fn=None,
    num_eval_samples: int = 10_000,
    data_dir: Optional[str] = None,
    normalization_std: float = 1.0,
    num_tica: Optional[int] = None,
    reference_samples: Optional[torch.Tensor] = None,
) -> EvalContext:
    """ts ``prepare_eval(sequence, stage)`` for an iF2D2 system.

    ``normalization_std`` is what the evaluator multiplies *generated* samples by
    (``destandardize_coords``). Pass 1.0 when handing over physical-nm samples, or the
    train std when handing over standardized ones. Reference frames are always physical.
    ``reference_samples`` may reuse the loader's held-out split in physical nm;
    otherwise the requested official split is loaded from the TS cache.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    if num_eval_samples < 1:
        raise ValueError("num_eval_samples must be positive")
    if num_tica is not None and num_tica < 1:
        raise ValueError("num_tica must be positive")
    if not np.isfinite(normalization_std) or normalization_std <= 0:
        raise ValueError("normalization_std must be finite and positive")
    true_all = load_split(system, split, data_dir) if reference_samples is None else reference_samples
    if true_all.ndim != 3 or tuple(true_all.shape[1:]) != (system.num_atoms, 3) or len(true_all) == 0:
        raise ValueError(f"Invalid {split} reference shape: {tuple(true_all.shape)}")
    topo = topology(system, data_dir)

    # ts fits TICA on the full split (time-ordered) BEFORE subsampling.
    tica_src = true_all if num_tica is None else true_all[:num_tica]
    tica_model = get_tica_model(tica_src.numpy(), topo)

    stride = max(1, len(true_all) // num_eval_samples)
    true = true_all[::stride][:num_eval_samples]

    owns_energy = energy_fn is None
    if owns_energy:
        energy_fn = build_potential(system, data_dir)
    true_E = energy_fn(true)
    if isinstance(true_E, np.ndarray):
        true_E = torch.from_numpy(true_E).float()
    true_E = true_E.reshape(-1).float()

    std_t = torch.tensor(float(normalization_std))
    return EvalContext(
        true_data=SamplesData(true, true_E),
        target_energy=TargetEnergy(energy_fn=energy_fn, normalization_std=std_t),
        normalization_std=std_t,
        system_cond=None,
        tica_model=tica_model,
        topology=topo,
    )
