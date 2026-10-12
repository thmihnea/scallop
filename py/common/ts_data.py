"""transferable-samplers dataloader for iF2D2, for every supported peptide system.

Reproduces their ``SinglePeptideDataModule`` train pipeline using the upstream
transforms: ``StandardizeTransform`` (per-sample mean-free + / scalar std computed on
the train split) + ``Random3DRotationTransform`` (uniform SO(3), train-only), with an
optional ``CenterOfMassTransform`` (translation augmentation, off by default). Their
official train/val/test splits come from HuggingFace via :mod:`common.ts_systems`.

System resolution, paths, energies and the evaluation context all live in
:mod:`common.ts_systems`; this module only owns the train-side loader.
"""
from __future__ import annotations

import numpy as np
import torch

from . import ts_systems
from . import distributed
from .tsx import (
    CenterOfMassTransform,
    Compose,
    Random3DRotationTransform,
    StandardizeTransform,
    TensorDataset,
)


def build_train_loader(train: torch.Tensor, std: float, bs: int, com_augmentation: bool = False,
                       num_workers: int = 2, seed: int = 0):
    transforms = [StandardizeTransform(std), Random3DRotationTransform()]
    if com_augmentation:
        transforms.append(CenterOfMassTransform())
    ds = TensorDataset(train, transform=Compose(transforms))
    sampler = None
    if distributed.active():
        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=distributed.world_size(), rank=distributed.rank(),
            shuffle=True, seed=seed, drop_last=True,
        )
    gen = torch.Generator().manual_seed(seed + distributed.rank())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
        num_workers=num_workers, drop_last=True, generator=gen,
        **({"multiprocessing_context": "spawn", "persistent_workers": True}
           if num_workers and distributed.active() else {}),
    )
    if len(loader) == 0:
        raise ValueError("Training shard is smaller than one local batch")
    return loader


class InfiniteTSLoader:
    """Wrap a (finite) DataLoader of ``{"x": ...}`` batches into an infinite tensor stream."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.epoch = 0
        self._it = iter(loader)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        try:
            batch = next(self._it)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.loader.sampler, "set_epoch"):
                self.loader.sampler.set_epoch(self.epoch)
            self._it = iter(self.loader)
            batch = next(self._it)
        x = batch["x"] if isinstance(batch, dict) else batch
        return x.to(self.device)


def load_peptide(target_or_cfg, bs: int, device, com_augmentation: bool = False,
                 num_workers: int = 2, seed: int = 0, data_dir: str | None = None):
    """Return the training iterator and measured std, validating all split paths."""
    system = ts_systems.resolve(target_or_cfg)
    if data_dir is None and not isinstance(target_or_cfg, str):
        problem = getattr(target_or_cfg, "problem", target_or_cfg)
        data_dir = getattr(problem, "ts_data_dir", None) or None
    paths = ts_systems.ensure_data(system, data_dir)

    train = torch.from_numpy(np.load(paths["train"])).float()
    std = ts_systems.compute_std(train)
    loader = build_train_loader(train, std, bs, com_augmentation, num_workers, seed)
    infinite = InfiniteTSLoader(loader, device)

    print(
        f"[ts_data] {system.trajectory_name} source={paths['source']} train={len(train)} "
        f"std={std:.6g} com_aug={com_augmentation}"
    )
    return infinite, std
