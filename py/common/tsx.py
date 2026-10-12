"""Leaf APIs from the installed transferable-samplers package.

The runtime dependency is pinned in requirements.txt. Importing leaf modules
avoids the unused Lightning training/datamodule and torchvision dependencies.
"""
# Dataloader
from transferable_samplers.data.datasets.tensor_dataset import TensorDataset  # noqa: E402
from transferable_samplers.data.transforms.standardize import StandardizeTransform  # noqa: E402
from transferable_samplers.data.transforms.rotation import Random3DRotationTransform  # noqa: E402
from transferable_samplers.data.transforms.center_of_mass import CenterOfMassTransform  # noqa: E402

# Evaluation
from transferable_samplers.evaluation.evaluator import PeptideEnsembleEvaluator  # noqa: E402
from transferable_samplers.samplers.snis_sampler import SNISSampler  # noqa: E402
from transferable_samplers.utils.dataclasses import (  # noqa: E402
    SamplesData,
    SourceEnergy,
    TargetEnergy,
    EvalContext,
)
from transferable_samplers.evaluation.metrics.ess import ess, normalized_ess  # noqa: E402
from transferable_samplers.data.preprocessing.tica import get_tica_model  # noqa: E402


class Compose:
    """Minimal replacement for torchvision.transforms.Compose (dict transforms)."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


__all__ = [
    "TensorDataset",
    "StandardizeTransform",
    "Random3DRotationTransform",
    "CenterOfMassTransform",
    "Compose",
    "PeptideEnsembleEvaluator",
    "SNISSampler",
    "SamplesData",
    "SourceEnergy",
    "TargetEnergy",
    "EvalContext",
    "ess",
    "normalized_ess",
    "get_tica_model",
]
