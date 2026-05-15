"""sae-forge — turn a Polygram-compressed SAE into a small, semantically-native transformer."""

from saeforge.basis import FeatureBasis
from saeforge.forge import ForgeFailed, ForgePipeline, ForgeResult
from saeforge.model import NativeModel
from saeforge.projector import SubspaceProjector
from saeforge.sweep import ParetoFrontierRow, sweep_pareto

__version__ = "0.3.0"

__all__ = [
    "FeatureBasis",
    "ForgeFailed",
    "ForgePipeline",
    "ForgeResult",
    "NativeModel",
    "ParetoFrontierRow",
    "SubspaceProjector",
    "__version__",
    "sweep_pareto",
]
