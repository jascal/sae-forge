"""sae-forge — turn a Polygram-compressed SAE into a small, semantically-native transformer."""

from saeforge.basis import FeatureBasis, RegrowController
from saeforge.forge import ForgeFailed, ForgePipeline, ForgeResult
from saeforge.forge_quality import QualityThresholds, QualityTier
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
    "QualityThresholds",
    "QualityTier",
    "RegrowController",
    "SubspaceProjector",
    "__version__",
    "sweep_pareto",
]
