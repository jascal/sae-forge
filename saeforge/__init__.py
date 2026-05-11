"""sae-forge — turn a Polygram-compressed SAE into a small, semantically-native transformer."""

from saeforge.basis import FeatureBasis, RegrowController
from saeforge.forge import ForgeFailed, ForgePipeline, ForgeResult
from saeforge.model import NativeModel
from saeforge.projector import SubspaceProjector

__version__ = "0.3.0"

__all__ = [
    "FeatureBasis",
    "ForgeFailed",
    "ForgePipeline",
    "ForgeResult",
    "NativeModel",
    "RegrowController",
    "SubspaceProjector",
    "__version__",
]
