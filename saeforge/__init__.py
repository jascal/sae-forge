"""sae-forge — turn a Polygram-compressed SAE into a small, semantically-native transformer."""

from saeforge.basis import FeatureBasis, RegrowController
from saeforge.calibration import (
    ANOMALOUS_TOKEN_IDS,
    compute_forged_logit_std,
    compute_host_logit_std,
    load_calibration_corpus,
    load_host_unembed,
    top1_is_anomalous,
)
from saeforge.forge import ForgeFailed, ForgePipeline, ForgeResult
from saeforge.forge_quality import QualityThresholds, QualityTier
from saeforge.model import NativeModel
from saeforge.polygram_diagnostics import (
    compute_redundancy_ratio,
    load_polygram_report,
    resolve_encoding_capacity,
)
from saeforge.projector import SubspaceProjector
from saeforge.sweep import ParetoFrontierRow, sweep_pareto
from saeforge.world_model import WorldModel

__version__ = "0.7.0"

__all__ = [
    "ANOMALOUS_TOKEN_IDS",
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
    "WorldModel",
    "__version__",
    "compute_forged_logit_std",
    "compute_host_logit_std",
    "compute_redundancy_ratio",
    "load_calibration_corpus",
    "load_host_unembed",
    "load_polygram_report",
    "resolve_encoding_capacity",
    "sweep_pareto",
    "top1_is_anomalous",
]
