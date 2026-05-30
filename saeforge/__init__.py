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
from saeforge.isf import (
    Recipe,
    best_auc_per_label,
    capability_pareto,
    ensemble_route,
    recipe_auc_matrix,
    salience_headroom,
)
from saeforge.model import NativeModel
from saeforge.polygram_diagnostics import (
    compute_redundancy_ratio,
    load_polygram_report,
    resolve_encoding_capacity,
)
from saeforge.projector import SubspaceProjector
from saeforge.sweep import ParetoFrontierRow, sweep_pareto
from saeforge.sweep_capability import sweep_pareto_capability
from saeforge.sweep_capability_progressive import (
    ConvergenceTrajectoryEntry,
    ProgressiveHistory,
    ProgressiveRecommendation,
    ProgressiveStageResult,
    sweep_pareto_capability_progressive,
)
from saeforge.training.concept_anchor import (
    LABEL_SOURCE_REGISTRY,
    LabelSource,
    register_label_source,
)
from saeforge.training.heads import focal_bce_loss
from saeforge.world_model import WorldModel

__version__ = "0.10.0"

__all__ = [
    "ANOMALOUS_TOKEN_IDS",
    "ConvergenceTrajectoryEntry",
    "FeatureBasis",
    "ForgeFailed",
    "ForgePipeline",
    "ForgeResult",
    "LABEL_SOURCE_REGISTRY",
    "LabelSource",
    "NativeModel",
    "ParetoFrontierRow",
    "ProgressiveHistory",
    "ProgressiveRecommendation",
    "ProgressiveStageResult",
    "QualityThresholds",
    "QualityTier",
    "Recipe",
    "RegrowController",
    "SubspaceProjector",
    "WorldModel",
    "__version__",
    "best_auc_per_label",
    "capability_pareto",
    "compute_forged_logit_std",
    "compute_host_logit_std",
    "compute_redundancy_ratio",
    "ensemble_route",
    "focal_bce_loss",
    "load_calibration_corpus",
    "load_host_unembed",
    "load_polygram_report",
    "recipe_auc_matrix",
    "register_label_source",
    "resolve_encoding_capacity",
    "salience_headroom",
    "sweep_pareto",
    "sweep_pareto_capability",
    "sweep_pareto_capability_progressive",
    "top1_is_anomalous",
]
