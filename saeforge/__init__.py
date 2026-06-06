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
    headroom_lift_analysis,
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
from saeforge.world_model import WorldModel

__version__ = "0.13.0"

__all__ = [
    "ANOMALOUS_TOKEN_IDS",
    "ConvergenceTrajectoryEntry",
    "FeatureBasis",
    "ForgeFailed",
    "ForgePipeline",
    "ForgeResult",
    "ForgedMoE",
    "ForgedMoEConfig",
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
    "forge_to_moe",
    "headroom_lift_analysis",
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


def __getattr__(name: str):
    # focal_bce_loss lives in saeforge.training.heads, which imports torch at
    # module load. Keep it out of the eager import graph (PEP 562) so a bare
    # `pip install sae-forge` — numpy / scipy / safetensors only, no torch —
    # can still `import saeforge` and run the CLI. The symbol resolves lazily
    # on first access, by which point the caller has opted into the [torch]
    # extra. heads.py is the only module in the package that forces torch at
    # import time, so this single lazy hop keeps the whole public surface
    # importable torch-free.
    if name == "focal_bce_loss":
        from saeforge.training.heads import focal_bce_loss

        return focal_bce_loss
    # sae-moe-forge: moe.py imports torch at load (ForgedMoE IS an
    # nn.Module), so keep it off the eager graph for the same reason as
    # focal_bce_loss — a torch-free `import saeforge` must stay cheap.
    if name in ("ForgedMoE", "ForgedMoEConfig", "forge_to_moe"):
        import saeforge.moe as _moe

        return getattr(_moe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
