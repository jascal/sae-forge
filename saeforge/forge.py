"""ForgePipeline — orchestrate basis load -> projection -> native model -> faithfulness eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from saeforge.basis import FeatureBasis
from saeforge.model import NativeModel
from saeforge.projector import SubspaceProjector


@dataclass
class ForgeResult:
    """Structured output of a ``ForgePipeline.run`` call."""

    model: NativeModel | None
    output_dir: Path
    n_params: int = 0
    faithfulness_kl: float | None = None
    extras: dict = field(default_factory=dict)


@dataclass
class ForgePipeline:
    """End-to-end forging pipeline.

    Stages, in order:

    1. ``stage_load_basis`` — load the Polygram-compressed checkpoint.
    2. ``stage_project`` — project host weights through ``projector``.
    3. ``stage_assemble`` — build the ``NativeModel`` from projected weights.
    4. ``stage_finetune`` — optional fine-tune against the host's outputs
       on ``finetune_prompts``. Skipped when the list is empty.
    5. ``stage_eval`` — measure faithfulness KL on ``eval_prompts``.

    Each stage is callable on its own so callers can stop after any step
    (e.g. inspect the projected weights without building a torch model).
    """

    basis: FeatureBasis
    projector: SubspaceProjector
    model: NativeModel | None = None
    host_model_id: str | None = None
    eval_prompts: list[str] = field(default_factory=list)
    finetune_prompts: list[str] = field(default_factory=list)
    dtype: str = "float32"
    device: str = "cpu"

    def run(self, output_dir: str | Path) -> ForgeResult:
        """Run every stage and write a structured artifact tree to ``output_dir``."""
        raise NotImplementedError(
            "ForgePipeline.run is the change-5 deliverable; "
            "see openspec/changes/forge-pipeline/proposal.md."
        )
