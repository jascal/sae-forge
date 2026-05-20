"""TrainingConfig and TrainingResult dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    """Knobs for `run_finetune`. Defaults target a Gemma-2-2B forge on 24GB GPU.

    For smaller hosts (GPT-2-small) raise ``peak_lr`` to ~1e-4 and disable
    ``gradient_checkpointing``.
    """

    total_steps: int = 1000
    warmup_steps: int = 100
    peak_lr: float = 5e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    batch_size: int = 8
    sequence_length: int = 512
    precision: str = "fp32"
    gradient_checkpointing: bool = False

    eval_every_steps: int = 100
    eval_input_ids: Any = None

    save_every_steps: int = 250
    save_dir: Path | None = None

    log_every_steps: int = 10

    # Host-distillation knobs (add-host-distillation-finetune-loss).
    # With distill_alpha=1.0 (default), the per-step loss is pure
    # corpus cross-entropy and the host forward is skipped entirely —
    # byte-identical to the pre-change training loop. With
    # distill_alpha < 1.0, run_finetune additionally runs a host
    # forward under no_grad on the same batch and the loss becomes
    # alpha * CE(corpus) + (1-alpha) * tau**2 * KL(host || forged).
    # Requires a non-None `host` argument at run_finetune call time.
    distill_alpha: float = 1.0
    distill_temperature: float = 2.0

    # Concept-anchoring knobs (add-concept-anchored-finetune).
    # With concept_alpha=0.0 (default), the entire concept-anchoring
    # branch is skipped — no label source is instantiated, no heads
    # constructed, no extra forward. Byte-identical to the pre-change
    # loop (and to distill_alpha alone). With concept_alpha > 0,
    # the total loss is composed as
    #   total = (1-concept_alpha)*L_existing + concept_alpha*L_concept
    # where L_existing is whatever the existing CE / CE+KL loss is.
    # L_concept is the dual-head focal-BCE sum from
    # saeforge/training/heads.py + concept_anchor.py.
    concept_alpha: float = 0.0
    concept_pool_weight: float = 1.0
    concept_channel_weight: float = 1.0
    concept_focal_gamma: float = 2.0
    concept_label_source: str = "polygram-clusters"
    concept_label_source_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(
                f"precision must be one of fp32 / bf16 / fp16; got {self.precision!r}"
            )
        if self.warmup_steps < 0 or self.total_steps < 1:
            raise ValueError("warmup_steps must be >=0 and total_steps must be >=1")
        if not (0.0 <= self.distill_alpha <= 1.0):
            raise ValueError(
                f"distill_alpha must lie in [0.0, 1.0]; got {self.distill_alpha}"
            )
        if self.distill_temperature <= 0:
            raise ValueError(
                f"distill_temperature must be > 0; got {self.distill_temperature}"
            )
        # Concept-anchoring validation. Defer the registry-key check until
        # concept_alpha > 0 so the default config doesn't pull in the
        # registry module at construction time (keeps imports cheap for
        # callers who never touch concept anchoring).
        if not (0.0 <= self.concept_alpha <= 1.0):
            raise ValueError(
                f"concept_alpha must lie in [0.0, 1.0]; got {self.concept_alpha}"
            )
        if self.concept_pool_weight < 0:
            raise ValueError(
                f"concept_pool_weight must be >= 0; got {self.concept_pool_weight}"
            )
        if self.concept_channel_weight < 0:
            raise ValueError(
                f"concept_channel_weight must be >= 0; got {self.concept_channel_weight}"
            )
        if self.concept_focal_gamma < 0:
            raise ValueError(
                f"concept_focal_gamma must be >= 0; got {self.concept_focal_gamma}"
            )
        if self.concept_alpha > 0:
            if self.concept_pool_weight == 0 and self.concept_channel_weight == 0:
                raise ValueError(
                    "concept_alpha > 0 requires at least one of "
                    "concept_pool_weight or concept_channel_weight to be > 0; "
                    f"got pool={self.concept_pool_weight}, "
                    f"channel={self.concept_channel_weight}"
                )
            from saeforge.training.concept_anchor import LABEL_SOURCE_REGISTRY

            if self.concept_label_source not in LABEL_SOURCE_REGISTRY:
                raise ValueError(
                    f"concept_label_source={self.concept_label_source!r} is "
                    f"not registered. Registered backends: "
                    f"{sorted(LABEL_SOURCE_REGISTRY)}"
                )


@dataclass
class TrainingResult:
    """Structured output of `run_finetune`."""

    final_loss: float
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    eval_history: list[tuple[int, float]] = field(default_factory=list)
    wall_seconds: float = 0.0
    n_steps_completed: int = 0
    save_paths: list[Path] = field(default_factory=list)
    converged: bool = False
    metadata: dict = field(default_factory=dict)
