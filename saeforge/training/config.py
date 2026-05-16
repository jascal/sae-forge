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
