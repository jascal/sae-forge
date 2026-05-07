"""Learning-rate schedules. Pure-numpy, no torch dep — separately unit-testable."""

from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine LR schedule with linear warmup.

    Properties:
    - ``step=0`` → ``peak_lr / warmup_steps`` (small but positive)
    - ``step=warmup_steps`` → ``peak_lr`` (peak)
    - ``step=total_steps`` → ``peak_lr * min_lr_ratio`` (floor)
    - ``step > total_steps`` → ``peak_lr * min_lr_ratio`` (clamped)
    - Monotonically non-increasing for ``step >= warmup_steps``
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress_denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / progress_denom
    progress = max(0.0, min(progress, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
