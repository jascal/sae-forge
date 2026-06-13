"""saeforge.training — fine-tune recipe (cosine LR + warmup, gradient
clipping, optional grad checkpointing, optional mixed precision, periodic
eval / save, structured loss tracking).
"""

from saeforge.training.config import TrainingConfig, TrainingResult
from saeforge.training.corpus import build_iterator, take
from saeforge.training.encoder import EncoderCalibrationReport, train_encoder
from saeforge.training.loop import run_finetune
from saeforge.training.replay import MixedIterator, ReplayBuffer
from saeforge.training.schedules import cosine_with_warmup
from saeforge.training.task_stream import (
    LabeledTaskStream,
    LossDriftTaskStream,
    TaskStream,
    TokenBudgetTaskStream,
)

__all__ = [
    "EncoderCalibrationReport",
    "LabeledTaskStream",
    "LossDriftTaskStream",
    "MixedIterator",
    "ReplayBuffer",
    "TaskStream",
    "TokenBudgetTaskStream",
    "TrainingConfig",
    "TrainingResult",
    "build_iterator",
    "cosine_with_warmup",
    "run_finetune",
    "take",
    "train_encoder",
]
