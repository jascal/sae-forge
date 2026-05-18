"""Faithfulness eval helpers — KL / cosine / GT-alignment scoring against a host model."""

from saeforge.eval.faithfulness import FaithfulnessTarget, faithfulness_kl
from saeforge.eval.targets import CosineTarget, GroundTruthTarget, KLTarget

__all__ = [
    "CosineTarget",
    "FaithfulnessTarget",
    "GroundTruthTarget",
    "KLTarget",
    "faithfulness_kl",
]
