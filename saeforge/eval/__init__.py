"""Faithfulness eval helpers — KL / cosine / GT-alignment scoring against a host model."""

from saeforge.eval.circuit_faithfulness import (
    assertion_cov95,
    circuit_kl,
    in_context_repeat,
    induction_predictable,
)
from saeforge.eval.faithfulness import FaithfulnessTarget, faithfulness_kl
from saeforge.eval.targets import CosineTarget, GroundTruthTarget, KLTarget

__all__ = [
    "CosineTarget",
    "FaithfulnessTarget",
    "GroundTruthTarget",
    "KLTarget",
    "assertion_cov95",
    "circuit_kl",
    "faithfulness_kl",
    "in_context_repeat",
    "induction_predictable",
]
