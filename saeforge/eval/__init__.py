"""Faithfulness eval helpers — KL between forged and host model on held-out prompts."""

from saeforge.eval.faithfulness import FaithfulnessTarget, faithfulness_kl

__all__ = ["FaithfulnessTarget", "faithfulness_kl"]
