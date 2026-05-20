"""Concept-anchoring heads for ``run_finetune``.

Transposed from econ-sae's Phase 6.2 dual-head + focal-loss recipe, which
lifted regime-tier mAUC from 0.595 to 0.991 in one training run.

- :class:`PooledConceptHead` reads ``mean_pool(residual, dim=T)`` and
  predicts all concepts. Catches *distributed* concept encodings.
- :class:`PerChannelConceptHead` reads the last ``n_concepts`` residual
  dims, one designated channel per concept. Catches *localised*
  concept encodings.
- :func:`focal_bce_loss` is the focal-BCE term used by both heads.

See ``openspec/changes/archive/add-concept-anchored-finetune/`` (after
archive) or the live ``openspec/changes/add-concept-anchored-finetune/``
for the load-bearing contract.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PooledConceptHead(nn.Module):
    """Mean-pool over time, then a single linear projection to per-concept
    logits.

    Input shape: ``(B, T, d_model)``. Output shape: ``(B, n_concepts)``.

    The pooled head catches concepts whose encoding is *distributed*
    across many residual-stream dims — the linear layer can recombine
    arbitrary directions in ``R^{d_model}`` into a per-concept score.
    """

    def __init__(self, d_model: int, n_concepts: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_concepts)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # residual: (B, T, d_model). Mean-pool over the time axis.
        pooled = residual.mean(dim=1)  # (B, d_model)
        return self.linear(pooled)


class PerChannelConceptHead(nn.Module):
    """Per-concept affine readout from a single designated residual dim.

    The trainer slices the *last* ``n_concepts`` residual dims and feeds
    them through this head. Each concept gets one ``(scale, bias)``
    parameter pair, producing a per-token, per-concept logit.

    Input shape: ``(B, T, n_concepts)`` (the sliced last-N dims).
    Output shape: ``(B, T, n_concepts)`` (per-concept logits).

    Why the last N dims: matches econ-sae Phase 5.2's recipe
    ("reserve the last 6 dimensions of h1 as direct regime channels").
    Deterministic, simple, no special configuration.
    """

    def __init__(self, n_concepts: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_concepts))
        self.bias = nn.Parameter(torch.zeros(n_concepts))

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        # channels: (..., n_concepts). Broadcast scale + bias across all
        # leading dims; n_concepts is the trailing axis.
        return channels * self.scale + self.bias


def focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal binary cross-entropy with logits.

    For each (logit, label) pair the per-element loss is

        BCE_with_logits(logit, label) * (1 - p_t) ** gamma

    where ``p_t = sigmoid(logit)`` if ``label == 1`` else ``1 - sigmoid(logit)``.

    Args:
        logits: raw logits of any shape.
        labels: target binary labels (0.0 or 1.0), same shape as ``logits``.
        gamma: focal-loss exponent. ``gamma=0.0`` reduces exactly to
            ``F.binary_cross_entropy_with_logits``. Phase 6.2 used ``2.0``.
        reduction: one of ``"none"``, ``"sum"``, ``"mean"`` (default).

    Returns:
        Scalar tensor (or unreduced loss tensor when ``reduction='none'``).
    """
    if gamma < 0:
        raise ValueError(f"focal_bce_loss: gamma must be >= 0; got {gamma}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            f"focal_bce_loss: reduction must be one of "
            f"'none' / 'sum' / 'mean'; got {reduction!r}"
        )

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none",
    )

    if gamma == 0.0:
        focal_weight: Any = 1.0
    else:
        # p_t = sigmoid(logits) when label==1; 1 - sigmoid(logits) when label==0.
        # Equivalent: p_t = label * p + (1 - label) * (1 - p).
        p = torch.sigmoid(logits)
        p_t = labels * p + (1.0 - labels) * (1.0 - p)
        focal_weight = (1.0 - p_t).pow(gamma)

    loss = focal_weight * bce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
