"""PolygramHeuristicRouter — the v1 ``sae-moe-forge`` router.

Wraps polygram 0.9.0's routing heuristic (``ExpertDictionary.route``:
sum feature activations per expert, take the top-k by summed score)
and vectorises it over leading batch dimensions in torch. The polygram
surface is per-vector numpy; this is a strict batched re-expression
with identical tie-breaking, **not** a re-implementation with different
semantics — equal-scoring experts are ordered by ascending expert index
to match polygram's ``np.argsort(..., kind="stable")``.

Zero trainable parameters: routing is a deterministic function of the
feature→expert partition. The router inherits any future polygram
heuristic change automatically when the caller re-clusters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PolygramHeuristicRouter(nn.Module):
    """Top-k expert router by summed per-expert feature activation.

    Parameters
    ----------
    feature_to_expert:
        ``(n_features,)`` integer map from feature index to expert
        index (the same partition the expert set uses).
    n_experts:
        Number of experts.
    """

    def __init__(self, feature_to_expert, n_experts: int):
        super().__init__()
        ft = np.asarray(feature_to_expert, dtype=np.int64)
        if ft.ndim != 1:
            raise ValueError(
                f"feature_to_expert must be 1-D (n_features,); got shape {ft.shape}"
            )
        self._n_experts = int(n_experts)
        self.register_buffer("feature_to_expert", torch.from_numpy(ft))

    @property
    def n_experts(self) -> int:
        return self._n_experts

    @property
    def n_features(self) -> int:
        return int(self.feature_to_expert.shape[0])

    def route(self, features: torch.Tensor, top_k: int) -> torch.Tensor:
        """Return ``(*batch, top_k)`` int64 expert indices, best first.

        ``features`` is ``(*batch, n_features)``. Scoring and ordering
        replicate ``ExpertDictionary.route`` per vector: scores are
        accumulated in float64 (matching polygram's float64
        ``np.add.at``) and ordered by descending score with stable
        ascending-index tie-breaking.
        """
        top_k = int(top_k)
        if not (1 <= top_k <= self._n_experts):
            raise ValueError(
                f"top_k={top_k} must satisfy 1 <= top_k <= n_experts="
                f"{self._n_experts}"
            )
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"features last dim {features.shape[-1]} != n_features "
                f"{self.n_features}"
            )
        batch_shape = features.shape[:-1]
        flat = features.reshape(-1, self.n_features)
        n_vectors = flat.shape[0]
        # Sum activations per expert. float64 to mirror polygram's
        # np.zeros(..., dtype=np.float64) + np.add.at accumulation, so
        # tie-breaking on near-equal scores matches the reference.
        scores = torch.zeros(
            (n_vectors, self._n_experts), dtype=torch.float64, device=features.device
        )
        idx = self.feature_to_expert.unsqueeze(0).expand(n_vectors, -1)
        scores.scatter_add_(1, idx, flat.to(torch.float64))
        # argsort(-scores, stable) is ascending of -scores == descending
        # of scores, ties broken by ascending original (expert) index —
        # identical to numpy argsort(-scores, kind="stable").
        order = torch.argsort(-scores, dim=-1, stable=True)
        topk = order[:, :top_k]
        return topk.reshape(*batch_shape, top_k).to(torch.int64)
