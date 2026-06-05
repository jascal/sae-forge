"""SubDictionaryExpertSet — the v1 ``sae-moe-forge`` expert implementation.

Each expert is a deterministic *slice* of the source SAE decoder: the
rows of ``W_dec`` whose features the polygram clustering assigned to
that expert. No new parameters are introduced — the forge is a pure
projection of the existing SAE into a routed form (see
``openspec/specs/sae-moe-forge``). The forward decodes only the
features belonging to the per-token top-k selected experts.

Vectorisation note
------------------

The decode is expressed as a single masked matmul rather than a
per-expert Python loop. Because the feature→expert partition is
**disjoint**, decoding the union of the selected experts is exactly
``(features * active_feature_mask) @ W_dec`` where a feature is active
iff its owning expert is among the token's top-k. This is
mathematically identical to summing per-expert sub-decodes, with no
Python-level loop over experts. The *counted* routed cost (which is
what the sparsity-gain acceptance band measures) is reported
separately by :meth:`effective_decode_cost`; the dense masked matmul is
the v1 kernel and a gather-based sparse kernel is a queued follow-up.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _as_int64_array(feature_to_expert) -> np.ndarray:
    arr = np.asarray(feature_to_expert, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(
            f"feature_to_expert must be 1-D (n_features,); got shape {arr.shape}"
        )
    return arr


class SubDictionaryExpertSet(nn.Module):
    """A flat SAE decoder partitioned into routable sub-dictionaries.

    Parameters
    ----------
    W_dec:
        ``(n_features, d_model)`` decoder matrix (rows are features).
    feature_to_expert:
        ``(n_features,)`` integer map from feature index to expert
        index. Every value must lie in ``[0, n_experts)``.
    n_experts:
        Number of experts (clusters).

    The module registers only buffers — no trainable parameters — so
    ``.parameters()`` is empty and ``.to(device)`` moves the decoder
    and the partition map together.
    """

    def __init__(self, W_dec, feature_to_expert, n_experts: int):
        super().__init__()
        W_dec_t = torch.as_tensor(np.asarray(W_dec), dtype=torch.float32)
        if W_dec_t.ndim != 2:
            raise ValueError(
                f"W_dec must be 2-D (n_features, d_model); got shape "
                f"{tuple(W_dec_t.shape)}"
            )
        ft = _as_int64_array(feature_to_expert)
        if ft.shape[0] != W_dec_t.shape[0]:
            raise ValueError(
                f"feature_to_expert length {ft.shape[0]} does not match "
                f"W_dec rows {W_dec_t.shape[0]}"
            )
        n_experts = int(n_experts)
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1; got {n_experts}")
        if ft.min(initial=0) < 0 or ft.max(initial=0) >= n_experts:
            raise ValueError(
                f"feature_to_expert values must lie in [0, {n_experts}); "
                f"got range [{int(ft.min())}, {int(ft.max())}]"
            )

        self._n_experts = n_experts
        sizes = np.bincount(ft, minlength=n_experts).astype(np.int64)

        self.register_buffer("W_dec", W_dec_t)
        self.register_buffer("feature_to_expert", torch.from_numpy(ft))
        self.register_buffer("expert_sizes", torch.from_numpy(sizes))

    @property
    def n_features(self) -> int:
        return int(self.W_dec.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_dec.shape[1])

    @property
    def n_experts(self) -> int:
        return self._n_experts

    @property
    def expert_feature_ids(self) -> list[torch.Tensor]:
        """Per-expert ``(n_features_e,)`` int64 tensors of feature ids.

        The union over experts is exactly ``range(n_features)`` and any
        two experts' id sets are disjoint (the partition invariant the
        polygram ``ExpertDictionary`` enforces upstream).
        """
        return [
            torch.nonzero(self.feature_to_expert == e, as_tuple=False).flatten()
            for e in range(self._n_experts)
        ]

    def _active_feature_mask(self, top_k_experts: torch.Tensor) -> torch.Tensor:
        """``(*batch, n_features)`` bool mask of features owned by a
        selected expert."""
        batch_shape = top_k_experts.shape[:-1]
        active_expert = torch.zeros(
            (*batch_shape, self._n_experts),
            dtype=torch.bool,
            device=top_k_experts.device,
        )
        active_expert.scatter_(-1, top_k_experts, True)
        # Gather each feature's owning-expert activity flag → per-feature mask.
        return active_expert.index_select(-1, self.feature_to_expert)

    def forward(
        self, features: torch.Tensor, top_k_experts: torch.Tensor
    ) -> torch.Tensor:
        """Decode ``features`` through the per-token selected experts.

        ``features`` is ``(*batch, n_features)``; ``top_k_experts`` is
        ``(*batch, k)`` int64. Returns ``(*batch, d_model)``.
        """
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"features last dim {features.shape[-1]} != n_features "
                f"{self.n_features}"
            )
        mask = self._active_feature_mask(top_k_experts).to(features.dtype)
        gated = features * mask
        return gated @ self.W_dec.to(features.dtype)

    def effective_decode_cost(self, top_k_experts: torch.Tensor) -> int:
        """Counted decoder-row touches across the batch for a routing.

        Equals ``sum over tokens of (sum of selected experts' sizes)`` —
        the honest routed decode cost the sparsity-gain band measures.
        For uniform clusters of size ``n_features / n_experts`` and
        ``k`` experts per token this is ``M * k * n_features / n_experts``.
        """
        return int(self._active_feature_mask(top_k_experts).sum().item())
