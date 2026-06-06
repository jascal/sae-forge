"""Augmented routing helper — per-layer kept-subspace dispatch over one adapter.

Two-basis forge preserves a per-layer composition subspace ``U_C`` plus a
global assertion subspace ``U_A`` inside the projection. Each block therefore
has its own effective decoder ``W_dec_eff`` (the ``U_C`` rows differ per layer);
non-block keys and blocks without a composition subspace use the
assertion-only base. Strategy mirrors ``_hybrid``: walk the adapter once per
distinct kept subspace, then route each emitted key to the walk that owns its
layer. Projection runs once per forge, so the extra walks are cheap.

The keyset matches the single-basis adapter output exactly — no keys added or
removed; augmentation only changes which kept subspace each weight is projected
through. See ``openspec/specs/composition-subspace-preserve``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from saeforge.basis import FeatureBasis

# GPT-2 (``h.<idx>``) and Llama/Gemma (``layers.<idx>``) block-key spellings.
_BLOCK_KEY_REGEX = re.compile(r"\.(?:h|layers)\.(\d+)\.")


def _layer_index_for_key(key: str) -> int | None:
    m = _BLOCK_KEY_REGEX.search(key)
    return int(m.group(1)) if m else None


def _basis_with_W(base: FeatureBasis, W_eff: np.ndarray) -> FeatureBasis:
    """A FeatureBasis identical to ``base`` but with decoder ``W_eff``.

    ``W_enc`` is dropped (the projector encodes via ``pinv(W_dec)``, not
    ``W_enc``); norms are recomputed from ``W_eff``.
    """
    norms = np.linalg.norm(W_eff, axis=1)
    return FeatureBasis(
        kept_ids=base.kept_ids,
        W_dec=np.ascontiguousarray(W_eff),
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=base.scale_compression_ratio,
        metadata=dict(base.metadata),
    )


def walk_augmented(
    host: Any,
    adapter,
    projector,
    *,
    augmented,
    attention_width: str = "host",
) -> dict[str, np.ndarray]:
    """Project ``host`` through per-layer augmented kept subspaces and route by layer.

    The returned dict has the same set of keys as the single-basis
    ``adapter.walk(host, projector)``. Each block weight comes from its layer's
    augmented projector (assertion + that layer's composition subspace); every
    other key comes from the assertion-only base projector.
    """
    from saeforge.projector import SubspaceProjector

    scale_boost = projector.scale_boost  # resolved to a float in __post_init__

    # Assertion-only base (layer index -1 never appears in the composition dict):
    # covers non-block keys and any block layer without a composition subspace.
    base_W, _ = augmented.kept_subspace(-1)
    base_proj = SubspaceProjector(_basis_with_W(projector.basis, base_W), scale_boost=scale_boost)
    base_out = adapter.walk(host, base_proj, attention_width=attention_width)

    comp = augmented.composition or {}
    per_layer_out: dict[int, dict[str, np.ndarray]] = {}
    for ell in comp:
        W_eff, _ = augmented.kept_subspace(ell)
        proj = SubspaceProjector(_basis_with_W(projector.basis, W_eff), scale_boost=scale_boost)
        out = adapter.walk(host, proj, attention_width=attention_width)
        if out.keys() != base_out.keys():
            raise RuntimeError(
                "walk_augmented: per-layer walk emitted a different key set than the base walk"
            )
        per_layer_out[ell] = out

    routed: dict[str, np.ndarray] = {}
    for key in base_out:
        idx = _layer_index_for_key(key)
        if idx is not None and idx in per_layer_out:
            routed[key] = per_layer_out[idx][key]
        else:
            routed[key] = base_out[key]
    return routed
