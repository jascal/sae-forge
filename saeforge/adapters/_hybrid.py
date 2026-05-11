"""Hybrid routing helper — three-basis dispatch over a single ArchitectureAdapter.

v1 strategy: walk the adapter three times (once per basis), then route each
emitted key through the basis whose layer region owns it. Wasteful by a factor
of 3 on projection work, but projection runs once per forge so the absolute
cost is small. The cleaner "one walk, per-key dispatch" alternative requires
patching the adapter API to take a routing callable; tracked as a follow-up
under ``multi-anchor-forge``.

Routing rules (matching ``HybridBasisBundle.basis_for_layer`` for block keys,
plus the documented non-block table):

- Block keys (matched by ``r"\\.h\\.(\\d+)\\."`` for GPT-2; family-specific
  regexes documented below): basis chosen by ``bundle.basis_for_layer(idx)``.
- Non-block keys with ``wte`` / ``wpe`` substrings: ``basis_embed``.
- Non-block keys with ``ln_f`` / ``lm_head`` substrings: ``basis_lm_head``.

The keyset of the hybrid output matches the single-basis adapter output for
the same host exactly — no keys added, no keys removed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.hybrid_basis import HybridBasisBundle

# GPT-2 / Llama / Gemma2 all use ``layer.<idx>`` or ``h.<idx>`` patterns.
# A unified regex covers both common spellings.
_BLOCK_KEY_REGEX = re.compile(r"\.(?:h|layers)\.(\d+)\.")


def _layer_index_for_key(key: str) -> int | None:
    """Return the block index encoded in ``key``, or ``None`` for non-block keys."""
    m = _BLOCK_KEY_REGEX.search(key)
    if m is None:
        return None
    return int(m.group(1))


def _basis_for_key(key: str, bundle: HybridBasisBundle) -> FeatureBasis:
    """Route a single emitted key to its basis using the design.md table."""
    idx = _layer_index_for_key(key)
    if idx is not None:
        return bundle.basis_for_layer(idx)
    # Non-block keys: embed-side and lm-head-side substrings.
    # Order matters — ``lm_head`` and ``ln_f`` belong to the lm-head region;
    # ``wte`` / ``wpe`` / ``embed_tokens`` to the embed region.
    if "lm_head" in key or ".ln_f." in key or key.endswith(".ln_f.weight") or key.endswith(".ln_f.bias"):
        return bundle.basis_lm_head
    if (
        "wte" in key
        or "wpe" in key
        or "embed_tokens" in key
        or "embed_positions" in key
    ):
        return bundle.basis_embed
    # Fallback: the mid basis. Conservative for keys that don't match the
    # documented patterns (e.g. an exotic adapter introduces a new
    # top-level key the hybrid router didn't anticipate). Documented in
    # design.md as the safe default.
    return bundle.basis_mid


def walk_hybrid(
    host: Any,
    adapter,
    projector,
    *,
    bundle: HybridBasisBundle,
    attention_width: str = "host",
) -> dict[str, np.ndarray]:
    """Project ``host`` through three bases and route each key by region.

    The returned dict has the same set of keys as ``adapter.walk(host, projector)``
    would have emitted with a single-basis projector. Each value comes from
    exactly one of the three bases per ``HybridBasisBundle.basis_for_layer``.
    """
    # Tied-embedding refusal is enforced one level up in ForgePipeline; this
    # helper assumes the bundle is shape-compatible (validated in
    # ``HybridBasisBundle.__post_init__``).
    from saeforge.projector import SubspaceProjector

    scale_boost = projector.scale_boost  # already resolved to a float in __post_init__

    proj_embed = SubspaceProjector(bundle.basis_embed, scale_boost=scale_boost)
    proj_mid = SubspaceProjector(bundle.basis_mid, scale_boost=scale_boost)
    proj_lm = SubspaceProjector(bundle.basis_lm_head, scale_boost=scale_boost)

    out_embed = adapter.walk(host, proj_embed, attention_width=attention_width)
    out_mid = adapter.walk(host, proj_mid, attention_width=attention_width)
    out_lm = adapter.walk(host, proj_lm, attention_width=attention_width)

    if not (out_embed.keys() == out_mid.keys() == out_lm.keys()):
        raise RuntimeError(
            "walk_hybrid: per-basis walks emitted different key sets — "
            f"embed={sorted(out_embed.keys())[:3]}..., "
            f"mid={sorted(out_mid.keys())[:3]}..., "
            f"lm={sorted(out_lm.keys())[:3]}..."
        )

    routed: dict[str, np.ndarray] = {}
    for key in out_mid.keys():
        basis = _basis_for_key(key, bundle)
        if basis is bundle.basis_embed:
            routed[key] = out_embed[key]
        elif basis is bundle.basis_lm_head:
            routed[key] = out_lm[key]
        else:
            routed[key] = out_mid[key]
    return routed
