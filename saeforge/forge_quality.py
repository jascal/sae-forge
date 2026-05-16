"""Forge feasibility diagnostics.

Surfaces structural-feasibility signals for the sweep pipeline: the host
model's residual width (``host_d_model``), the rank of the kept-features
basis (``basis_rank``), the ratio between them, and a categorical
"quality tier" (``saturated`` / ``good`` / ``undersized`` / ``degenerate``).
These let downstream consumers filter sweep rows like
``jq 'select(.quality_tier == "good" or .quality_tier == "saturated")'``
without an expert prior on what a "trustable" ``faithfulness_kl`` looks
like for a given setup.

See ``openspec/changes/add-forge-quality-diagnostics/`` for the design
rationale, threshold derivation, and lifecycle contract.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class QualityTier(str, Enum):
    """Four-tier categorical describing the rank ratio of the kept-features
    basis against the host residual width. Inherits from ``str`` so the
    enum value round-trips through JSON.
    """

    SATURATED = "saturated"
    GOOD = "good"
    UNDERSIZED = "undersized"
    DEGENERATE = "degenerate"


@dataclass(frozen=True)
class QualityThresholds:
    """Three boundary values that partition ``quality_ratio`` into four tiers.

    **Empirical anchor for the defaults**: PR #33's live N=32 Rung4 smoke
    (jbloom GPT-2 layer-8 SAE, threshold=0.95) produced rank-1 bases against
    GPT-2's 768-dim residual stream — ``quality_ratio ≈ 1/768 ≈ 0.0013`` —
    and the resulting forge faithfulness_kl was ~6.99 (near-random output
    entropy). That puts the empirically-observed "catastrophic" floor two
    orders of magnitude below the ``undersized``/``degenerate`` boundary of
    ``0.0625`` (= 1/16). The other two boundaries (``saturated`` = 1.0
    "basis fully spans residual"; ``good`` = 0.5 "half-coverage") are
    symmetric anchors above that empirical floor. Cross-host empirical
    calibration is open follow-up work (see
    ``openspec/changes/archive/2026-05-15-add-forge-quality-diagnostics/tasks.md``
    §10.2).

    Tweakable via the CLI ``--quality-tier-thresholds`` flag for callers
    running specific research. See `design.md` Decision 3.

    Invariants enforced in ``__post_init__``: ``saturated > good > undersized >= 0``.
    """

    saturated: float = 1.0
    good: float = 0.5
    undersized: float = 0.0625

    def __post_init__(self) -> None:
        if not (self.saturated > self.good > self.undersized >= 0):
            raise ValueError(
                f"QualityThresholds: must satisfy saturated > good > undersized >= 0; "
                f"got saturated={self.saturated}, good={self.good}, "
                f"undersized={self.undersized}"
            )


# Allowlist of `model_type` values whose `AutoConfig.hidden_size` field is
# canonically the residual-stream width. Non-standard architectures resolve
# but get a caveat in the advisory (per design.md Risks).
_RESIDUAL_STREAM_LM_MODEL_TYPES: frozenset[str] = frozenset({
    "gpt2", "llama", "gemma", "gemma2", "pythia", "gpt_neox",
    "mistral", "qwen", "qwen2", "qwen3",
})


# ---------------------------------------------------------------------------
# Computation primitives
# ---------------------------------------------------------------------------


def compute_basis_rank(w_dec_kept: np.ndarray) -> int:
    """Numerical rank of the kept-features basis matrix.

    Uses ``numpy.linalg.matrix_rank`` with the default tolerance (machine-
    precision-aware). Reflects the *actual* span of the basis, which can be
    less than the row count if surviving features have linearly dependent
    decoder rows (rare but possible).

    Edge cases:
    - **Empty input** (``shape[0] == 0``): raises ``ValueError``. Callers
      with no surviving features should detect this upstream and emit a
      ``basis_rank=0`` row directly rather than calling this function.
      ``basis_rank_from_safetensors`` handles the all-zeroed-W_dec case
      and returns 0 without calling here.
    - **All-zero rows**: returns 0 (every row is zero, no span). Caller's
      polygram pipeline should not produce this normally — the polygram
      compressor zeroes *non-representative* rows but keeps at least one
      representative per cluster.
    - **Linearly dependent rows**: returns the true span (lower than row
      count). E.g. four rows where row 4 = 2 * row 0 → rank 3.
    - **Very small matrices** (1 or 2 rows): handled correctly by
      ``matrix_rank``'s machine-precision tolerance. A single non-zero
      row has rank 1; a single zero row has rank 0.
    """
    if w_dec_kept.shape[0] == 0:
        raise ValueError("compute_basis_rank: input has 0 rows")
    return int(np.linalg.matrix_rank(w_dec_kept))


def classify_quality(
    basis_rank: int,
    host_d_model: int,
    thresholds: QualityThresholds | None = None,
) -> tuple[float, QualityTier]:
    """Compute ``(quality_ratio, quality_tier)`` from a rank + d_model.

    ``quality_ratio = basis_rank / host_d_model``. Tier boundaries follow
    ``thresholds`` (defaults to ``QualityThresholds()``).
    """
    if thresholds is None:
        thresholds = QualityThresholds()
    if host_d_model < 1:
        raise ValueError(
            f"classify_quality: host_d_model must be >= 1; got {host_d_model}"
        )
    if basis_rank < 0:
        raise ValueError(
            f"classify_quality: basis_rank must be >= 0; got {basis_rank}"
        )

    ratio = float(basis_rank) / float(host_d_model)
    if ratio >= thresholds.saturated:
        tier = QualityTier.SATURATED
    elif ratio >= thresholds.good:
        tier = QualityTier.GOOD
    elif ratio >= thresholds.undersized:
        tier = QualityTier.UNDERSIZED
    else:
        tier = QualityTier.DEGENERATE
    return ratio, tier


# ---------------------------------------------------------------------------
# Host d_model resolution
# ---------------------------------------------------------------------------


def resolve_host_d_model(host_model_id: str) -> tuple[int | None, str | None]:
    """Resolve the host transformer's residual stream width via
    ``transformers.AutoConfig`` (config-only fetch, no weight load).

    Returns ``(d_model, model_type)`` where ``model_type`` is the
    AutoConfig's ``model_type`` attribute (e.g. ``"gpt2"``, ``"llama"``)
    used to gate the residual-stream-LM caveat in the advisory. Both
    values may be ``None`` when resolution fails (network error, gated
    model, ``hidden_size`` attribute missing — typical of non-transformer
    or encoder-decoder hosts).

    Failures are logged to stderr (single line, prefixed) and do NOT
    raise — the sweep should proceed with diagnostics disabled rather
    than aborting.
    """
    try:
        # Lazy-imported so the module can be imported without the [torch]
        # extra. The function is no-op when transformers is unavailable.
        import transformers
    except ImportError:
        print(
            "saeforge.forge_quality: transformers not installed; "
            "host d_model resolution skipped",
            file=sys.stderr,
        )
        return None, None

    try:
        config = transformers.AutoConfig.from_pretrained(host_model_id)
    except Exception as exc:  # noqa: BLE001 — defensive; we log + return None
        print(
            f"saeforge.forge_quality: AutoConfig.from_pretrained({host_model_id!r}) "
            f"failed ({exc!r}); diagnostics disabled",
            file=sys.stderr,
        )
        return None, None

    d_model = getattr(config, "hidden_size", None)
    if d_model is None:
        print(
            f"saeforge.forge_quality: {host_model_id!r} config has no "
            "hidden_size attribute (non-transformer or encoder-decoder host?); "
            "diagnostics disabled",
            file=sys.stderr,
        )
        return None, getattr(config, "model_type", None)

    return int(d_model), getattr(config, "model_type", None)


# ---------------------------------------------------------------------------
# Pre-flight advisory
# ---------------------------------------------------------------------------


def advise_sweep_quality(
    *,
    encodings: list[tuple[str, Path]],
    host_d_model: int,
    thresholds: QualityThresholds,
    manifest_loader: Callable[[Path], dict[int, Any]],
    basis_rank_loader: Callable[[Path], int],
    model_type: str | None = None,
) -> str | None:
    """Build the pre-flight advisory string, or ``None`` if no warning is warranted.

    Examines each encoding's smallest-K materialised SAE, computes that K's
    ``basis_rank``, classifies quality. If ANY encoding's smallest-K tier is
    ``undersized`` or ``degenerate``, returns a stderr-ready multi-line
    advisory naming the affected encoding, the computed numbers, and a
    suggested K floor derived from the manifest (or a recommendation to
    re-run polygram compression with larger K targets if no K in the
    manifest meets the ``good`` threshold).

    ``manifest_loader`` and ``basis_rank_loader`` are injected so this
    function is testable without a full polygram fixture; production
    wiring uses ``saeforge.sweep._load_pareto_manifest`` and a
    ``safetensors``-backed loader.
    """
    lines: list[str] = []

    if model_type is not None and model_type not in _RESIDUAL_STREAM_LM_MODEL_TYPES:
        lines.append(
            f"Note: host_d_model resolved as {host_d_model} via "
            f"AutoConfig.hidden_size (model_type={model_type!r}); "
            "interpretation as residual-stream width assumes a standard "
            "transformer architecture and may be misleading for "
            "encoder-decoder, encoder-only, or non-LM hosts."
        )

    for label, enc_path in encodings:
        manifest = manifest_loader(enc_path)
        if not manifest:
            # No manifest → can't enumerate per-K kept counts; skip advisory
            # for this encoding (the row-level diagnostics still populate
            # via the SAE-direct fallback path).
            continue

        sorted_ks = sorted(manifest.keys())
        smallest_k = sorted_ks[0]

        # Locate the smallest-K checkpoint file. Mirrors sweep's enumeration
        # layout — pareto/k_{K}.safetensors under the encoding dir.
        ckpt_path = enc_path / "pareto" / f"k_{smallest_k}.safetensors"
        if not ckpt_path.is_file():
            ckpt_path = enc_path / f"k_{smallest_k}.safetensors"
        if not ckpt_path.is_file():
            continue  # can't locate; defer to row-level diagnostics

        try:
            basis_rank = basis_rank_loader(ckpt_path)
        except Exception:  # noqa: BLE001
            continue

        ratio, tier = classify_quality(basis_rank, host_d_model, thresholds)
        if tier in (QualityTier.SATURATED, QualityTier.GOOD):
            continue  # this encoding looks fine

        # Find a suggested K floor: smallest K whose entry's
        # n_features_kept is >= good_threshold * host_d_model.
        good_floor = int(thresholds.good * host_d_model)
        suggested_k: int | None = None
        for k in sorted_ks:
            entry = manifest[k]
            kept = getattr(entry, "n_features_kept", None)
            if kept is None and isinstance(entry, dict):
                kept = entry.get("n_features_kept")
            if kept is not None and kept >= good_floor:
                suggested_k = k
                break

        block = [
            f"  encoding={label}: smallest K={smallest_k} basis_rank={basis_rank} "
            f"host_d_model={host_d_model} quality_ratio={ratio:.4f} "
            f"quality_tier={tier.value!r}",
        ]
        if suggested_k is not None:
            block.append(
                f"    suggested K floor: K={suggested_k} (smallest K whose "
                f"n_features_kept >= {good_floor} = host_d_model * "
                f"{thresholds.good:g})"
            )
        else:
            block.append(
                f"    No K target in the supplied manifest meets the 'good' "
                f"threshold (basis_rank >= host_d_model/2 = {good_floor}). "
                f"Consider re-running 'polygram compress --pareto' with "
                f"larger K targets — e.g. include values closer to your "
                f"SAE's full feature count."
            )
        lines.extend(block)

    if not lines:
        return None

    header = (
        "saeforge sweep-pareto: forge-quality advisory — one or more "
        "encodings' smallest K is undersized or degenerate."
    )
    footer = (
        "  'degenerate' describes the rank ratio, not the validity of the "
        "run; exploratory low-rank smokes remain valid for impl validation."
    )
    return "\n".join([header, *lines, footer])


def basis_rank_from_safetensors(sae_checkpoint: Path) -> int:
    """Load ``W_dec`` from a polygram-compressed safetensors checkpoint and
    return its numerical rank restricted to surviving (non-zero) rows.

    Helper used by ``advise_sweep_quality``'s injected loader and by
    ``sweep.sweep_pareto``'s row-population path.
    """
    from safetensors.numpy import load_file

    state = load_file(str(sae_checkpoint))
    if "W_dec" not in state:
        raise KeyError(
            f"basis_rank_from_safetensors: {sae_checkpoint} missing 'W_dec'"
        )
    w_dec = state["W_dec"]
    nonzero_rows = (w_dec != 0).any(axis=1)
    w_dec_kept = w_dec[nonzero_rows]
    if w_dec_kept.shape[0] == 0:
        return 0
    return compute_basis_rank(w_dec_kept)
