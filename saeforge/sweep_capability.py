"""sweep_pareto_capability — Pareto sweep over downstream-capability AUC.

Wraps :func:`saeforge.sweep_pareto`'s machinery (encoding × width ×
scale_boost cube; per-cell forge; frontier.jsonl emission) but
substitutes :class:`saeforge.eval.targets.DownstreamCapabilityTarget`
for the default faithfulness metric. Each row carries the existing
forge / polygram diagnostics PLUS the new capability fields
documented in
``add-downstream-capability-target/specs/pareto-sweep/spec.md``.

Use this entry point when the sweep's goal is to **retain a downstream
task** (e.g. bio-sae's GO/Pfam/EC feature discrimination) rather than
to **minimise residual-cosine drift from host**. Bio-sae's 2026-05-22
investigation showed those are different Pareto frontiers and the
cosine-driven one misranks forges for capability-bound users.

Pipeline per cell:

  1. Build :class:`FeatureBasis` from the SAE checkpoint at
     ``target_n_features_kept`` (slice the SAE's W_dec to its top-N
     rows by L2 norm — simple, no polygram dependency).
  2. Forge the host via :class:`ForgePipeline` using the
     ``DownstreamCapabilityTarget`` constructed from ``dataset``.
  3. Compute retained metrics from the target's side-channel
     ``host_pf_auc`` / ``forge_pf_auc`` arrays.
  4. Populate a :class:`ParetoFrontierRow` (the same dataclass
     ``sweep_pareto`` emits) with the capability fields filled in.
  5. Append to ``frontier.jsonl`` under ``output_dir``.

Host-extraction caching: identical inputs across cells produce the
same host activations, so we cache once per
``(host_model_id, sequences_hash, aggregator, max_seq_len)`` cube to
amortise the per-cell host forward across the sweep. Opt-out via
``cache_host=False``.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from saeforge.datasets._host_cache import HostCacheKey, HostExtractionCache
from saeforge.sweep import ParetoFrontierRow, _finite_or_none


@dataclass(frozen=True)
class _SweepCell:
    """One (encoding, target_n_features_kept, scale_boost) cell."""

    encoding_label: str
    target_n_features_kept: int
    scale_boost: "float | str"


@dataclass(frozen=True)
class _EncodingState:
    """Per-encoding loaded SAE state for the cell loop.

    Constructed by :func:`_load_encoding_state` from an SAE state
    dict. The invariants below are checked in ``__post_init__``
    so callers constructing this directly (e.g. tests with hand-
    written tensors) fail loudly on shape mismatches rather than
    deferring to opaque downstream errors.
    """

    label: str
    sae_checkpoint: Path
    W_dec_full: np.ndarray   # (n_features, d_model), float64
    row_norms: np.ndarray    # (n_features,), float64
    order: np.ndarray        # argsort descending of row_norms
    partition_block_ids: np.ndarray | None

    def __post_init__(self) -> None:
        n_features = self.W_dec_full.shape[0]
        if self.row_norms.shape != (n_features,):
            raise ValueError(
                f"_EncodingState[{self.label!r}]: row_norms shape "
                f"{self.row_norms.shape} does not match W_dec_full "
                f"row count ({n_features},)"
            )
        if self.order.shape != (n_features,):
            raise ValueError(
                f"_EncodingState[{self.label!r}]: order shape "
                f"{self.order.shape} does not match W_dec_full "
                f"row count ({n_features},)"
            )
        if (
            self.partition_block_ids is not None
            and self.partition_block_ids.shape != (n_features,)
        ):
            raise ValueError(
                f"_EncodingState[{self.label!r}]: partition_block_ids "
                f"shape {self.partition_block_ids.shape} does not match "
                f"W_dec_full row count ({n_features},)"
            )


def _load_encoding_state(label: str, path: "str | Path") -> _EncodingState:
    """Load one encoding's SAE state for the cell loop.

    Accepts two key conventions (detected by presence):

    - **reference** (`add-downstream-capability-target` / bio-sae): ``decoder.weight`` shaped
      ``(d_model, n_features)`` — transposed to ``(n_features, d_model)``.
    - **SAELens** (LM SAEs, e.g. jbloom GPT-2): ``W_dec`` already shaped ``(n_features, d_model)``
      (`add-causal-host-capability-sweep`).

    plus the optional ``partition_block_ids`` key (added by
    ``add-partition-encoding-capability-validation``). ``.safetensors`` checkpoints (SAELens) are read with the
    safetensors loader; ``.pt`` with ``torch.load``. Computes row_norms + argsort order once per encoding.

    Raises ``ValueError`` when ``partition_block_ids`` shape doesn't match decoder rows.
    """
    import torch

    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sae_state = load_file(str(path))
    else:
        sae_state = torch.load(str(path), map_location="cpu", weights_only=True)
    if "decoder.weight" in sae_state:
        W_dec_full = sae_state["decoder.weight"].numpy().T.astype(np.float64)  # (d_model, n_features) -> (n_features, d_model)
    elif "W_dec" in sae_state:
        W_dec_full = np.asarray(sae_state["W_dec"], dtype=np.float64)  # SAELens: already (n_features, d_model)
    else:
        raise ValueError(
            f"sweep_pareto_capability: SAE checkpoint {path} has neither 'decoder.weight' (reference) nor "
            f"'W_dec' (SAELens) key; got {sorted(sae_state.keys())[:8]!r}..."
        )
    row_norms = np.linalg.norm(W_dec_full, axis=1)
    order = np.argsort(-row_norms)
    partition_block_ids: np.ndarray | None = None
    if "partition_block_ids" in sae_state:
        partition_block_ids = (
            sae_state["partition_block_ids"].numpy().astype(np.int64)
        )
        if partition_block_ids.shape != (W_dec_full.shape[0],):
            raise ValueError(
                f"sweep_pareto_capability: partition_block_ids for "
                f"encoding {label!r} has shape "
                f"{partition_block_ids.shape}, expected "
                f"({W_dec_full.shape[0]},) to match decoder.weight rows."
            )
    return _EncodingState(
        label=label,
        sae_checkpoint=Path(path),
        W_dec_full=W_dec_full,
        row_norms=row_norms,
        order=order,
        partition_block_ids=partition_block_ids,
    )


def _readout_aligned_order(
    W_dec_full: np.ndarray,
    u_matrix: np.ndarray,
    gain: "np.ndarray | None" = None,
    rank: int = 64,
) -> np.ndarray:
    """Order decoder rows by alignment with the model's READOUT (decode-decision) geometry, not row L2
    norm (change add-capability-trained-encoder, task 3.1). The readout subspace is the top-``rank`` right
    singular directions of ``gain⊙U`` (the unembed the model actually reads through). Each W_dec row scores
    by the energy of its projection onto that subspace; descending argsort. Decode-aligned features rank
    first — the fieldrun finding that the readout-aligned decision directions, not raw decoder norm, govern
    behaviour. ``u_matrix``: (vocab, d_model) unembed; ``gain``: (d_model,) final-norm gain (default ones)."""
    U = np.asarray(u_matrix, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != W_dec_full.shape[1]:
        raise ValueError(
            f"_readout_aligned_order: u_matrix must be (vocab, d_model) with "
            f"d_model={W_dec_full.shape[1]}; got {U.shape}"
        )
    if gain is not None:
        U = U * np.asarray(gain, dtype=np.float64)[None, :]
    r = int(min(rank, U.shape[0], U.shape[1]))
    Vt = np.linalg.svd(U, full_matrices=False)[2][:r]          # (r, d_model) readout subspace
    proj = (np.asarray(W_dec_full, dtype=np.float64) @ Vt.T)   # (n_features, r)
    score = np.linalg.norm(proj, axis=1)                       # projection energy per feature
    return np.argsort(-score)


def _resolve_basis_order(
    enc_state: "_EncodingState",
    *,
    basis_order: str,
    u_matrix: "np.ndarray | None",
    gain: "np.ndarray | None",
    host_model_id: str,
    readout_fallback: "str | None",
) -> np.ndarray:
    """Return the feature ordering for the width slice per the basis_order policy (task 3.2, Decision 5).
    ``row_norm`` ⇒ the existing L2-norm order. ``readout_aligned`` ⇒ order by readout-geometry alignment,
    sourcing ``u_matrix`` from the arg or ``load_host_unembed(host_model_id)``; if neither is available
    (encoder-only families), RAISE unless ``readout_fallback='downstream_decode'`` (then warn once + use the
    SAE's own decode geometry). Never silently revert to row_norm."""
    if basis_order == "row_norm":
        return enc_state.order
    U = u_matrix
    if U is None:
        try:
            from saeforge import load_host_unembed
            U = load_host_unembed(host_model_id)
        except Exception:
            U = None
    if U is not None:
        return _readout_aligned_order(enc_state.W_dec_full, U, gain)
    # No readout geometry available (encoder-only family / no unembed).
    if readout_fallback == "downstream_decode":
        warnings.warn(
            f"sweep_pareto_capability(basis_order='readout_aligned'): no host unembed available for "
            f"'{host_model_id}' (encoder-only family). Falling back to the downstream encoder's decode "
            f"geometry (the SAE's own W_dec directions) per readout_fallback='downstream_decode'.",
            UserWarning,
            stacklevel=2,
        )
        return _readout_aligned_order(enc_state.W_dec_full, enc_state.W_dec_full, None)
    raise ValueError(
        f"basis_order='readout_aligned' needs a readout geometry, but no u_matrix was supplied and "
        f"load_host_unembed('{host_model_id}') returned none (encoder-only family). Supply u_matrix=... "
        f"or pass readout_fallback='downstream_decode' to order by the SAE's own decode geometry."
    )


def _normalize_encodings_arg(
    encodings: "list[tuple[str, str | Path]] | None",
    sae_checkpoint: "str | Path | None",
) -> list[tuple[str, Path]]:
    """Reconcile the new ``encodings`` kwarg with the legacy
    ``sae_checkpoint`` kwarg.

    Per add-multi-encoding-capability-sweep/specs/pareto-sweep/spec.md:

    - encodings provided → canonical multi-encoding list.
    - encodings=None AND sae_checkpoint provided →
      [("raw_slice", sae_checkpoint)] (v0.9.x back-compat).
    - both None → ValueError.
    - both provided → ValueError (explicit is better than ambiguous).

    Encoding labels SHALL be unique; duplicates raise ValueError.
    """
    if encodings is not None and sae_checkpoint is not None:
        raise ValueError(
            "sweep_pareto_capability: pass either `encodings=[(label, "
            "path), ...]` (multi-encoding) OR `sae_checkpoint=PATH` "
            "(single-encoding), not both. The legacy sae_checkpoint "
            "kwarg is internally sugar for encodings=[('raw_slice', "
            "sae_checkpoint)]; to compare raw_slice against other "
            "encodings, list it explicitly in `encodings`. See "
            "openspec/changes/add-multi-encoding-capability-sweep/"
            "design.md Decision 1."
        )
    if encodings is None and sae_checkpoint is None:
        raise ValueError(
            "sweep_pareto_capability: pass either `encodings=[(label, "
            "path), ...]` or `sae_checkpoint=PATH`. Neither was given."
        )
    if encodings is None:
        # Back-compat: single-encoding via sae_checkpoint.
        return [("raw_slice", Path(sae_checkpoint))]  # type: ignore[arg-type]
    # Multi-encoding path.
    seen: set[str] = set()
    normalized: list[tuple[str, Path]] = []
    for label, path in encodings:
        if label in seen:
            raise ValueError(
                f"sweep_pareto_capability: duplicate encoding label "
                f"{label!r} in encodings list. Labels SHALL be unique."
            )
        seen.add(label)
        normalized.append((label, Path(path)))
    return normalized


def sweep_pareto_capability(
    sae_checkpoint: "str | Path | None" = None,
    host_model_id: str | None = None,
    dataset: Any = None,  # CapabilityDataset
    *,
    widths: list[int] | None = None,
    encodings: "list[tuple[str, str | Path]] | list[str] | None" = None,
    scale_boosts: list["float | str"] | None = None,
    output_dir: "str | Path | None" = None,
    cache_host: bool = True,
    max_seq_len: int = 512,
    device: str = "cpu",
    host_layer: "int | None" = None,
    train_encoder: bool = False,
    train_objective: str = "proxy",
    basis_order: str = "row_norm",
    readout_fallback: "str | None" = None,
    host_encoder: "Any | None" = None,
    u_matrix: "np.ndarray | None" = None,
    gain: "np.ndarray | None" = None,
    train_steps: int = 300,
    train_lr: float = 1e-3,
    train_seed: int = 0,
) -> list[ParetoFrontierRow]:
    """Run a capability-aware Pareto sweep over the basis-config cube.

    **Multi-encoding API (added by add-multi-encoding-capability-sweep).**
    Pass ``encodings=[(label, path), ...]`` to compare multiple
    encodings in a single sweep run. Each (label, path) pair is
    loaded once; the per-cell loop iterates over
    `(encoding, width, scale_boost)`. Per-cell rows carry the
    encoding's label in ``ParetoFrontierRow.encoding_label``.

    The legacy single-encoding ``sae_checkpoint=PATH`` keyword is
    retained as sugar (equivalent to
    ``encodings=[("raw_slice", sae_checkpoint)]``).

    Parameters
    ----------
    sae_checkpoint:
        Legacy single-encoding path. Mutually exclusive with
        ``encodings``.
    host_model_id:
        HF id of the host model to forge.
    dataset:
        :class:`saeforge.datasets.CapabilityDataset` carrying
        sequences + labels + encoder + aggregator + min_prevalence.
    widths:
        List of basis widths to sweep. Each width slices each
        encoding's W_dec to its top-N rows by L2 norm (or
        per-tier proportionally when partition_block_ids is set).
    encodings:
        List of ``(label, sae_checkpoint_path)`` pairs. Mutually
        exclusive with ``sae_checkpoint``. A legacy list of bare
        encoding labels (strings) is accepted with a deprecation
        warning and treated as informational labels with the
        legacy single-encoding path.
    scale_boosts:
        List of SubspaceProjector scale_boost values. Floats or
        the literal ``"auto"``. Defaults to ``[1.0]``.
    output_dir:
        Where to write ``frontier.jsonl`` and the host-extraction
        cache.
    cache_host:
        When True (default), cache host activations across cells
        AND across encodings. Host activations are encoding-
        independent so the cache is shared.

    Returns
    -------
    List of :class:`ParetoFrontierRow`, one per cell. Multi-encoding
    sweeps produce rows whose ``encoding_label`` field identifies
    which encoding was used.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distinguish the new multi-encoding shape (list of (label, path)
    # tuples) from the legacy informational-labels shape (list of
    # str). The legacy shape preserves v0.8.x / v0.9.x byte-equivalent
    # behaviour — encoding labels are just stamped on rows without
    # actually changing the basis.
    legacy_encoding_labels: list[str] | None = None
    if encodings is not None and len(encodings) > 0 and isinstance(encodings[0], str):
        legacy_encoding_labels = list(encodings)
        encodings = None  # type: ignore[assignment]
    if scale_boosts is None or not scale_boosts:
        scale_boosts = [1.0]
    if basis_order not in ("row_norm", "readout_aligned"):
        raise ValueError(
            f"basis_order must be 'row_norm' or 'readout_aligned'; got {basis_order!r}"
        )
    if train_objective not in ("proxy", "full_forge"):
        raise ValueError(
            f"train_objective must be 'proxy' or 'full_forge'; got {train_objective!r}"
        )

    # Resolve the (label, path) list. Per the openspec, exactly one of
    # encodings / sae_checkpoint must be provided (legacy informational
    # labels excepted — those still need sae_checkpoint).
    if legacy_encoding_labels is not None:
        if sae_checkpoint is None:
            raise ValueError(
                "sweep_pareto_capability: legacy string-list encodings "
                "(informational labels) require sae_checkpoint to be set."
            )
        loaded_encodings = [
            _load_encoding_state(label, sae_checkpoint)
            for label in legacy_encoding_labels
        ]
    else:
        encoding_pairs = _normalize_encodings_arg(encodings, sae_checkpoint)
        loaded_encodings = [
            _load_encoding_state(label, path)
            for label, path in encoding_pairs
        ]

    # Validate dataset shape early (catches bad fixtures before we
    # pay any forge cost). Under pooled feed, each sequence yields
    # one labelled item. Under residue feed, each sequence yields
    # multiple labelled rows; the strict per-residue count is verified
    # post-extraction once we know what the tokenizer actually emits.
    n_items = dataset.labels.shape[0]
    if dataset.feed == "pooled" and len(dataset.sequences) != n_items:
        raise ValueError(
            f"sweep_pareto_capability(feed='pooled'): dataset.sequences "
            f"({len(dataset.sequences)}) and dataset.labels ({n_items} "
            f"rows) must align"
        )
    if dataset.feed == "residue" and n_items < len(dataset.sequences):
        raise ValueError(
            f"sweep_pareto_capability(feed='residue'): dataset.labels "
            f"({n_items} rows) must be >= len(sequences) "
            f"({len(dataset.sequences)}) — each protein contributes "
            f">=1 residue row."
        )

    # ---- Stage 1: host-extraction cache + once-per-sweep host load. ----
    cache = HostExtractionCache(
        output_dir / "host_cache",
        enabled=cache_host,
    )
    key = HostCacheKey.from_inputs(
        # host_layer changes the activations; fold it into the key's identity (hash-only — the real
        # host_model_id below drives loading) so different layers don't collide in the same cache dir.
        host_model_id=host_model_id if host_layer is None else f"{host_model_id}#L{host_layer}",
        sequences=list(dataset.sequences),
        aggregator=dataset.aggregator,
        max_seq_len=max_seq_len,
        feed=dataset.feed,
    )

    host_X: torch.Tensor | None = None
    if cache.has(key):
        host_X = cache.load(key)
    # Lazy-load host only if we need to extract (cache miss).

    # ---- Stage 2: per-encoding SAE state already loaded above. ----
    # `loaded_encodings` carries one _EncodingState per encoding;
    # each has its own W_dec_full / row_norms / order /
    # partition_block_ids. The host-extraction cache is shared across
    # encodings (host activations are encoding-independent).

    # ---- Stage 3: host-baseline metrics (compute once per sweep). ----
    # Note: host metrics need the encoder applied AFTER extraction;
    # cache stores raw host_X (post-aggregator d_model vectors).
    if host_X is None:
        host_X = _extract_host_activations(
            host_model_id=host_model_id,
            sequences=list(dataset.sequences),
            aggregator=dataset.aggregator,
            max_seq_len=max_seq_len,
            device=device,
            feed=dataset.feed,
            host_layer=host_layer,
        )
        cache.save(key, host_X)

    # Under residue feed, host_X rows MUST align with dataset.labels
    # rows (each row = one residue, protein-major). Misalignment
    # surfaces silently as nonsense AUCs downstream; check loudly here.
    if dataset.feed == "residue" and host_X.shape[0] != dataset.labels.shape[0]:
        raise RuntimeError(
            f"sweep_pareto_capability(feed='residue'): host extraction "
            f"yielded {host_X.shape[0]} residue rows but dataset.labels "
            f"has {dataset.labels.shape[0]} rows. Mismatch usually means "
            f"the dataset's max_seq_len doesn't match the bundle's "
            f"build-time max_seq_len, or sequences contain non-canonical "
            f"residues the tokenizer maps differently than expected."
        )

    Y = _filter_labels(dataset.labels, dataset.min_prevalence)
    with torch.no_grad():
        host_z = dataset.encoder(host_X.float())
    host_pf_auc = _best_auc_per_feature(host_z.detach().cpu().numpy(), Y)
    host_mauc = float(np.nanmean(host_pf_auc))
    valid = np.isfinite(host_pf_auc)
    host_cov95 = float((host_pf_auc[valid] >= 0.95).mean()) if valid.any() else 0.0

    # ---- Stage 4: sweep cells, per-encoding. ----
    # The cell loop iterates over (encoding × width × scale_boost).
    # Each encoding has its own W_dec / row_norms / order /
    # partition_block_ids; the host extraction is shared across
    # encodings (already done above).
    cells = [
        (enc_state, _SweepCell(
            encoding_label=enc_state.label,
            target_n_features_kept=w,
            scale_boost=s,
        ))
        for enc_state in loaded_encodings
        for w in widths
        for s in scale_boosts
    ]
    # Resolve the per-encoding feature ordering for the width slice once (basis_order policy, task 3.1/3.2).
    orders = {
        enc.label: _resolve_basis_order(
            enc, basis_order=basis_order, u_matrix=u_matrix, gain=gain,
            host_model_id=host_model_id, readout_fallback=readout_fallback,
        )
        for enc in loaded_encodings
    }
    rows: list[ParetoFrontierRow] = []
    frontier_path = output_dir / "frontier.jsonl"
    if frontier_path.exists():
        frontier_path.unlink()  # fresh file per invocation

    for enc_state, cell in cells:
        t0 = time.monotonic()
        try:
            row = _run_capability_cell(
                cell=cell,
                sae_checkpoint=enc_state.sae_checkpoint,
                host_model_id=host_model_id,
                dataset=dataset,
                Y=Y,
                W_dec_full=enc_state.W_dec_full,
                row_norms=enc_state.row_norms,
                order=orders[enc_state.label],
                host_pf_auc=host_pf_auc,
                host_mauc=host_mauc,
                host_cov95=host_cov95,
                device=device,
                host_layer=host_layer,
                partition_block_ids=enc_state.partition_block_ids,
                train_encoder=train_encoder,
                train_objective=train_objective,
                host_encoder=host_encoder if host_encoder is not None else dataset.encoder,
                host_X=host_X,
                output_dir=output_dir,
                basis_order=basis_order if basis_order != "row_norm" else None,
                train_steps=train_steps,
                train_lr=train_lr,
                train_seed=train_seed,
            )
        except Exception as exc:  # noqa: BLE001 — surface failures per-row
            row = ParetoFrontierRow(
                encoding_label=cell.encoding_label,
                target_n_features_kept=cell.target_n_features_kept,
                n_features_kept_actual=None,
                pareto_reached_target=None,
                faithfulness_kl=None,
                perplexity=None,
                final_fine_tune_loss=None,
                sae_checkpoint=str(enc_state.sae_checkpoint),
                forged_model_path=None,
                elapsed_seconds=time.monotonic() - t0,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
        with frontier_path.open("a") as fh:
            fh.write(json.dumps(row.to_json_dict()) + "\n")

    return rows


def _slice_partition_aware(
    *,
    row_norms: np.ndarray,
    partition_block_ids: np.ndarray,
    target_n_features_kept: int,
) -> np.ndarray:
    """Partition-aware basis slicing.

    Allocates ``target_n_features_kept`` across the unique tiers in
    ``partition_block_ids`` proportionally to per-tier feature counts,
    then takes top-K by row norm WITHIN each tier. The proportional
    allocation uses largest-fractional-remainder rounding for the last
    slots; ties broken by lowest tier id for determinism.

    Per spec
    ``add-partition-encoding-capability-validation/specs/pareto-sweep/spec.md``:

    1. tier_sizes[t] = count of features in tier t.
    2. proportional[t] = target * tier_sizes[t] / sum(tier_sizes).
    3. floor allocation; remainder slots distributed by largest
       fractional residual.
    4. within each tier, kept features = top-K by row norm.

    Returns a sorted int64 array of kept feature ids. ``len(result)``
    SHALL equal ``target_n_features_kept``.
    """
    n_features_total = row_norms.shape[0]
    if partition_block_ids.shape[0] != n_features_total:
        raise ValueError(
            f"partition_block_ids has {partition_block_ids.shape[0]} "
            f"entries but row_norms has {n_features_total}"
        )
    unique_tiers = sorted(set(int(t) for t in partition_block_ids))
    tier_sizes = np.array(
        [int((partition_block_ids == t).sum()) for t in unique_tiers],
        dtype=np.int64,
    )
    total_size = int(tier_sizes.sum())
    if target_n_features_kept > total_size:
        raise ValueError(
            f"target_n_features_kept={target_n_features_kept} exceeds "
            f"sum(tier_sizes)={total_size}"
        )
    # Step 2-3: proportional allocation + remainder distribution.
    proportional = (
        target_n_features_kept * tier_sizes.astype(np.float64) / total_size
    )
    allocated = np.floor(proportional).astype(np.int64)
    remaining_slots = target_n_features_kept - int(allocated.sum())
    residuals = proportional - allocated
    # Largest residual wins remaining slots; lowest tier id breaks ties.
    if remaining_slots > 0:
        # argsort ascending: take from the end; lowest tier-id breaks tie via
        # stable sort (numpy's default is stable).
        order_by_residual = np.argsort(-residuals, kind="stable")
        for i in order_by_residual[:remaining_slots]:
            allocated[i] += 1
    # Step 4: within each tier, top-K by row norm.
    kept_ids: list[int] = []
    for tier_idx, tier in enumerate(unique_tiers):
        k = int(allocated[tier_idx])
        if k == 0:
            continue
        tier_features = np.flatnonzero(partition_block_ids == tier)
        # Sort tier features by descending row norm; take top-K.
        tier_norms = row_norms[tier_features]
        sorted_indices = tier_features[np.argsort(-tier_norms, kind="stable")]
        kept_ids.extend(int(fid) for fid in sorted_indices[:k])
    result = np.sort(np.array(kept_ids, dtype=np.int64))
    assert result.shape[0] == target_n_features_kept, (
        f"partition-aware slicing produced {result.shape[0]} features, "
        f"requested {target_n_features_kept}"
    )
    return result


def _run_capability_cell(
    *,
    cell: _SweepCell,
    sae_checkpoint: Path,
    host_model_id: str,
    dataset: Any,
    Y: np.ndarray,
    W_dec_full: np.ndarray,
    row_norms: np.ndarray,
    order: np.ndarray,
    host_pf_auc: np.ndarray,
    host_mauc: float,
    host_cov95: float,
    device: str,
    host_layer: "int | None" = None,
    partition_block_ids: np.ndarray | None = None,
    train_encoder: bool = False,
    train_objective: str = "proxy",
    host_encoder: Any = None,
    host_X: Any = None,
    output_dir: Path | None = None,
    basis_order: str | None = None,
    train_steps: int = 300,
    train_lr: float = 1e-3,
    train_seed: int = 0,
) -> ParetoFrontierRow:
    """Run one sweep cell. Returns a populated ParetoFrontierRow.

    When ``partition_block_ids`` is provided, the basis is sliced
    per-tier proportionally (per
    ``add-partition-encoding-capability-validation`` spec) instead
    of flat-by-row-norm. The proportional allocation uses largest-
    fractional-remainder rounding for deterministic tie-breaking.
    """
    import torch

    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector
    from saeforge.utils.host_loader import load_host_for_forge

    t0 = time.monotonic()
    n_features = cell.target_n_features_kept
    if n_features > W_dec_full.shape[0]:
        raise ValueError(
            f"sweep cell width {n_features} exceeds SAE width "
            f"{W_dec_full.shape[0]}"
        )
    if partition_block_ids is not None:
        kept = _slice_partition_aware(
            row_norms=row_norms,
            partition_block_ids=partition_block_ids,
            target_n_features_kept=n_features,
        )
    else:
        kept = np.sort(order[:n_features])
    W_dec_slice = W_dec_full[kept]
    basis = FeatureBasis(
        kept_ids=kept.astype(np.int64),
        W_dec=W_dec_slice,
        merged_norms=row_norms[kept].astype(np.float64),
        original_norms=row_norms[kept].astype(np.float64),
    )
    host = load_host_for_forge(host_model_id)
    adapter = adapter_for(host)
    W_dec_t = torch.from_numpy(W_dec_slice.astype(np.float32))

    def _forge_pf_auc(projector: "SubspaceProjector") -> np.ndarray:
        """Run the full forge with ``projector`` and return per-feature AUC vs Y (decode→encode→AUC)."""
        weights = projector.project_module(host, attention_width="host")
        config = adapter.build_native_config(host, basis.n_features)
        config.forward_mode = "native_in_basis"
        model = NativeModel.from_projected_weights(config, weights)
        model._move(dtype="float32", device=device)
        forged_h = _extract_forged_activations(
            model.torch_module, host, list(dataset.sequences), device=device,
            aggregator=dataset.aggregator, max_seq_len=512, feed=dataset.feed,
            host_layer=host_layer,
        )
        forged_d = forged_h @ W_dec_t
        with torch.no_grad():
            forge_z = dataset.encoder(forged_d.float())
        return _best_auc_per_feature(forge_z.detach().cpu().numpy(), Y)

    # Baseline (pinv) forge — always computed. The trained encoder, when on, is scored against this.
    projector = SubspaceProjector(basis=basis, scale_boost=cell.scale_boost)
    forge_pf_auc = _forge_pf_auc(projector)
    forge_mauc = float(np.nanmean(forge_pf_auc))

    # Capability-trained encoder (add-capability-trained-encoder, task 3.2): fit E on the activation proxy,
    # apply via encoder_override, and re-score the FULL forge against the pinv baseline (held-out, the gate).
    retained_mauc_trained = retained_mauc_pinv_baseline = delta_heldout = None
    encoder_trained = overfit_flag = False
    encoder_artifact_path = None
    if train_encoder:
        from saeforge.training import train_encoder as _train_encoder
        host_acts = host_X.detach().cpu().numpy() if hasattr(host_X, "detach") else np.asarray(host_X)
        sb = float(SubspaceProjector(basis=basis, scale_boost=cell.scale_boost).scale_boost)
        train_kw: dict = dict(steps=train_steps, lr=train_lr, seed=train_seed, scale_boost=sb)
        if train_objective == "full_forge":
            # Train E through the FULL differentiable forge (add-full-forge-encoder-training), not the proxy.
            from saeforge.forge_diff import DifferentiableEsm2Forge
            forge = DifferentiableEsm2Forge(host, basis, scale_boost=sb, device=device)
            train_kw.update(objective="forge_distill", forge=forge,
                            sequences=list(dataset.sequences), feed=dataset.feed)
        E, report = _train_encoder(
            basis=basis, host_acts=host_acts, host_encoder=host_encoder, labels=Y, **train_kw,
        )
        trained_proj = SubspaceProjector(basis=basis, scale_boost=cell.scale_boost, encoder_override=E)
        trained_pf_auc = _forge_pf_auc(trained_proj)
        trained_mauc = float(np.nanmean(trained_pf_auc))
        retained_mauc_trained = trained_mauc / host_mauc if host_mauc > 0 else None
        retained_mauc_pinv_baseline = forge_mauc / host_mauc if host_mauc > 0 else None
        if retained_mauc_trained is not None and retained_mauc_pinv_baseline is not None:
            delta_heldout = retained_mauc_trained - retained_mauc_pinv_baseline
        encoder_trained = True
        overfit_flag = bool(report.overfit_flag)
        if output_dir is not None:
            art = Path(output_dir) / f"{cell.encoding_label}_w{n_features}_s{cell.scale_boost}.encoder.npy"
            np.save(art, E)
            encoder_artifact_path = str(art)
    valid = np.isfinite(forge_pf_auc)
    forge_cov95 = float((forge_pf_auc[valid] >= 0.95).mean()) if valid.any() else 0.0

    # Per-feature gap distribution.
    drops = host_pf_auc - forge_pf_auc
    drops_finite = drops[np.isfinite(drops)]
    if drops_finite.size:
        gap_median = float(np.median(drops_finite))
        gap_p25 = float(np.percentile(drops_finite, 25))
        gap_p75 = float(np.percentile(drops_finite, 75))
        gap_p95 = float(np.percentile(drops_finite, 95))
        n_above_0_1 = int((drops_finite > 0.1).sum())
        n_negative = int((drops_finite < 0).sum())
    else:
        gap_median = gap_p25 = gap_p75 = gap_p95 = None
        n_above_0_1 = n_negative = None

    aggregator_label = (
        dataset.aggregator if isinstance(dataset.aggregator, str)
        else getattr(dataset.aggregator, "__name__", repr(dataset.aggregator))
    )
    elapsed = time.monotonic() - t0
    return ParetoFrontierRow(
        encoding_label=cell.encoding_label,
        target_n_features_kept=n_features,
        n_features_kept_actual=int(basis.n_features),
        pareto_reached_target=True,
        faithfulness_kl=None,
        perplexity=None,
        final_fine_tune_loss=None,
        sae_checkpoint=str(sae_checkpoint),
        forged_model_path=None,
        elapsed_seconds=elapsed,
        error_message=None,
        host_d_model=int(W_dec_full.shape[1]),
        basis_rank=int(basis.n_features),
        host_baseline_mauc=_finite_or_none(host_mauc),
        host_baseline_cov95=_finite_or_none(host_cov95),
        forge_mauc=_finite_or_none(forge_mauc),
        forge_cov95=_finite_or_none(forge_cov95),
        retained_mauc_vs_host=(
            _finite_or_none(forge_mauc / host_mauc) if host_mauc > 0 else None
        ),
        retained_cov95_vs_host=(
            _finite_or_none(forge_cov95 / host_cov95) if host_cov95 > 0 else None
        ),
        gap_median=gap_median,
        gap_p25=gap_p25,
        gap_p75=gap_p75,
        gap_p95=gap_p95,
        n_features_gap_above_0_1=n_above_0_1,
        n_features_negative_gap=n_negative,
        capability_aggregator=aggregator_label,
        capability_min_prevalence=int(dataset.min_prevalence),
        retained_mauc_trained=_finite_or_none(retained_mauc_trained),
        retained_mauc_pinv_baseline=_finite_or_none(retained_mauc_pinv_baseline),
        delta_heldout=_finite_or_none(delta_heldout),
        encoder_trained=encoder_trained,
        overfit_flag=overfit_flag,
        basis_order=basis_order,
        encoder_artifact_path=encoder_artifact_path,
    )


def _filter_labels(labels: np.ndarray, min_prevalence: int) -> np.ndarray:
    if min_prevalence <= 0:
        return labels
    n_pos = labels.sum(axis=0)
    keep = np.flatnonzero(n_pos >= min_prevalence)
    return labels[:, keep]


def _best_auc_per_feature(z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-label best-AUC over latents (Mann-Whitney rank-sum).

    Returns ``(V,)`` array; NaN for labels with no positives or no
    negatives. Mirrors the kernel in
    ``saeforge.eval.targets.downstream_capability._best_auc_per_feature``
    so the sweep wrapper doesn't need to import the target's
    internals.
    """
    n, k = z.shape
    Y_f = Y.astype(np.float64)
    n_pos = Y_f.sum(axis=0)
    n_neg = n - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    u_offset = n_pos * (n_pos + 1) / 2.0
    denom = np.where(valid, n_pos * n_neg, 1.0)
    rank_template = np.arange(1, n + 1, dtype=np.float64)[:, None]
    order = z.argsort(axis=0)
    ranks = np.empty((n, k), dtype=np.float64)
    col_idx = np.arange(k)[None, :]
    ranks[order, col_idx] = rank_template
    s_pos = Y_f.T @ ranks
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s_pos - u_offset[:, None]) / denom[:, None]
    sym = np.maximum(auc, 1.0 - auc)
    sym = np.where(valid[:, None], sym, -np.inf)
    best = sym.max(axis=1)
    return np.where(valid, best, np.nan)


def _extract_host_activations(
    host_model_id: str,
    sequences: list[str],
    aggregator: "str | Any",
    max_seq_len: int,
    device: str,
    *,
    feed: str = "pooled",
    host_layer: "int | None" = None,
):
    """Run the host over sequences; return activations shaped per ``feed``.

    When ``host_layer`` is set (**causal LM hosts**, whose SAE lives mid-model), read the residual stream at
    ``hidden_states[host_layer]`` (= ``blocks.{host_layer}.hook_resid_pre``) via ``output_hidden_states=True``,
    with **no** CLS/EOS strip (causal LMs have none). When ``host_layer is None`` the original encoder-only
    (ESM-2) path runs byte-identically (``host.esm`` / ``last_hidden_state`` / ``[0, 1:-1, :]``).

    - ``feed="pooled"`` (default): mean-pool per protein. Returns
      ``(n_proteins, d_model)``.
    - ``feed="residue"``: keep per-residue states (CLS / EOS stripped),
      concatenated across proteins. Returns
      ``(n_total_residues, d_model)``. The protein-major ordering
      matches what bio-sae's bundle's ``labels_residue_Y`` carries
      (``residue_index[:, 0]`` ascending; positions monotone within
      each protein).

    Aggregator is consumed downstream by the encoder + scoring step;
    this helper is feed-only. The two pool orders
    (``pool_then_encode`` / ``encode_then_pool``) produce identical
    host activations at this step under ``feed="pooled"`` — the
    composition difference manifests at score time.
    """
    import torch
    from transformers import AutoTokenizer

    from saeforge.utils.host_loader import load_host_for_forge

    host = load_host_for_forge(host_model_id)
    causal = host_layer is not None
    inner = host if causal else (host.esm if hasattr(host, "esm") else host)
    inner.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(host_model_id)
    chunks: list = []
    with torch.no_grad():
        for seq in sequences:  # batch-size 1 per item ([0] drops the singleton batch dim)
            enc = tokenizer(seq[:max_seq_len], return_tensors="pt").to(device)
            if causal:
                out = inner(input_ids=enc["input_ids"], output_hidden_states=True)
                # hidden_states[L] == blocks.{L}.hook_resid_pre; no CLS/EOS to strip on a causal LM.
                h = out.hidden_states[host_layer][0].cpu().float()
            else:
                out = inner(input_ids=enc["input_ids"])
                h = out.last_hidden_state[0, 1:-1, :].cpu().float()
            if feed == "pooled":
                chunks.append(h.mean(dim=0, keepdim=True))
            else:  # feed == "residue"
                chunks.append(h)  # (L_i, d_model), no pooling
    return torch.cat(chunks, dim=0)


# Causal mid-layer extraction: family → callable(forged_module, layer) returning the block whose **input**
# is ``resid_pre[layer]``. A forward-pre-hook on that block captures the forged residual at the SAE's hook
# point. NOTE the forged module runs ``native_in_basis`` (its ``hidden_size == n_features``), so the captured
# residual lives in **basis space** (width N), NOT ``d_model`` — which is exactly why the caller remaps it
# with ``forged_h @ W_dec`` (N → d_model). The pre-hook input is the forged counterpart of the tensor the
# host side reads as ``hidden_states[layer]``, keeping host/forged aligned. To extend a family, add its
# block accessor here (e.g. Llama-style: ``lambda m, layer: m.model.layers[layer]``).
_FORGED_BLOCK_ACCESSORS = {
    "gpt2": lambda m, layer: m.transformer.h[layer],
    "gpt_neox": lambda m, layer: m.gpt_neox.layers[layer],
}


def _extract_forged_activations(
    forged_module,
    host,
    sequences: list[str],
    *,
    device: str,
    aggregator: "str | Any",
    max_seq_len: int,
    feed: str = "pooled",
    host_layer: "int | None" = None,
):
    """Run forged module over sequences; shape output per ``feed``.

    Mirrors :func:`_extract_host_activations`'s shape contract. When ``host_layer`` is set (**causal LM**), the
    forged module emits only final logits, so the residual at the SAE's layer is captured via a
    forward-pre-hook on the family's block (see ``_FORGED_BLOCK_ACCESSORS``) — no forward-signature change.
    Because the forged module runs ``native_in_basis`` (``hidden_size == n_features``), the captured residual
    is in **basis space**: shape ``(seq, N)`` where ``N`` is the basis width, **not** ``d_model``. That is
    exactly why the caller's ``forged_h @ W_dec`` step (N → d_model) is needed, mirroring ESM-2 (no CLS/EOS
    strip). ``host_layer is None`` keeps the encoder-only path byte-identical.

    Extraction is **batch-size-1 per sequence** (one forward per item; the ``[0]`` / ``[0, 1:-1, :]`` indexing
    drops the singleton batch dim). True batching would need padding + an attention mask and is out of scope.
    """
    import torch
    from transformers import AutoTokenizer

    forged_module.to(device).eval()
    causal = host_layer is not None
    # Tokenizer id from the host's own config. The ESM default is a *legacy encoder-path* fallback only — on
    # the causal path we require the host's name rather than silently tokenizing LM text with an ESM vocab.
    tok_id = getattr(getattr(host, "config", None), "_name_or_path", None)
    if tok_id is None and not causal:
        tok_id = "facebook/esm2_t6_8M_UR50D"
    if tok_id is None:
        raise ValueError(
            "_extract_forged_activations: causal mid-layer extraction needs the host tokenizer id "
            "(host.config._name_or_path); none found on the host config."
        )
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    block, captured, handle = None, {}, None
    if causal:
        family = getattr(getattr(forged_module, "config", None), "family", None)
        accessor = _FORGED_BLOCK_ACCESSORS.get(family)
        if accessor is None:
            raise NotImplementedError(
                f"_extract_forged_activations: causal mid-layer extraction is implemented for "
                f"{sorted(_FORGED_BLOCK_ACCESSORS)}; family {family!r} is a follow-up — add its block "
                f"accessor to _FORGED_BLOCK_ACCESSORS (the block whose input is resid_pre[layer])."
            )
        block = accessor(forged_module, host_layer)

        def _prehook(_mod, args):
            captured["h"] = args[0].detach()  # block INPUT == basis-space resid_pre[host_layer]

        handle = block.register_forward_pre_hook(_prehook)

    chunks: list = []
    try:
        with torch.no_grad():
            for seq in sequences:  # batch-size 1 per item (see docstring)
                enc = tokenizer(seq[:max_seq_len], return_tensors="pt").to(device)
                if causal:
                    forged_module(enc["input_ids"])  # logits discarded; the hook captures resid_pre
                    h = captured["h"][0].cpu().float()  # (seq, N) basis-space, no strip
                else:
                    h = forged_module(enc["input_ids"])[0, 1:-1, :].cpu().float()
                if feed == "pooled":
                    chunks.append(h.mean(dim=0, keepdim=True))
                else:  # feed == "residue"
                    chunks.append(h)
    finally:
        if handle is not None:
            handle.remove()
    return torch.cat(chunks, dim=0)
