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


def sweep_pareto_capability(
    sae_checkpoint: "str | Path",
    host_model_id: str,
    dataset: Any,  # CapabilityDataset
    *,
    widths: list[int],
    encodings: list[str] | None = None,
    scale_boosts: list["float | str"] | None = None,
    output_dir: "str | Path",
    cache_host: bool = True,
    max_seq_len: int = 512,
    device: str = "cpu",
) -> list[ParetoFrontierRow]:
    """Run a capability-aware Pareto sweep over the basis-config cube.

    Parameters
    ----------
    sae_checkpoint:
        Path to the SAE state dict (the same artifact
        :class:`CapabilityDataset.from_bio_sae` consumed). Used here
        to construct the basis at each width.
    host_model_id:
        HF id of the host model to forge.
    dataset:
        :class:`saeforge.datasets.CapabilityDataset` carrying
        sequences + labels + encoder + aggregator + min_prevalence.
    widths:
        List of basis widths to sweep. Each width slices the SAE's
        W_dec to its top-N rows by L2 norm.
    encodings:
        List of encoding labels (currently informational; defaults
        to ``["raw_slice"]`` because this wrapper uses a row-norm
        slice, not a polygram encoding). Future revisions can hook
        polygram-compressed bases per encoding.
    scale_boosts:
        List of SubspaceProjector scale_boost values. Floats or
        the literal ``"auto"``. Defaults to ``[1.0]``.
    output_dir:
        Where to write ``frontier.jsonl`` and the host-extraction
        cache.
    cache_host:
        When True (default), cache host activations across cells.
        Opt-out for non-deterministic hosts or scarce disk.

    Returns
    -------
    List of :class:`ParetoFrontierRow`, one per cell, with the
    capability fields populated. Also written to
    ``output_dir/frontier.jsonl``.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sae_checkpoint = Path(sae_checkpoint)
    if encodings is None or not encodings:
        encodings = ["raw_slice"]
    if scale_boosts is None or not scale_boosts:
        scale_boosts = [1.0]

    # Validate dataset shape early (catches bad fixtures before we
    # pay any forge cost).
    n_items = dataset.labels.shape[0]
    if len(dataset.sequences) != n_items:
        raise ValueError(
            f"sweep_pareto_capability: dataset.sequences ({len(dataset.sequences)}) "
            f"and dataset.labels ({n_items} rows) must align"
        )

    # ---- Stage 1: host-extraction cache + once-per-sweep host load. ----
    cache = HostExtractionCache(
        output_dir / "host_cache",
        enabled=cache_host,
    )
    key = HostCacheKey.from_inputs(
        host_model_id=host_model_id,
        sequences=list(dataset.sequences),
        aggregator=dataset.aggregator,
        max_seq_len=max_seq_len,
    )

    host_X: torch.Tensor | None = None
    if cache.has(key):
        host_X = cache.load(key)
    # Lazy-load host only if we need to extract (cache miss).

    # ---- Stage 2: load SAE state for basis construction. ----
    sae_state = torch.load(str(sae_checkpoint), map_location="cpu", weights_only=True)
    W_dec_full = sae_state["decoder.weight"].numpy().T.astype(np.float64)  # (n, d)
    row_norms = np.linalg.norm(W_dec_full, axis=1)
    order = np.argsort(-row_norms)

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
        )
        cache.save(key, host_X)

    Y = _filter_labels(dataset.labels, dataset.min_prevalence)
    with torch.no_grad():
        host_z = dataset.encoder(host_X.float())
    host_pf_auc = _best_auc_per_feature(host_z.detach().cpu().numpy(), Y)
    host_mauc = float(np.nanmean(host_pf_auc))
    valid = np.isfinite(host_pf_auc)
    host_cov95 = float((host_pf_auc[valid] >= 0.95).mean()) if valid.any() else 0.0

    # ---- Stage 4: sweep cells. ----
    cells = [
        _SweepCell(encoding_label=e, target_n_features_kept=w, scale_boost=s)
        for e in encodings for w in widths for s in scale_boosts
    ]
    rows: list[ParetoFrontierRow] = []
    frontier_path = output_dir / "frontier.jsonl"
    if frontier_path.exists():
        frontier_path.unlink()  # fresh file per invocation

    for cell in cells:
        t0 = time.monotonic()
        try:
            row = _run_capability_cell(
                cell=cell,
                sae_checkpoint=sae_checkpoint,
                host_model_id=host_model_id,
                dataset=dataset,
                Y=Y,
                W_dec_full=W_dec_full,
                row_norms=row_norms,
                order=order,
                host_pf_auc=host_pf_auc,
                host_mauc=host_mauc,
                host_cov95=host_cov95,
                device=device,
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
                sae_checkpoint=str(sae_checkpoint),
                forged_model_path=None,
                elapsed_seconds=time.monotonic() - t0,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
        with frontier_path.open("a") as fh:
            fh.write(json.dumps(row.to_json_dict()) + "\n")

    return rows


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
) -> ParetoFrontierRow:
    """Run one sweep cell. Returns a populated ParetoFrontierRow."""
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
    kept = np.sort(order[:n_features])
    W_dec_slice = W_dec_full[kept]
    basis = FeatureBasis(
        kept_ids=kept.astype(np.int64),
        W_dec=W_dec_slice,
        merged_norms=row_norms[kept].astype(np.float64),
        original_norms=row_norms[kept].astype(np.float64),
    )
    projector = SubspaceProjector(basis=basis, scale_boost=cell.scale_boost)

    host = load_host_for_forge(host_model_id)
    adapter = adapter_for(host)
    weights = projector.project_module(host, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    config.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(config, weights)
    model._move(dtype="float32", device=device)

    forged_h = _extract_forged_activations(
        model.torch_module,
        host,
        list(dataset.sequences),
        device=device,
        aggregator=dataset.aggregator,
        max_seq_len=512,
    )
    W_dec_t = torch.from_numpy(W_dec_slice.astype(np.float32))
    forged_d = forged_h @ W_dec_t
    with torch.no_grad():
        forge_z = dataset.encoder(forged_d.float())
    forge_pf_auc = _best_auc_per_feature(forge_z.detach().cpu().numpy(), Y)
    forge_mauc = float(np.nanmean(forge_pf_auc))
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
):
    """Run host ESM-2 over sequences; return per-protein activations
    aggregated according to ``aggregator``.

    Only ``pool_then_encode`` and ``encode_then_pool`` are honored at
    the host-extraction step (the encoder runs downstream in
    ``sweep_pareto_capability`` regardless of aggregator). Returns
    shape ``(n_proteins, d_model)`` in both cases — the encoder + pool
    composition difference manifests at score time, not at host
    extraction.
    """
    import torch
    from transformers import AutoTokenizer

    from saeforge.utils.host_loader import load_host_for_forge

    host = load_host_for_forge(host_model_id)
    inner = host.esm if hasattr(host, "esm") else host
    inner.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(host_model_id)
    chunks: list = []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq[:max_seq_len], return_tensors="pt").to(device)
            out = inner(input_ids=enc["input_ids"])
            h = out.last_hidden_state[0, 1:-1, :].cpu().float()
            chunks.append(h.mean(dim=0, keepdim=True))
    return torch.cat(chunks, dim=0)  # (n_proteins, d_model)


def _extract_forged_activations(
    forged_module,
    host,
    sequences: list[str],
    *,
    device: str,
    aggregator: "str | Any",
    max_seq_len: int,
):
    """Run forged module over sequences; mean-pool per protein."""
    import torch
    from transformers import AutoTokenizer

    forged_module.to(device).eval()
    tok_id = (
        getattr(host.config, "_name_or_path", None)
        or "facebook/esm2_t6_8M_UR50D"
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    chunks: list = []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq[:max_seq_len], return_tensors="pt").to(device)
            h = forged_module(enc["input_ids"])[0, 1:-1, :].cpu().float()
            chunks.append(h.mean(dim=0, keepdim=True))
    return torch.cat(chunks, dim=0)
