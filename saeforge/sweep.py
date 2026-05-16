"""Pareto sweep driver — forge across per-K materialised SAE checkpoints.

Consumes the artifacts produced by ``polygram compress --pareto --pareto-materialize``
(a ``pareto.json`` manifest plus ``pareto/k_{K}.safetensors`` files), runs the
forge pipeline once per K (optionally across multiple labelled encodings), and
emits one JSONL row per ``(encoding, target_n_features_kept)`` capturing kept-feature
count, downstream KL, perplexity, and faithfulness.

The driver is sequential, resumable via append-only JSONL scan, and isolates
per-row failures: one bad row writes ``error_message`` and the sweep continues;
``sweep_pareto`` raises ``RuntimeError`` at the end if any row errored.

See ``openspec/specs/pareto-sweep/spec.md`` for the row contract and lifecycle
states (success / frontier-only / row failure).
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from saeforge.auto_materialise import AutoMaterialiseSpec  # noqa: F401
    from saeforge.forge import ForgePipeline
    from saeforge.forge_quality import QualityThresholds  # noqa: F401


_K_FROM_FILENAME = re.compile(r"^k_(\d+)\.safetensors$")


@dataclass(frozen=True)
class ParetoFrontierRow:
    """One row of the sweep frontier output.

    Three lifecycle states are normative (see the spec table): **success**
    (forge ran), **frontier-only** (``frontier_only=True``, no forge), and
    **row failure** (forge raised). Downstream consumers SHALL filter on
    ``error_message is None`` before reading metric fields.
    """

    encoding_label: str
    target_n_features_kept: int
    n_features_kept_actual: int | None
    pareto_reached_target: bool | None
    faithfulness_kl: float | None
    perplexity: float | None
    final_fine_tune_loss: float | None
    sae_checkpoint: str
    forged_model_path: str | None
    elapsed_seconds: float
    error_message: str | None
    # Forge-feasibility diagnostics, populated when the sweep can resolve
    # the host's residual width. See ``saeforge.forge_quality`` and the
    # ``add-forge-quality-diagnostics`` capability for the contract.
    host_d_model: int | None = None
    basis_rank: int | None = None
    quality_ratio: float | None = None
    quality_tier: str | None = None
    # Methodological provenance, populated when the sweep ran under
    # `--auto-materialise`. See ``saeforge.auto_materialise`` and the
    # ``add-auto-materialise-sweep`` capability for the contract.
    validation_threshold: float | None = None
    encoding_class: str | None = None
    validation_eval_overlap: bool | None = None

    def __post_init__(self) -> None:
        if int(self.target_n_features_kept) < 1:
            raise ValueError(
                f"ParetoFrontierRow: target_n_features_kept must be >= 1; "
                f"got {self.target_n_features_kept}"
            )
        if float(self.elapsed_seconds) < 0:
            raise ValueError(
                f"ParetoFrontierRow: elapsed_seconds must be >= 0; "
                f"got {self.elapsed_seconds}"
            )
        if self.n_features_kept_actual is not None and int(self.n_features_kept_actual) < 0:
            raise ValueError(
                f"ParetoFrontierRow: n_features_kept_actual must be >= 0 or None; "
                f"got {self.n_features_kept_actual}"
            )
        if self.host_d_model is not None and int(self.host_d_model) < 1:
            raise ValueError(
                f"ParetoFrontierRow: host_d_model must be >= 1 or None; "
                f"got {self.host_d_model}"
            )
        if self.basis_rank is not None and int(self.basis_rank) < 0:
            raise ValueError(
                f"ParetoFrontierRow: basis_rank must be >= 0 or None; "
                f"got {self.basis_rank}"
            )
        if self.quality_ratio is not None and float(self.quality_ratio) < 0:
            raise ValueError(
                f"ParetoFrontierRow: quality_ratio must be >= 0 or None; "
                f"got {self.quality_ratio}"
            )
        if self.quality_tier is not None:
            # Lazy-imported to avoid circular import with forge_quality if
            # this module is ever consumed before forge_quality finishes.
            from saeforge.forge_quality import QualityTier

            valid = {t.value for t in QualityTier}
            if self.quality_tier not in valid:
                raise ValueError(
                    f"ParetoFrontierRow: quality_tier must be one of "
                    f"{sorted(valid)} or None; got {self.quality_tier!r}"
                )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "encoding_label": self.encoding_label,
            "target_n_features_kept": int(self.target_n_features_kept),
            "n_features_kept_actual": (
                int(self.n_features_kept_actual)
                if self.n_features_kept_actual is not None
                else None
            ),
            "pareto_reached_target": self.pareto_reached_target,
            "faithfulness_kl": _finite_or_none(self.faithfulness_kl),
            "perplexity": _finite_or_none(self.perplexity),
            "final_fine_tune_loss": _finite_or_none(self.final_fine_tune_loss),
            "sae_checkpoint": str(self.sae_checkpoint),
            "forged_model_path": (
                str(self.forged_model_path)
                if self.forged_model_path is not None
                else None
            ),
            "elapsed_seconds": float(self.elapsed_seconds),
            "error_message": self.error_message,
            "host_d_model": (
                int(self.host_d_model) if self.host_d_model is not None else None
            ),
            "basis_rank": (
                int(self.basis_rank) if self.basis_rank is not None else None
            ),
            "quality_ratio": _finite_or_none(self.quality_ratio),
            "quality_tier": self.quality_tier,
            "validation_threshold": _finite_or_none(self.validation_threshold),
            "encoding_class": self.encoding_class,
            "validation_eval_overlap": self.validation_eval_overlap,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "ParetoFrontierRow":
        return cls(
            encoding_label=str(data["encoding_label"]),
            target_n_features_kept=int(data["target_n_features_kept"]),
            n_features_kept_actual=(
                int(data["n_features_kept_actual"])
                if data.get("n_features_kept_actual") is not None
                else None
            ),
            pareto_reached_target=data.get("pareto_reached_target"),
            faithfulness_kl=(
                float(data["faithfulness_kl"])
                if data.get("faithfulness_kl") is not None
                else None
            ),
            perplexity=(
                float(data["perplexity"])
                if data.get("perplexity") is not None
                else None
            ),
            final_fine_tune_loss=(
                float(data["final_fine_tune_loss"])
                if data.get("final_fine_tune_loss") is not None
                else None
            ),
            sae_checkpoint=str(data["sae_checkpoint"]),
            forged_model_path=(
                str(data["forged_model_path"])
                if data.get("forged_model_path") is not None
                else None
            ),
            elapsed_seconds=float(data["elapsed_seconds"]),
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
            host_d_model=(
                int(data["host_d_model"])
                if data.get("host_d_model") is not None
                else None
            ),
            basis_rank=(
                int(data["basis_rank"])
                if data.get("basis_rank") is not None
                else None
            ),
            quality_ratio=(
                float(data["quality_ratio"])
                if data.get("quality_ratio") is not None
                else None
            ),
            quality_tier=(
                str(data["quality_tier"])
                if data.get("quality_tier") is not None
                else None
            ),
            validation_threshold=(
                float(data["validation_threshold"])
                if data.get("validation_threshold") is not None
                else None
            ),
            encoding_class=(
                str(data["encoding_class"])
                if data.get("encoding_class") is not None
                else None
            ),
            validation_eval_overlap=(
                bool(data["validation_eval_overlap"])
                if data.get("validation_eval_overlap") is not None
                else None
            ),
        )


def _finite_or_none(x: float | None) -> float | None:
    """JSON-friendly: convert non-finite floats to None for stable JSONL."""
    if x is None:
        return None
    f = float(x)
    if f != f or f == float("inf") or f == float("-inf"):  # noqa: PLR0124
        return None
    return f


# ---------------------------------------------------------------------------
# Manifest + checkpoint enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ManifestEntry:
    target_k: int
    n_features_kept: int
    reached_target: bool


def _load_pareto_manifest(checkpoint_dir: Path) -> dict[int, _ManifestEntry]:
    """Load ``pareto.json`` from the directory or its parent.

    ``polygram compress --pareto-materialize --out <dir>`` writes
    ``<dir>/pareto.json`` (the manifest) and ``<dir>/pareto/k_{K}.safetensors``
    (the per-K SAEs). The driver accepts either ``<dir>`` or ``<dir>/pareto``
    as ``--encoding LABEL:PATH``, so look in both.

    Returns ``{}`` when no manifest is found — callers fall back to counting
    surviving features from the SAE checkpoint directly.
    """
    candidates = [checkpoint_dir / "pareto.json", checkpoint_dir.parent / "pareto.json"]
    for p in candidates:
        if p.is_file():
            return _parse_pareto_manifest(p)
    return {}


def _parse_pareto_manifest(path: Path) -> dict[int, _ManifestEntry]:
    """Parse polygram's ``pareto.json`` into a per-K lookup.

    The JSON schema (polygram 0.4.0 ``ParetoReport._serialize`` /
    ``_outcome_to_dict``) emits each outcome as a flat object with
    ``target_k``, ``reached_target``, ``clusters`` (list of cluster dicts),
    and ``feature_ids``. ``n_features_kept`` is the count of cluster
    representatives — ``len(outcome.clusters)`` — matching polygram's
    own ``CompressionPlan.n_features_kept`` semantic (one survivor per
    cluster, plus all the singleton features outside any cluster which
    are not modelled here because the manifest's `feature_ids` already
    enumerates only the features touched by compression).

    We parse the JSON directly rather than calling
    ``polygram.ParetoReport.from_json`` so a schema mismatch surfaces as
    a focused ``KeyError`` rather than an opaque polygram-side validation
    error.
    """
    payload = json.loads(path.read_text())
    out: dict[int, _ManifestEntry] = {}
    for outcome in payload.get("outcomes", []):
        target_k = int(outcome["target_k"])
        out[target_k] = _ManifestEntry(
            target_k=target_k,
            n_features_kept=int(len(outcome["clusters"])),
            reached_target=bool(outcome["reached_target"]),
        )
    return out


def _count_surviving_features(sae_checkpoint: Path) -> int:
    """Count non-zero feature rows in ``W_dec`` of a polygram-compressed SAE.

    Used as the fallback when ``pareto.json`` is absent. Reads
    ``safetensors`` directly rather than going through
    ``polygram.sae_import``, so the sweep driver doesn't require polygram at
    enumeration time (CI on the no-extras install would otherwise return
    None here and miss the fallback). The `W_dec` key contract is the same
    one ``FeatureBasis.from_polygram_checkpoint`` reads from.
    """
    from safetensors.numpy import load_file

    state = load_file(str(sae_checkpoint))
    if "W_dec" not in state:
        raise KeyError(f"sweep: {sae_checkpoint} is missing required key 'W_dec'")
    w_dec = state["W_dec"]
    # Survivor = any non-zero entry on the decoder row.
    nonzero_rows = (w_dec != 0).any(axis=1)
    return int(nonzero_rows.sum())


def _enumerate_checkpoints(
    encoding_path: Path,
) -> list[tuple[int, Path]]:
    """Resolve a per-encoding path into a list of ``(K, ckpt_path)``.

    ``encoding_path`` is either a single ``.safetensors`` file (degenerate
    single-K sweep — K is determined later from the SAE metadata) or a
    directory. Directories are searched at the root and under a ``pareto/``
    subdirectory for files matching ``k_{K}.safetensors``.

    Returns the list sorted ascending by K. Raises ``FileNotFoundError`` when
    nothing matches.
    """
    if encoding_path.is_file():
        if not encoding_path.name.endswith(".safetensors"):
            raise ValueError(
                f"sweep_pareto: --encoding path {encoding_path} is a file but "
                f"does not end in .safetensors"
            )
        # Single-file: K read from SAE metadata downstream (see _resolve_single_file_k).
        return [(_resolve_single_file_k(encoding_path), encoding_path)]

    if not encoding_path.is_dir():
        raise FileNotFoundError(
            f"sweep_pareto: --encoding path does not exist: {encoding_path}"
        )

    # Directory: enumerate k_{K}.safetensors files at the root and under pareto/.
    found: list[tuple[int, Path]] = []
    search_dirs = [encoding_path, encoding_path / "pareto"]
    seen_k: set[int] = set()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for child in sorted(d.iterdir()):
            m = _K_FROM_FILENAME.match(child.name)
            if m is None:
                continue
            k = int(m.group(1))
            if k in seen_k:
                continue  # prefer the first directory in search order
            seen_k.add(k)
            found.append((k, child))

    if not found:
        raise FileNotFoundError(
            f"sweep_pareto: no k_<K>.safetensors files under {encoding_path} "
            f"or {encoding_path / 'pareto'}"
        )
    found.sort(key=lambda kv: kv[0])
    return found


def _resolve_single_file_k(path: Path) -> int:
    """For a single ``.safetensors`` file passed as ``--encoding LABEL:FILE``,
    count surviving features and treat that as ``target_n_features_kept``.

    The contract: row's ``target_n_features_kept`` equals the actual survivor
    count, ``n_features_kept_actual`` equals the same, ``pareto_reached_target``
    is ``None`` (no manifest).
    """
    return _count_surviving_features(path)


# ---------------------------------------------------------------------------
# Resumability — append-only JSONL scan
# ---------------------------------------------------------------------------


def _load_completed_rows(frontier_path: Path) -> set[tuple[str, int]]:
    """Read existing ``frontier.jsonl`` and return ``{(label, K), ...}``.

    Truncated last lines (mid-write crashes) are discarded and the file is
    rewritten without them so subsequent appends produce a cleanly parseable
    file. Failure rows (``error_message`` populated) are NOT counted as
    completed — they are retryable.
    """
    if not frontier_path.is_file():
        return set()

    lines = frontier_path.read_text().splitlines(keepends=False)
    completed: set[tuple[str, int]] = set()
    last_index_valid = -1
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Truncated final line — drop everything from here on.
            break
        last_index_valid = idx
        if row.get("error_message") is None:
            completed.add(
                (str(row["encoding_label"]), int(row["target_n_features_kept"]))
            )

    # Rewrite if we trimmed anything (truncated trailing line, or stray blanks
    # after the last valid line).
    if last_index_valid + 1 < len(lines):
        keep = [line for line in lines[: last_index_valid + 1] if line.strip()]
        frontier_path.write_text("\n".join(keep) + ("\n" if keep else ""))

    return completed


# ---------------------------------------------------------------------------
# Pipeline basis swap (per row)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _basis_swap(pipeline: "ForgePipeline", sae_checkpoint: Path) -> Iterator[None]:
    """Temporarily rebuild ``pipeline.basis`` and ``pipeline.projector`` from
    ``sae_checkpoint`` for the duration of one forge call.

    ``ForgePipeline`` is bound to a specific basis at construction time. To
    sweep multiple SAEs through one pipeline (reusing its host model, eval
    config, fine-tune knobs, …) we hot-swap the basis + projector around each
    ``pipeline.run`` call and restore the originals afterwards. Byte-identity
    with a freshly-constructed pipeline holds because the same
    ``FeatureBasis.from_polygram_checkpoint`` + ``SubspaceProjector(basis)``
    factories are used.
    """
    from saeforge.basis import FeatureBasis
    from saeforge.projector import SubspaceProjector

    original_basis = pipeline.basis
    original_projector = pipeline.projector
    pipeline.basis = FeatureBasis.from_polygram_checkpoint(sae_checkpoint)
    pipeline.projector = SubspaceProjector(pipeline.basis)
    try:
        yield
    finally:
        pipeline.basis = original_basis
        pipeline.projector = original_projector


# ---------------------------------------------------------------------------
# Per-row diagnostics
# ---------------------------------------------------------------------------


def _compute_row_diagnostics(
    ckpt_path: Path,
    host_d_model: int | None,
    thresholds: "QualityThresholds | None",
) -> tuple[int | None, float | None, str | None]:
    """Return ``(basis_rank, quality_ratio, quality_tier_value)`` for a row.

    All three values are ``None`` when ``host_d_model`` is unresolved.
    Otherwise ``basis_rank`` is computed from the SAE checkpoint's
    surviving-feature ``W_dec`` rows, and the ratio + tier follow.
    """
    if host_d_model is None:
        return None, None, None
    try:
        from saeforge.forge_quality import (
            basis_rank_from_safetensors,
            classify_quality,
        )

        basis_rank = basis_rank_from_safetensors(ckpt_path)
    except Exception:  # noqa: BLE001 — diagnostic, not load-bearing
        return None, None, None
    if basis_rank == 0:
        return 0, 0.0, "degenerate"
    ratio, tier = classify_quality(basis_rank, host_d_model, thresholds)
    return basis_rank, ratio, tier.value


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def sweep_pareto(
    pipeline: "ForgePipeline",
    *,
    encodings: list[tuple[str, Path]],
    output_dir: Path,
    frontier_only: bool = False,
    quality_floor: float | None = None,
    quality_thresholds: "QualityThresholds | None" = None,
    host_d_model_override: int | None = None,
    auto_materialise_specs: "list[AutoMaterialiseSpec] | None" = None,
    validation_prompts: Path | None = None,
    validation_threshold: float = 0.7,
    validation_jaccard_threshold: float = 0.3,
    layer: int | None = None,
    targets: list[int] | None = None,
    score_field: str = "polygram_overlap",
    rep_selection: str = "scale_aware",
    assign_phase_knobs: bool = False,
    validation_eval_overlap: bool = False,
    force_rematerialise: bool = False,
    plan_only: bool = False,
    **forge_kwargs: Any,
) -> Path:
    """Run the forge pipeline across per-K SAE checkpoints.

    Parameters
    ----------
    pipeline:
        A constructed :class:`saeforge.forge.ForgePipeline`. Its host model,
        eval prompts, fine-tune knobs, etc. are reused for every row.
    encodings:
        List of ``(label, path)`` tuples. ``path`` is either a
        ``.safetensors`` file (degenerate single-K row) or a directory
        containing ``k_{K}.safetensors`` files (and optionally a
        ``pareto.json`` manifest).
    output_dir:
        Sweep output root. ``frontier.jsonl`` is written here; per-row forge
        outputs land under ``<output_dir>/<label>/k_{K}/``.
    frontier_only:
        When ``True``, do not invoke ``pipeline.run`` — emit rows with only
        the manifest-derived fields populated.
    **forge_kwargs:
        Passed through to ``pipeline.run`` per row.

    Returns
    -------
    Path
        Absolute path to the resulting ``frontier.jsonl``.

    Raises
    ------
    RuntimeError
        At the *end* of the sweep if any row's forge raised. Per-row failures
        do not abort — they are recorded with ``error_message`` populated and
        the sweep continues. The error names the count of failed rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = output_dir / "frontier.jsonl"

    # Auto-materialise pre-step. When auto_materialise_specs is supplied,
    # run polygram's validator + plan_pareto + apply chain per spec,
    # writing artifacts to `<output_dir>/_materialised/<label>/`. Then
    # override the `encodings` arg's interpretation: each (label, path)
    # gets remapped to the materialised dir for the subsequent sweep
    # loop. Methodological provenance fields (validation_threshold,
    # encoding_class, validation_eval_overlap) are accumulated per label
    # and propagated to every row.
    per_label_provenance: dict[str, dict[str, Any]] = {}
    if auto_materialise_specs is not None:
        if validation_prompts is None:
            raise ValueError(
                "sweep_pareto: auto_materialise_specs requires "
                "validation_prompts to be set"
            )
        if layer is None:
            raise ValueError(
                "sweep_pareto: auto_materialise_specs requires layer to be set"
            )
        if targets is None or not targets:
            raise ValueError(
                "sweep_pareto: auto_materialise_specs requires non-empty targets"
            )
        host_model_id = getattr(pipeline, "host_model_id", None)
        if host_model_id is None:
            raise ValueError(
                "sweep_pareto: auto_materialise_specs requires pipeline.host_model_id"
            )

        from saeforge.auto_materialise import (
            AutoMaterialiseSpec as _AutoMaterialiseSpec,  # noqa: F401
            estimate_prompt_token_count,
            format_plan_only_block,
            is_cache_hit as _is_cache_hit,
            materialise as _materialise,
            compute_cache_key as _compute_cache_key,
        )

        # --plan-only: short-circuit BEFORE doing any expensive work.
        if plan_only:
            plan_blocks: list[str] = []
            for spec in auto_materialise_specs:
                cache_key = _compute_cache_key(
                    spec=spec,
                    validation_prompts_path=validation_prompts,
                    validation_threshold=validation_threshold,
                    jaccard_threshold=validation_jaccard_threshold,
                    layer=layer,
                    model_name=host_model_id,
                    targets=targets,
                    score_field=score_field,
                    rep_selection=rep_selection,
                    assign_phase_knobs=assign_phase_knobs,
                )
                materialised_dir = output_dir / "_materialised" / spec.label
                if force_rematerialise:
                    cache_hit, diff_fields = False, ["forced"]
                else:
                    cache_hit, diff_fields = _is_cache_hit(materialised_dir, cache_key)
                n_prompts, avg_tokens = estimate_prompt_token_count(validation_prompts)
                plan_blocks.append(
                    format_plan_only_block(
                        spec=spec,
                        cache_key=cache_key,
                        diff_fields=diff_fields,
                        cache_hit=cache_hit,
                        n_prompts=n_prompts,
                        avg_prompt_tokens=avg_tokens,
                    )
                )

            print("sweep-pareto --plan-only: per-encoding plan", file=sys.stderr)
            for block in plan_blocks:
                print(block, file=sys.stderr)
            return frontier_path  # No frontier.jsonl written under --plan-only

        # Real materialisation pass — produces per-label materialised dirs.
        remapped_encodings: list[tuple[str, Path]] = []
        for spec in auto_materialise_specs:
            materialised_dir, cache_key, _diff = _materialise(
                spec,
                validation_prompts_path=validation_prompts,
                validation_threshold=validation_threshold,
                jaccard_threshold=validation_jaccard_threshold,
                layer=layer,
                model_name=host_model_id,
                targets=targets,
                score_field=score_field,
                rep_selection=rep_selection,
                output_root=output_dir,
                force_rematerialise=force_rematerialise,
                assign_phase_knobs=assign_phase_knobs,
            )
            remapped_encodings.append((spec.label, materialised_dir))
            per_label_provenance[spec.label] = {
                "validation_threshold": float(validation_threshold),
                "encoding_class": spec.encoding_class,
                "validation_eval_overlap": bool(validation_eval_overlap),
            }
        encodings = remapped_encodings

    # Forge-quality diagnostics: resolve host d_model once, build the
    # advisory, enforce --quality-floor if set. All three are best-effort
    # (skipped silently when host d_model can't be resolved).
    from saeforge.forge_quality import (
        QualityThresholds as _QualityThresholds,
        advise_sweep_quality,
        basis_rank_from_safetensors,
        resolve_host_d_model,
    )

    thresholds = quality_thresholds if quality_thresholds is not None else _QualityThresholds()
    host_d_model: int | None
    host_model_type: str | None = None
    if host_d_model_override is not None:
        host_d_model = int(host_d_model_override)
    else:
        host_model_id = getattr(pipeline, "host_model_id", None)
        if host_model_id is not None:
            host_d_model, host_model_type = resolve_host_d_model(host_model_id)
        else:
            host_d_model = None

    if host_d_model is not None:
        advisory = advise_sweep_quality(
            encodings=encodings,
            host_d_model=host_d_model,
            thresholds=thresholds,
            manifest_loader=_load_pareto_manifest,
            basis_rank_loader=basis_rank_from_safetensors,
            model_type=host_model_type,
        )
        if advisory is not None:
            print(advisory, file=sys.stderr)

        if quality_floor is not None:
            # Refuse the sweep BEFORE any forge work when any encoding's
            # smallest-K basis falls below the floor.
            for label, enc_path in encodings:
                checkpoints = _enumerate_checkpoints(Path(enc_path))
                smallest_k, smallest_ckpt = checkpoints[0]
                try:
                    rank = basis_rank_from_safetensors(smallest_ckpt)
                except Exception:  # noqa: BLE001
                    continue
                ratio = rank / host_d_model
                if ratio < quality_floor:
                    raise RuntimeError(
                        f"sweep_pareto: quality_floor={quality_floor} rejects "
                        f"encoding={label!r} at smallest K={smallest_k} "
                        f"(basis_rank={rank}, host_d_model={host_d_model}, "
                        f"ratio={ratio:.4f}). Re-run with a higher K floor "
                        f"or drop the --quality-floor flag to proceed anyway."
                    )

    completed = _load_completed_rows(frontier_path)
    failures = 0

    with frontier_path.open("a") as fh:
        for label, enc_path in encodings:
            checkpoints = _enumerate_checkpoints(Path(enc_path))
            manifest = _load_pareto_manifest(Path(enc_path))

            for target_k, ckpt_path in checkpoints:
                if (label, target_k) in completed:
                    continue

                entry = manifest.get(target_k)
                basis_rank, quality_ratio, quality_tier = _compute_row_diagnostics(
                    ckpt_path, host_d_model, thresholds
                )
                provenance = per_label_provenance.get(label, {})
                row = _process_row(
                    pipeline=pipeline,
                    label=label,
                    target_k=target_k,
                    ckpt_path=ckpt_path,
                    manifest_entry=entry,
                    sweep_output_dir=output_dir,
                    frontier_only=frontier_only,
                    forge_kwargs=forge_kwargs,
                    host_d_model=host_d_model,
                    basis_rank=basis_rank,
                    quality_ratio=quality_ratio,
                    quality_tier=quality_tier,
                    provenance_validation_threshold=provenance.get("validation_threshold"),
                    provenance_encoding_class=provenance.get("encoding_class"),
                    provenance_validation_eval_overlap=provenance.get(
                        "validation_eval_overlap"
                    ),
                )
                fh.write(json.dumps(row.to_json_dict()) + "\n")
                fh.flush()
                if row.error_message is not None:
                    failures += 1

    if failures > 0:
        raise RuntimeError(
            f"sweep_pareto: {failures} row(s) failed; see "
            f"{frontier_path} for details"
        )
    return frontier_path


def _process_row(
    *,
    pipeline: "ForgePipeline",
    label: str,
    target_k: int,
    ckpt_path: Path,
    manifest_entry: _ManifestEntry | None,
    sweep_output_dir: Path,
    frontier_only: bool,
    forge_kwargs: dict[str, Any],
    host_d_model: int | None = None,
    basis_rank: int | None = None,
    quality_ratio: float | None = None,
    quality_tier: str | None = None,
    provenance_validation_threshold: float | None = None,
    provenance_encoding_class: str | None = None,
    provenance_validation_eval_overlap: bool | None = None,
) -> ParetoFrontierRow:
    """Build one frontier row — manifest-only when ``frontier_only``, otherwise
    invoke ``pipeline.run`` inside a try/except.

    The four diagnostic fields (``host_d_model``, ``basis_rank``,
    ``quality_ratio``, ``quality_tier``) are computed pre-forge by the caller
    and passed in; this function just propagates them onto every emitted row
    regardless of lifecycle state.
    """
    n_features_actual: int | None
    reached: bool | None
    if manifest_entry is not None:
        n_features_actual = manifest_entry.n_features_kept
        reached = manifest_entry.reached_target
    else:
        # Fallback: count surviving features from the SAE directly.
        try:
            n_features_actual = _count_surviving_features(ckpt_path)
        except Exception:  # noqa: BLE001 — best-effort fallback
            n_features_actual = None
        reached = None

    if frontier_only:
        return ParetoFrontierRow(
            encoding_label=label,
            target_n_features_kept=target_k,
            n_features_kept_actual=n_features_actual,
            pareto_reached_target=reached,
            faithfulness_kl=None,
            perplexity=None,
            final_fine_tune_loss=None,
            sae_checkpoint=str(ckpt_path.resolve()),
            forged_model_path=None,
            elapsed_seconds=0.0,
            error_message=None,
            host_d_model=host_d_model,
            basis_rank=basis_rank,
            quality_ratio=quality_ratio,
            quality_tier=quality_tier,
            validation_threshold=provenance_validation_threshold,
            encoding_class=provenance_encoding_class,
            validation_eval_overlap=provenance_validation_eval_overlap,
        )

    row_output_dir = sweep_output_dir / label / f"k_{target_k}"
    started = time.monotonic()
    try:
        with _basis_swap(pipeline, ckpt_path):
            result = pipeline.run(
                output_dir=row_output_dir,
                **forge_kwargs,
            )
    except Exception as exc:  # noqa: BLE001 — per-row isolation by design
        elapsed = time.monotonic() - started
        return ParetoFrontierRow(
            encoding_label=label,
            target_n_features_kept=target_k,
            n_features_kept_actual=n_features_actual,
            pareto_reached_target=reached,
            faithfulness_kl=None,
            perplexity=None,
            final_fine_tune_loss=None,
            sae_checkpoint=str(ckpt_path.resolve()),
            forged_model_path=None,
            elapsed_seconds=elapsed,
            error_message=repr(exc),
            host_d_model=host_d_model,
            basis_rank=basis_rank,
            quality_ratio=quality_ratio,
            quality_tier=quality_tier,
            validation_threshold=provenance_validation_threshold,
            encoding_class=provenance_encoding_class,
            validation_eval_overlap=provenance_validation_eval_overlap,
        )

    elapsed = time.monotonic() - started
    extras = getattr(result, "extras", {}) or {}
    return ParetoFrontierRow(
        encoding_label=label,
        target_n_features_kept=target_k,
        n_features_kept_actual=n_features_actual,
        pareto_reached_target=reached,
        faithfulness_kl=getattr(result, "faithfulness_kl", None),
        perplexity=_finite_or_none(extras.get("perplexity")),
        final_fine_tune_loss=_finite_or_none(extras.get("final_loss")),
        sae_checkpoint=str(ckpt_path.resolve()),
        forged_model_path=str(Path(result.output_dir).resolve()),
        elapsed_seconds=elapsed,
        error_message=None,
        host_d_model=host_d_model,
        basis_rank=basis_rank,
        quality_ratio=quality_ratio,
        quality_tier=quality_tier,
        validation_threshold=provenance_validation_threshold,
        encoding_class=provenance_encoding_class,
        validation_eval_overlap=provenance_validation_eval_overlap,
    )
