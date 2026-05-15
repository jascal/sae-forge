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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from saeforge.forge import ForgePipeline


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
# Driver
# ---------------------------------------------------------------------------


def sweep_pareto(
    pipeline: "ForgePipeline",
    *,
    encodings: list[tuple[str, Path]],
    output_dir: Path,
    frontier_only: bool = False,
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
                row = _process_row(
                    pipeline=pipeline,
                    label=label,
                    target_k=target_k,
                    ckpt_path=ckpt_path,
                    manifest_entry=entry,
                    sweep_output_dir=output_dir,
                    frontier_only=frontier_only,
                    forge_kwargs=forge_kwargs,
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
) -> ParetoFrontierRow:
    """Build one frontier row — manifest-only when ``frontier_only``, otherwise
    invoke ``pipeline.run`` inside a try/except.
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
    )
