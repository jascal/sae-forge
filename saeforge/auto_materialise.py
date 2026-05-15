"""Auto-materialise pre-step for `sweep-pareto`.

Drives the polygram-side compression pipeline (BehaviouralValidator →
Compressor.plan_pareto → Compressor.apply) into a deterministic on-disk
layout under ``<output-dir>/_materialised/<label>/`` so the subsequent
sweep loop can consume the materialised per-K SAEs unchanged. Includes a
content-addressed cache: reruns with identical inputs skip the (expensive)
validator pass entirely.

See ``openspec/changes/archive/.../add-auto-materialise-sweep/`` for the
design rationale, leakage-firewall constraint, and cache key inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Encoding class registry — names map to polygram classes. Restricted to
# the four classes whose `from_sae_lens` path returns a plain Dictionary
# (not ClusteredDictionary, which BehaviouralValidator can't accept).
# See design Decision 7.
_ENCODING_CLASS_REGISTRY: dict[str, str] = {
    "MPSRung1": "MPSRung1",
    "Rung3": "Rung3",
    "Rung4": "Rung4",
    "HEA_Rung2": "HEA_Rung2",
}


@dataclass(frozen=True)
class AutoMaterialiseSpec:
    """One encoding's auto-materialise configuration.

    ``label`` is the user-supplied encoding label (e.g. ``"mps"``,
    ``"rung4"``); appears in frontier rows and as the materialised
    directory's subdir name. ``sae_checkpoint`` is the path to an
    uncompressed SAE-Lens checkpoint (W_enc, W_dec, b_enc, b_dec).
    ``encoding_class`` is one of the four supported polygram encoding
    class names; ``encoding_kwargs`` carries class-specific arguments
    (e.g. ``{"n_qubits": 5}`` for HEA_Rung2).
    """

    label: str
    sae_checkpoint: Path
    encoding_class: str = "MPSRung1"
    encoding_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("AutoMaterialiseSpec: label must be non-empty")
        if self.encoding_class not in _ENCODING_CLASS_REGISTRY:
            raise ValueError(
                f"AutoMaterialiseSpec: unknown encoding_class "
                f"{self.encoding_class!r}; supported: "
                f"{sorted(_ENCODING_CLASS_REGISTRY)}"
            )


def _resolve_encoding_class(class_name: str) -> type:
    """Map an encoding class name to the polygram class object."""
    import polygram

    if class_name not in _ENCODING_CLASS_REGISTRY:
        raise ValueError(
            f"resolve_encoding_class: unknown {class_name!r}; "
            f"supported: {sorted(_ENCODING_CLASS_REGISTRY)}"
        )
    return getattr(polygram, class_name)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's contents. Used in cache keys so the
    cache is content-addressed (renaming a file doesn't invalidate it).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_cache_key(
    *,
    spec: AutoMaterialiseSpec,
    validation_prompts_path: Path,
    validation_threshold: float,
    jaccard_threshold: float,
    layer: int,
    model_name: str,
    targets: list[int],
    score_field: str,
    rep_selection: str,
) -> dict[str, Any]:
    """Compute the cache key for a materialisation run.

    Content-addressed via SHA-256 of the SAE checkpoint and the validation
    prompts file — moving or renaming a file with identical content does
    NOT invalidate the cache. Validator-tuning fields, encoding choice,
    layer, model_name, and target K list are all included so any change
    invalidates the cache.
    """
    return {
        "sae_checkpoint_sha256": _file_sha256(spec.sae_checkpoint),
        "sae_checkpoint_path": str(spec.sae_checkpoint),
        "validation_prompts_sha256": _file_sha256(validation_prompts_path),
        "validation_prompts_path": str(validation_prompts_path),
        "validation_threshold": float(validation_threshold),
        "jaccard_threshold": float(jaccard_threshold),
        "encoding_class": spec.encoding_class,
        "encoding_kwargs": dict(spec.encoding_kwargs),
        "layer": int(layer),
        "model_name": str(model_name),
        "targets": sorted(int(k) for k in targets),
        "score_field": str(score_field),
        "rep_selection": str(rep_selection),
    }


def is_cache_hit(
    materialised_dir: Path,
    expected_key: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check whether the materialised directory satisfies the expected
    cache key. Returns ``(hit, diff_fields)``; on miss, ``diff_fields``
    lists the cache-key fields that differ from disk (used in
    ``--plan-only`` advisory output to show why a rerun is needed).

    Returns ``(False, ["cold"])`` when ``auto_materialise_meta.json`` is
    absent. Returns ``(False, [...])`` listing any disk-vs-expected
    differences when the meta exists. Returns ``(True, [])`` when all
    cache-key fields match AND every expected ``pareto/k_<K>.safetensors``
    is present.
    """
    meta_path = materialised_dir / "auto_materialise_meta.json"
    if not meta_path.is_file():
        return False, ["cold"]
    try:
        on_disk = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, ["meta_unreadable"]

    diff_fields = [k for k in expected_key if on_disk.get(k) != expected_key[k]]
    if diff_fields:
        return False, diff_fields

    # All cache-key fields match — verify per-K files are present.
    pareto_dir = materialised_dir / "pareto"
    missing_files = [
        k for k in expected_key["targets"]
        if not (pareto_dir / f"k_{k}.safetensors").is_file()
    ]
    if missing_files:
        return False, [f"missing_k_{k}" for k in missing_files]

    return True, []


# ---------------------------------------------------------------------------
# Materialisation driver
# ---------------------------------------------------------------------------


def materialise(
    spec: AutoMaterialiseSpec,
    *,
    validation_prompts_path: Path,
    validation_threshold: float,
    jaccard_threshold: float,
    layer: int,
    model_name: str,
    targets: list[int],
    score_field: str,
    rep_selection: str,
    output_root: Path,
    force_rematerialise: bool = False,
) -> tuple[Path, dict[str, Any], list[str]]:
    """Run polygram's validator + plan_pareto + apply chain for one encoding.

    Returns ``(materialised_dir, cache_key, diff_fields)``:

    - ``materialised_dir`` — the directory containing ``validation_report.json``,
      ``pareto.json``, ``pareto/k_<K>.safetensors`` files, and
      ``auto_materialise_meta.json``.
    - ``cache_key`` — the cache-key dict computed for this run (returned
      so callers can surface it under ``--plan-only``).
    - ``diff_fields`` — the cache-status indicator. Empty list when the
      sweep was a cache hit and nothing was rebuilt; ``["cold"]`` /
      ``["validation_threshold", ...]`` / etc. on cache miss describing
      what differed.

    ``force_rematerialise=True`` skips the cache-hit check and runs the
    full chain regardless. Existing files are overwritten in place.

    The validation prompts file is read as one prompt per non-empty line
    (matching polygram's validate CLI convention).
    """
    materialised_dir = output_root / "_materialised" / spec.label
    materialised_dir.mkdir(parents=True, exist_ok=True)

    expected_key = compute_cache_key(
        spec=spec,
        validation_prompts_path=validation_prompts_path,
        validation_threshold=validation_threshold,
        jaccard_threshold=jaccard_threshold,
        layer=layer,
        model_name=model_name,
        targets=targets,
        score_field=score_field,
        rep_selection=rep_selection,
    )

    if not force_rematerialise:
        hit, diff_fields = is_cache_hit(materialised_dir, expected_key)
        if hit:
            return materialised_dir, expected_key, []
    else:
        diff_fields = ["forced"]

    # Cache miss (or force) → run the full chain.
    _run_materialisation_chain(
        spec=spec,
        validation_prompts_path=validation_prompts_path,
        validation_threshold=validation_threshold,
        jaccard_threshold=jaccard_threshold,
        layer=layer,
        model_name=model_name,
        targets=targets,
        score_field=score_field,
        rep_selection=rep_selection,
        materialised_dir=materialised_dir,
    )

    (materialised_dir / "auto_materialise_meta.json").write_text(
        json.dumps(expected_key, indent=2, sort_keys=True)
    )
    return materialised_dir, expected_key, diff_fields


def _run_materialisation_chain(
    *,
    spec: AutoMaterialiseSpec,
    validation_prompts_path: Path,
    validation_threshold: float,
    jaccard_threshold: float,
    layer: int,
    model_name: str,
    targets: list[int],
    score_field: str,
    rep_selection: str,
    materialised_dir: Path,
) -> None:
    """Run polygram's BehaviouralValidator + Compressor.plan_pareto + apply
    against ``spec``, writing artifacts to ``materialised_dir``.
    """
    from polygram import (
        BehaviouralValidator,
        Compressor,
        CompressionConfig,
        ValidationConfig,
        from_sae_lens,
        load_sae_safetensors,
    )

    encoding_cls = _resolve_encoding_class(spec.encoding_class)
    encoding_instance = encoding_cls(**spec.encoding_kwargs)

    # Read prompts: one per non-empty line, '#' comments skipped.
    prompts = _read_validation_prompts(validation_prompts_path)
    if not prompts:
        raise ValueError(
            f"auto_materialise: validation prompts file {validation_prompts_path} "
            f"is empty"
        )

    # Slot ids 0..N-1: the sliced SAE's feature ids are positional, not the
    # original SAE-Lens ids the user may have stride-sampled from.
    # Callers who want the original ids should slice upstream.
    records = load_sae_safetensors(spec.sae_checkpoint)
    slot_ids = sorted(records.keys())

    dictionary, _selection_report = from_sae_lens(
        records, slot_ids, encoding=encoding_instance
    )
    validator = BehaviouralValidator(
        dictionary=dictionary,
        sae_checkpoint=spec.sae_checkpoint,
        feature_ids=slot_ids,
        prompts=prompts,
        layer=layer,
        model_name=model_name,
        config=ValidationConfig(
            polygram_overlap_threshold=validation_threshold,
            jaccard_threshold=jaccard_threshold,
        ),
    )
    report = validator.run()
    report.to_json(materialised_dir / "validation_report.json")

    compressor = Compressor(
        validation_report=report,
        sae_checkpoint=spec.sae_checkpoint,
        config=CompressionConfig(
            strategy="merge",
            rep_selection=rep_selection,
            score_field=score_field,
        ),
    )
    pareto = compressor.plan_pareto(targets)
    (materialised_dir / "pareto.json").write_text(pareto.to_json())

    pareto_dir = materialised_dir / "pareto"
    pareto_dir.mkdir(exist_ok=True)
    for outcome in pareto.outcomes:
        ckpt_path = pareto_dir / f"k_{outcome.target_k}.safetensors"
        compressor.apply(plan=outcome.plan, output_checkpoint=ckpt_path)


def _read_validation_prompts(path: Path) -> list[str]:
    """One prompt per non-empty line; ``#``-prefixed lines are comments."""
    if not path.is_file():
        raise FileNotFoundError(
            f"auto_materialise: validation prompts file not found: {path}"
        )
    prompts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


# ---------------------------------------------------------------------------
# --plan-only output formatting
# ---------------------------------------------------------------------------


def format_plan_only_block(
    *,
    spec: AutoMaterialiseSpec,
    cache_key: dict[str, Any],
    diff_fields: list[str],
    cache_hit: bool,
    n_prompts: int,
    avg_prompt_tokens: float,
) -> str:
    """Format the per-encoding ``--plan-only`` stderr block."""
    if cache_hit:
        status = "HIT"
    else:
        status = "MISS (" + ", ".join(diff_fields) + ")"

    estimated_forwards = int(n_prompts * avg_prompt_tokens)

    lines = [
        f"  label={spec.label}",
        f"    cache_status={status}",
        f"    sae_sha256={cache_key['sae_checkpoint_sha256']}",
        f"    validation_prompts_sha256={cache_key['validation_prompts_sha256']}",
        f"    targets={cache_key['targets']}",
        f"    encoding_class={cache_key['encoding_class']}",
        f"    encoding_kwargs={cache_key['encoding_kwargs']}",
        f"    validator_forward_count_estimate={estimated_forwards}",
    ]
    return "\n".join(lines)


def estimate_prompt_token_count(prompts_path: Path) -> tuple[int, float]:
    """Rough estimate of (n_prompts, avg_tokens_per_prompt) for the
    ``--plan-only`` validator-cost estimate. Uses whitespace tokenisation
    as a cheap approximation — the true tokenizer-driven count would
    require loading the host's tokenizer, which we don't want during
    ``--plan-only`` (the whole point is to not invoke heavy machinery).
    """
    if not prompts_path.is_file():
        return 0, 0.0
    prompts = _read_validation_prompts(prompts_path)
    if not prompts:
        return 0, 0.0
    total_tokens = sum(len(p.split()) for p in prompts)
    return len(prompts), total_tokens / len(prompts)
