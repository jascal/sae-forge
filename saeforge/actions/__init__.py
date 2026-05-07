"""Action functions bound to the SaeForge FSM.

Every action takes ``(ctx: dict, payload: dict | None) -> dict | None``
and returns a delta that the orca-runtime-python ``OrcaMachine`` merges
into the machine context.

Compress / regrow / fine-tune are no-op pass-throughs in v0.1. The
v0.2 milestone swaps them for real Polygram + HF trainer calls.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def _log(ctx: dict, name: str, extra: dict | None = None) -> None:
    entry = {"action": name, "wall_clock_ms": int(time.monotonic() * 1000)}
    if extra:
        entry.update(extra)
    ctx.setdefault("transitions_log", []).append(entry)


def load_sae_and_corpus(ctx: dict, _payload: dict | None = None) -> dict:
    sae = Path(ctx["sae_checkpoint"])
    if not sae.is_file():
        raise FileNotFoundError(f"sae_checkpoint not found: {sae}")
    _log(ctx, "load_sae_and_corpus")
    return {"current_sae_path": str(sae)}


def compress_with_polygram(ctx: dict, _payload: dict | None = None) -> dict:
    _log(ctx, "compress_with_polygram", {"quantum_aware": ctx.get("quantum_aware", False)})
    return {
        "compressed_sae_path": ctx["current_sae_path"],
        "current_feature_count": ctx.get("current_feature_count", 0),
    }


def perform_regrowth(ctx: dict, _payload: dict | None = None) -> dict:
    _log(ctx, "perform_regrowth")
    return {"regrown_sae_path": ctx["compressed_sae_path"]}


def project_to_subspace(ctx: dict, _payload: dict | None = None) -> dict:
    """Pure projection step. Builds the projected weights and writes them as a checkpoint."""
    from saeforge import FeatureBasis, NativeModel, SubspaceProjector
    from saeforge.model import _config_from_host

    sae_path = ctx.get("regrown_sae_path") or ctx.get("compressed_sae_path") or ctx["current_sae_path"]
    basis = FeatureBasis.from_polygram_checkpoint(sae_path)
    projector = SubspaceProjector(basis)

    host = ctx.pop("_host_model", None)
    if host is None:
        from saeforge.utils.lazy import require_extra

        transformers = require_extra("transformers", "torch")
        host = transformers.GPT2LMHeadModel.from_pretrained(ctx["host_model_id"]).eval()

    weights = projector.project_module(host)
    config = _config_from_host(host, basis.n_features)
    model = NativeModel.from_projected_weights(config, weights)

    output_dir = Path(ctx["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    projected_dir = output_dir / "projected"
    model.save_pretrained(projected_dir)

    _log(ctx, "project_to_subspace", {"n_features": basis.n_features})
    return {
        "projected_weights_path": str(projected_dir),
        "current_feature_count": basis.n_features,
        "_host_model": host,
        "_native_model": model,
    }


def fine_tune_model(ctx: dict, _payload: dict | None = None) -> dict:
    _log(ctx, "fine_tune_model")
    return {"finetuned_model_path": ctx["projected_weights_path"]}


def evaluate_faithfulness(ctx: dict, _payload: dict | None = None) -> dict:
    """Compute the per-token KL between the forged native model and the host."""
    from saeforge.forge import _kl_from_input_ids

    host = ctx.get("_host_model")
    forged = ctx.get("_native_model")
    eval_input_ids = ctx.get("_eval_input_ids")
    if host is None or forged is None or eval_input_ids is None:
        kl = 0.0
    else:
        kl = _kl_from_input_ids(forged, host, eval_input_ids, device=ctx.get("device", "cpu"))
    perplexity = float(np.exp(kl)) if kl >= 0 else float("inf")
    iters = ctx.get("iterations", 1)
    current = ctx.get("current_iter", 0)
    min_faith = ctx.get("min_faithfulness", 0.0)
    best_perp = ctx.get("best_perplexity", float("inf"))
    should_continue = bool(
        current + 1 < iters
        and (kl >= min_faith if min_faith == 0.0 else kl <= min_faith * -1)
        and perplexity < best_perp
    )
    _log(
        ctx,
        "evaluate_faithfulness",
        {"faithfulness": kl, "perplexity": perplexity, "should_continue": should_continue},
    )
    return {
        "faithfulness": float(kl),
        "perplexity": perplexity,
        "should_continue": should_continue,
    }


def rotate_for_next_iter(ctx: dict, _payload: dict | None = None) -> dict:
    next_input = ctx.get("regrown_sae_path") or ctx.get("compressed_sae_path") or ctx["current_sae_path"]
    _log(ctx, "rotate_for_next_iter", {"next_iter": ctx.get("current_iter", 0) + 1})
    return {
        "current_sae_path": next_input,
        "current_iter": ctx.get("current_iter", 0) + 1,
        "best_perplexity": min(
            ctx.get("best_perplexity", float("inf")),
            ctx.get("perplexity", float("inf")),
        ),
    }


def save_final_model(ctx: dict, _payload: dict | None = None) -> dict:
    """Persist the forged model from the projected stage to ``output_dir/forged``."""
    output_dir = Path(ctx["output_dir"])
    forged_dir = output_dir / "forged"
    model = ctx.get("_native_model")
    if model is not None:
        model.save_pretrained(forged_dir)
        n_params = model.num_parameters()
    else:
        n_params = ctx.get("n_params", 0)
    _log(ctx, "save_final_model", {"n_params": n_params})
    import json

    summary = {
        "host_model_id": ctx.get("host_model_id"),
        "n_params": n_params,
        "faithfulness_kl": ctx.get("faithfulness"),
        "n_features": ctx.get("current_feature_count"),
        "iterations": ctx.get("current_iter", 0) + 1,
    }
    (output_dir / "forge_result.json").write_text(json.dumps(summary, indent=2))
    return {"final_model_path": str(forged_dir), "n_params": n_params}


def log_error(ctx: dict, payload: dict | None = None) -> dict:
    msg = (payload or {}).get("error", ctx.get("error_message", "unknown error"))
    _log(ctx, "log_error", {"error": msg})
    return {"error_message": str(msg)}


ACTION_TABLE: dict[str, Any] = {
    "load_sae_and_corpus": load_sae_and_corpus,
    "compress_with_polygram": compress_with_polygram,
    "perform_regrowth": perform_regrowth,
    "project_to_subspace": project_to_subspace,
    "fine_tune_model": fine_tune_model,
    "evaluate_faithfulness": evaluate_faithfulness,
    "rotate_for_next_iter": rotate_for_next_iter,
    "save_final_model": save_final_model,
    "log_error": log_error,
}
