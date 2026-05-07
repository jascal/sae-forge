"""Action functions bound to the SaeForge FSM.

Every action takes ``(ctx: dict, payload: dict | None) -> dict | None``
and returns a delta that the orca-runtime-python ``OrcaMachine`` merges
into the machine context.

Actions gate their work on the presence of input fields in ``ctx``:

- ``compress_with_polygram`` runs Polygram's ``Compressor`` when
  ``ctx["validation_report_path"]`` is set; pass-through otherwise.
- ``perform_regrowth`` runs Polygram's ``Regrower`` when ``regrow_count
  > 0`` AND a compression report is reachable; pass-through otherwise.
- ``fine_tune_model`` runs N steps of LM training when
  ``ctx["_finetune_input_ids"]`` is set; pass-through otherwise.

The byte-equivalence with the imperative orchestrator holds for the
no-input case (the projection-only path). Real production runs supply
the gating inputs and the actions actually do work.
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
    """Run Polygram's Compressor against the current SAE when a validation report is supplied.

    Gating: ``ctx["validation_report_path"]`` must point to a polygram
    ``ValidationReport`` JSON. When absent, the action is a pass-through —
    the FSM treats the input SAE as already-compressed and forwards
    ``current_sae_path`` to ``compressed_sae_path`` unchanged.
    """
    report_path = ctx.get("validation_report_path")
    if not report_path:
        _log(ctx, "compress_with_polygram", {"mode": "passthrough"})
        return {
            "compressed_sae_path": ctx["current_sae_path"],
            "current_feature_count": ctx.get("current_feature_count", 0),
        }

    from polygram import Compressor, ValidationReport

    output_dir = Path(ctx["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "compressed.safetensors"

    confirmer = "quantum_interference" if ctx.get("quantum_aware", False) else None
    validation = ValidationReport.from_json(report_path)
    compressor_kwargs: dict[str, Any] = {
        "validation_report": validation,
        "sae_checkpoint": Path(ctx["current_sae_path"]),
        "strategy": ctx.get("compression_strategy", "merge"),
        "rep_selection": ctx.get("rep_selection", "scale_aware"),
    }
    if confirmer is not None:
        compressor_kwargs["confirmer"] = confirmer
    compressor = Compressor(**{k: v for k, v in compressor_kwargs.items() if k != "confirmer"})
    result = compressor.run(output_path)

    report = result.report
    _log(
        ctx,
        "compress_with_polygram",
        {
            "mode": "polygram",
            "n_features_kept": report.n_features_kept,
            "n_features_zeroed": report.n_features_zeroed,
            "scale_compression_ratio": report.scale_compression_ratio,
            "quantum_aware": ctx.get("quantum_aware", False),
        },
    )
    # Match FeatureBasis.from_polygram_checkpoint's auto-locator: look for
    # `<stem>_compression_report.json` next to the checkpoint.
    compression_report_path = output_dir / "compressed_compression_report.json"
    report.to_json(compression_report_path)
    return {
        "compressed_sae_path": str(output_path),
        "current_feature_count": report.n_features_kept,
        "compression_report_path": str(compression_report_path),
    }


def perform_regrowth(ctx: dict, _payload: dict | None = None) -> dict:
    """Regrow zeroed slots via Polygram's Regrower when a compression report and prompts are supplied."""
    if ctx.get("regrow_count", 0) == 0 or not ctx.get("compression_report_path"):
        _log(ctx, "perform_regrowth", {"mode": "passthrough"})
        return {"regrown_sae_path": ctx["compressed_sae_path"]}

    from polygram import CompressionReport, Regrower

    output_dir = Path(ctx["output_dir"])
    output_path = output_dir / "regrown.safetensors"
    report = CompressionReport.from_json(ctx["compression_report_path"])
    prompts = ctx.get("regrow_prompts") or [""] * 16

    regrower = Regrower.from_compression_report(
        report,
        sae_checkpoint=Path(ctx["compressed_sae_path"]),
        strategy=ctx.get("regrow_strategy", "residual_kmeans"),
        prompts=prompts,
        layer=ctx.get("regrow_layer", 10),
        model_name=ctx.get("host_model_id") or "gpt2",
        seed=ctx.get("regrow_seed", 0),
    )
    result = regrower.run(output_path)
    _log(ctx, "perform_regrowth", {"mode": "polygram", "n_regrown": len(result.report.populations)})
    return {"regrown_sae_path": str(output_path)}


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

    attention_width = ctx.get("attention_width", "host")
    weights = projector.project_module(host, attention_width=attention_width)
    config = _config_from_host(host, basis.n_features, attention_width=attention_width)
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
    """Fine-tune the forged native model.

    Three modes:
    - **Recipe**: when `ctx["finetune_corpus"]` or `ctx["_finetune_iterator"]`
      is supplied, delegate to `saeforge.training.run_finetune` with the full
      cosine-LR-with-warmup, gradient-clipping, optional grad-checkpointing,
      optional mixed-precision recipe.
    - **v0.1 fallback (smoke)**: when only `ctx["_finetune_input_ids"]` is
      supplied, run the original 4-step single-batch loop. Preserves
      byte-equivalence with v0.1 forged outputs for the safety-net test.
    - **Pass-through**: when none of the above, no fine-tune happens.
    """
    model = ctx.get("_native_model")
    if model is None:
        _log(ctx, "fine_tune_model", {"mode": "passthrough"})
        return {"finetuned_model_path": ctx["projected_weights_path"]}

    corpus = ctx.get("finetune_corpus")
    iterator = ctx.get("_finetune_iterator")
    if corpus is not None or iterator is not None:
        return _run_recipe_fine_tune(ctx, model, corpus, iterator)

    input_ids = ctx.get("_finetune_input_ids")
    if input_ids is None:
        _log(ctx, "fine_tune_model", {"mode": "passthrough"})
        return {"finetuned_model_path": ctx["projected_weights_path"]}

    return _run_v01_smoke_fine_tune(ctx, model, input_ids)


def _run_v01_smoke_fine_tune(ctx: dict, model, input_ids) -> dict:
    """v0.1 4-step smoke loop. Preserves the byte-equivalence safety net."""
    from saeforge.utils.lazy import require_extra

    torch = require_extra("torch", "torch")
    F = torch.nn.functional

    n_steps = ctx.get("finetune_steps", 4)
    lr = ctx.get("finetune_lr", 1e-3)
    device = ctx.get("device", "cpu")

    module = model.torch_module.to(device).train()
    optim = torch.optim.AdamW(module.parameters(), lr=lr)
    input_ids = input_ids.to(device)
    losses: list[float] = []
    for _ in range(n_steps):
        logits = module(input_ids)
        targets = input_ids[:, 1:]
        preds = logits[:, :-1].reshape(-1, logits.size(-1))
        loss = F.cross_entropy(preds, targets.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    module.eval()

    output_dir = Path(ctx["output_dir"])
    finetuned_dir = output_dir / "finetuned"
    model.save_pretrained(finetuned_dir)
    _log(
        ctx,
        "fine_tune_model",
        {"mode": "trained", "n_steps": n_steps, "loss_first": losses[0], "loss_last": losses[-1]},
    )
    return {"finetuned_model_path": str(finetuned_dir), "_finetune_losses": losses}


def _run_recipe_fine_tune(ctx: dict, model, corpus, iterator) -> dict:
    """v0.3 recipe path: delegate to saeforge.training.run_finetune."""
    from saeforge.training import TrainingConfig, build_iterator, run_finetune

    output_dir = Path(ctx["output_dir"])
    finetuned_dir = output_dir / "finetuned"

    if iterator is None:
        # Build iterator from corpus path or HF dataset name. We need a
        # tokenizer; pull it from the host model's id when available.
        from saeforge.utils.lazy import require_extra

        transformers = require_extra("transformers", "torch")
        host_id = ctx.get("host_model_id") or "gpt2"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            host_id if host_id != "<in-memory>" else "gpt2"
        )
        iterator = build_iterator(
            corpus,
            tokenizer,
            batch_size=ctx.get("finetune_batch_size", 8),
            sequence_length=ctx.get("finetune_seq_len", 512),
        )

    config = TrainingConfig(
        total_steps=ctx.get("finetune_total_steps", ctx.get("finetune_steps", 1000)),
        warmup_steps=ctx.get("finetune_warmup_steps", 100),
        peak_lr=ctx.get("finetune_peak_lr", ctx.get("finetune_lr", 5e-5)),
        weight_decay=ctx.get("finetune_weight_decay", 0.01),
        batch_size=ctx.get("finetune_batch_size", 8),
        sequence_length=ctx.get("finetune_seq_len", 512),
        precision=ctx.get("finetune_precision", "fp32"),
        gradient_checkpointing=ctx.get("finetune_grad_checkpoint", False),
        eval_every_steps=ctx.get("finetune_eval_every", 100),
        eval_input_ids=ctx.get("_eval_input_ids"),
        save_every_steps=ctx.get("finetune_save_every", 250),
        save_dir=ctx.get("finetune_save_dir") or finetuned_dir / "checkpoints",
        log_every_steps=ctx.get("finetune_log_every", 10),
    )

    host = ctx.get("_host_model")
    result = run_finetune(model, host, iterator, config)

    model.save_pretrained(finetuned_dir)
    _log(
        ctx,
        "fine_tune_model",
        {
            "mode": "recipe",
            "n_steps": result.n_steps_completed,
            "final_loss": result.final_loss,
            "wall_seconds": result.wall_seconds,
            "n_eval_samples": len(result.eval_history),
            "n_saves": len(result.save_paths),
            "converged": result.converged,
            "oom_batch_halved": result.metadata.get("oom_batch_halved", False),
        },
    )
    return {
        "finetuned_model_path": str(finetuned_dir),
        "_finetune_losses": [loss for (_, loss) in result.loss_history],
        "_finetune_eval_history": result.eval_history,
    }


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
        "compress_mode": _last_log_extra(ctx, "compress_with_polygram", "mode"),
        "finetune_mode": _last_log_extra(ctx, "fine_tune_model", "mode"),
    }
    (output_dir / "forge_result.json").write_text(json.dumps(summary, indent=2))
    return {"final_model_path": str(forged_dir), "n_params": n_params}


def log_error(ctx: dict, payload: dict | None = None) -> dict:
    msg = (payload or {}).get("error", ctx.get("error_message", "unknown error"))
    _log(ctx, "log_error", {"error": msg})
    return {"error_message": str(msg)}


def _last_log_extra(ctx: dict, action_name: str, key: str):
    for entry in reversed(ctx.get("transitions_log", [])):
        if entry.get("action") == action_name and key in entry:
            return entry[key]
    return None


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
