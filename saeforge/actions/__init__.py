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
    entry = {
        "action": name,
        "wall_clock_ms": int(time.monotonic() * 1000),
        "machine_path": ctx.get("_machine_path", "stream"),
    }
    if extra:
        entry.update(extra)
    ctx.setdefault("transitions_log", []).append(entry)


def load_and_scan(ctx: dict, payload: dict | None = None) -> dict:
    """Composed RefineMachine entry action: load_sae_and_corpus + scan_activations.

    Replaces the v0.2 two-state pair (``loaded`` + ``activations_scanned``)
    with one transition action that runs both helpers in order. The
    ``transitions_log`` records both inner action names — preserving the
    v0.2 log shape and the byte-equivalence contract.

    Each helper returns a ctx delta dict (or ``None``); we merge them in
    order so later writes override earlier ones, matching the runtime's
    standard ``ctx.update(result)`` behavior.
    """
    delta: dict = {}
    for helper in (load_sae_and_corpus, scan_activations):
        result = helper(ctx, payload)
        if isinstance(result, dict):
            delta.update(result)
            ctx.update(result)  # let the second helper see the first's writes
    return delta


_POLYGRAM_VERSION_HINT = (
    "sae-forge requires polygram>=0.9.0 (which adds cluster_experts / "
    "ExpertDictionary to the public surface; tuning-config dataclasses "
    "have been required since 0.1.0). Upgrade with "
    "`pip install -U 'polygram>=0.9.0'`."
)


def _import_polygram_symbols(*names: str):
    """Import the named attributes from ``polygram``; raise an ImportError
    that points the user at the right ``polygram`` version when one of
    the new symbols (``CompressionConfig``, ``EpochCompressionConfig``,
    ``RegrowConfig``) is missing.
    """
    try:
        import polygram
    except ImportError as exc:
        raise ImportError(
            "sae-forge action needs the `polygram` package. "
            + _POLYGRAM_VERSION_HINT
        ) from exc

    resolved = []
    for name in names:
        if not hasattr(polygram, name):
            raise ImportError(
                f"polygram is installed but does not export {name!r}. "
                + _POLYGRAM_VERSION_HINT
            )
        resolved.append(getattr(polygram, name))
    return resolved


def load_sae_and_corpus(ctx: dict, _payload: dict | None = None) -> dict:
    sae = Path(ctx["sae_checkpoint"])
    if not sae.is_file():
        raise FileNotFoundError(f"sae_checkpoint not found: {sae}")
    _log(ctx, "load_sae_and_corpus")
    return {"current_sae_path": str(sae)}


def scan_activations(ctx: dict, _payload: dict | None = None) -> dict:
    """Score features and select a protected set; pass-through under v0.2 defaults.

    True no-op (no basis load, no torch import) when ``protect_top_k == 0``
    so the v0.1 byte-equivalence test continues to pass.

    When ``protect_top_k > 0``, scores features against the SAE basis and
    writes ``feature_usage`` + ``protected_features``. The v0.2.0 scorer
    uses **direction L2 norms** as the importance proxy — this is
    deterministic, fast, requires no host-model forward, and gives a
    sensible per-feature ranking out of the box. Activation-driven
    scoring (true ``mean_act`` against host residuals) is a refinement
    tracked in tasks.md §12.2.
    """
    if ctx.get("protect_top_k", 0) == 0:
        _log(ctx, "scan_activations", {"mode": "passthrough"})
        return {}

    score_strategy = ctx.get("protect_score", "mean_act")
    if score_strategy not in ("mean_act", "usage", "grad_importance"):
        raise ValueError(
            f"unknown protect_score {score_strategy!r}; must be "
            "'mean_act' | 'usage' | 'grad_importance'"
        )

    from saeforge import FeatureBasis
    from saeforge.utils.lazy import require_extra

    torch = require_extra("torch", "torch")

    basis = FeatureBasis.from_polygram_checkpoint(ctx["current_sae_path"])
    n_features = basis.n_features
    top_k = min(int(ctx["protect_top_k"]), n_features)

    directions = torch.as_tensor(basis.W_dec, dtype=torch.float32)
    # All three strategies fall back to direction L2 in v0.2.0; the
    # strategy is honored as a config knob so callers can request a
    # different scorer once activation capture lands. Recording which
    # strategy was *requested* is informative for follow-up debugging.
    feature_usage = directions.norm(dim=1).tolist()

    indexed = sorted(enumerate(feature_usage), key=lambda kv: -kv[1])
    protected = [i for i, _ in indexed[:top_k]]

    _log(
        ctx,
        "scan_activations",
        {
            "mode": "scored",
            "n_features": n_features,
            "n_protected": len(protected),
            "score": score_strategy,
            "scorer_impl": "direction_l2_v0_2",
        },
    )
    return {
        "feature_usage": feature_usage,
        "protected_features": protected,
    }


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
        delta = {
            "compressed_sae_path": ctx["current_sae_path"],
            "current_feature_count": ctx.get("current_feature_count", 0),
        }
        # Pass-through compress is still one basis-loop pass when no
        # regrow follows. Increment so the basis_loop guards exit.
        if ctx.get("regrow_count", 0) == 0:
            delta["inner_refine_idx"] = ctx.get("inner_refine_idx", 0) + 1
        return delta

    Compressor, ValidationReport, CompressionConfig = _import_polygram_symbols(
        "Compressor", "ValidationReport", "CompressionConfig"
    )

    output_dir = Path(ctx["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "compressed.safetensors"

    validation = ValidationReport.from_json(report_path)

    # Protected features (structural EWC). In v0.2.0 we ship the
    # ValidationReport-postfilter workaround: any feature index in
    # ctx["protected_features"] is marked as confirmed in every
    # validation pair so Polygram's Compressor cannot drop it. The
    # do_not_remove kwarg is the preferred long-term path; tracked
    # in tasks.md §10.4.
    protected = ctx.get("protected_features") or []
    if protected:
        _apply_protected_postfilter(validation, protected)

    # Reconstitute a CompressionConfig from ctx (serialised via .to_dict()
    # by ForgePipeline._build_context). When the key is absent, build a
    # default — its (strategy, rep_selection) defaults match what
    # sae-forge has always passed explicitly.
    compression_dict = ctx.get("compression")
    base_config = (
        CompressionConfig.from_dict(compression_dict)
        if compression_dict is not None
        else CompressionConfig()
    )

    # ``quantum_aware`` is an FSM-level toggle; it overrides the
    # config's ``confirmer`` field when set so the FSM's quantum-aware
    # decision is the source of truth on a per-run basis.
    if ctx.get("quantum_aware", False):
        from dataclasses import replace

        config = replace(base_config, confirmer="quantum_interference")
    else:
        config = base_config

    compressor = Compressor(
        validation_report=validation,
        sae_checkpoint=Path(ctx["current_sae_path"]),
        config=config,
    )
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
    delta = {
        "compressed_sae_path": str(output_path),
        "current_feature_count": report.n_features_kept,
        "compression_report_path": str(compression_report_path),
    }
    if ctx.get("regrow_count", 0) == 0:
        delta["inner_refine_idx"] = ctx.get("inner_refine_idx", 0) + 1
    return delta


def _apply_protected_postfilter(validation, protected_indices: list[int]) -> None:
    """Mark protected feature indices as confirmed in every validation pair.

    **Mutates ``validation.pairs`` in place.** This is intentional —
    Polygram's Compressor consumes the ValidationReport object directly,
    so the mutation needs to be visible to the same call. Callers should
    NOT persist the same ``validation`` object back to disk after this
    runs without re-loading from JSON; the on-disk report should remain
    the authoritative pre-protection record. The compress action follows
    this contract by loading via ``ValidationReport.from_json`` (fresh
    object every call), mutating, and discarding.

    Tolerant of older ValidationReport schemas that lack a ``confirmed``
    field per pair — skips silently if the schema doesn't match (the
    protection is best-effort until the upstream ``do_not_remove`` kwarg
    lands; tracked in tasks.md §10.4).
    """
    if not hasattr(validation, "pairs"):
        return
    proto_set = set(protected_indices)
    for pair in validation.pairs:
        # Common schema shapes: pair.confirmed (bool) or pair.feature_a /
        # pair.feature_b indices. We pessimistically skip any pair that
        # involves a protected index — that prevents the merge/removal.
        feature_a = getattr(pair, "feature_a", None)
        feature_b = getattr(pair, "feature_b", None)
        if feature_a in proto_set or feature_b in proto_set:
            if hasattr(pair, "confirmed"):
                pair.confirmed = True


def perform_regrowth(ctx: dict, _payload: dict | None = None) -> dict:
    """Regrow zeroed slots via Polygram's Regrower when a compression report is supplied.

    The ``regrow`` ctx key (a serialised :class:`polygram.RegrowConfig`)
    is required whenever ``regrow_count > 0``. The pre-change
    ``layer=10`` / ``model_name="gpt2"`` ctx fallbacks were a footgun on
    non-GPT-2 hosts; polygram-tuning-config removed the corresponding
    polygram-side defaults so callers must now declare them explicitly.

    The per-cycle count is read from ``effective_regrow_count`` when
    present (written by ``adapt_and_regrow`` under
    ``adaptive_regrow=True``) and falls back to the configured
    ``regrow_count`` otherwise. The gate that turns the action into a
    pass-through is the configured ``regrow_count == 0`` — unchanged
    from v0.2.
    """
    if ctx.get("regrow_count", 0) == 0 or not ctx.get("compression_report_path"):
        _log(ctx, "perform_regrowth", {"mode": "passthrough"})
        # Even pass-through regrowth completes one basis-loop round.
        return {
            "regrown_sae_path": ctx["compressed_sae_path"],
            "inner_refine_idx": ctx.get("inner_refine_idx", 0) + 1,
        }

    if ctx.get("regrow") is None:
        raise ValueError(
            "perform_regrowth: regrow_count > 0 requires ctx['regrow'] to be "
            "set (a serialised polygram.RegrowConfig). Set "
            "ForgePipeline(regrow=RegrowConfig(model_name=..., layer=...)) "
            "or pass --regrow-layer / --regrow-strategy on the CLI."
        )

    CompressionReport, Regrower, RegrowConfig = _import_polygram_symbols(
        "CompressionReport", "Regrower", "RegrowConfig"
    )

    output_dir = Path(ctx["output_dir"])
    output_path = output_dir / "regrown.safetensors"
    report = CompressionReport.from_json(ctx["compression_report_path"])

    config = RegrowConfig.from_dict(ctx["regrow"])
    # ``prompts`` historically got a 16-empty-string fallback when omitted
    # so the regrower could capture residuals without a prompt corpus.
    # Honour that here when neither config.prompts nor a separate ctx
    # entry supply one.
    prompts = config.prompts if config.prompts is not None else tuple([""] * 16)

    # Per-cycle count: prefer the adaptive controller's
    # ``effective_regrow_count`` (when set by ``adapt_and_regrow``) over
    # the configured ``regrow_count``. ``top_k`` is the polygram-side
    # field that caps the regrow population.
    effective = ctx.get("effective_regrow_count")
    top_k = int(effective) if effective is not None else int(ctx["regrow_count"])

    regrower = Regrower.from_compression_report(
        report,
        sae_checkpoint=Path(ctx["compressed_sae_path"]),
        config=config,
        prompts=list(prompts),
        top_k=top_k,
    )
    result = regrower.run(output_path)
    _log(
        ctx,
        "perform_regrowth",
        {
            "mode": "polygram",
            "n_regrown": len(result.report.populations),
            "top_k": top_k,
            "source": "effective" if effective is not None else "configured",
        },
    )
    return {
        "regrown_sae_path": str(output_path),
        "inner_refine_idx": ctx.get("inner_refine_idx", 0) + 1,
    }


def _compute_effective_regrow_count(ctx: dict) -> dict:
    """Run the RegrowController against ctx and stash the result.

    Mutates ``ctx['effective_regrow_count']`` AND appends an
    ``adapt_regrow_count`` entry to ``transitions_log``. Returns a
    dict delta for orca-runtime to merge back. The two writes (ctx
    mutation + delta return) are kept in lock-step so callers that
    consume ctx directly (the composed action) and callers that rely
    on the runtime's delta merge (the FSM dispatch) both see the new
    field.
    """
    from saeforge.basis import RegrowController

    kept = int(ctx.get("current_feature_count", 0))
    target = int(ctx.get("n_features_target", 0))
    regrow_count = int(ctx.get("regrow_count", 0))
    regrow_max = int(ctx.get("regrow_max", 0))
    damping = float(ctx.get("regrow_damping", 0.5))

    value = RegrowController.next_count(
        n_features_kept=kept,
        n_features_target=target,
        regrow_count=regrow_count,
        regrow_max=regrow_max,
        regrow_damping=damping,
    )
    gap = max(0, target - kept)
    _log(
        ctx,
        "adapt_regrow_count",
        {"value": value, "gap": gap, "target": target},
    )
    ctx["effective_regrow_count"] = value
    return {"effective_regrow_count": value}


def adapt_and_regrow(ctx: dict, payload: dict | None = None) -> dict:
    """Composed BasisMachine action: optionally compute ``effective_regrow_count`` then regrow.

    Three paths, in order:

    1. **Disabled toggle** — ``ctx['adaptive_regrow']`` is falsy. The
       controller is NOT invoked. ``effective_regrow_count`` is NOT
       written. The call reduces to ``perform_regrowth(ctx, payload)``
       — byte-identical to v0.2.
    2. **Cold start** — ``adaptive_regrow=True`` but no prior
       compression has populated ``current_feature_count`` (it is
       ``None`` or ``0``). The controller is NOT invoked; the action
       falls through to ``perform_regrowth`` using the configured
       ``regrow_count``. No ``adapt_regrow_count`` log entry is
       appended.
    3. **Enabled, warm** — ``adaptive_regrow=True`` AND a prior
       compression populated ``current_feature_count``. The controller
       runs, writes ``effective_regrow_count`` to ctx, appends an
       ``adapt_regrow_count`` log entry, and then calls
       ``perform_regrowth`` which reads the just-written field.

    The ``transitions_log`` records two consecutive entries per
    enabled-warm cycle: ``adapt_regrow_count`` then
    ``perform_regrowth``. Existing readers indexed by action name see
    one extra entry per regrow cycle.
    """
    if not ctx.get("adaptive_regrow"):
        return perform_regrowth(ctx, payload)

    # Cold-start: skip the controller entirely on the first cycle.
    # ``current_feature_count`` is unset (None) or zero (the initial
    # ctx default) until the first ``compress_with_polygram`` runs in
    # polygram mode. Pass-through compression also leaves it at zero;
    # in that case there's nothing meaningful for the controller to
    # target on the first pass, so we use the configured ``regrow_count``.
    if not ctx.get("current_feature_count"):
        return perform_regrowth(ctx, payload)

    delta: dict = {}
    delta.update(_compute_effective_regrow_count(ctx))
    inner_delta = perform_regrowth(ctx, payload)
    if isinstance(inner_delta, dict):
        delta.update(inner_delta)
    return delta


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
    """v0.1 4-step smoke loop. Preserves the byte-equivalence safety net.

    Also accumulates ``tokens_seen_in_task`` for the v0.2 token_budget
    trigger and adds the consumed input_ids to the replay buffer when
    one is registered.
    """
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
    tokens_seen = 0
    for _ in range(n_steps):
        logits = module(input_ids)
        targets = input_ids[:, 1:]
        preds = logits[:, :-1].reshape(-1, logits.size(-1))
        loss = F.cross_entropy(preds, targets.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        tokens_seen += int(input_ids.numel())
    module.eval()

    _add_to_replay_buffer(ctx, input_ids)

    output_dir = Path(ctx["output_dir"])
    finetuned_dir = output_dir / "finetuned"
    model.save_pretrained(finetuned_dir)
    _log(
        ctx,
        "fine_tune_model",
        {"mode": "trained", "n_steps": n_steps, "loss_first": losses[0], "loss_last": losses[-1]},
    )
    return {
        "finetuned_model_path": str(finetuned_dir),
        "_finetune_losses": losses,
        "tokens_seen_in_task": ctx.get("tokens_seen_in_task", 0) + tokens_seen,
    }


def _add_to_replay_buffer(ctx: dict, sequence) -> None:
    """Add a sequence to the registered replay buffer, if any."""
    buffer = ctx.get("_replay_buffer")
    if buffer is None:
        return
    buffer.add(sequence, task_id=ctx.get("task_idx", 0))


def _wrap_iterator_for_continual(ctx: dict, iterator):
    """Compose replay-mixing + token counting + buffer-add around the iterator.

    The returned generator yields the same shape as the input. As a side
    effect each yielded batch increments ``ctx['tokens_seen_in_task']``
    and is added to the replay buffer when one is registered.

    When ``replay_ratio == 0`` or the buffer is missing/empty, no replay
    mixing happens (counting and buffer-add still run).
    """
    from saeforge.training import MixedIterator

    buffer = ctx.get("_replay_buffer")
    replay_ratio = ctx.get("replay_ratio", 0.0)
    if buffer is not None and replay_ratio > 0:
        iterator = MixedIterator(iterator, buffer, replay_ratio=replay_ratio)

    def _instrumented():
        for batch in iterator:
            ctx["tokens_seen_in_task"] = ctx.get("tokens_seen_in_task", 0) + _batch_tokens(batch)
            if buffer is not None:
                buffer.add(batch, task_id=ctx.get("task_idx", 0))
            yield batch

    return _instrumented()


def _batch_tokens(batch) -> int:
    """Best-effort token count for a batch — handles tensors, dicts, tuples."""
    # Common cases: a tensor of input_ids, a dict with 'input_ids', a
    # (input_ids, labels) tuple. Anything else: count 0 and let the user
    # fix the iterator if precise budgeting matters.
    if hasattr(batch, "numel"):
        return int(batch.numel())
    if isinstance(batch, dict):
        ids = batch.get("input_ids")
        if ids is not None and hasattr(ids, "numel"):
            return int(ids.numel())
    if isinstance(batch, (list, tuple)) and batch:
        first = batch[0]
        if hasattr(first, "numel"):
            return int(first.numel())
    return 0


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

    # Wrap the iterator with replay mixing + token counting + buffer add.
    iterator = _wrap_iterator_for_continual(ctx, iterator)

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
        distill_alpha=ctx.get("finetune_distill_alpha", 1.0),
        distill_temperature=ctx.get("finetune_distill_temperature", 2.0),
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
    """Compute faithfulness via the active target, then derive
    ``should_continue`` and ``advance_stream``.

    Target dispatch:

    1. ``ctx["_faithfulness_target"]`` overrides everything when set
       (the user-supplied ``ForgePipeline(faithfulness=...)`` path).
    2. Otherwise, the family-based default policy in
       :func:`saeforge.eval.targets._default_target_for` picks
       :class:`~saeforge.eval.targets.KLTarget` for LM families and
       :class:`~saeforge.eval.targets.CosineTarget` for
       ``whisper_encoder``. Unknown family raises ``ValueError``.

    The ``faithfulness`` ctx field carries the target's score; the
    ``perplexity`` ctx field carries the target's perplexity analog
    (``exp(score)`` for ``better_when="lower"`` targets; ``1 - score``
    for ``better_when="higher"`` targets, clamped at 0). The
    ``should_continue`` predicate consults ``target.better_when``:

    - ``"lower"`` (KL, MSE): ``min_faithfulness`` follows the v0.1
      negation convention — pass a negative threshold to encode
      "max allowed score" (``min_faithfulness=-0.05`` continues only
      while ``score ≤ 0.05``).
    - ``"higher"`` (cosine, GT-alignment): ``min_faithfulness`` is
      the minimum required score — pass ``min_faithfulness=0.95``
      to require ``score ≥ 0.95``.

    The ``advance_stream`` stream-loop computation is target-agnostic
    (it consumes the target-supplied perplexity analog). Both
    predicate fields are written here so the FSM's ``eval_done``
    guards can be a flat ctx read. The stream-loop dominance contract
    (``advance_stream == true`` wins over ``should_continue``) lives
    in the guard expression ``refine_same_shard``, which requires
    ``advance_stream == false``.
    """
    from saeforge.eval.targets import _default_target_for

    host = ctx.get("_host_model")
    forged = ctx.get("_native_model")
    family = (
        forged.config.family
        if forged is not None and hasattr(forged, "config")
        else None
    )

    target = ctx.get("_faithfulness_target")
    if target is None:
        # Default policy: family dispatch into the built-in targets.
        # Pre-loading bootstrap paths can hit `evaluate_faithfulness`
        # before a native model exists in ctx; v0.4 silently fell to
        # the LM/KL arm and returned 0.0. Preserve that — only consult
        # family dispatch when a forged model is actually present.
        if forged is None:
            from saeforge.eval.targets import KLTarget

            target = KLTarget()
        else:
            target = _default_target_for(family)

    # Preserve the v0.4 defensive zero: when the forged module or its
    # required eval input is missing the target's score() would raise,
    # but the legacy action returned 0.0 in those cases so the
    # imperative byte-identity test could pass with no eval inputs.
    if forged is None or _missing_required_eval_input(ctx, target):
        score = 0.0
        perplexity = _default_perplexity_for(target, score)
    else:
        score, perplexity = target.score(forged=forged, host=host, ctx=ctx)

    should_continue = _should_continue_for(target, ctx, score, perplexity)
    advance_stream = _compute_advance_stream(ctx, score, perplexity)

    _log(
        ctx,
        "evaluate_faithfulness",
        {
            "faithfulness": score,
            "perplexity": perplexity,
            "should_continue": should_continue,
            "advance_stream": advance_stream,
            "target": getattr(target, "name", "?"),
        },
    )
    return {
        "faithfulness": float(score),
        "perplexity": float(perplexity),
        "should_continue": should_continue,
        "advance_stream": advance_stream,
    }


def _missing_required_eval_input(ctx: dict, target: Any) -> bool:
    """Replicate the v0.4 defensive-zero gate.

    Built-in targets read fixed ctx keys; if they're absent the legacy
    action returned 0.0 instead of raising. Replicate that policy for
    the two built-ins. Third-party targets are expected to raise their
    own ``KeyError`` from inside ``score`` per the protocol contract,
    so this helper only checks the names sae-forge ships.
    """
    name = getattr(target, "name", None)
    if name == "kl":
        return ctx.get("_eval_input_ids") is None
    if name == "cosine":
        return ctx.get("_eval_audio_features") is None
    return False


def _default_perplexity_for(target: Any, score: float) -> float:
    """Match the perplexity analog the target would have produced for
    ``score`` — used only on the defensive-zero path so the FSM's
    ``perplexity < best_perplexity`` check stays well-defined.
    """
    if getattr(target, "better_when", "lower") == "higher":
        return max(0.0, 1.0 - score)
    return float(np.exp(score)) if score >= 0 else float("inf")


def _should_continue_for(target: Any, ctx: dict, score: float, perplexity: float) -> bool:
    """Loop-continuation predicate, target-aware.

    ``better_when == "lower"`` preserves the v0.4 KL semantics exactly:
    ``min_faithfulness == 0.0`` keeps the gate open as long as ``score
    >= 0.0``; ``min_faithfulness < 0.0`` encodes a max allowed score via
    the legacy ``score <= min_faithfulness * -1`` predicate. ``better_when
    == "higher"`` preserves the v0.4 cosine semantics: ``score >=
    min_faithfulness``.
    """
    iters = ctx.get("iterations", 1)
    current = ctx.get("current_iter", 0)
    min_faith = ctx.get("min_faithfulness", 0.0)
    best_perp = ctx.get("best_perplexity", float("inf"))
    if getattr(target, "better_when", "lower") == "higher":
        gate = score >= min_faith
    else:
        gate = score >= min_faith if min_faith == 0.0 else score <= min_faith * -1
    return bool(current + 1 < iters and gate and perplexity < best_perp)


def _compute_advance_stream(ctx: dict, kl: float, perplexity: float) -> bool:
    """Decide whether the stream loop should advance to the next task.

    Implements the three ``task_trigger`` modes from the spec. The
    function appends to ``ctx['recent_eval_losses']`` (capped at 3) for
    the loss_delta path so subsequent calls have the window they need.
    All three modes share the budget guard ``task_idx + 1 < n_tasks``.
    """
    n_tasks = ctx.get("n_tasks", 1)
    task_idx = ctx.get("task_idx", 0)
    if task_idx + 1 >= n_tasks:
        return False

    trigger = ctx.get("task_trigger", "labeled")
    if trigger == "labeled":
        return True

    if trigger == "token_budget":
        budget = ctx.get("token_budget_per_task", 0)
        seen = ctx.get("tokens_seen_in_task", 0)
        return budget > 0 and seen >= budget

    if trigger == "loss_delta":
        # Use perplexity as the held-out probe signal — already computed
        # from the same KL the action just measured. Append, then check.
        history = list(ctx.get("recent_eval_losses", []))
        history.append(perplexity)
        history = history[-3:]
        ctx["recent_eval_losses"] = history
        if len(history) < 3:
            return False
        threshold = ctx.get("loss_delta_threshold", 0.0)
        prior_mean = (history[0] + history[1]) / 2.0
        return (history[-1] - prior_mean) > threshold

    raise ValueError(
        f"unknown task_trigger {trigger!r}; expected "
        "'labeled' | 'token_budget' | 'loss_delta'"
    )


def advance_to_next_task(ctx: dict, _payload: dict | None = None) -> dict:
    """Stream-loop advance: install next task's iterator, reset per-task counters.

    Loud-warns when ``n_tasks > 1`` but no TaskStream is registered. In
    that case the FSM still advances ``task_idx``, but the next fine-tune
    will reuse the previous task's iterator (likely already exhausted) —
    the loud warning surfaces the misconfiguration immediately instead of
    silently producing a degenerate run.
    """
    import warnings

    from saeforge.training import task_stream

    handle = ctx.get("task_iterator_id") or ""
    next_iterator = None
    if handle:
        try:
            stream = task_stream.get(handle)
            next_iterator = stream.next()
        except KeyError:
            next_iterator = None
    elif ctx.get("n_tasks", 1) > 1:
        warnings.warn(
            f"advance_to_next_task: n_tasks={ctx.get('n_tasks')} > 1 but no "
            "TaskStream is registered (ForgePipeline(task_stream=...) was not "
            "set). The FSM will advance task_idx but the next fine-tune will "
            "reuse the previous task's iterator. Pass a TaskStream to "
            "ForgePipeline to enable real cross-shard advancement.",
            UserWarning,
            stacklevel=2,
        )

    delta = {
        "task_idx": ctx.get("task_idx", 0) + 1,
        "inner_refine_idx": 0,
        "tokens_seen_in_task": 0,
        "current_iter": 0,
        "recent_eval_losses": [],
        # Carry forward `final_model_path`, `current_sae_path`, and
        # `protected_features` implicitly — they are not reset here.
    }
    if next_iterator is not None:
        delta["_finetune_iterator"] = next_iterator
    _log(
        ctx,
        "advance_to_next_task",
        {"task_idx": delta["task_idx"], "next_iterator_set": next_iterator is not None},
    )
    return delta


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

    target = ctx.get("_faithfulness_target")
    target_name = getattr(target, "name", None)
    if target_name is None:
        # Reconstruct the family-default's name so the JSON metadata
        # carries it even on the no-explicit-target path. Mirrors the
        # dispatch in `evaluate_faithfulness`.
        from saeforge.eval.targets import _default_target_for

        forged = ctx.get("_native_model")
        family = (
            forged.config.family
            if forged is not None and hasattr(forged, "config")
            else None
        )
        try:
            target_name = _default_target_for(family).name
        except ValueError:
            target_name = None
    score = ctx.get("faithfulness")
    summary: dict[str, Any] = {
        "host_model_id": ctx.get("host_model_id"),
        "n_params": n_params,
        "faithfulness": score,
        "faithfulness_target_name": target_name,
        "n_features": ctx.get("current_feature_count"),
        "iterations": ctx.get("current_iter", 0) + 1,
        "compress_mode": _last_log_extra(ctx, "compress_with_polygram", "mode"),
        "finetune_mode": _last_log_extra(ctx, "fine_tune_model", "mode"),
    }
    # Back-compat shim: every consumer of forge_result.json's
    # `faithfulness_kl` field keeps reading it through one minor
    # version. Populated when the active target is "kl"; null
    # otherwise. Removed alongside ForgeResult.faithfulness_kl.
    summary["faithfulness_kl"] = score if target_name == "kl" else None
    (output_dir / "forge_result.json").write_text(json.dumps(summary, indent=2))
    return {"final_model_path": str(forged_dir), "n_params": n_params}


def log_error(ctx: dict, payload: dict | None = None) -> dict:
    msg = (payload or {}).get("error", ctx.get("error_message", "unknown error"))
    _log(ctx, "log_error", {"error": msg})
    delta: dict = {"error_message": str(msg)}
    # error_origin_machine: the deepest sub-machine that originated the
    # error wins. Each machine's log_error writes its own name as a
    # fallback; once written, subsequent log_error calls (from machines
    # higher up the bubble chain) preserve the existing value.
    if not ctx.get("error_origin_machine"):
        path = ctx.get("_machine_path", "stream")
        delta["error_origin_machine"] = path.rsplit("/", 1)[-1]
    return delta


def _last_log_extra(ctx: dict, action_name: str, key: str):
    for entry in reversed(ctx.get("transitions_log", [])):
        if entry.get("action") == action_name and key in entry:
            return entry[key]
    return None


ACTION_TABLE: dict[str, Any] = {
    "load_sae_and_corpus": load_sae_and_corpus,
    "scan_activations": scan_activations,
    "load_and_scan": load_and_scan,
    "compress_with_polygram": compress_with_polygram,
    "perform_regrowth": perform_regrowth,
    "adapt_and_regrow": adapt_and_regrow,
    "project_to_subspace": project_to_subspace,
    "fine_tune_model": fine_tune_model,
    "evaluate_faithfulness": evaluate_faithfulness,
    "advance_to_next_task": advance_to_next_task,
    "rotate_for_next_iter": rotate_for_next_iter,
    "save_final_model": save_final_model,
    "log_error": log_error,
}
