"""run_finetune — the actual training loop. AdamW + cosine LR + clipping +
optional grad checkpointing + optional mixed precision + periodic eval/save.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from statistics import mean

from saeforge.training.config import TrainingConfig, TrainingResult
from saeforge.training.corpus import take
from saeforge.training.schedules import cosine_with_warmup
from saeforge.utils.lazy import require_extra


def run_finetune(model, host, iterator, config: TrainingConfig) -> TrainingResult:
    """Run ``config.total_steps`` of LM cross-entropy training on ``model``.

    ``model`` is a `saeforge.NativeModel`. ``host`` is the source HF model
    (used for periodic faithfulness eval, and — when
    `config.distill_alpha < 1.0` — for every-step host-distillation
    KL). Pass `None` to skip eval; passing `None` with
    `distill_alpha < 1.0` raises before any batches are consumed.
    ``iterator`` is any iterable yielding ``(batch_size,
    sequence_length)`` int64 token tensors — typically built via
    `saeforge.training.build_iterator`.
    """
    if config.distill_alpha < 1.0 and host is None:
        raise ValueError(
            "run_finetune: distill_alpha < 1.0 requires a non-None "
            "`host` model (the distillation teacher); got "
            f"distill_alpha={config.distill_alpha} and host=None"
        )

    torch = require_extra("torch", "torch")
    F = torch.nn.functional

    module = model.torch_module
    device = next(module.parameters()).device

    # Concept-anchoring setup (add-concept-anchored-finetune).
    # When concept_alpha > 0, instantiate the label source via the
    # registry, run one-time calibration, construct the two heads, and
    # add their parameters to the optimiser's param set BEFORE the
    # optimiser is built. concept_alpha == 0.0 is the disable path —
    # the entire block is skipped (no instantiation, no extra params).
    concept_state: dict = {}
    if config.concept_alpha > 0:
        from saeforge.training.concept_anchor import LABEL_SOURCE_REGISTRY
        from saeforge.training.heads import PerChannelConceptHead, PooledConceptHead

        label_source_cls = LABEL_SOURCE_REGISTRY[config.concept_label_source]
        label_source = label_source_cls(**config.concept_label_source_kwargs)
        # prepare(...) consumes `calibration_batches` from the iterator;
        # see docs/concept-anchoring.md for the iterator-consumption caveat.
        n_concepts = int(label_source.prepare(module, iterator))
        d_model = int(model.config.hidden_size)
        if n_concepts > d_model:
            raise ValueError(
                f"concept-anchoring: per-channel head needs at least "
                f"n_concepts (={n_concepts}) residual dims, but the model "
                f"has hidden_size={d_model}. Either drop concept_channel_weight "
                f"to 0 (pooled-only) or pick a polygram basis with fewer "
                f"clusters."
            )
        pooled_head = PooledConceptHead(d_model=d_model, n_concepts=n_concepts).to(device)
        channel_head = PerChannelConceptHead(n_concepts=n_concepts).to(device)
        concept_state = {
            "label_source": label_source,
            "pooled_head": pooled_head,
            "channel_head": channel_head,
            "n_concepts": n_concepts,
        }

    # Byte-identity branch: when concept anchoring is off, build the
    # optimiser exactly as the pre-change loop did (module.parameters()
    # iterator directly, NOT wrapped in a param-groups list). Even
    # though the param-groups form is semantically equivalent, AdamW's
    # internal state initialization is sensitive to the iterator vs
    # list-of-groups distinction in ways that change downstream
    # numerics — caught by `test_forge_result_digest_per_family`.
    if concept_state:
        optim_param_groups: list[dict] = [
            {"params": list(module.parameters())},
            {
                "params": list(concept_state["pooled_head"].parameters())
                + list(concept_state["channel_head"].parameters()),
            },
        ]
        optim = torch.optim.AdamW(
            optim_param_groups,
            lr=config.peak_lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    else:
        optim = torch.optim.AdamW(
            module.parameters(),
            lr=config.peak_lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

    if config.gradient_checkpointing:
        _enable_grad_checkpointing(module)

    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[
        config.precision
    ]
    scaler = (
        torch.amp.GradScaler(device.type)
        if config.precision == "fp16" and device.type == "cuda"
        else None
    )

    loss_history: list[tuple[int, float]] = []
    eval_history: list[tuple[int, float]] = []
    save_paths: list = []
    metadata: dict = {"oom_batch_halved": False}
    if concept_state:
        metadata["concept_anchoring"] = {
            "n_concepts": concept_state["n_concepts"],
            "label_source": config.concept_label_source,
            "concept_alpha": config.concept_alpha,
            "concept_pool_weight": config.concept_pool_weight,
            "concept_channel_weight": config.concept_channel_weight,
            "concept_focal_gamma": config.concept_focal_gamma,
        }

    module.train()
    t0 = time.monotonic()
    final_loss = float("nan")
    n_steps_completed = 0

    for step, batch in enumerate(take(iterator, config.total_steps)):
        lr = cosine_with_warmup(
            step, config.total_steps, config.warmup_steps, config.peak_lr, config.min_lr_ratio
        )
        for group in optim.param_groups:
            group["lr"] = lr

        batch = batch.to(device)
        try:
            with _autocast(device.type, autocast_dtype):
                # Concept anchoring needs the residual stream — the
                # native module's forward returns logits only, so we
                # capture the residual at the lm_head's input via a
                # forward pre-hook. This avoids broadening every
                # adapter's forward signature with
                # ``output_hidden_states``.
                #
                # ADAPTER CONTRACT: this hook assumes the architecture's
                # final unembed (a.k.a. LM head) is an attribute named
                # ``lm_head`` on the module and is called as
                # ``lm_head(residual)`` — i.e. its first positional arg
                # IS the post-final-layer residual stream. The
                # ForgedGPT2 / ForgedLlamaFamily / ForgedGemma2 /
                # ForgedQwen3 native classes all conform; custom
                # architectures that register an alternate unembed name
                # (or wrap the head in a different positional contract)
                # MUST either rename to ``lm_head`` OR drop in their own
                # residual-capture hook before invoking ``run_finetune``
                # with ``concept_alpha > 0``.
                residual = None
                hook_handle = None
                if concept_state:
                    captured: dict = {}

                    def _capture_residual(_module, args, _captured=captured):
                        # args[0] is the residual being fed to lm_head
                        if args:
                            _captured["residual"] = args[0]

                    lm_head = getattr(module, "lm_head", None)
                    if lm_head is None:
                        raise RuntimeError(
                            "concept anchoring expects `module.lm_head` to "
                            "exist for residual-stream capture; the active "
                            "adapter does not expose one. See the ADAPTER "
                            "CONTRACT comment in run_finetune (loop.py) for "
                            "the contract and the workarounds for custom "
                            "architectures."
                        )
                    hook_handle = lm_head.register_forward_pre_hook(_capture_residual)
                try:
                    logits = module(batch)
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()
                if concept_state:
                    residual = captured.get("residual")
                ce_loss = _shift_lm_loss(logits, batch, F)

                if config.distill_alpha < 1.0:
                    # Host-distillation: teacher forward under no_grad
                    # on the same batch + autocast context. Gradients
                    # flow only through the student.
                    #
                    # Future optimization: when the corpus iterator is
                    # deterministic (static corpus, fixed shuffle seed),
                    # host_logits could be precomputed once per epoch
                    # and cached. The deferred caching path documented
                    # in add-host-distillation-finetune-loss design.md
                    # Decision 5 would plug in here, keyed on
                    # `(batch_hash, step)`. Out of scope for v1.
                    with torch.no_grad():
                        host_out = host(input_ids=batch)
                        host_logits = (
                            host_out.logits
                            if hasattr(host_out, "logits")
                            else host_out[0]
                        )
                    # Hinton-style soft-label KL with tau^2 rescaling.
                    # Direction matches saeforge/eval/faithfulness.py
                    # (KL(host || forged)) so the training objective
                    # is the same quantity the eval reports.
                    tau = config.distill_temperature
                    kd_loss = (tau ** 2) * F.kl_div(
                        F.log_softmax(logits / tau, dim=-1),
                        F.softmax(host_logits / tau, dim=-1),
                        reduction="batchmean",
                    )
                    existing_loss = (
                        config.distill_alpha * ce_loss
                        + (1.0 - config.distill_alpha) * kd_loss
                    )
                else:
                    existing_loss = ce_loss

                # Concept-anchoring branch.
                # L_concept = β_pool·focal_BCE(pool_logits, max_T(labels))
                #           + β_chan·focal_BCE(channel_logits, labels)
                # total = (1-α)·existing + α·L_concept
                concept_loss_value: float | None = None
                if concept_state:
                    from saeforge.training.heads import focal_bce_loss

                    if residual is None:
                        raise RuntimeError(
                            "concept anchoring active but the student forward "
                            "did not return hidden_states; check that "
                            "`module(batch, output_hidden_states=True)` is "
                            "supported by this architecture."
                        )
                    label_source = concept_state["label_source"]
                    pooled_head = concept_state["pooled_head"]
                    channel_head = concept_state["channel_head"]
                    n_concepts = concept_state["n_concepts"]

                    # Labels are computed under no_grad through the
                    # frozen polygram projection; gradients flow only
                    # through the student → heads → loss chain.
                    with torch.no_grad():
                        labels = label_source.labels_for_batch(
                            batch, hidden_states=residual.detach()
                        )
                    pool_labels = labels.amax(dim=1)  # (B, n_concepts)

                    pool_logits = pooled_head(residual)
                    pool_loss = focal_bce_loss(
                        pool_logits, pool_labels, gamma=config.concept_focal_gamma,
                    )

                    channel_input = residual[..., -n_concepts:]
                    channel_logits = channel_head(channel_input)
                    channel_loss = focal_bce_loss(
                        channel_logits, labels, gamma=config.concept_focal_gamma,
                    )

                    concept_loss = (
                        config.concept_pool_weight * pool_loss
                        + config.concept_channel_weight * channel_loss
                    )
                    loss = (
                        (1.0 - config.concept_alpha) * existing_loss
                        + config.concept_alpha * concept_loss
                    )
                    concept_loss_value = float(concept_loss.detach())
                else:
                    loss = existing_loss

            # Parameters to clip: the module always, plus the two
            # concept heads when concept anchoring is active.
            clip_params = list(module.parameters())
            if concept_state:
                clip_params.extend(concept_state["pooled_head"].parameters())
                clip_params.extend(concept_state["channel_head"].parameters())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(clip_params, config.max_grad_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, config.max_grad_norm)
                optim.step()
            optim.zero_grad(set_to_none=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # MPS reports OOM as RuntimeError("MPS backend out of memory…");
            # CUDA has its own subclass. Filter RuntimeError to OOM-only so we
            # don't swallow unrelated failures.
            if isinstance(e, RuntimeError) and not isinstance(e, torch.cuda.OutOfMemoryError):
                if "out of memory" not in str(e).lower():
                    raise
            if metadata["oom_batch_halved"]:
                raise
            metadata["oom_batch_halved"] = True
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            optim.zero_grad(set_to_none=True)
            continue

        final_loss = float(loss.item())
        n_steps_completed = step + 1
        if step % config.log_every_steps == 0 or step == config.total_steps - 1:
            loss_history.append((step, final_loss))
            if concept_loss_value is not None:
                # Stash the per-step concept loss alongside the total so
                # analysts can audit "how much of the loss is concept
                # anchoring contributing right now". Lives in metadata so
                # we don't widen the LossHistory schema for a knob most
                # callers don't enable.
                metadata.setdefault("concept_loss_history", []).append(
                    (step, concept_loss_value)
                )

        if (
            config.eval_input_ids is not None
            and host is not None
            and step > 0
            and step % config.eval_every_steps == 0
        ):
            module.eval()
            with torch.no_grad():
                kl = _eval_kl(module, host, config.eval_input_ids, device)
            module.train()
            eval_history.append((step, kl))

        if (
            config.save_dir is not None
            and step > 0
            and step % config.save_every_steps == 0
        ):
            ckpt_path = Path(config.save_dir) / f"step-{step:06d}"
            model.save_pretrained(ckpt_path)
            save_paths.append(ckpt_path)

    module.eval()
    return TrainingResult(
        final_loss=final_loss,
        loss_history=loss_history,
        eval_history=eval_history,
        wall_seconds=time.monotonic() - t0,
        n_steps_completed=n_steps_completed,
        save_paths=save_paths,
        converged=_check_convergence(loss_history),
        metadata=metadata,
    )


def _shift_lm_loss(logits, input_ids, F) -> "torch.Tensor":  # noqa: F821 — torch
    targets = input_ids[:, 1:]
    preds = logits[:, :-1].reshape(-1, logits.size(-1))
    return F.cross_entropy(preds, targets.reshape(-1))


def _autocast(device_type: str, dtype):
    import torch

    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


def _enable_grad_checkpointing(module) -> None:
    """Wrap each transformer block in torch.utils.checkpoint.checkpoint.

    We patch each block's ``forward`` to route through ``checkpoint``,
    halving activation memory at the cost of recompute on backward.

    Family-aware: dispatches via ``adapter_for_family(module.config.family)``
    to find the right block list and embedding parameter for the
    architecture (GPT-2 uses ``module.transformer.{h, wte}``;
    Llama / Gemma-2 use ``module.model.{layers, embed_tokens}``).
    Pre-fix this hardcoded the GPT-2 layout and crashed inside the FSM
    on Llama / Gemma-2 hosts, surfacing as a silent
    ``final_state: failed`` in the run summary with KL=0.0.
    """
    from torch.utils.checkpoint import checkpoint

    from saeforge.adapters import adapter_for_family

    adapter = adapter_for_family(module.config.family)
    blocks, embedding_param = adapter.grad_checkpoint_targets(module)
    for block in blocks:
        original_forward = block.forward

        def checkpointed_forward(x, _orig=original_forward):
            return checkpoint(_orig, x, use_reentrant=False)

        block.forward = checkpointed_forward
    # Activation checkpointing requires at least one input to require
    # grad — the embedding output isn't itself a leaf tensor, so we mark
    # the embedding *weight* as requires_grad. (It's already part of the
    # param set; this just routes gradient through it.)
    embedding_param.requires_grad_(True)


def _eval_kl(forged_module, host, eval_input_ids, device) -> float:
    import torch

    F = torch.nn.functional
    eval_input_ids = eval_input_ids.to(device)
    forged_logits = forged_module(eval_input_ids)
    host_module = host.to(device).eval() if hasattr(host, "to") else host
    host_out = host_module(input_ids=eval_input_ids)
    host_logits = host_out.logits if hasattr(host_out, "logits") else host_out[0]
    log_q = F.log_softmax(forged_logits, dim=-1)
    log_p = F.log_softmax(host_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1).mean()
    return float(kl.item())


def _check_convergence(loss_history: list, window: int = 100, threshold: float = 0.01) -> bool:
    if len(loss_history) < 2 * window:
        return False
    recent = [entry[1] for entry in loss_history[-window:]]
    prior = [entry[1] for entry in loss_history[-2 * window : -window]]
    if not recent or not prior:
        return False
    prior_mean = mean(prior)
    if prior_mean == 0.0:
        return False
    return abs(mean(recent) - prior_mean) / abs(prior_mean) < threshold
