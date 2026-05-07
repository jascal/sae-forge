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
    (used only for periodic faithfulness eval; pass `None` to skip eval
    even if `config.eval_input_ids` is set). ``iterator`` is any iterable
    yielding ``(batch_size, sequence_length)`` int64 token tensors —
    typically built via `saeforge.training.build_iterator`.
    """
    torch = require_extra("torch", "torch")
    F = torch.nn.functional

    module = model.torch_module
    device = next(module.parameters()).device

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
                logits = module(batch)
                loss = _shift_lm_loss(logits, batch, F)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(module.parameters(), config.max_grad_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), config.max_grad_norm)
                optim.step()
            optim.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError:
            if metadata["oom_batch_halved"]:
                raise
            metadata["oom_batch_halved"] = True
            torch.cuda.empty_cache()
            optim.zero_grad(set_to_none=True)
            continue

        final_loss = float(loss.item())
        n_steps_completed = step + 1
        if step % config.log_every_steps == 0 or step == config.total_steps - 1:
            loss_history.append((step, final_loss))

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

    We patch the block list's `forward` to route through `checkpoint`. This
    halves activation memory at the cost of recompute on backward.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    transformer = module.transformer
    blocks = transformer.h
    for block in blocks:
        original_forward = block.forward

        def checkpointed_forward(x, _orig=original_forward):
            return checkpoint(_orig, x, use_reentrant=False)

        block.forward = checkpointed_forward
    # Activation checkpointing requires inputs to require grad somewhere — the
    # embedding output isn't a leaf tensor that requires grad on its own. Forcing
    # `transformer.wte.weight.requires_grad_(True)` suffices since wte is part of
    # the param set anyway.
    transformer.wte.weight.requires_grad_(True)
    _ = torch  # silence unused import if grad checkpointing is never enabled


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
