# Design: forge fine-tune recipe

## Module layout

```
saeforge/
  training/
    __init__.py        # re-exports
    config.py          # TrainingConfig, TrainingResult
    schedules.py       # cosine_with_warmup
    corpus.py          # build_iterator, lazy `datasets`
    loop.py            # run_finetune — the actual loop
```

Why split: `loop.py` ends up around 100 lines once you fold in
gradient checkpointing, mixed precision, periodic eval, periodic
saves, and structured logging. Keeping `schedules` and `corpus`
separate lets each be unit-tested without spinning up torch.

## TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Optimization
    total_steps: int = 1000
    warmup_steps: int = 100
    peak_lr: float = 5e-5
    min_lr_ratio: float = 0.1       # cosine decays to peak_lr * min_lr_ratio
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95              # Llama / Gemma convention
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Throughput
    batch_size: int = 8
    sequence_length: int = 512
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    gradient_checkpointing: bool = False

    # Eval
    eval_every_steps: int = 100
    eval_input_ids: torch.Tensor | None = None  # if None, eval is skipped

    # Checkpointing
    save_every_steps: int = 250
    save_dir: Path | None = None

    # Logging
    log_every_steps: int = 10        # how often to record loss in result.loss_history
```

Defaults target a Gemma-2-2B forge on a 24GB GPU. For small hosts
(GPT-2-small) the user overrides `peak_lr` upward (1e-4) and may
disable `gradient_checkpointing`.

## TrainingResult

```python
@dataclass
class TrainingResult:
    final_loss: float
    loss_history: list[tuple[int, float]]    # [(step, loss), ...]
    eval_history: list[tuple[int, float]]    # [(step, kl), ...]
    wall_seconds: float
    n_steps_completed: int
    save_paths: list[Path]                   # checkpoints written during the run
    converged: bool                          # whether loss plateaued (heuristic)
```

`converged` is set when the trailing-100-step loss EMA changes by
less than 1% across two consecutive 100-step windows. Heuristic only;
not an early-exit signal in v0.3.

## The loop

Pseudo:

```python
def run_finetune(model, host, iterator, config):
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    if config.gradient_checkpointing:
        _enable_grad_checkpointing(model.torch_module)

    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[config.precision]
    scaler = torch.cuda.amp.GradScaler() if config.precision == "fp16" else None

    loss_history = []
    eval_history = []
    save_paths = []
    t0 = time.monotonic()

    for step, batch in enumerate(_take(iterator, config.total_steps)):
        lr = cosine_with_warmup(step, config.total_steps, config.warmup_steps,
                                config.peak_lr, config.min_lr_ratio)
        for group in optim.param_groups:
            group["lr"] = lr

        with _autocast(autocast_dtype):
            logits = model.torch_module(batch)
            loss = _shift_lm_loss(logits, batch)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optim.step()
        optim.zero_grad(set_to_none=True)

        if step % config.log_every_steps == 0:
            loss_history.append((step, float(loss.item())))

        if config.eval_input_ids is not None and step % config.eval_every_steps == 0 and step > 0:
            kl = _kl_from_input_ids(model, host, config.eval_input_ids)
            eval_history.append((step, kl))

        if config.save_dir is not None and step % config.save_every_steps == 0 and step > 0:
            ckpt_path = config.save_dir / f"step-{step:06d}"
            model.save_pretrained(ckpt_path)
            save_paths.append(ckpt_path)

    return TrainingResult(
        final_loss=float(loss.item()),
        loss_history=loss_history,
        eval_history=eval_history,
        wall_seconds=time.monotonic() - t0,
        n_steps_completed=config.total_steps,
        save_paths=save_paths,
        converged=_check_convergence(loss_history),
    )
```

Real implementation will handle iterator-exhaustion (rewind by
restarting `iterator` if it's an iterator factory; raise otherwise)
and OOM via batch-halving retry (single-shot retry, not exhaustive).

## Cosine schedule

Closed-form, no torch dep:

```python
def cosine_with_warmup(step, total_steps, warmup_steps, peak_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
```

Properties verified by tests:
- step=0 → peak_lr / warmup_steps (small)
- step=warmup_steps → peak_lr (peak)
- step=total_steps → peak_lr * min_lr_ratio (floor)
- step > total_steps → peak_lr * min_lr_ratio (clamped, no NaN)

## Gradient checkpointing

Apply via `torch.utils.checkpoint.checkpoint_sequential` over the
transformer block list (`model.torch_module.transformer.h`). Saves
~50% activations memory; trades off ~25% wall time. Worthwhile when:
- forged model has ≥6 layers AND
- forward batch × seq is large enough that activations dominate over
  optimizer state.

For GPT-2 toy: not worth it. For Gemma-2-2B: roughly the difference
between fitting in 24GB and OOMing.

The wrapping happens at training-loop start, scoped to the loop only —
inference paths stay un-checkpointed for speed.

## Mixed precision

Three modes, selected by `config.precision`:

- **fp32** (default): no autocast, no scaler. Maximum precision,
  highest memory.
- **bf16**: `torch.autocast(device_type, dtype=torch.bfloat16)`,
  no scaler (bf16's exponent range eliminates the need for loss
  scaling). Recommended for Apple Silicon and modern CUDA.
- **fp16**: `torch.autocast(...)` + `GradScaler`. Backward-compat
  for older NVIDIA cards (V100, T4).

The autocast wraps only the forward pass + loss; backward and optimizer
step run at native param dtype. Standard pattern.

## Corpus iterator

`build_iterator` accepts:

- A pre-tokenized `Iterable[torch.Tensor]` (each tensor shape
  `(batch_size, sequence_length)`, int64): used as-is.
- A HuggingFace dataset name (str like `"HuggingFaceFW/fineweb-edu"`):
  lazy-imports `datasets`, streams in chunks, tokenizes with the host
  tokenizer, batches to `batch_size × sequence_length`.

The HF path:
1. `datasets.load_dataset(name, streaming=True)`
2. Filter empty/short examples (< sequence_length tokens)
3. Tokenize with the host tokenizer (`return_tensors="pt"`,
   `truncation=True`, `padding="max_length"`)
4. Batch into `(batch_size, sequence_length)` tensors
5. Yield indefinitely (loops back when source exhausts; total_steps
   bounds the actual consumption)

## Eval during fine-tune

Every `eval_every_steps`, run faithfulness KL on
`config.eval_input_ids`. Same `_kl_from_input_ids` helper the v0.1 FSM
already uses. Switches the model to `eval()` for the eval pass, back
to `train()` after.

The eval set is held-out — caller's responsibility to not include it
in the training corpus. v0.3 doesn't enforce this; tests do, and the
example documents it.

## Why not the HuggingFace Trainer

We could plug in `transformers.Trainer` and inherit a lot of plumbing
(checkpoint saving, logging integrations, distributed training). I'm
not going there in v0.3 because:

1. **Trainer bakes in transformers' model class assumptions.** Our
   `NativeModel` isn't a `PreTrainedModel` and integrating cleanly
   would mean either subclassing `PreTrainedModel` (a lot of API
   surface to inherit) or writing adapter glue.
2. **The recipe is short enough.** The whole loop is ~80 lines. Trainer
   would replace it with ~5 lines of config but pull in a 5000-line
   dependency surface that we can't easily customize when the recipe
   needs to evolve.
3. **Multi-GPU is out of scope anyway.** Trainer's main value-add
   beyond what we ship here is its DDP / FSDP integration. We're
   single-GPU in v0.3 by design.

If the v1.x roadmap adds multi-GPU and the loop grows beyond ~150
lines, revisit Trainer integration as a separate
`forge-trainer-integration` change.

## OOM handling

Single-shot batch-size halving on OOM, no retry beyond that:

```python
try:
    forward + loss + backward
except torch.cuda.OutOfMemoryError:
    if not retried:
        torch.cuda.empty_cache()
        config.batch_size //= 2
        retried = True
        # rebuild iterator with new batch_size
    else:
        raise
```

This catches the common "I picked a too-big batch" mistake without
turning the loop into a retry forest. v0.3 logs the halving in
`TrainingResult.metadata` so the user knows it happened.

For Apple Silicon: `torch.cuda.OutOfMemoryError` doesn't fire on MPS;
MPS just kills the process. Document this in the recipe section.

## Convergence heuristic

`_check_convergence(loss_history)`:

```python
def _check_convergence(loss_history, window=100, threshold=0.01):
    if len(loss_history) < 2 * window:
        return False
    recent = [l for s, l in loss_history[-window:]]
    prior = [l for s, l in loss_history[-2*window:-window]]
    if not recent or not prior:
        return False
    return abs(mean(recent) - mean(prior)) / mean(prior) < threshold
```

Returns True when the trailing-100-step loss EMA changes by less than
1% vs the previous 100-step window. Heuristic — useful for the
`TrainingResult.converged` field but not used as an early-exit
trigger in v0.3 (we always run total_steps).

Future extension (`forge-finetune-early-stop`): treat `converged=True`
as an early-exit signal, configurable via `config.early_stop_on_converge`.
Keeping it out of v0.3 because the convergence heuristic itself needs
empirical validation on real runs first.

## Open questions deferred

- **Optimal default LR per host scale.** v0.3 uses `5e-5` (Gemma's
  reported recipe). For GPT-2-small the empirical optimum may be
  higher (1e-4). Track via `docs/research/forge-finetune-lr-sweep.md`
  once we have data.
- **Corpus quality matters.** Fineweb-edu vs C4 vs The Pile likely
  produce different recovery curves. Empirical question; not a v0.3
  spec concern.
- **Distillation as additional loss term.** Adding `KL(host || forged)`
  to the LM CE loss with some coefficient may speed recovery. Lands
  as `forge-distill-recipe` if signal warrants.
