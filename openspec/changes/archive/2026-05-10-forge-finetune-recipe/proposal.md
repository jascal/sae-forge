## Why

The v0.1 `fine_tune_model` action runs 4 AdamW steps with a constant
learning rate against a pre-tokenized batch held in `ctx`. That's a
smoke test, not a recipe. It runs to completion, the loss usually
decreases, and that's enough to verify the action wires up — but it
won't recover capability after projection, won't converge to anything
meaningful on a real corpus, and won't fit a Gemma-2 host's
activations on 24GB hardware.

This change adds a real fine-tune recipe behind the same FSM action so
runs against `examples/forge_gpt2_real_sae.py` and the planned
`examples/forge_gemma2_2b.py` produce capability-recovering forged
models, not structurally-correct-but-broken ones.

The bottleneck this lifts: today, the only published demo of sae-forge
goes "compress 32 features → forge → KL=9.5". With this change the
demo becomes "compress 256 features → forge → fine-tune 1000 steps on
Fineweb-edu → KL approaches host baseline". That's the difference
between a structurally correct prototype and a research artifact.

## What Changes

### New `saeforge.training` module

- `saeforge/training/__init__.py` — re-exports `TrainingConfig`,
  `TrainingResult`, `run_finetune`.
- `saeforge/training/config.py` — `TrainingConfig` dataclass with the
  full recipe knob set (peak_lr, warmup_steps, total_steps, batch_size,
  sequence_length, weight_decay, beta1, beta2, max_grad_norm,
  precision, eval_every_steps, eval_input_ids, save_every_steps,
  gradient_checkpointing).
- `saeforge/training/schedules.py` — `cosine_with_warmup(step,
  total_steps, warmup_steps, peak_lr, min_lr_ratio=0.1) -> float`. One
  closed-form LR-schedule helper, no torch dep.
- `saeforge/training/loop.py` — `run_finetune(model, host, iterator,
  config) -> TrainingResult` with the full loop: AdamW with weight
  decay, gradient clipping, cosine LR schedule, optional gradient
  checkpointing, optional bf16/fp16 autocast, periodic faithfulness
  eval, periodic checkpoint saves, structured per-step loss log.
- `saeforge/training/corpus.py` — `build_iterator(source: str | Path | Iterable,
  tokenizer, batch_size, sequence_length) -> Iterator[Tensor]`.
  Accepts (in priority order):
  1. A local file path to a `.txt` (one document per line) or
     `.jsonl` (with a `"text"` field per line) — first-class for
     work / proprietary data scenarios where nothing should hit HF.
  2. A local directory of such files — read recursively.
  3. A HuggingFace dataset name (lazy-imports `datasets`) — for
     public corpora.
  4. Any pre-tokenized iterable yielding `(batch_size, sequence_length)`
     int64 tensors — for users with their own tokenization pipeline.
  Returns batches of `(batch_size, sequence_length)` int64 input_ids on demand.

### Modified FSM action

- `saeforge/actions/__init__.py`'s `fine_tune_model` becomes a thin
  ~20-line wrapper that:
  - reads training knobs from ctx (`finetune_steps`, `finetune_lr`,
    `finetune_corpus`, `finetune_batch_size`, `finetune_seq_len`,
    `finetune_precision`, `finetune_grad_checkpoint`, etc.),
  - calls `run_finetune` (which handles all the loop logic),
  - logs the structured `TrainingResult` (loss curve, final loss,
    eval KL trajectory, wall time) into `ctx["transitions_log"]`,
  - falls back to the v0.1 4-step pass-through when no corpus and no
    `_finetune_iterator` are supplied (preserving the byte-equivalence
    safety net for tests that don't supply a corpus).

### ForgePipeline knob expansion

`ForgePipeline` gains the same knobs (`finetune_corpus`,
`finetune_batch_size`, `finetune_seq_len`, `finetune_precision`,
`finetune_grad_checkpoint`, `eval_every_steps`, `save_every_steps`)
defaulting to v0.1 behaviour when omitted. The CLI gets matching
flags.

### New example

`examples/forge_gemma2_2b.py` — the headline demo:

- Pulls `google/gemma-scope-2b-pt-res` SAE checkpoints from HF (one
  layer; configurable).
- Slices to a configurable feature subset (default 256).
- Runs `polygram.EpochCompressor` against Gemma-2-2B forward passes
  on a Fineweb-edu slice.
- Forges with `attention_width="host"` (v0.2 feature_native is
  separate).
- Fine-tunes via the new recipe: 1000 AdamW steps, batch=8, seq=512,
  bf16, cosine LR with 100-step warmup, gradient checkpointing.
- Eval every 100 steps; save every 250 steps.
- Wall clock target: ~30–90 min on a 24GB GPU.

The script accepts `--device cuda` and `--device mps` cleanly; default
is `cpu` for a degraded but-still-runnable smoke run.

## Capabilities

### New Capabilities

- `forge-finetune-recipe`: A real LM-cross-entropy fine-tune loop
  with cosine LR + warmup, gradient clipping, optional gradient
  checkpointing, optional mixed precision (bf16/fp16), periodic
  faithfulness eval, periodic checkpoint saves, and structured loss
  tracking. Accepts a HuggingFace dataset name or any pre-tokenized
  iterable.

### Modified Capabilities

- `forge-outer-loop-fsm`: `fine_tune_model` action delegates to
  `saeforge.training.run_finetune` when a corpus is supplied; falls
  back to the v0.1 single-batch loop when only `_finetune_input_ids`
  is supplied (preserving the byte-equivalence safety net).
- `forge-pipeline`: `ForgePipeline` gains the recipe knobs as fields;
  `_run_synthetic_fsm` threads them onto ctx; `run` and `run_synthetic`
  pass them through.

## Impact

- ~250 new lines under `saeforge/training/` across config, schedule,
  loop, corpus.
- ~30-line rewrite of `fine_tune_model` action.
- ~10 new fields on `ForgePipeline`.
- ~150-line `examples/forge_gemma2_2b.py`.
- 6 new tests:
  - cosine schedule shape correctness (warmup + decay)
  - run_finetune on tiny synthetic iterator reduces loss monotonically
  - bf16 autocast doesn't break fp32 numerics-sensitive paths
  - gradient checkpointing produces identical gradients to non-checkpointed
    (within fp tolerance) — a unit-level correctness check
  - periodic eval logs are structured correctly
  - missing-corpus fallback preserves v0.1 single-batch behaviour
- README "Quickstart" + "How it works" gain a "Fine-tune recipe"
  subsection.
- `[torch]` extra bumps `transformers>=4.46` (required for Gemma-2
  support); `[recipe]` extra optional, pulls `datasets>=2.16`.
- AGENTS.md: add v0.3 (`forge-finetune-recipe`) to the milestone list.

## Data handling guarantees

The recipe SHALL be local-only by default. Specifically:

- **No automatic uploads.** Forged checkpoints, loss curves, eval
  outputs all stay under `output_dir`. Nothing is pushed to HF Hub,
  W&B, or any remote service unless the caller explicitly opts in
  (no such opt-in ships in v0.3).
- **No telemetry.** sae-forge ships zero phone-home behaviour.
  Polygram and orca-runtime-python are vetted for the same.
- **HF cache stays local.** Models downloaded from HF go to
  `~/.cache/huggingface/` per HF's standard layout. Nothing about
  the training corpus, the SAE, or the forged outputs is sent to HF.
- **Local-corpus-first ergonomics.** The recipe's primary corpus
  input is a local file or directory path. HF dataset names are
  supported but explicitly second-priority — work / proprietary data
  flows are the design target, not an afterthought.
- **Derivative-work disclosure.** Forged checkpoints fine-tuned on a
  corpus are derivative works of that corpus under most jurisdictions'
  copyright doctrine. v0.3 documents this in the recipe section's
  output README; users handling restricted data are responsible for
  treating the forged checkpoint with the same restrictions.

These guarantees are spec-level, not implementation niceties. Tests
verify that no network call is made when a local-path corpus is used
(except the once-per-host model + tokenizer download, which is
unavoidable and cached).

## Out of scope

- **Multi-GPU training (DDP / FSDP).** Single-GPU only in v0.3. Real
  multi-GPU training lands as a separate `forge-multi-gpu` change once
  there's a workload (probably 8B+) that actually requires it.
- **LoRA / adapter fine-tuning.** The recipe trains the full forged
  model. LoRA is a follow-up `forge-lora-recipe` change; useful when
  the forged model is large enough that full fine-tune memory is the
  bottleneck.
- **Distillation losses.** v0.3 trains plain LM cross-entropy on the
  corpus. Adding a KL-to-host distillation term is a knob worth
  exploring but lands in `forge-distill-recipe`. The current loss
  cleanly recovers capability when the basis spans enough of the
  residual; distillation matters more for aggressive compressions
  where projection error is large.
- **Curriculum / data mixing.** v0.3 streams a single dataset.
  Multi-corpus mixing (e.g. Fineweb + code + math) is a follow-up.
- **Hyperparameter search.** v0.3 ships sensible defaults derived from
  Gemma's published recipe; HP sweeps are out of scope and live in
  `docs/research/` write-ups when run.

## Migration path

- **v0.1 (shipped)**: 4-step smoke fine-tune.
- **v0.3 (this change)**: real recipe; `fine_tune_model` action
  delegates when a corpus is supplied; v0.1 behaviour preserved for
  callers without a corpus.
- **v1.0**: corpus becomes a required input for the FSM's
  `fine_tune_model` to actually run; the no-corpus pass-through stays
  as an explicit "skip-finetune" mode behind a `--no-finetune` flag.
