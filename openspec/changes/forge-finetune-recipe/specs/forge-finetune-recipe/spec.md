# forge-finetune-recipe Specification

## Purpose

Defines the v0.3 fine-tune recipe — a real LM cross-entropy training
loop with cosine LR + warmup, gradient clipping, optional gradient
checkpointing, optional mixed precision, periodic faithfulness eval,
and structured loss tracking. Replaces the v0.1 4-step smoke
fine-tune behind the same FSM action surface, with v0.1 behaviour
preserved as a fallback when no corpus is supplied.

## Requirements

### Requirement: Cosine schedule with warmup is monotone after warmup

`saeforge.training.schedules.cosine_with_warmup(step, total_steps,
warmup_steps, peak_lr, min_lr_ratio=0.1)` SHALL produce a learning
rate schedule that:

- Linearly ramps from `peak_lr / warmup_steps` at `step=0` to
  `peak_lr` at `step=warmup_steps`.
- Cosine-decays from `peak_lr` at `step=warmup_steps` to
  `peak_lr * min_lr_ratio` at `step=total_steps`.
- Is monotonically non-increasing for `step >= warmup_steps`.
- Returns `peak_lr * min_lr_ratio` for any `step >= total_steps`
  (clamped — no negative LRs, no NaN).

#### Scenario: schedule corner values

- **GIVEN** `total_steps=100, warmup_steps=10, peak_lr=1e-3,
  min_lr_ratio=0.1`
- **WHEN** the schedule is queried at `step=0`, `step=10`,
  `step=100`, `step=200`
- **THEN** the values are `1e-4`, `1e-3`, `1e-4`, `1e-4` within
  floating-point tolerance

#### Scenario: monotone decay after warmup

- **WHEN** the schedule is sampled at every step from `warmup_steps`
  to `total_steps + 50`
- **THEN** consecutive samples satisfy `lr[i+1] <= lr[i]`

### Requirement: Local file corpora work without HuggingFace

`saeforge.training.corpus.build_iterator(source, tokenizer, batch_size,
sequence_length)` SHALL accept any of:

1. A local `.txt` file path (one document per line).
2. A local `.jsonl` file path (each line a JSON object with a `"text"`
   field).
3. A local directory path containing `.txt` / `.jsonl` files (read
   recursively).
4. A HuggingFace dataset name (e.g. `"HuggingFaceFW/fineweb-edu"`).
5. A pre-tokenized `Iterable[torch.Tensor]` of shape
   `(batch_size, sequence_length)` int64 tensors.

For sources 1–3, no HuggingFace `datasets` import SHALL occur. For
source 4, `datasets` is lazy-imported. For source 5, the iterable is
yielded as-is.

#### Scenario: local .txt corpus runs without datasets installed

- **GIVEN** a temporary `.txt` file with 100 short documents, an HF
  tokenizer for `gpt2`, `batch_size=2`, `sequence_length=16`, and the
  `datasets` package uninstalled (simulated via monkeypatch)
- **WHEN** `build_iterator` is called and the first 5 batches are
  consumed
- **THEN** five `(2, 16)` int64 tensors are produced
- **AND** `datasets` was never imported

#### Scenario: HF dataset path lazy-imports datasets

- **GIVEN** the `datasets` module is available
- **WHEN** `build_iterator("HuggingFaceFW/fineweb-edu", ...)` is called
- **THEN** `datasets` appears in `sys.modules` after the first batch is
  consumed (not before — the import is lazy at iterator construction)

### Requirement: Local-corpus runs make no network calls

When the corpus source is a local file or directory AND the host
model + tokenizer are already cached, the entire fine-tune run
(corpus iterator + training loop + periodic eval + periodic save)
SHALL make zero outbound network connections. Verified by tests via
a `socket.socket` monkeypatch that raises on `connect`.

#### Scenario: local-only fine-tune is offline-safe

- **GIVEN** a local `.txt` corpus, a host model already cached at
  `~/.cache/huggingface/hub/`, and a `socket.socket.connect`
  monkeypatch that raises `OSError` on any call
- **WHEN** `run_finetune(...)` runs to completion
- **THEN** no exception is raised — the patched `connect` is never
  invoked

### Requirement: run_finetune reduces loss on a tractable corpus

`run_finetune(model, host, iterator, config)` SHALL produce a
`TrainingResult` whose `loss_history[-1][1] < loss_history[0][1]`
when run on a synthetic corpus drawn from the model's own
distribution for at least 50 steps with sensible defaults. This is
the baseline "training does anything" smoke check.

#### Scenario: synthetic-corpus loss decreases over 50 steps

- **GIVEN** a tiny `NativeModel` and an iterator yielding random
  token batches drawn from `[0, vocab_size)`, `total_steps=50`
- **WHEN** `run_finetune` runs
- **THEN** `result.loss_history[-1][1] < result.loss_history[0][1]`

### Requirement: Periodic eval is logged at the right cadence

When `config.eval_input_ids is not None` and `eval_every_steps=N`,
`run_finetune` SHALL produce a `TrainingResult.eval_history` with
exactly one entry per N completed steps (excluding step 0). Each
entry is `(step, kl)` where `kl` is the faithfulness KL on
`eval_input_ids` at that step.

#### Scenario: eval cadence is exact

- **GIVEN** `total_steps=100, eval_every_steps=25,
  eval_input_ids=<some tensor>`
- **WHEN** `run_finetune` runs
- **THEN** `result.eval_history` contains exactly entries at steps
  `25, 50, 75, 100` (no entry at step 0)

### Requirement: Periodic saves write checkpoints at the right cadence

When `config.save_dir is not None` and `save_every_steps=N`,
`run_finetune` SHALL save a `NativeModel` checkpoint at every step
divisible by N (excluding step 0). Each checkpoint goes to
`save_dir / f"step-{step:06d}"`. Paths are recorded in
`TrainingResult.save_paths`.

### Requirement: Gradient checkpointing produces matching gradients

When `config.gradient_checkpointing=True`, the gradients computed via
`torch.utils.checkpoint.checkpoint_sequential` over the transformer
block list SHALL match the gradients computed without checkpointing
on the same inputs and seed, within `1e-4` absolute tolerance per
parameter. This is a unit-level correctness check separate from the
end-to-end loss-decreases assertion.

#### Scenario: checkpointed and non-checkpointed gradients agree

- **GIVEN** a tiny `NativeModel`, a fixed `(batch, seq)` input, and a
  fixed RNG seed
- **WHEN** the loss-and-backward pass is run twice — once with
  `gradient_checkpointing=True`, once without — and gradients are
  collected for every parameter
- **THEN** `torch.allclose(grad_a, grad_b, atol=1e-4)` for every
  parameter

### Requirement: Mixed precision (bf16) preserves correctness within tolerance

When `config.precision="bf16"`, `run_finetune` SHALL produce a final
loss within `0.05` absolute of the fp32 final loss on the same input
sequence and seed. (The tolerance is loose because bf16 introduces
real numerical drift; this is a sanity check, not a strict bit-for-bit
match.)

### Requirement: FSM action delegates when corpus is supplied

The `fine_tune_model` action in `saeforge/actions/__init__.py` SHALL:

- Call `run_finetune` when `ctx["finetune_corpus"]` is truthy OR
  `ctx["_finetune_iterator"]` is supplied.
- Fall back to the v0.1 4-step single-batch loop when only
  `ctx["_finetune_input_ids"]` is supplied (preserving the
  byte-equivalence safety net for tests that don't supply a corpus).
- Pass-through (no fine-tune) when none of those is supplied.

#### Scenario: corpus-supplied path delegates to recipe

- **GIVEN** a `ForgePipeline` with `finetune_corpus="/path/to/local.txt"`
  and `orchestrator="fsm"`
- **WHEN** `run_synthetic` runs
- **THEN** the `transitions_log` contains a `fine_tune_model` entry
  with `mode="recipe"` (not `"trained"` from v0.1) and a non-empty
  loss history

#### Scenario: no-corpus path preserves v0.1 byte-equivalence

- **GIVEN** the v0.1 byte-equivalence test setup (no corpus, no
  iterator, only `_finetune_input_ids`)
- **WHEN** the FSM runs
- **THEN** the SHA-256 of `forged/model.safetensors` matches the v0.1
  baseline exactly

### Requirement: No automatic uploads

`run_finetune`, `build_iterator`, and the FSM action SHALL NOT push
any output to remote services (HuggingFace Hub, W&B, comet, etc.).
All outputs (checkpoints, loss curves, eval logs) stay under the
configured `output_dir` / `save_dir`. v0.3 ships with no opt-in
remote-upload pathway; that's a future change if it's wanted.

#### Scenario: no upload calls anywhere

- **WHEN** the full forge + fine-tune + save pipeline runs
- **THEN** no call to `huggingface_hub.HfApi.upload_*`,
  `wandb.init`, or any other known telemetry/upload entry point is
  made (verified via mock-patching)
