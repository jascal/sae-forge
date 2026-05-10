## 1. saeforge.training scaffolding

- [ ] 1.1 Create `saeforge/training/__init__.py` re-exporting `TrainingConfig`, `TrainingResult`, `run_finetune`, `cosine_with_warmup`, `build_iterator`
- [ ] 1.2 Create `saeforge/training/config.py` with `TrainingConfig` and `TrainingResult` dataclasses per `design.md`
- [ ] 1.3 Create `saeforge/training/schedules.py` with `cosine_with_warmup(step, total_steps, warmup_steps, peak_lr, min_lr_ratio=0.1) -> float`. Pure-numpy / no torch dep
- [ ] 1.4 Create `saeforge/training/corpus.py` with `build_iterator(source, tokenizer, batch_size, sequence_length)`. Handle local `.txt` / `.jsonl` files first; lazy-import `datasets` only when source is an HF dataset name; pass-through pre-tokenized iterables unchanged
- [ ] 1.5 Create `saeforge/training/loop.py` with `run_finetune(model, host, iterator, config) -> TrainingResult`. AdamW + weight decay + clipping + cosine LR + optional gradient checkpointing + optional bf16/fp16 autocast + periodic eval + periodic save + structured loss log

## 2. FSM action upgrade

- [ ] 2.1 Rewrite `fine_tune_model` action in `saeforge/actions/__init__.py` to delegate to `run_finetune` when `ctx["finetune_corpus"]` or `ctx["_finetune_iterator"]` is supplied; preserve the v0.1 4-step single-batch fallback when only `_finetune_input_ids` is present (byte-equivalence safety net)
- [ ] 2.2 Log the structured `TrainingResult` (final loss, loss curve length, eval trajectory, wall seconds, n_steps, save count, converged flag) into `ctx["transitions_log"]`
- [ ] 2.3 Map `OutOfMemoryError` to a single batch-size halving retry; on second OOM raise to the FSM error handler

## 3. ForgePipeline integration

- [ ] 3.1 Add fields to `ForgePipeline`: `finetune_corpus`, `finetune_total_steps`, `finetune_warmup_steps`, `finetune_peak_lr`, `finetune_batch_size`, `finetune_seq_len`, `finetune_precision`, `finetune_grad_checkpoint`, `finetune_eval_every`, `finetune_save_every`, `finetune_save_dir`. All default to v0.1 behaviour (no recipe, single-batch fallback)
- [ ] 3.2 Thread the new fields through `_run_synthetic_fsm` onto ctx; `run` and `run_synthetic` pass them through to the FSM
- [ ] 3.3 Add matching CLI flags to `sae-forge forge`: `--finetune-corpus`, `--finetune-steps`, `--finetune-lr`, `--finetune-batch`, `--finetune-seq`, `--finetune-precision`, `--finetune-grad-checkpoint`, `--finetune-eval-every`, `--finetune-save-every`

## 4. Tests

- [ ] 4.1 `tests/test_training_schedules.py`: cosine-with-warmup correctness — `step=0` returns small fraction of peak_lr, `step=warmup_steps` returns peak_lr, `step=total_steps` returns peak_lr * min_lr_ratio, monotonically non-increasing after warmup, no NaN past total_steps
- [ ] 4.2 `tests/test_training_corpus.py`: `build_iterator` reads local `.txt` correctly (one doc per line); reads local `.jsonl` correctly (`text` field); reads local directory recursively; pre-tokenized iterable passes through unchanged; HF dataset path lazy-imports (test does not actually network — uses a stubbed `datasets` module)
- [ ] 4.3 `tests/test_training_loop.py`: `run_finetune` on a tiny synthetic iterator over a tiny model reduces loss monotonically (within noise tolerance); periodic eval is logged at the right step count; periodic saves write checkpoints at the right step count; convergence heuristic flags steady-loss runs as converged
- [ ] 4.4 `tests/test_training_loop.py` cont.: bf16 autocast doesn't break correctness vs fp32 (forward outputs match within bf16 tolerance); gradient checkpointing produces gradients matching non-checkpointed (within fp tolerance) — unit-level correctness check
- [ ] 4.5 `tests/test_fsm_finetune_recipe.py`: `_run_synthetic_fsm` with `finetune_corpus` set runs the recipe; without corpus and with only `_finetune_input_ids` falls back to v0.1 behaviour (preserving byte-equivalence)
- [ ] 4.6 `tests/test_local_only.py`: with a local-file corpus and an already-cached host model, no network call is made during the entire run (verified via `socket` monkeypatch raising on connect)

## 5. Example: forge_gemma2_2b.py

- [ ] 5.1 Add `examples/forge_gemma2_2b.py` covering: pull `google/gemma-scope-2b-pt-res` SAE for one configurable layer; slice to 256 features; run `EpochCompressor` against Gemma-2-2B forwards on a Fineweb-edu slice; forge with `attention_width="host"`; fine-tune 1000 steps via the new recipe at bf16 + grad checkpointing; eval every 100 steps
- [ ] 5.2 Document expected wall time per device (`cpu`: do-not-attempt; `mps` 24GB: ~30-90 min; `cuda` 24GB: ~10-30 min)
- [ ] 5.3 The script accepts `--corpus` flag for a local file path or HF dataset name, defaulting to `HuggingFaceFW/fineweb-edu` (sample-10BT subset)

## 6. Docs

- [ ] 6.1 README "How it works" gains a "Fine-tune recipe" subsection naming the cosine LR + warmup + grad checkpointing default config and pointing at `examples/forge_gemma2_2b.py`
- [ ] 6.2 README "Apple Silicon" subsection adds: "with v0.3 forge-finetune-recipe, real fine-tunes (1k steps batch=8 seq=512) become tractable on M-series 24GB+"
- [ ] 6.3 README "Linux + CUDA" tier table gains a v0.3 column showing what each tier unlocks once the recipe ships
- [ ] 6.4 New `docs/finetune-recipe.md` covering: which corpus formats are accepted, how local-only data handling works, what the convergence heuristic means, when to enable gradient checkpointing, when to use bf16 vs fp32
- [ ] 6.5 Update `AGENTS.md` to add the v0.3 milestone

## 7. Dependencies

- [ ] 7.1 Bump `[torch]` extra's `transformers>=4.46` (Gemma-2 support)
- [ ] 7.2 Add a new `[recipe]` extra pulling `datasets>=2.16`. Optional — local-corpus paths don't require it
- [ ] 7.3 Document the data-handling guarantees in `AGENTS.md` (no telemetry, no automatic uploads, HF cache stays local)

## 8. OpenSpec scaffolding

- [x] 8.1 `openspec/changes/forge-finetune-recipe/proposal.md`
- [x] 8.2 `openspec/changes/forge-finetune-recipe/design.md`
- [x] 8.3 `openspec/changes/forge-finetune-recipe/tasks.md` (this file)
- [x] 8.4 `openspec/changes/forge-finetune-recipe/specs/forge-finetune-recipe/spec.md`
