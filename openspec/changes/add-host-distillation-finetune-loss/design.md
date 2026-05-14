## Context

`run_finetune(model, host, iterator, config)` in `saeforge/training/loop.py` is the v0.3 trainer. Today's loop, per step:

1. Sample a batch from `iterator`.
2. Forward the student (`module(batch) → logits`).
3. Compute LM cross-entropy: `_shift_lm_loss(logits, batch, F)`.
4. Backward + step.

The `host` arg is currently used only inside `_eval_kl` (called periodically every `eval_every_steps`). It's a loaded HF model with frozen weights, present in scope at every training step but unused during the loss computation.

`saeforge/eval/faithfulness.py::faithfulness_kl` computes mean per-token KL between host and forged logits — but with `torch.no_grad()` around the *student* forward, so it can't be used as a training loss directly. Lifting it into a trainable path is the load-bearing change in this proposal.

## Goals / Non-Goals

**Goals**:

- Make `faithfulness_kl(host, forged)` a *training* loss component, not just an eval metric.
- Default config (`distill_alpha=1.0`) preserves v0.3 LM-CE training byte-identically.
- Single-knob α-blend: `loss = α·CE + (1-α)·τ²·KL`. Standard Hinton-style temperature scaling.

**Non-goals**:

- No new pipeline stage. No `forge distill` CLI. This is a `TrainingConfig` field, full stop.
- No feature-space / hidden-state distillation. Only response-based logit KL.
- No relation-based distillation (pairwise feature similarities).
- No Matryoshka partial-alignment. sae-forge's forged transformer has one feature basis; no core/peripheral split.
- No new metadata file or checkpoint filename. The training config (which records `distill_alpha`) already lands in the run metadata.

## Decisions

### Decision 1: α=1.0 default — byte-identical to v0.3

The KD path is opt-in. With `α = 1.0`, the `(1 - α) · τ² · KL` term is exactly zero, but we go further: skip the host forward pass entirely when `α >= 1.0` to avoid the cost. This makes the default zero-cost vs v0.3, which is the load-bearing property for the byte-equivalence test.

### Decision 2: Reject `host=None` + `α<1.0` at config-validation time

`TrainingConfig.__post_init__` raises if `distill_alpha < 1.0` and there's no way to thread a teacher through. (Whether `host` will be `None` is known only at `run_finetune` call time, so a defensive runtime check there too.) Two-layer guard — config-time for the common path, runtime for the edge case.

### Decision 3: KL direction — KL(host ‖ forged), not KL(forged ‖ host)

Mirror `faithfulness_kl`'s direction. The student is approximating the teacher's distribution; `KL(p_teacher ‖ p_student)` is the standard formulation (mode-seeking from the student's perspective; penalises the student putting low probability where the teacher puts high probability). Matches Hinton et al. 2015 and matches the existing eval metric so training-time and eval-time numbers are directly comparable.

### Decision 4: Soft + hard label blend — `α · CE_hard + (1-α) · τ² · KL_soft`

The `τ²` factor restores the gradient magnitude of the soft-target term to a scale comparable with the hard-label CE (per Hinton — softening logits by τ reduces gradients by 1/τ², so we multiply by τ² to compensate). Standard. Not a user-tunable secret knob; documented in the docstring.

### Decision 5: Single host forward per step, no caching

Don't try to cache host logits across epochs. Two reasons:

1. The iterator may not be deterministic (HuggingFace `streaming=True`, random shuffling). Cache keys would be brittle.
2. The host forward is the cost dominator, but it's not infinite. A naive forward per step is acceptable for v1.

A future optimization could pre-tokenize and cache host logits when the corpus is static — out of scope.

### Decision 6: Same autocast context for host and student

Run the host forward inside the same `_autocast(device.type, autocast_dtype)` block as the student. Two reasons:

1. Memory budget: bf16/fp16 cuts the host forward's working set in half on a 24GB card. Without autocast, Llama-3-8B as a teacher likely OOMs.
2. Numerical consistency: KL between two log-softmaxes is more sensitive to scale than to absolute magnitude; matching precision across the two forwards keeps the KL well-conditioned.

### Decision 7: Smoke test exercises α=0.5 — proves gradient path works

The byte-equivalence test already pins α=1.0 (default). A new smoke test at α=0.5 verifies:

1. `total_loss` is finite (not NaN/inf even when host forward produces vanishingly small logits).
2. `total_loss.backward()` succeeds (gradients flow).
3. The student's parameters move after one optimizer step (sanity check that the loss is reaching the student).

No empirical claim about "KD improves faithfulness_kl" — that's an A/B research run, not a smoke test.

## Open Questions

- **Should we warn if `α < 1.0` and the student-host vocab dimensions don't match?** Defensive shape check at the KL site is mandatory. Whether to also pre-validate at config-time is open. Defer: the shape check catches it loudly enough on first batch.
- **Should `distill_temperature` validate `τ > 0`?** Yes — silently accepting τ ≤ 0 would NaN the softmax. Add `__post_init__` validation. (Already in tasks.md.)
- **What's the right default α for users who turn it on without thinking?** Literature defaults vary (Hinton 0.9 hard / 0.1 soft, modern Matryoshka SAE papers use 0.5/0.5). Pick `0.5` and document that users should sweep this for their domain. The default α=1.0 (off) means nobody hits the "wrong default" path by accident.

## Risks / Trade-offs

- **Per-step compute doubles when α < 1.0.** Host forward is the cost dominator. Documented in `docs/finetune-recipe.md`. The α=1.0 default keeps cost the same as v0.3.
- **Vocab mismatch.** Per `SubspaceProjector`, the forged transformer inherits the host's tokenizer and unembedding shape. Realistically can't mismatch unless someone hand-constructs the projector wrong. Defensive shape check in the KL loss site for the long tail.
- **The KD term won't help if the corpus already covers the host's distribution.** If the fine-tune corpus IS what the host was trained on, KD is mostly redundant. Expected use: small custom corpora where corpus CE undersamples the host's actual behavior — that's where distillation buys signal.

## Migration Plan

Two-way migration:

- **Existing users (α=1.0 default)**: nothing changes. The byte-equivalence test pins this.
- **New users (α<1.0)**: opt in by passing `distill_alpha=0.5` (or whatever) to `TrainingConfig`. `host` must be non-None (config-time check catches missing teacher).

No deprecation needed — additive parameters with safe defaults.
