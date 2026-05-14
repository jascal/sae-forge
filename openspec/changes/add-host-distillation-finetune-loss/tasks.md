## 1. `TrainingConfig` extensions

- [ ] 1.1 Add `distill_alpha: float = 1.0` to `saeforge/training/config.py::TrainingConfig`. Validate `0.0 <= distill_alpha <= 1.0` in `__post_init__`.
- [ ] 1.2 Add `distill_temperature: float = 2.0`. Validate `distill_temperature > 0` in `__post_init__`.
- [ ] 1.3 If `distill_alpha < 1.0`, document in the docstring that `host` MUST be non-None when calling `run_finetune`. (Runtime check enforces this; see §2.)

## 2. `run_finetune` loss extension

- [ ] 2.1 In `saeforge/training/loop.py::run_finetune`, after the existing student-forward + CE computation, add a `if config.distill_alpha < 1.0:` branch.
- [ ] 2.2 Inside the branch, raise `ValueError` at the start of `run_finetune` if `host is None` and `config.distill_alpha < 1.0` (defensive — config-time validation can't catch this).
- [ ] 2.3 Run the host forward under `torch.no_grad()` on the same `batch` and same device, inside the same `_autocast` context as the student forward.
- [ ] 2.4 Compute `kd_loss = (distill_temperature ** 2) * F.kl_div(log_softmax(forged_logits / τ, dim=-1), softmax(host_logits / τ, dim=-1), reduction="batchmean")`. Direction is `KL(host ‖ forged)` — matches `faithfulness_kl`.
- [ ] 2.5 Combine: `loss = config.distill_alpha * ce_loss + (1.0 - config.distill_alpha) * kd_loss`.
- [ ] 2.6 The α=1.0 default path skips the host forward entirely (cheap conditional at the top of the branch — `if alpha >= 1.0: skip`).

## 3. Tests

- [ ] 3.1 `tests/training/test_distillation.py::test_alpha_one_byte_identical` — train 10 steps with α=1.0, snapshot the loss history; train 10 steps on the same fixture without the new fields, snapshot. Assert byte-identical loss history (load-bearing v0.3 invariant).
- [ ] 3.2 `tests/training/test_distillation.py::test_alpha_half_gradients_flow` — α=0.5, τ=2.0, one optimizer step on a toy host+forged pair. Assert `total_loss.requires_grad`, the student's first parameter moves after `optim.step()`, and `host` parameters are unchanged.
- [ ] 3.3 `tests/training/test_distillation.py::test_distill_alpha_validation` — `TrainingConfig(distill_alpha=-0.1)` raises; `distill_alpha=1.5` raises; `distill_temperature=0.0` raises.
- [ ] 3.4 `tests/training/test_distillation.py::test_distill_alpha_lt_one_requires_host` — `run_finetune(model, host=None, iterator, config=TrainingConfig(distill_alpha=0.5))` raises `ValueError` immediately, before consuming any batches.
- [ ] 3.5 `tests/training/test_distillation.py::test_alpha_zero_pure_kd` — α=0.0 trains on pure KL only (no CE term contributes). Assert one step succeeds and produces a finite gradient.
- [ ] 3.6 End-to-end: `tests/forge/test_pipeline_distill.py::test_pipeline_with_distill_smoke` — run `ForgePipeline.run(...)` with `finetune_distill_alpha=0.5` on the toy GPT-2 fixture, total_steps=4. Assert run completes, `result.faithfulness_kl` is finite, and the run metadata records the α/τ values.

## 4. `ForgePipeline` plumbing

- [ ] 4.1 Add `finetune_distill_alpha: float = 1.0` and `finetune_distill_temperature: float = 2.0` to `ForgePipeline`'s constructor kwargs, mirroring the existing `finetune_*` knobs.
- [ ] 4.2 Thread them through into the `TrainingConfig` instance the pipeline builds for its fine-tune step.
- [ ] 4.3 The default (1.0, 2.0) preserves byte-identity with the existing pipeline tests.

## 5. Docs

- [ ] 5.1 Add a "Host distillation" subsection to `docs/finetune-recipe.md`, after the "Quick reference" block, covering:
  - When to use it (small custom corpora where the host distribution isn't well-sampled by the corpus).
  - The α/τ semantics + recommended starting values (α=0.5, τ=2.0).
  - Per-step compute cost: roughly 2× the α=1.0 path on Llama/Gemma-scale hosts (host forward is dominant).
  - How to disable: omit the fields, or set α=1.0 explicitly.
- [ ] 5.2 Add `CHANGELOG.md` entry under unreleased: "**Host distillation in fine-tune** — `TrainingConfig` gains `distill_alpha` (default 1.0 = pure LM-CE, byte-identical to v0.3) and `distill_temperature` (default 2.0). When `distill_alpha < 1.0`, the loss becomes `α·CE(corpus) + (1-α)·τ²·KL(host ‖ forged)`. `ForgePipeline` exposes the same knobs as `finetune_distill_alpha` / `finetune_distill_temperature`."

## 6. Validation

- [ ] 6.1 Run `openspec validate add-host-distillation-finetune-loss --strict`.
- [ ] 6.2 Run `pytest` full suite; verify no regressions on the v0.3 byte-equivalence tests.
- [ ] 6.3 Run `ruff check` clean on the modified files.
- [ ] 6.4 Run the existing `tests/forge/test_pipeline_smoke.py` (or its analogue) with default config; confirm unchanged.

## 7. What this change explicitly defers

- [ ] 7.1 Feature-space distillation (cosine/L2 on sparse codes). Out of scope; logit KL is the lowest-effort, highest-precedent variant.
- [ ] 7.2 Relation-based distillation (pairwise feature similarities — DistilRoBERTa-style). Substantial new code; defer until logit-KD proves valuable.
- [ ] 7.3 Cached host logits for static corpora. Per-step host forward is acceptable for v1; optimization later when measurements show it's binding.
- [ ] 7.4 Matryoshka partial alignment. sae-forge's forged transformer has one feature basis; the proposal's original "core vs peripheral" idea doesn't apply here.
- [ ] 7.5 A full hyperparameter sweep over α and τ. That's a research-track follow-up after this change lands; not a prerequisite.
