## Why

The `forge-finetune-recipe` (v0.3) trains the forged transformer with pure LM cross-entropy on a corpus. `faithfulness_kl(host, forged)` already exists as an **eval-only** metric — we measure how close the forged model's logits are to the host's, but we don't *optimize* that distance directly.

This is leaving signal on the table. The forged transformer's purpose is to imitate the host model's behavior in the SAE's surviving feature basis. If we have a teacher (the host) available during fine-tune anyway, we can distill its logits into the student (forged) directly, alongside or instead of corpus CE.

The lift is small:

- `host` is already passed into `run_finetune` (currently used only periodically inside `_eval_kl`).
- `faithfulness_kl` is already implemented (`saeforge/eval/faithfulness.py`).
- The trainer already loops over batches with the host model in scope.

The only missing piece is the **trainable** path: replacing/augmenting the per-batch CE loss with `α · CE(corpus) + (1-α) · τ² · KL(host_logits/τ ‖ forged_logits/τ)`.

This is the right shape for sae-forge: a single-knob extension of the existing fine-tune objective, with `α = 1.0` preserving v0.3 behavior byte-identically.

## What Changes

### Scope (small)

Add a host-distillation term to `forge-finetune-recipe`'s loss. `α = 1.0` (default) is byte-identical to v0.3 LM-CE training.

### New `TrainingConfig` fields

- `distill_alpha: float = 1.0` — interpolation between corpus CE (α) and host-distillation KL (1-α). Default 1.0 = pure CE = byte-identical to v0.3.
- `distill_temperature: float = 2.0` — softmax temperature for KL. Standard literature default.

### Modified `run_finetune` loop

When `distill_alpha < 1.0` and `host is not None`:

1. Run the host model forward on the same `batch` under `torch.no_grad()`. Same device, same autocast context as the student forward.
2. Compute `kd_loss = τ² · KL(softmax(host_logits / τ) ‖ log_softmax(forged_logits / τ))` (reduction = mean over tokens, matching `faithfulness_kl`'s normalization).
3. Compute `total_loss = α · ce_loss + (1 - α) · kd_loss`.

The host forward is `no_grad` — gradients flow only through the student. The host stays in `eval()` mode and is loaded only once.

When `α == 1.0` (default), the KD branch is skipped entirely — no host forward, no extra compute, no behavioral change vs v0.3.

When `host is None` and `α < 1.0`, raise at `TrainingConfig.__post_init__` time with a clear error: distillation requires the teacher.

### What this PR explicitly does NOT do

- **No new CLI command.** No `forge distill` subcommand. This is a fine-tune config extension, not a new pipeline stage.
- **No new checkpoint filename.** Output remains the standard `forged.safetensors` (or whatever `save_dir` resolves to). The training config goes into the metadata as it already does.
- **No Matryoshka partial-alignment path.** sae-forge's forged transformer has a single feature basis; there's no "core vs peripheral features" distinction to apply.
- **No feature-space or relation-based distillation.** Only response-based (logit KL). The other variants would require new code; logit KL is one-line.
- **No `register_profile` hook.** `TrainingConfig` is the integration point; profiles can pass through these fields the same way they pass `peak_lr`.
- **No new evaluation metric.** `faithfulness_kl` is already the right eval; this proposal makes it the training objective too.

## Impact

- **Affected specs:** `training` (new capability).
- **Affected code:**
  - `saeforge/training/config.py` — two new fields + validation
  - `saeforge/training/loop.py` — KD loss branch in `run_finetune`
  - `tests/training/test_distillation.py` (new) — α=1.0 byte-identity, α<1.0 gradient flow, α<1.0+host=None rejection, end-to-end smoke
- **Affected docs:**
  - `docs/finetune-recipe.md` — new section "Host distillation"
  - `CHANGELOG.md`
- **Closes:** the externally-proposed "Add Knowledge-Distillation Refinement Loop" idea, rescoped to sae-forge's actual architecture.

## Risks

- **Host forward doubles per-step compute.** When `α < 1.0`, every training step runs both student forward+backward and a host forward. For Llama/Gemma-scale hosts on a 24GB card, this is the limiting factor. The `α = 1.0` default keeps the cost-aware path the same; opt-in for users who can afford it. Document the cost trade-off in `docs/finetune-recipe.md`.
- **Vocab mismatch between host and forged.** The forged transformer inherits the host's tokenizer and unembedding by construction (per `SubspaceProjector`), so vocab and logit dimensionality must already match. Defensive shape check at the KL call site catches misconfiguration early.
- **Temperature scaling subtleties.** The `τ²` rescaling is standard practice (Hinton et al. 2015) and the right normalization when mixing distillation with hard-label CE. Calling it out in the docstring so readers don't tune it as a free knob.
