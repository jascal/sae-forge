# add-host-distillation-finetune-loss Specification

## Purpose

The `add-host-distillation-finetune-loss` capability extends `forge-finetune-recipe`'s loss with an opt-in host-model distillation term. With the default `distill_alpha = 1.0`, training is byte-identical to v0.3 LM cross-entropy. With `distill_alpha < 1.0`, the per-step loss becomes a Hinton-style blend of corpus CE and `τ²`-scaled KL between the frozen host model's logits and the forged student's logits.

This is **not** a new pipeline stage. It's a single-knob extension of the existing `TrainingConfig` and `run_finetune` loop. No new CLI, no new artifact filenames, no new metadata format.

## ADDED Requirements

### Requirement: `TrainingConfig` accepts host-distillation parameters

`saeforge.training.TrainingConfig` SHALL accept two new fields:

- `distill_alpha: float = 1.0` — interpolation between corpus CE (weight `α`) and host-distillation KL (weight `1-α`).
- `distill_temperature: float = 2.0` — softmax temperature applied to both host and forged logits before computing KL.

The class's `__post_init__` SHALL validate `0.0 <= distill_alpha <= 1.0` and raise `ValueError` with a clear message on out-of-range values.

The class's `__post_init__` SHALL validate `distill_temperature > 0` and raise `ValueError` on non-positive values.

The defaults (`distill_alpha=1.0`, `distill_temperature=2.0`) SHALL preserve byte-identical training to v0.3 — when `α=1.0`, the KD branch is skipped entirely (no host forward), so the optimizer state evolution is bit-equal to the pre-change loop on the same inputs.

#### Scenario: default config preserves v0.3 byte-identity

- **WHEN** `TrainingConfig()` is constructed with no `distill_*` arguments
- **THEN** `distill_alpha == 1.0` and `distill_temperature == 2.0`
- **AND** `run_finetune(model, host, iterator, config)` produces a loss history bit-equal to the pre-change implementation on the same fixture

#### Scenario: out-of-range `distill_alpha` rejected

- **WHEN** `TrainingConfig(distill_alpha=-0.1)` or `TrainingConfig(distill_alpha=1.5)` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the offending value and the valid range `[0.0, 1.0]`

#### Scenario: non-positive `distill_temperature` rejected

- **WHEN** `TrainingConfig(distill_temperature=0.0)` or `TrainingConfig(distill_temperature=-1.0)` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the offending value

### Requirement: `run_finetune` applies host-distillation loss when `distill_alpha < 1.0`

When `config.distill_alpha < 1.0`, `saeforge.training.run_finetune` SHALL:

1. Verify `host is not None` at the top of the function. If `host is None` and `distill_alpha < 1.0`, raise `ValueError` BEFORE consuming any batches from `iterator`.
2. For each training step, after computing the student forward pass and `ce_loss`, run a `torch.no_grad()` host forward on the same `batch`, in the same autocast context as the student forward.
3. Compute `kd_loss = (distill_temperature ** 2) * KL(softmax(host_logits / τ) ‖ softmax(forged_logits / τ))`, normalized per token (matching `faithfulness_kl`'s reduction). KL direction is `host ‖ forged` — mode-seeking from the student's perspective.
4. Combine: `total_loss = distill_alpha * ce_loss + (1 - distill_alpha) * kd_loss`.
5. Run `total_loss.backward()` — gradients flow only through the student (host is `no_grad`).

When `config.distill_alpha >= 1.0`, `run_finetune` SHALL skip the host forward entirely and use `ce_loss` directly as the training loss (zero additional compute vs v0.3).

#### Scenario: `host=None` + `distill_alpha<1.0` rejected before training starts

- **WHEN** `run_finetune(model, host=None, iterator, config=TrainingConfig(distill_alpha=0.5))` is invoked
- **THEN** the function raises `ValueError` naming `distill_alpha` and `host`, BEFORE consuming any batches from `iterator`

#### Scenario: gradients flow through the student under α=0.5

- **WHEN** `run_finetune` is called with `distill_alpha=0.5`, a real `host` model, and a toy iterator yielding one batch
- **THEN** after one step, `student_model.parameters()` differ from their pre-step values
- **AND** `host.parameters()` are unchanged
- **AND** the recorded `total_loss` is finite (not NaN, not inf)

#### Scenario: α=0.0 trains on pure KL

- **WHEN** `run_finetune` is called with `distill_alpha=0.0`
- **THEN** the per-step loss equals `(1.0 - 0.0) * kd_loss == kd_loss`
- **AND** training proceeds without consuming any signal from the corpus labels (only from the host's logit distribution)

### Requirement: `ForgePipeline` exposes the same knobs

`saeforge.ForgePipeline` SHALL accept `finetune_distill_alpha: float = 1.0` and `finetune_distill_temperature: float = 2.0` constructor kwargs, mirroring the existing `finetune_*` plumbing pattern.

The pipeline SHALL thread these into the `TrainingConfig` instance it constructs for its fine-tune step. Defaults (`1.0`, `2.0`) SHALL preserve byte-identity with the existing `ForgePipeline` smoke and integration tests.

#### Scenario: `ForgePipeline` defaults preserve v0.3 behavior

- **WHEN** `ForgePipeline(...)` is constructed without `finetune_distill_*` arguments and `.run()` is called
- **THEN** the run produces a forged transformer bit-equal to the pre-change pipeline on the same fixture

#### Scenario: `ForgePipeline` propagates distillation knobs

- **WHEN** `ForgePipeline(..., finetune_distill_alpha=0.5, finetune_distill_temperature=2.0).run()` is called
- **THEN** the internally-constructed `TrainingConfig` carries `distill_alpha=0.5` and `distill_temperature=2.0`
- **AND** the resulting run metadata records both values
