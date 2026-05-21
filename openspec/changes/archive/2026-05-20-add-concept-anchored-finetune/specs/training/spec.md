# training Specification (delta)

## ADDED Requirements

### Requirement: `TrainingConfig` accepts concept-anchoring parameters

`saeforge.training.TrainingConfig` SHALL accept six new fields covering concept-anchored fine-tune:

- `concept_alpha: float = 0.0` — overall weight on the concept-anchoring term. With the default `0.0`, the entire concept branch is skipped and training is byte-identical to the pre-change loop (and byte-identical to the `add-host-distillation-finetune-loss` path when its own defaults apply).
- `concept_pool_weight: float = 1.0` — relative weight of the pooled-head loss within the concept term.
- `concept_channel_weight: float = 1.0` — relative weight of the per-channel-head loss within the concept term.
- `concept_focal_gamma: float = 2.0` — focal-loss γ exponent. `0.0` reduces to plain BCE.
- `concept_label_source: str = "polygram-clusters"` — registry key naming the `LabelSource` backend.
- `concept_label_source_kwargs: dict[str, Any]` — backend-specific configuration; default empty dict.

`__post_init__` SHALL validate:
- `0.0 <= concept_alpha <= 1.0`; out-of-range raises `ValueError`.
- `concept_pool_weight >= 0` AND `concept_channel_weight >= 0`.
- When `concept_alpha > 0`: at least one of `concept_pool_weight` / `concept_channel_weight` SHALL be `> 0`. Otherwise raise `ValueError("concept_alpha > 0 requires non-zero pool or channel weight")`.
- `concept_focal_gamma >= 0`; negative values raise `ValueError`.
- When `concept_alpha > 0`: `concept_label_source` SHALL be a key in `saeforge.training.concept_anchor.LABEL_SOURCE_REGISTRY`; unknown keys raise `ValueError` listing the registered keys. When `concept_alpha == 0.0`, this check is skipped (branch is inactive, so misspellings don't matter).

The defaults SHALL preserve byte-identical training to the pre-change loop — when `concept_alpha == 0.0`, no label source is instantiated, no heads are constructed, no extra per-step work is done.

#### Scenario: default config preserves byte-identity

- **WHEN** `TrainingConfig()` is constructed with no `concept_*` arguments
- **THEN** `concept_alpha == 0.0` and the other concept fields hold their documented defaults
- **AND** `run_finetune(model, host, iterator, config)` produces a loss history bit-equal to a baseline trained without the new fields on the same fixture

#### Scenario: `concept_alpha` out-of-range rejected

- **WHEN** `TrainingConfig(concept_alpha=-0.1)` or `TrainingConfig(concept_alpha=1.5)` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the offending value and the valid range `[0.0, 1.0]`

#### Scenario: positive `concept_alpha` with zero weights rejected

- **WHEN** `TrainingConfig(concept_alpha=0.5, concept_pool_weight=0.0, concept_channel_weight=0.0)` is constructed
- **THEN** `__post_init__` raises `ValueError` whose message names both `concept_pool_weight` and `concept_channel_weight`

#### Scenario: unknown `concept_label_source` rejected when active

- **WHEN** `TrainingConfig(concept_alpha=0.5, concept_label_source="bogus")` is constructed
- **THEN** `__post_init__` raises `ValueError` listing the registered backend keys (at minimum `"polygram-clusters"`)

#### Scenario: unknown `concept_label_source` tolerated when inactive

- **WHEN** `TrainingConfig(concept_alpha=0.0, concept_label_source="bogus")` is constructed
- **THEN** no error is raised; the misspelling is harmless because the concept branch is inactive

### Requirement: `run_finetune` applies concept-anchoring loss when `concept_alpha > 0`

When `config.concept_alpha > 0`, `saeforge.training.run_finetune` SHALL:

1. Before the main training loop:
   - Look up `LabelSourceCls = LABEL_SOURCE_REGISTRY[config.concept_label_source]`.
   - Instantiate the backend with `config.concept_label_source_kwargs`.
   - Call `n_concepts = label_source.prepare(model, iterator)`; record `n_concepts` in the run metadata.
   - Construct `PooledConceptHead(d_model=model.config.hidden_size, n_concepts=n_concepts)` and `PerChannelConceptHead(n_concepts=n_concepts)` on the model's device.
   - Append both heads' parameters to the optimiser as a new parameter group.
2. Per training step, after the student forward already required for CE / distillation:
   - Fetch the last-layer residual stream `residual` (shape `[B, T, d_model]`).
   - Compute `labels = label_source.labels_for_batch(batch, hidden_states=residual.detach())` → shape `[B, T, n_concepts]`, dtype float, values in `{0.0, 1.0}`.
   - Aggregate `pool_labels = labels.max(dim=1).values` → shape `[B, n_concepts]`.
   - `pool_logits = pooled_head(residual)` → `[B, n_concepts]`; `pool_loss = focal_bce_loss(pool_logits, pool_labels, gamma=concept_focal_gamma)`.
   - `channel_input = residual[..., -n_concepts:]` → `[B, T, n_concepts]`.
   - `channel_logits = per_channel_head(channel_input)` → `[B, T, n_concepts]`; `channel_loss = focal_bce_loss(channel_logits, labels, gamma=concept_focal_gamma)`.
   - `concept_loss = concept_pool_weight * pool_loss + concept_channel_weight * channel_loss`.
   - Compose with the existing loss (CE alone, or CE + distillation KL): `total_loss = (1.0 - concept_alpha) * existing_loss + concept_alpha * concept_loss`.
3. `total_loss.backward()` — gradients flow through the student AND the two heads. The host (if any) stays `no_grad`.

When `config.concept_alpha == 0.0`, none of the above runs (no label source instantiated, no heads constructed, no per-step concept work).

#### Scenario: `concept_alpha == 0.0` is byte-identical

- **WHEN** `run_finetune(model, host=None, iterator, config=TrainingConfig(concept_alpha=0.0))` runs for 10 steps
- **THEN** the recorded loss history is bit-equal to a baseline run on the same fixture without the new fields

#### Scenario: gradients flow through student + heads under positive `concept_alpha`

- **GIVEN** `concept_alpha=0.1`, a fake polygram basis with `n_clusters=4`, a toy student model, and a single-batch iterator
- **WHEN** one optimiser step completes
- **THEN** `total_loss.requires_grad` is True
- **AND** the student's first parameter changes value
- **AND** the pooled head's parameters change value
- **AND** the per-channel head's parameters change value
- **AND** the host's parameters (if a host is present from the distillation branch) are unchanged

#### Scenario: positive `concept_alpha` with `host=None` is supported

- **WHEN** `run_finetune(model, host=None, iterator, config=TrainingConfig(concept_alpha=0.1, distill_alpha=1.0))` runs
- **THEN** training proceeds without raising; the concept branch is independent of the distillation branch

#### Scenario: `n_concepts <= 1` from `prepare` is rejected

- **GIVEN** a polygram basis whose `metadata["n_clusters"]` is `0` or `1`
- **WHEN** `PolygramClusterLabelSource.prepare(...)` runs
- **THEN** the source raises `ValueError` naming the polygram report and suggesting a higher-rung encoding; the fine-tune fails fast before any training step

#### Scenario: loss history records `concept_loss` when branch is active

- **GIVEN** a run with `concept_alpha=0.1` and the standard loss-history surface
- **WHEN** any training step completes
- **THEN** the loss-history entry for that step has a finite `concept_loss` field
- **AND** an inactive-branch run (`concept_alpha=0.0`) records `concept_loss` as `None` for every step

### Requirement: `ForgePipeline` exposes concept-anchoring knobs

`saeforge.ForgePipeline` SHALL accept six new constructor kwargs mirroring the `TrainingConfig` fields:

- `finetune_concept_alpha: float = 0.0`
- `finetune_concept_pool_weight: float = 1.0`
- `finetune_concept_channel_weight: float = 1.0`
- `finetune_concept_focal_gamma: float = 2.0`
- `finetune_concept_label_source: str = "polygram-clusters"`
- `finetune_concept_label_source_kwargs: dict | None = None`

The pipeline SHALL thread these into the `TrainingConfig` it constructs. When `finetune_concept_alpha > 0` and the chosen label source requires the polygram basis (the v1 `polygram-clusters` backend does), the pipeline SHALL inject the polygram basis it already holds into `concept_label_source_kwargs["polygram_basis"]` automatically. Direct callers of `run_finetune` who set `concept_alpha > 0` with the polygram backend are responsible for supplying the basis themselves.

Defaults SHALL preserve byte-identity with the pre-change `ForgePipeline` smoke and integration tests.

#### Scenario: `ForgePipeline` defaults preserve byte-identity

- **WHEN** `ForgePipeline(...)` is constructed without `finetune_concept_*` arguments and `.run()` is called
- **THEN** the run produces a forged transformer bit-equal to the pre-change pipeline on the same fixture

#### Scenario: `ForgePipeline` injects the polygram basis when concept anchoring is active

- **WHEN** `ForgePipeline(..., finetune_concept_alpha=0.1).run()` is called with the v1 `polygram-clusters` backend
- **THEN** the `TrainingConfig` the pipeline constructs has `concept_label_source_kwargs["polygram_basis"]` set to the pipeline's polygram basis
- **AND** `PolygramClusterLabelSource.prepare(...)` succeeds without the caller needing to pass the basis manually


## ADDED Capability: concept-anchoring

### Requirement: `LabelSource` protocol and registry

`saeforge.training.concept_anchor` SHALL expose:

- `class LabelSource(Protocol)` with two methods:
  - `prepare(self, model: nn.Module, iterator: Iterable) -> int` — runs one-time calibration; returns `n_concepts`.
  - `labels_for_batch(self, batch, hidden_states: Tensor | None) -> Tensor` — returns multi-hot labels, shape `[B, T, n_concepts]`, dtype float, values in `{0.0, 1.0}`.
- `LABEL_SOURCE_REGISTRY: dict[str, type[LabelSource]]` — module-level registry, initially empty and populated by decorator at import time.
- `register_label_source(name: str)` decorator — registers the decorated class under `name`; raises `ValueError` on duplicate registration.

The registry SHALL be the single source of truth for `TrainingConfig.concept_label_source` validation. Adding a new backend requires only `@register_label_source("name")` on a `LabelSource` subclass.

#### Scenario: registry contains the v1 backend

- **WHEN** `saeforge.training.concept_anchor` is imported
- **THEN** `"polygram-clusters" in LABEL_SOURCE_REGISTRY` is True
- **AND** `LABEL_SOURCE_REGISTRY["polygram-clusters"]` is `PolygramClusterLabelSource`

#### Scenario: custom backend registration

- **WHEN** `@register_label_source("custom")` is applied to a class implementing the `LabelSource` protocol
- **THEN** `LABEL_SOURCE_REGISTRY["custom"]` is the class
- **AND** registering a second class under the same name raises `ValueError`

### Requirement: `PolygramClusterLabelSource` backend

The v1 bundled backend SHALL:

- Be registered under the key `"polygram-clusters"`.
- Accept a `polygram_basis: FeatureBasis` argument plus optional `calibration_batches: int = 32` and `firing_threshold: float = 0.5`.
- In `prepare`:
  - Run `calibration_batches` no-grad student forwards.
  - Project last-layer residuals into the polygram feature space via the basis's pseudoinverse.
  - Determine `n_concepts = polygram_basis.metadata["n_clusters"]`; raise `ValueError` (with a message naming the polygram report path and suggesting a higher-rung encoding) when `n_clusters` is `None`, `0`, or `1`.
  - Freeze a per-cluster firing threshold (v1: the `firing_threshold` scalar applied uniformly).
  - Return `n_concepts`.
- In `labels_for_batch`:
  - Raise `ValueError` when `hidden_states is None` (the backend relies on the caller passing residuals).
  - Project hidden states into the polygram feature space; threshold against the frozen per-cluster thresholds; return the multi-hot tensor.

The backend SHALL NOT refresh its frozen state during training — labels remain fixed for the run's duration.

#### Scenario: `prepare` returns the polygram cluster count

- **GIVEN** a fake `FeatureBasis` with `metadata["n_clusters"]=4` and a calibration iterator yielding 32+ batches
- **WHEN** `PolygramClusterLabelSource(polygram_basis=basis).prepare(model, iterator)` runs
- **THEN** the return value is `4`

#### Scenario: `prepare` rejects trivial cluster count

- **GIVEN** a fake basis with `metadata["n_clusters"]=1`
- **WHEN** `PolygramClusterLabelSource(polygram_basis=basis).prepare(model, iterator)` runs
- **THEN** `ValueError` is raised; the message names the polygram report path and recommends a higher-rung encoding

#### Scenario: `labels_for_batch` returns multi-hot of the documented shape

- **GIVEN** a prepared `PolygramClusterLabelSource` with `n_concepts=4`, a batch with `B=2`, `T=16`, and a residual tensor of shape `[2, 16, d_model]`
- **WHEN** `labels_for_batch(batch, hidden_states=residual)` is called
- **THEN** the returned tensor has shape `[2, 16, 4]`, dtype float, and every value is either `0.0` or `1.0`

#### Scenario: `labels_for_batch` requires hidden states

- **WHEN** `labels_for_batch(batch, hidden_states=None)` is called
- **THEN** the backend raises `ValueError` whose message explains that the polygram backend needs the residual stream from the caller

### Requirement: `PooledConceptHead`, `PerChannelConceptHead`, `focal_bce_loss`

`saeforge.training.heads` SHALL expose three primitives:

- `PooledConceptHead(d_model, n_concepts)` — `nn.Module` whose `forward(residual: Tensor)` consumes `[B, T, d_model]`, mean-pools over T, and returns `[B, n_concepts]` logits.
- `PerChannelConceptHead(n_concepts)` — `nn.Module` whose `forward(channels: Tensor)` consumes a `[..., n_concepts]` slice of the residual stream and returns the same shape, with per-concept scalar weight + bias parameters.
- `focal_bce_loss(logits, labels, *, gamma: float, reduction: str = "mean") -> Tensor` — focal-weighted BCE, equivalent to `BCEWithLogitsLoss(reduction="none")` × `(1 - p_t) ** gamma`, then reduced. With `gamma == 0.0`, identical to plain BCE-with-logits (modulo float arithmetic) to within `1e-6` tolerance.

#### Scenario: `PooledConceptHead` shape contract

- **GIVEN** `PooledConceptHead(d_model=768, n_concepts=6)`
- **WHEN** `forward(residual)` is called with `residual.shape == [2, 16, 768]`
- **THEN** the output shape is `[2, 6]`

#### Scenario: `PerChannelConceptHead` shape contract

- **GIVEN** `PerChannelConceptHead(n_concepts=6)`
- **WHEN** `forward(channels)` is called with `channels.shape == [2, 16, 6]`
- **THEN** the output shape is `[2, 16, 6]`

#### Scenario: `focal_bce_loss` reduces to BCE when γ=0

- **GIVEN** any logits + labels pair
- **WHEN** `focal_bce_loss(logits, labels, gamma=0.0)` is compared to `F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")`
- **THEN** the two values agree to within `1e-6`

#### Scenario: `focal_bce_loss` down-weights confident terms when γ=2

- **GIVEN** a logit/label pair where `p_t ≈ 0.99` for the correct class
- **WHEN** `focal_bce_loss(...)` is computed with γ=2.0 and reduction="none"
- **THEN** the focal weight applied to that element is approximately `(1 - 0.99)² ≈ 1e-4` times the plain-BCE value (within numerical precision)
