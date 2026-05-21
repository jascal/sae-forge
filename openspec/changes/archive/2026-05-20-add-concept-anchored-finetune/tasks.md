## 1. `saeforge/training/heads.py` module

- [ ] 1.1 Create `saeforge/training/heads.py` with module-level docstring linking to this proposal and noting the Phase 6.2 lineage from econ-sae.
- [ ] 1.2 Add `class PooledConceptHead(nn.Module)`:
  - `__init__(self, d_model: int, n_concepts: int)`
  - `forward(self, residual: Tensor) -> Tensor` — input shape `[B, T, d_model]`, applies `mean_pool` over dim T, returns shape `[B, n_concepts]`.
- [ ] 1.3 Add `class PerChannelConceptHead(nn.Module)`:
  - `__init__(self, n_concepts: int)`
  - Per-concept scalar weight + bias (two `nn.Parameter` of shape `[n_concepts]`).
  - `forward(self, channels: Tensor) -> Tensor` — input shape `[..., n_concepts]` (the designated last-N residual dims), returns same shape (per-concept logits).
- [ ] 1.4 Add `def focal_bce_loss(logits, labels, *, gamma: float, reduction: str = "mean") -> Tensor`:
  - Wraps `BCEWithLogitsLoss(reduction="none")`, multiplies elementwise by `(1 - p_t) ** gamma` where `p_t = softplus_sigmoid(logits)` evaluated with respect to the label.
  - Supports `reduction in {"none", "sum", "mean"}`; default `"mean"`.
  - When `gamma == 0.0`, the focal weight is constant 1, reducing to plain BCE.
- [ ] 1.5 Unit tests covering the shape contracts and the `gamma=0` reduction (see §4.2).

## 2. `saeforge/training/concept_anchor.py` module

- [ ] 2.1 Create `saeforge/training/concept_anchor.py`.
- [ ] 2.2 Define `class LabelSource(Protocol)`:
  - `prepare(self, model: nn.Module, iterator: Iterable, **kwargs) -> int` — returns `n_concepts`.
  - `labels_for_batch(self, batch, hidden_states: Tensor | None) -> Tensor` — returns multi-hot tensor of shape `[B, T, n_concepts]`, dtype float (in `{0.0, 1.0}`).
  - Docstring documents the contract: `prepare` is called once; `labels_for_batch` is called per training step; `hidden_states` is the student's last-layer activations from the forward already computed (backend may reuse them, or ignore and recompute).
- [ ] 2.3 Define `LABEL_SOURCE_REGISTRY: dict[str, type[LabelSource]] = {}`.
- [ ] 2.4 Define `def register_label_source(name: str)` decorator. Adds the decorated class to `LABEL_SOURCE_REGISTRY[name]`. Raises `ValueError` on duplicate registration.
- [ ] 2.5 Implement `@register_label_source("polygram-clusters") class PolygramClusterLabelSource(LabelSource)`:
  - `__init__(self, polygram_basis: FeatureBasis, calibration_batches: int = 32, firing_threshold: float = 0.5)` — accepts the polygram basis (passed by the caller; the loop has it available via the `ForgePipeline`'s context).
  - `prepare(self, model, iterator)`:
    - Iterates `calibration_batches` batches from `iterator` (uses `itertools.islice`; restores iterator state if possible — see §2.6).
    - For each batch: run student forward under `torch.no_grad()`, project hidden states into the polygram feature space via `polygram_basis.pseudoinverse @ hidden`.
    - Records the per-cluster firing probability distribution (histogram or per-cluster mean) across all calibration tokens.
    - Determines `n_concepts = polygram_basis.metadata["n_clusters"]` (raises if `n_clusters` is missing or `<= 1`, with a message naming the polygram report and suggesting a higher-rung encoding).
    - Stores per-cluster thresholds (default: `firing_threshold` directly; future extension can be quantile-based).
    - Returns `n_concepts`.
  - `labels_for_batch(self, batch, hidden_states)`:
    - If `hidden_states is None`, raise (the polygram backend depends on the residual stream being available; the loop is expected to pass it in).
    - Project hidden states into the polygram feature space; threshold against the per-cluster thresholds; return multi-hot tensor.
- [ ] 2.6 Iterator-state caveat: `calibration_batches` consumes from the training iterator. The loop documentation in §3 SHALL be explicit that the iterator must be either resettable (e.g., a `DataLoader` rather than a raw `iter()` chain), OR the caller must pre-slice the calibration data and pass a fresh iterator. The polygram backend's `prepare` does NOT silently re-iterate; behaviour is what `iterator` exposes.

## 3. `TrainingConfig` extensions

- [ ] 3.1 Add six new fields to `saeforge/training/config.py::TrainingConfig`:
  - `concept_alpha: float = 0.0`
  - `concept_pool_weight: float = 1.0`
  - `concept_channel_weight: float = 1.0`
  - `concept_focal_gamma: float = 2.0`
  - `concept_label_source: str = "polygram-clusters"`
  - `concept_label_source_kwargs: dict[str, Any] = field(default_factory=dict)`
- [ ] 3.2 `__post_init__` validation (extend existing block):
  - `0.0 <= concept_alpha <= 1.0`; out-of-range raises `ValueError` naming the value and valid range.
  - `concept_pool_weight >= 0` AND `concept_channel_weight >= 0`.
  - When `concept_alpha > 0`: at least one of `concept_pool_weight`, `concept_channel_weight` SHALL be `> 0`; otherwise raise `ValueError("concept_alpha > 0 requires non-zero pool or channel weight")`.
  - `concept_focal_gamma >= 0`.
  - When `concept_alpha > 0`: `concept_label_source` SHALL be in `LABEL_SOURCE_REGISTRY`; unknown keys raise `ValueError` listing the registered keys.

## 4. `run_finetune` loop extension

- [ ] 4.1 In `saeforge/training/loop.py::run_finetune`, before the main training loop, add a guarded setup block:
  - When `config.concept_alpha > 0`:
    - Look up `LabelSourceCls = LABEL_SOURCE_REGISTRY[config.concept_label_source]`.
    - Instantiate `label_source = LabelSourceCls(**config.concept_label_source_kwargs)` (the polygram backend's `__init__` expects `polygram_basis` — the loop fetches this from `model.config.polygram_basis` or an equivalent accessor exposed by `ForgePipeline`; see §5).
    - `n_concepts = label_source.prepare(model, iterator)`.
    - Construct `pooled_head = PooledConceptHead(d_model=model.config.hidden_size, n_concepts=n_concepts)` and `channel_head = PerChannelConceptHead(n_concepts=n_concepts)`; move to the model's device.
    - Add both heads' parameters to the optimiser via a parameter-group append.
    - Record `n_concepts` and label-source state in the run metadata.
- [ ] 4.2 In the per-step loop, when `config.concept_alpha > 0`:
  - After the existing student forward (the one that produces logits + final hidden states), fetch `residual = hidden_states[-1]` (last-layer, full `[B, T, d_model]`).
  - `labels = label_source.labels_for_batch(batch, hidden_states=residual.detach())` → shape `[B, T, n_concepts]`.
  - `pool_labels = labels.max(dim=1).values` → `[B, n_concepts]`.
  - `pool_logits = pooled_head(residual)` → `[B, n_concepts]`.
  - `pool_loss = focal_bce_loss(pool_logits, pool_labels, gamma=config.concept_focal_gamma)`.
  - `channel_input = residual[..., -n_concepts:]` → `[B, T, n_concepts]`.
  - `channel_logits = channel_head(channel_input)` → `[B, T, n_concepts]`.
  - `channel_loss = focal_bce_loss(channel_logits, labels, gamma=config.concept_focal_gamma)`.
  - `concept_loss = config.concept_pool_weight * pool_loss + config.concept_channel_weight * channel_loss`.
  - `total_loss = (1 - config.concept_alpha) * existing_loss + config.concept_alpha * concept_loss`.
  - `existing_loss` is whatever the loop already computes (CE alone, or CE + distillation KL when `distill_alpha < 1.0`).
- [ ] 4.3 When `config.concept_alpha == 0.0`, the concept branch is skipped entirely (no setup, no per-step work, no head construction) — byte-identical to the current loop.
- [ ] 4.4 Wire the loss-tracking surface (`LossHistory`) to record `concept_loss` per step when the branch is active. New field on the loss-history struct; default `None` per step when inactive.

## 5. `ForgePipeline` plumbing

- [ ] 5.1 Add six matching kwargs to `ForgePipeline.__init__`, mirroring the existing `finetune_*` knobs:
  - `finetune_concept_alpha: float = 0.0`
  - `finetune_concept_pool_weight: float = 1.0`
  - `finetune_concept_channel_weight: float = 1.0`
  - `finetune_concept_focal_gamma: float = 2.0`
  - `finetune_concept_label_source: str = "polygram-clusters"`
  - `finetune_concept_label_source_kwargs: dict | None = None` (parsed to `{}` default)
- [ ] 5.2 Thread them into the `TrainingConfig` instance the pipeline builds.
- [ ] 5.3 Make the polygram basis available to label sources: extend the loop's call signature so the `TrainingConfig.concept_label_source_kwargs` dict can receive `polygram_basis` automatically when `ForgePipeline` constructs the config (the pipeline knows the basis; users invoking `run_finetune` directly must pass it themselves via `concept_label_source_kwargs={"polygram_basis": ...}`).
- [ ] 5.4 The defaults preserve byte-identity with the existing pipeline tests.

## 6. Tests

### 6.1 Heads

- [ ] 6.1.1 `tests/training/test_concept_heads.py::test_pooled_head_shape`: input `[2, 16, 768]`, n_concepts=6 → output `[2, 6]`.
- [ ] 6.1.2 `test_per_channel_head_shape`: input `[2, 16, 6]` (the last-6 dims), n_concepts=6 → output `[2, 16, 6]`.
- [ ] 6.1.3 `test_focal_bce_gamma_zero_equals_bce`: γ=0 produces the same value as `F.binary_cross_entropy_with_logits` to within 1e-6.
- [ ] 6.1.4 `test_focal_bce_gamma_two_down_weights_confident`: with γ=2 and a logit/label pair where p_t ≈ 0.99, the focal loss is ≈ (0.01²) × BCE; verify the ratio.
- [ ] 6.1.5 `test_focal_bce_gradient_flows`: γ=2, requires_grad=True logits, backward through `focal_bce_loss(...)` produces finite gradients of the right shape.

### 6.2 Concept-anchor module

- [ ] 6.2.1 `tests/training/test_concept_anchor.py::test_registry_lookup`: assert `"polygram-clusters" in LABEL_SOURCE_REGISTRY` and `LABEL_SOURCE_REGISTRY["polygram-clusters"]` is `PolygramClusterLabelSource`.
- [ ] 6.2.2 `test_register_label_source_decorator`: define a stub LabelSource via the decorator; assert it appears in the registry; remove it.
- [ ] 6.2.3 `test_register_label_source_rejects_duplicate`: registering the same key twice raises `ValueError`.
- [ ] 6.2.4 `test_polygram_cluster_label_source_prepare_returns_n_concepts`: with a fake `FeatureBasis` whose `metadata["n_clusters"]=4`, `prepare` returns 4 after consuming the requested calibration batches.
- [ ] 6.2.5 `test_polygram_cluster_label_source_rejects_trivial_cluster_count`: `metadata["n_clusters"]=1` → `prepare` raises with a message naming the polygram report path.
- [ ] 6.2.6 `test_labels_for_batch_shape`: prepared backend + a fake hidden-states tensor → output shape `[B, T, n_concepts]`, values in `{0.0, 1.0}`.
- [ ] 6.2.7 `test_labels_for_batch_requires_hidden_states`: `labels_for_batch(batch, hidden_states=None)` raises with a clear error.

### 6.3 TrainingConfig

- [ ] 6.3.1 `tests/training/test_concept_config.py::test_concept_alpha_default_zero`: default `TrainingConfig()` has `concept_alpha == 0.0` and all other concept fields at their documented defaults.
- [ ] 6.3.2 `test_concept_alpha_out_of_range_rejected`: `concept_alpha=-0.1` → `ValueError`; `concept_alpha=1.5` → `ValueError`.
- [ ] 6.3.3 `test_concept_alpha_positive_requires_nonzero_weight`: `concept_alpha=0.5, concept_pool_weight=0, concept_channel_weight=0` → `ValueError`.
- [ ] 6.3.4 `test_concept_focal_gamma_negative_rejected`: `concept_focal_gamma=-0.5` → `ValueError`.
- [ ] 6.3.5 `test_concept_label_source_unknown_rejected`: `concept_alpha=0.5, concept_label_source="bogus"` → `ValueError` listing the registered keys.
- [ ] 6.3.6 `test_concept_label_source_not_validated_when_alpha_zero`: `concept_alpha=0.0, concept_label_source="bogus"` → no error (validation skipped because the branch will be inactive).

### 6.4 run_finetune integration

- [ ] 6.4.1 `tests/training/test_concept_anchor_loop.py::test_alpha_zero_byte_identical`: train 10 steps with `concept_alpha=0.0` and an otherwise-default config; snapshot the loss history; compare to a baseline run without the new fields (byte-identical loss history). Load-bearing v0.3 + distillation invariant.
- [ ] 6.4.2 `test_alpha_zero_skips_label_source_construction`: monkeypatch `PolygramClusterLabelSource.__init__` to raise; with `concept_alpha=0.0`, training proceeds without instantiating the source (the raise is never hit).
- [ ] 6.4.3 `test_alpha_positive_constructs_heads_and_trains`: `concept_alpha=0.1` with a fake polygram basis (n_clusters=4), one training step:
  - Heads are constructed; their parameters are added to the optimiser.
  - `total_loss` is finite.
  - Student's first parameter moves after `optim.step()`.
  - Pooled head and per-channel head parameters move.
  - The loss-history record for the step has `concept_loss` populated as a finite float.
- [ ] 6.4.4 `test_concept_loss_recorded_in_metadata`: after a run with `concept_alpha=0.1`, the run metadata dict (returned by the training-config to-metadata path) contains `concept_alpha`, `concept_label_source`, `n_concepts`.
- [ ] 6.4.5 `test_iterator_not_silently_advanced_during_calibration`: when the iterator is a real DataLoader, `prepare`'s calibration batches consume from the same iterator the training loop reads from. The test asserts the documented behaviour: the training loop sees fewer than expected steps because calibration consumed first. (Documentation in `docs/concept-anchoring.md` warns about this; the test enforces it.)

### 6.5 ForgePipeline plumbing

- [ ] 6.5.1 `tests/forge/test_pipeline_concept.py::test_pipeline_defaults_byte_identical`: existing `ForgePipeline(...)` smoke + new kwargs unset → byte-equivalent to pre-change pipeline.
- [ ] 6.5.2 `test_pipeline_concept_alpha_threads_to_training_config`: `ForgePipeline(..., finetune_concept_alpha=0.1).run()` produces a `TrainingConfig` with `concept_alpha == 0.1`.
- [ ] 6.5.3 `test_pipeline_polygram_basis_threaded_to_label_source`: under `finetune_concept_alpha=0.1`, the polygram basis is passed into `concept_label_source_kwargs` automatically; smoke run completes; metadata records `n_concepts`.

### 6.6 End-to-end smoke

- [ ] 6.6.1 `tests/forge/test_pipeline_concept_smoke.py::test_concept_smoke_e2e`: toy GPT-2 + 4-feature polygram fixture; `ForgePipeline.run(..., finetune_concept_alpha=0.1, finetune_concept_focal_gamma=2.0, total_steps=4)`; assert run completes, `result.faithfulness_kl` is finite, metadata records the concept-anchoring config and n_concepts.

## 7. Docs

- [ ] 7.1 Add a "Concept anchoring" section to `docs/finetune-recipe.md`, after the planned "Host distillation" section, covering:
  - When to use it (polygram-cluster preservation during fine-tune; future: external concept tagging).
  - The dual-head structure (pooled + per-channel) and why both.
  - Focal loss + γ; recommended starting γ=2.0.
  - Recommended starting `concept_alpha=0.1`; warning against `> 0.3` without per-host validation.
  - The iterator-consumption caveat for calibration batches.
  - How to disable: omit the fields, or set `concept_alpha=0.0`.
- [ ] 7.2 Create `docs/concept-anchoring.md` (new short doc):
  - Lineage: econ-sae Phase 5.1 → 6.2 dual-head + focal loss recipe.
  - The `LabelSource` protocol and the registry.
  - The bundled `polygram-clusters` backend in detail.
  - How to add a custom label source via `@register_label_source(...)`.
  - Worked example: GPT-2 toy fixture with polygram-clusters at `concept_alpha=0.1`, before/after fine-tune metric snapshots.
  - Forward pointer: `add-concept-tag-corpus-preprocessing` will add corpus-tag and LLM-tagger backends.
- [ ] 7.3 `CHANGELOG.md` entry under `[Unreleased]` → `### Added (add-concept-anchored-finetune)`.

## 8. Validation

- [ ] 8.1 `openspec validate add-concept-anchored-finetune --strict` is green.
- [ ] 8.2 Full `pytest` suite passes; new tests cover §6.1 through §6.6.
- [ ] 8.3 `ruff check` clean on touched files.
- [ ] 8.4 Live MBP smoke (GPT-2 + a real polygram-compressed SAE):
  - Baseline run: `concept_alpha=0.0`. Record final faithfulness_kl and polygram_redundancy_ratio of the final SAE.
  - Concept-anchored run: `concept_alpha=0.1`, same seed.
  - Falsifiable check: post-fine-tune `polygram_redundancy_ratio` SHALL be strictly higher (clusters more concentrated) at matched faithfulness_kl ± 5%. If both rises happen, the recipe is doing what it claims; if KL regresses by more than 5%, document and triage.
- [ ] 8.5 `openspec archive add-concept-anchored-finetune` after merge.

## 9. What this change explicitly defers

- [ ] 9.1 Corpus-tag-based label sources (spaCy NER, POS, sentiment). Queued as `add-concept-tag-corpus-preprocessing`.
- [ ] 9.2 LLM-tagger-based label sources (safety, human-interest dimensions tagged by a stronger model). Folds into the corpus-preprocessing proposal as a tagger backend.
- [ ] 9.3 Cross-feature attention block at the d→f boundary (the structural follow-up — econ-sae's `AttnWorldModel` analog). Substantial new code; deferred to a separate v2 proposal.
- [ ] 9.4 Configurable `concept_channel_indices` for non-default channel-slot assignment. Last-N reservation is sufficient for v1.
- [ ] 9.5 Refreshing labels mid-fine-tune (`label_refresh_interval`). Phase 6.2's recipe used frozen labels; we follow.
- [ ] 9.6 Provenance fields on `ParetoFrontierRow`. Concept-anchoring is per-run config; lives in run metadata.
- [ ] 9.7 Automatic hyperparameter sweep for γ / β / α. Defaults match Phase 6.2; analysts tune manually if needed.
- [ ] 9.8 A `--concept-alpha` CLI flag on top-level commands. The `ForgePipeline` kwarg is the integration point; CLI exposure can come later if there's demand.
