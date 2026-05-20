## Why

The `forge-finetune-recipe` (v0.3) and its planned host-distillation extension (`add-host-distillation-finetune-loss`) train the forged transformer with `α · CE(corpus) + (1-α) · τ² · KL(host ‖ forged)`. Both terms minimise behavioural divergence from the host. Neither term *demands that the forged model's residual stream organise around named, interpretable concepts*.

The empirical case for adding a supervised concept term comes from `econ-sae`. With a frozen substrate and unsupervised SAE training, conjunctive feature recovery mean-AUC plateaued at 0.842. Adding a **dual-head + focal-weighted supervised loss** during world-model training — pooled head over the substrate plus a per-channel head reading designated h1 dims, with focal loss for class imbalance — lifted regime-tier mAUC from 0.595 to **0.991** (6/6 features at AUC ≥ 0.95) and conjunctive-tier mAUC from 0.842 to **0.989** (7/8) in a *single training run*. Polygram compression of the supervised SAE then showed cluster count saturating at exactly the supervised concept count — direct quantitative evidence the substrate concentrated around the labelled concepts.

The asymmetry vs econ-sae matters: there, the substrate was trained from scratch. In sae-forge, the forged model inherits weights via `SubspaceProjector` from a frozen host, and the polygram basis is fixed before fine-tune begins. A supervised head can't redefine the feature axes. What it *can* do:

1. **Sharpen activation patterns** so labelled concepts fire monosemantically on the existing basis.
2. **Counter the distillation gradient's smearing** — pure KL distillation pulls all features toward the host's distribution, which is fine for behavioural fidelity but leaves polygram-cluster structure free to dilute. A focal-weighted concept term *preserves* cluster sharpness during fine-tune.
3. **Make concept anchoring legible** — a frontier.jsonl row that says "this forge was anchored to N concepts via label source X" carries provenance the analyst can audit.

The proposal extends `TrainingConfig` and `run_finetune` with one concept-anchoring term, structurally analogous to `add-host-distillation-finetune-loss`'s `distill_alpha` extension: one new scalar (`concept_alpha`) plus a small set of shape knobs, defaulting to **off** (`concept_alpha=0.0`), preserving byte-identical training with v0.3 and the distillation extension.

## What Changes

### Scope (small loss extension + label-source interface)

Add a supervised concept-anchoring term to `forge-finetune-recipe`'s loss. Defaults preserve current behaviour byte-identically. The proposal lands one bundled label-source backend (polygram-cluster pseudo-labels) and defines the interface for richer backends (e.g., `add-concept-tag-corpus-preprocessing` will add corpus-tag backends).

### New `TrainingConfig` fields

All default to values that produce zero behaviour change:

- **`concept_alpha: float = 0.0`** — overall weight on the concept-anchoring loss. `0.0` skips the entire branch (no label-source instantiation, no extra forward, byte-identical training). Valid range `[0.0, 1.0]`.
- **`concept_pool_weight: float = 1.0`** — relative weight of the pooled head's loss within the concept term. Pooled head reads the full residual stream and predicts all concepts; catches *distributed* concept encodings.
- **`concept_channel_weight: float = 1.0`** — relative weight of the per-channel head's loss within the concept term. Per-channel head reads only the last `n_concepts` residual dims (one designated channel per concept); catches *localised* concept encodings.
- **`concept_focal_gamma: float = 2.0`** — focal loss γ. `0.0` reduces to plain BCE. Phase 6.2 used γ=2.0.
- **`concept_label_source: str = "polygram-clusters"`** — registry key for the label source. v1 ships only `"polygram-clusters"`; the registry exists so follow-up proposals can add `"corpus-tags"` etc. without touching the loss code.
- **`concept_label_source_kwargs: dict[str, Any] = field(default_factory=dict)`** — backend-specific config. For `"polygram-clusters"`: optional `calibration_batches: int = 32`, `firing_threshold: float = 0.5`.

Validation in `__post_init__`:
- `0.0 <= concept_alpha <= 1.0`; out-of-range raises `ValueError`.
- `concept_pool_weight >= 0` and `concept_channel_weight >= 0`; at least one SHALL be `> 0` when `concept_alpha > 0`.
- `concept_focal_gamma >= 0`.
- `concept_label_source` SHALL be a registered key; unknown keys raise `ValueError` naming the registered options.

### Modified `run_finetune` loop

When `config.concept_alpha > 0`:

1. **Setup (once, before the training loop)**: instantiate `label_source = LABEL_SOURCE_REGISTRY[config.concept_label_source](model, iterator, **config.concept_label_source_kwargs)`. The source's `prepare(...)` method runs any required calibration (e.g., the polygram-cluster backend runs a forward pass over `calibration_batches` batches to record per-cluster firing distributions on the *pre-fine-tune* forged model). The number of concepts `n_concepts` is determined here; the per-channel head reserves the last `n_concepts` residual-stream dims.
2. **Per step**: after the student forward, fetch the per-token residual stream (the model's last hidden state). Compute:
   - `labels = label_source.labels_for_batch(batch)` → tensor shape `[B, T, n_concepts]` (multi-hot, in `{0, 1}`).
   - `pool_logits = pooled_head(mean_pool(residual, dim=T))` → `[B, n_concepts]`. Aggregate labels by max over T to make `pool_labels` → `[B, n_concepts]`.
   - `channel_logits = per_channel_head(residual[..., -n_concepts:])` → `[B, T, n_concepts]`.
   - `pool_loss = focal_bce(pool_logits, pool_labels, gamma=γ)`.
   - `channel_loss = focal_bce(channel_logits, labels, gamma=γ)`.
   - `concept_loss = β_pool · pool_loss + β_chan · channel_loss`.
3. Combine with the existing loss: `total_loss = (1 - concept_alpha) · L_existing + concept_alpha · concept_loss`. `L_existing` is whatever the v0.3 + distillation loss is (`α · CE + (1-α) · τ² · KL`); the concept term layers on top without disturbing the existing α/τ semantics.
4. `total_loss.backward()` — gradients flow through both the student and the two heads.

When `config.concept_alpha == 0.0` (default), the label source is never instantiated, the heads are never constructed, no extra forward is run — byte-identical to the pre-change loop.

### New module `saeforge/training/concept_anchor.py`

- `class LabelSource(Protocol)` — `prepare(self, model, iterator) -> int` (returns `n_concepts`); `labels_for_batch(self, batch) -> Tensor`.
- `LABEL_SOURCE_REGISTRY: dict[str, type[LabelSource]]` — module-level registry.
- `@register_label_source("polygram-clusters")` decorator.
- `class PolygramClusterLabelSource(LabelSource)` — the v1 backend. `prepare` runs `calibration_batches` student forwards under `torch.no_grad()`, records per-token cluster firing in the polygram-compressed feature space, derives a fixed firing threshold per cluster, and stores the assignment table. `labels_for_batch` re-runs the student forward in eval mode on the same batch's input ids, projects to the polygram basis via the existing pseudoinverse, thresholds, and returns the multi-hot tensor.

### New module `saeforge/training/heads.py`

- `class PooledConceptHead(nn.Module)` — `Linear(d_model, n_concepts)` after mean-pool over the time axis.
- `class PerChannelConceptHead(nn.Module)` — per-concept `(scale, bias)` reading a single designated residual dim. Output shape `[..., n_concepts]`.
- `def focal_bce_loss(logits, labels, *, gamma, reduction="mean") -> Tensor` — `BCEWithLogitsLoss` weighted by `(1 - p_t) ** gamma` per element, per Phase 6.2.

### `ForgePipeline` exposes the same knobs

`ForgePipeline` SHALL accept matching `finetune_concept_*` kwargs in its constructor and thread them into the `TrainingConfig` it builds. Default values preserve byte-identity with existing pipeline tests.

### What this PR explicitly does NOT do

- **No corpus-tag backend.** v1 ships `polygram-clusters` only. Other backends (`corpus-tags`, `host-probe`, custom user-defined) land via the registry in follow-up proposals; `add-concept-tag-corpus-preprocessing` is queued next.
- **No new CLI subcommand.** Same surface as `add-host-distillation-finetune-loss` — config extension, not a new pipeline stage.
- **No cross-feature attention block.** That's the structural follow-up (the "attention-over-features" analog of econ-sae's `AttnWorldModel`); intentionally deferred — loss-level intervention is the smaller, faster bet.
- **No supervised SAE re-training.** sae-forge consumes polygram-compressed SAEs; this proposal does not re-train the SAE itself.
- **No concept-anchoring metric on the row schema.** Provenance (which label source, n_concepts, calibration size) is recorded in fine-tune run metadata, not in `ParetoFrontierRow`. Row schema is for sweep diagnostics; this is per-run config.
- **No automatic γ / β sweep.** Defaults match Phase 6.2's working recipe. Tuning is a knob for analysts who want to push further; no auto-tuner.

## Capabilities

### Modified Capabilities

- `training`: `TrainingConfig` gains six new fields for concept anchoring; `run_finetune` gains a concept-anchoring branch active when `concept_alpha > 0`. Defaults preserve byte-identity with the v0.3 LM-CE path and the planned distillation extension. `ForgePipeline` exposes matching `finetune_concept_*` kwargs.

### Added Capabilities

- `concept-anchoring`: a new capability covering the `LabelSource` protocol, the registry, the v1 `polygram-clusters` backend, the pooled + per-channel heads, and the focal-BCE helper. Future label-source backends (corpus-tag, host-probe) extend this capability without modifying `training`.

## Impact

- **New module**: `saeforge/training/concept_anchor.py` — `LabelSource` protocol, `LABEL_SOURCE_REGISTRY`, `register_label_source` decorator, `PolygramClusterLabelSource`.
- **New module**: `saeforge/training/heads.py` — `PooledConceptHead`, `PerChannelConceptHead`, `focal_bce_loss`.
- **Modified**:
  - `saeforge/training/config.py` — six new `TrainingConfig` fields + `__post_init__` validation.
  - `saeforge/training/loop.py` — concept-anchor branch in `run_finetune`; one-time label-source setup before the main loop; per-batch loss composition.
  - `saeforge/forge.py` — `ForgePipeline` gains six matching kwargs; threads them into the `TrainingConfig` it constructs.
  - `saeforge/__init__.py` — export `register_label_source`, `LabelSource`, `LABEL_SOURCE_REGISTRY`.
- **New tests**:
  - `tests/training/test_concept_anchor.py` — `concept_alpha=0.0` byte-identity, validation, registry lookup, `PolygramClusterLabelSource.prepare` calibration shape, `labels_for_batch` returns the expected multi-hot.
  - `tests/training/test_concept_heads.py` — pooled/per-channel forward shapes, focal_bce_loss math (γ=0 reduces to BCE, γ=2 down-weights high-confidence terms).
  - `tests/training/test_concept_finetune_smoke.py` — end-to-end one-step `run_finetune` with `concept_alpha=0.1`, `concept_label_source="polygram-clusters"`, toy GPT-2 + 4-feature polygram fixture. Asserts loss is finite, student parameters move, heads' parameters move.
- **Affected docs**:
  - `docs/finetune-recipe.md` — new "Concept anchoring" section after "Host distillation".
  - `docs/concept-anchoring.md` (new, short) — explains the dual-head structure, the polygram-clusters backend, recommended starting values (`concept_alpha=0.1`, γ=2.0), and how to add a custom label source via the registry decorator.
  - `CHANGELOG.md`.
- **No breaking changes**: all new fields default to disable-everything values; `concept_alpha=0.0` is byte-identical to the pre-change loop. `ForgePipeline` byte-identity preserved when the new kwargs are omitted.

## Risks

- **Per-step cost.** When `concept_alpha > 0`, every step computes two extra linear projections (pooled + per-channel) plus a no-grad polygram-projection inside `labels_for_batch`. The polygram projection is the dominant cost. Mitigation: `PolygramClusterLabelSource.labels_for_batch` reuses cached per-batch hidden states from the student forward (no second forward pass); the polygram-projection is one `Linear` op. Per-step cost overhead < 5% for typical GPT-2-scale forges; documented in `docs/concept-anchoring.md`.
- **Label-source noise contaminates the gradient.** The polygram-clusters backend infers labels from the *pre-fine-tune* forged model — those labels are necessarily approximate. Focal loss with γ=2 handles label noise well (down-weights high-confidence terms), and `concept_alpha=0.1` keeps the term as nudge, not driver. Documented.
- **Concept anchoring fights distillation.** A high `concept_alpha` against a host that doesn't naturally encode the chosen concepts would degrade KL. Mitigation: defaults to `0.0` (off); recommended starting value `0.1` (small fraction of total loss); documentation explicitly warns against `concept_alpha > 0.3` without per-host validation.
- **Per-channel head reserves the *last* n_concepts residual dims.** This is structurally consequential — those dims become concept-supervised "slots" while the rest of the basis stays distillation-driven. Mitigation: the choice of "last N" matches econ-sae Phase 5.2's recipe and is deterministic; advanced users can override which dims are channels via a future `concept_channel_indices` kwarg (out of scope for v1, noted in docs).
- **Registry is a new public surface.** External code may register custom label sources via `@register_label_source(...)`. Mitigation: the protocol is small (two methods); semver guarantees apply to the protocol shape, not to the v1 backend's internals.
- **Behaviour change when `concept_alpha > 0` and `host is None`.** The polygram-clusters backend requires the forged model (not the host) — it works without `host`. Other label sources may require the host or external data; the registry's `prepare(model, iterator)` signature is host-agnostic. Validation: `__post_init__` does not require `host` for concept anchoring; runtime errors from a specific backend are the backend's responsibility (label sources document their requirements).
