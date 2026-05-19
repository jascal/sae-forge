# sae-moe-forge Specification

## Purpose

The `sae-moe-forge` capability defines the public surface for
turning a polygram-compressed SAE into a routed mixture-of-experts.
Each expert is a coherent cluster of SAE features; a per-token
router selects the top-k experts that should participate in
reconstructing the residual stream. The result is a torch module
whose decode cost scales as `k / n_experts` of the flat-SAE decode
while preserving the feature-level interpretability the underlying
SAE delivers.

v1 ships:

- One expert implementation: `"sub_dictionary"` — each expert is a
  deterministic slice of the source SAE decoder. No new parameters.
- One router implementation: `"polygram_heuristic"` — wraps
  polygram 0.9.0's `ExpertDictionary.route` (sum activations per
  expert, top-k). Zero trainable parameters.
- Standalone module surface: `ForgedMoE` is a `torch.nn.Module`
  with `forward`, `route`, `expert_load`. Not yet wired into the
  forged-transformer residual stream — that's a queued follow-up.

The v1 contract is **structurally correct + computationally honest +
inference-only**. Quality claims (versus a learned router, versus a
distilled expert) belong to follow-up proposals.

## Requirements

### Requirement: `forge_to_moe` entry point

`saeforge.forge_to_moe` SHALL be the single public entry for
constructing a `ForgedMoE`. Its signature SHALL be:

```python
def forge_to_moe(
    basis: FeatureBasis,
    expert_dictionary: ExpertDictionary | None = None,
    *,
    k_experts: int = 2,
    expert_type: str = "sub_dictionary",
    router_type: str = "polygram_heuristic",
    coherence_threshold: float = 0.3,
    max_features_per_expert: int | None = None,
) -> ForgedMoE
```

`forge_to_moe` SHALL:

- When `expert_dictionary` is supplied: use it directly. Raise
  `ValueError` when `expert_dictionary.n_features != basis.n_features`.
- When `expert_dictionary is None`: load the polygram `Dictionary`
  from `basis.polygram_checkpoint_path` via
  `polygram.load_sae_safetensors`, then call
  `polygram.cluster_experts(...)` with the supplied
  `coherence_threshold` and `max_features_per_expert`. Raise
  `ValueError` with a clear pointer at the queued
  `add-moe-explicit-cluster-construction` follow-up when
  `polygram_checkpoint_path` is unavailable.
- Validate `1 <= k_experts <= expert_dictionary.n_experts`.
- Validate `expert_type ∈ {"sub_dictionary"}`. Other values
  (`"tiny_mlp"`, `"residual_block"`) SHALL raise
  `NotImplementedError` naming the queued follow-up proposal.
- Validate `router_type ∈ {"polygram_heuristic"}`. Other values
  (`"linear"`, `"mlp"`) SHALL raise `NotImplementedError` naming
  the queued follow-up proposal.

#### Scenario: explicit ExpertDictionary path

- **GIVEN** a `FeatureBasis` with `n_features=128` and an
  `ExpertDictionary` with `n_experts=4, n_features=128`
- **WHEN** `forge_to_moe(basis, expert_dictionary=ed, k_experts=2)`
  is called
- **THEN** the returned `ForgedMoE` SHALL have `config.n_experts=4`,
  `config.k_experts=2`, `config.expert_type="sub_dictionary"`,
  `config.router_type="polygram_heuristic"`.

#### Scenario: missing polygram checkpoint path

- **GIVEN** a `FeatureBasis` constructed via `FeatureBasis(...)`
  directly (no `polygram_checkpoint_path`)
- **WHEN** `forge_to_moe(basis)` is called without
  `expert_dictionary`
- **THEN** the call SHALL raise `ValueError`
- **AND** the message SHALL name `add-moe-explicit-cluster-
  construction` as the queued follow-up that would auto-cluster
  from `basis.W_dec` alone.

#### Scenario: invalid k_experts

- **GIVEN** an `ExpertDictionary` with `n_experts=4`
- **WHEN** `forge_to_moe(basis, expert_dictionary=ed, k_experts=8)`
  is called
- **THEN** the call SHALL raise `ValueError` naming the legal
  range `[1, 4]`.

### Requirement: `ForgedMoE` module contract

`saeforge.ForgedMoE` SHALL be a `torch.nn.Module` with:

- `forward(residual: Tensor, *, track_load: bool = False) -> Tensor`.
  `residual` has shape `(*batch_dims, d_model)`; output has the same
  shape. Computes `features = encoder(residual)`, then
  `top_k_experts = router.route(features, k_experts)`, then
  per-expert decoding masked by `top_k_experts`. Output is the
  summed reconstruction across the k selected experts per token.
- `route(residual: Tensor) -> Tensor` returning a
  `(*batch_dims, k_experts)` int64 tensor of selected expert
  indices, ordered by descending routing score per token.
- `expert_load() -> Tensor | None` returning a `(n_experts,)`
  float tensor of token-slot fractions if `track_load=True` was
  set on the most recent `forward` call; `None` otherwise.
- `.config: ForgedMoEConfig` exposing the v1 contract surface
  (n_features, d_model, n_experts, k_experts, expert_type,
  router_type, source_basis_checkpoint).

The module SHALL NOT register trainable parameters in v1. Calling
`.parameters()` SHALL return an empty iterator. Buffers (`W_enc`,
`b_enc`, per-expert decoder slices, feature-to-expert maps) are
registered as `nn.Buffer` so device-moves work.

#### Scenario: k=n_experts collapses to flat SAE

- **GIVEN** a `ForgedMoE` with `n_experts=4, k_experts=4` built
  from a basis whose flat decoder is `W_dec`
- **WHEN** `forge_moe(residual)` is called on a 256-token
  calibration batch
- **THEN** the output SHALL equal `features @ W_dec` (the flat
  SAE reconstruction) within mean-squared-error per coordinate
  `<= 1e-5`.

#### Scenario: routed reconstruction has correct shape

- **GIVEN** a `ForgedMoE` with `d_model=64, n_features=128,
  n_experts=8, k_experts=2`
- **WHEN** `forge_moe(residual)` is called with `residual.shape =
  (4, 32, 64)`
- **THEN** the output SHALL have shape `(4, 32, 64)`
- **AND** `forge_moe.route(residual).shape` SHALL equal `(4, 32, 2)`
- **AND** each row of `forge_moe.route(residual)` SHALL contain
  exactly `k_experts=2` distinct integer values in
  `[0, n_experts=8)`.

### Requirement: sub-dictionary expert correctness

The `"sub_dictionary"` expert implementation SHALL satisfy:

- Each expert `e` holds an integer index list
  `expert_feature_ids[e]` partitioning `range(n_features)`. The
  union over `e` is exactly `range(n_features)`; no feature
  belongs to more than one expert.
- Each expert holds a row slice `expert_W_dec[e]` of
  `basis.W_dec[expert_feature_ids[e]]`. No copy of the encoder
  side; encoder is shared.
- The forward over k selected experts SHALL produce
  `sum_e (features[..., expert_feature_ids[e]] @ expert_W_dec[e])`
  summed over selected `e`.

This is the *deterministic projection of the flat SAE into a
routed form* — no learned re-projection, no quantisation, no
distillation.

#### Scenario: feature partition is complete and disjoint

- **GIVEN** a `ForgedMoE.experts: SubDictionaryExpertSet` with
  `n_features=100, n_experts=4`
- **WHEN** the union of `experts.expert_feature_ids[e]` over
  `e ∈ [0, 4)` is computed
- **THEN** the union SHALL equal `set(range(100))`
- **AND** the intersection of any two distinct experts' feature
  ids SHALL be empty.

### Requirement: polygram-heuristic router parity with polygram

`PolygramHeuristicRouter.route(features, top_k)` SHALL produce
results equivalent to `ExpertDictionary.route(activations, top_k)`
applied per-vector, within float tolerance.

The torch implementation SHALL batch-vectorise over the leading
batch dimensions; the polygram surface is per-vector numpy. The
batched implementation SHALL be a strict vectorisation, not a
re-implementation with different tie-breaking — equal scoring
experts SHALL be ordered identically to polygram's stable argsort.

#### Scenario: torch router matches polygram per-vector

- **GIVEN** an `ExpertDictionary` `ed`, a `(B, n_features)` float
  activation batch
- **WHEN** the torch router's `.route(activations_torch, top_k)`
  is compared to `[ed.route(act_i, top_k) for act_i in
  activations_np]`
- **THEN** the two results SHALL be element-wise equal as int64
  tensors / lists for the same input.

### Requirement: mechanical-correctness acceptance gate (universal)

On any fixture the implementation accepts, `forge_to_moe` SHALL
satisfy three mechanical bands measured on a 256-token calibration
batch:

- **Band A — fidelity collapse.** With `k_experts = n_experts`,
  routed reconstruction MSE per coordinate vs flat-SAE
  reconstruction SHALL be `<= 1e-5`. The 2026-05-19 prototype
  measured ~10^-12 across all fixtures.
- **Band B — sparsity gain.** With `k_experts = 2`, the counted
  decoder-row-touch ratio (routed cost / flat cost) SHALL be in
  the band `[2/n_experts - 0.05, 2/n_experts + 0.05]`. Reflects
  cluster-size uniformity; off-band values indicate uneven
  partition shape, not a routing bug.
- **Band D — round-trip stability.** `ForgedMoE.config.to_dict()
  → from_dict()` reconstructs an equal config; the reconstructed
  module produces byte-identical reconstruction on the same input.

### Requirement: faithfulness acceptance gate (basis-split)

Routed reconstruction quality is bounded by the basis's cluster
structure. The faithfulness gate has two arms:

- **Band C-strict (clusterable basis).** When the basis has
  intra-cluster cosine median > 0.5 (computed inside
  `forge_to_moe` and exposed as
  `ForgedMoE.coherence_diagnostic.median_intra_cluster_cosine`),
  routed-vs-flat MSE at `k_experts = 2` SHALL be `<= 0.5x`
  flat-vs-host MSE. The 2026-05-19 prototype's synthetic clusterable
  fixture (4 clusters of 32, intra-cosine ~0.96) measured 0.12x,
  giving 4x headroom.
- **Band C-advisory (any basis).** The routed-vs-flat MSE and
  ratio SHALL be reported on every `ForgedMoE` instance via
  `ForgedMoE.faithfulness_report(host_residual)`. No pass/fail
  gate — a user-facing diagnostic. On near-isotropic bases the
  prototype measured ratio ≈ 4.6x (K=211 jbloom GPT-2 L8 at E=9).

#### Scenario: synthetic clusterable basis hits Band C-strict

- **GIVEN** a basis with `n_features=128, d_model=768` built from
  4 deliberate cosine-coherent clusters (intra-cluster cosine
  median > 0.9 after `cluster_experts(coherence_threshold=0.5)`)
- **WHEN** `forge_to_moe(basis, expert_dictionary=ed, k_experts=2)`
  is constructed and forward over a 256-token calibration batch
- **THEN** `routed_vs_flat_mse / flat_vs_host_mse` SHALL be `<= 0.5`.

#### Scenario: near-isotropic basis advisory only

- **GIVEN** a basis whose intra-cluster cosine median is `<= 0.5`
  after polygram clustering
- **WHEN** `forge_to_moe(...)` is constructed
- **THEN** `ForgedMoE.coherence_diagnostic` SHALL flag the basis
  as low-coherence
- **AND** the Band C-strict gate SHALL NOT apply
- **AND** `ForgedMoE.faithfulness_report(host)` SHALL still
  populate the ratio for the user to inspect.

The reproducer is `scripts/prototype_sae_moe_forge.py`. Its
outputs commit to `reports/moe_forge/`. The smoke-results
audit lives in this change's `smoke-results.md`, which lands
BEFORE production code per the openspec cadence.

#### Scenario: band B sparsity check

- **GIVEN** the GPT-2 L8 K=211 basis clustered into `n_experts=8`
  with `k_experts=2`
- **WHEN** the prototype computes `effective_decode_cost` over the
  256-token calibration batch
- **THEN** the ratio of `effective_decode_cost / flat_decode_cost`
  SHALL be within `[0.20, 0.30]` (the `k/E = 0.25` band ± cluster-
  size variance).

### Requirement: out-of-scope behaviour surfaces clean errors

The capability surfaces queued follow-ups via
`NotImplementedError` rather than silent fallback:

- `expert_type="tiny_mlp"` → `NotImplementedError` naming
  `add-moe-tiny-mlp-experts`.
- `expert_type="residual_block"` → `NotImplementedError` naming
  `add-moe-residual-block-experts`.
- `router_type="linear"` → `NotImplementedError` naming
  `add-moe-trained-router`.
- `router_type="mlp"` → `NotImplementedError` naming
  `add-moe-trained-router`.
- `forge_to_moe(basis)` without `expert_dictionary` and without
  `basis.polygram_checkpoint_path` → `ValueError` naming
  `add-moe-explicit-cluster-construction`.

#### Scenario: tiny_mlp expert type raises clean error

- **WHEN** `forge_to_moe(basis, expert_dictionary=ed,
  expert_type="tiny_mlp")` is called
- **THEN** the call SHALL raise `NotImplementedError`
- **AND** the error message SHALL name `add-moe-tiny-mlp-experts`
  as the queued follow-up.
