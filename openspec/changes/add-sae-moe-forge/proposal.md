# Forge a clustered SAE into a routed mixture-of-experts

> **When to use this**: best on coherence-trained SAEs or
> polygram-clustered-during-compression outputs (where decoder rows
> form natural cosine clusters). Still functional with an advisory
> degradation diagnostic on typical isotropic SAEs — the forge
> mechanics work universally, but routing's faithfulness is
> bounded by basis cluster structure (see *Falsifiable acceptance
> gate* below for the empirical split).

## Why

Polygram 0.9.0 (PR #87) promotes `cluster_experts` and
`ExpertDictionary` to the public surface. The polygram module
docstring states the architectural split explicitly: *"Trained MLP
router — belongs in sae-forge (where torch lives). Bio-specific
scoring (GO enrichment, motif overlap) — downstream."* This is the
sae-forge half: take a polygram `ExpertDictionary` (or auto-cluster
inside the forge) and produce a torch module that routes residual-
stream activations to the top-k most-relevant experts and
reconstructs from their sub-dictionaries.

The motivation is two-fold:

1. **Runtime efficiency.** A flat SAE with N features reconstructs
   via `act @ W_enc` then `(act_features) @ W_dec` — both linear maps
   over all N features per token. An MoE-forged SAE clusters those N
   features into E experts of size N/E and routes each token to the
   top-k experts (k ≤ E), reducing per-token decode cost to
   approximately `k * (N / E)` rows — for `k=2, E=16` that's a 8×
   compute saving per token.
2. **Interpretability by routing.** Each expert is a cluster of
   coherent features (decoder rows with high pairwise cosine, in
   the cosine-clustering MVP). The expert-id sequence per token IS
   an interpretation track — analogous to "which concept cluster did
   this token activate." Steering, ablation, and downstream
   evaluation become per-expert rather than per-feature.

Neither claim is new to the literature; the contribution here is
making both *immediately usable* against sae-forge's existing
polygram-compressed SAEs and slotting into the established
forge→eval pipeline.

This is a **new abstraction layer** in sae-forge. It is NOT the
same as the existing `qwen3_moe` adapter, which forges a host
transformer that already has MoE blocks (router + N expert MLPs
inside each layer). That code projects HOST weights through a flat
basis. This proposal does the opposite: takes a flat basis and
adds a routing layer ON TOP, producing a routed SAE that can be
inserted into a host or used as a standalone reconstructor.

The two stories are orthogonal and can compose: a Qwen3-MoE host
forged through `qwen3_moe_adapter` can later have its SAE replaced
with a `ForgedMoE`. v1 ships only the standalone SAE-side MoE; the
composition is a queued follow-up.

## What Changes

### New `forge_to_moe` entry point

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

Single public entry; defaults are "the recipe that works without
training." Resolves the input as follows:

- When `expert_dictionary` is supplied, it's used directly. Its
  `n_features` must match `basis.n_features`.
- When `expert_dictionary` is `None`, `forge_to_moe` calls
  `polygram.cluster_experts(dictionary, basis.W_dec,
  method="cosine", coherence_threshold=coherence_threshold,
  max_features_per_expert=max_features_per_expert)` internally.
  This requires loading the polygram `Dictionary` from the same
  checkpoint that produced `basis`; sae-forge's
  `FeatureBasis.from_polygram_checkpoint` already retains the path,
  so we load it once via `polygram.load_sae_safetensors`.

`k_experts` is the top-k count for routing. Must be in
`[1, expert_dictionary.n_experts]`.

`expert_type` accepts one value in v1: `"sub_dictionary"`. Other
values (`"tiny_mlp"`, `"residual_block"`) raise `NotImplementedError`
pointing at the queued follow-up proposals (`add-moe-tiny-mlp-experts`,
`add-moe-residual-block-experts`).

`router_type` accepts one value in v1: `"polygram_heuristic"`. Other
values (`"linear"`, `"mlp"`) raise `NotImplementedError` pointing at
`add-moe-trained-router`.

### New `ForgedMoE` module

A `torch.nn.Module` returned by `forge_to_moe`. Surface:

- `forward(residual: Tensor) -> Tensor` — given `(B, T, d_model)`
  residual activations, returns `(B, T, d_model)` reconstruction.
  Internally: encode to basis features via the source basis's
  encoder; route to top-k experts via the heuristic router; decode
  through the selected experts' decoder sub-blocks; sum.
- `route(residual: Tensor) -> Tensor` — given `(B, T, d_model)`,
  returns `(B, T, k_experts)` int64 tensor of selected expert
  indices per token.
- `expert_load() -> Tensor` — returns `(n_experts,)` float tensor
  of the fraction of routed-token-slots each expert received on the
  most recent forward. Diagnostic; populated only when
  `track_load=True` is passed to `forward`.
- `.config: ForgedMoEConfig` — a frozen dataclass recording
  `n_features`, `n_experts`, `k_experts`, `expert_type`,
  `router_type`, and the source basis checkpoint path. Round-trips
  to JSON for save/load.

### Sub-dictionary expert path (v1 MVP)

Each expert is a *slice* of the source basis: rows of `W_dec`
indexed by the features the polygram cluster assigned to that
expert. The encoder (`W_enc` and `b_enc` from the SAE checkpoint)
is held intact at the basis level; routing selects which decoder
rows participate per token.

Forward (sketch):

```
features = encoder(residual)                  # (B, T, n_features)
top_k_experts = router.route(features)        # (B, T, k)
decoded = 0
for e in 0..n_experts:
    mask = (top_k_experts == e).any(-1)       # (B, T)
    if mask.any():
        f_e = features[..., expert_e_feature_ids]   # (B, T, n_features_e)
        d_e = f_e @ W_dec[expert_e_feature_ids]      # (B, T, d_model)
        decoded = decoded + mask.unsqueeze(-1) * d_e
return decoded
```

The vectorisation lives in the actual implementation — the sketch is
to fix what "sub_dictionary" means: **a deterministic slice of the
existing SAE decoder, gated by per-token routing**. No new
parameters are introduced. The forge is a pure projection of the
existing SAE into a routed form.

### Polygram-heuristic router (v1 MVP)

Wraps `ExpertDictionary.route(activations, top_k)` from polygram
0.9.0. Heuristic = "sum of feature activations per expert; pick
top-k by sum." Zero trainable parameters; deterministic per
input. The router lives inside `ForgedMoE` but delegates to the
polygram surface for the routing computation, so a polygram-side
change to the heuristic (e.g., coactivation-method block formation
when polygram lands that) flows through automatically.

### Out of scope, deliberately

- **Trained routers (linear, MLP).** v1 ships heuristic only. The
  training story (loss = reconstruction + sparsity + optional load
  balance) is queued as `add-moe-trained-router`.
- **`tiny_mlp` / `residual_block` expert types.** Both require
  distillation training (extract a small MLP from each expert's
  cluster). Queued as `add-moe-tiny-mlp-experts`.
- **Matryoshka-style nested experts.** Requires the trained-router
  capability first (since matryoshka relies on graded routing).
  Queued as `add-moe-matryoshka` after `add-moe-trained-router`.
- **`.steer(expert_ids, strength)`.** Steering surface for
  interpretability. Cheap to add but introduces a new API contract
  about *how* steering composes with the reconstruction; deferred
  to `add-moe-steering` so v1 doesn't lock in the wrong shape.
- **`.export_to_hf()` / `to_safetensors()` interop.** The MoE
  module saves via the existing `torch.save`/`safetensors` path;
  HF Hub interop (uploading a model card + tokeniser glue) is
  out of v1 scope and lives in `add-moe-hub-export`.
- **`.evaluate_downstream(task)` (folding pLDDT, GO probing).**
  Bio-specific evaluation. v1 ships standalone reconstruction-fidelity
  metrics; the bio evaluation surface lives downstream in bio-sae
  (per polygram's `experts.py` docstring: "Bio-specific scoring —
  downstream").
- **Per-residue routing for protein sequences.** Per-token routing
  on a `(B, T, d_model)` residual stream is already per-residue
  for protein language models — variable-length sequences work
  exactly as for variable-length text. No protein-specific code is
  needed.
- **Integration with `forward_mode="host_wrapped"` / `NativeModel`.**
  The MoE module is standalone in v1. Inserting it into a forged
  transformer as a residual-stream rerouting layer is a queued
  `add-moe-as-residual-stream-layer` follow-up.

## Falsifiable acceptance gate

A 2026-05-19 prototype run (`scripts/prototype_sae_moe_forge.py`,
results in this change's `smoke-results.md`) measured the bands
below on two fixtures: the GPT-2 layer-8 jbloom-sliced K=211 basis
(near-isotropic in decoder geometry — only 5,257 of 78,729 row
pairs above cos > 0.15) and a synthetic clusterable basis (4
deliberate cosine clusters of 32 features each). The bands below
reflect the post-prototype revisions; the original draft of this
proposal specified a single universal `5×` faithfulness bound,
which the prototype showed splits cleanly by basis quality.

**Mechanical bands** (apply universally, gating):

- **Band A — fidelity collapse.** For `k_experts=n_experts`,
  routed reconstruction MSE per coordinate vs flat-SAE
  reconstruction SHALL be ≤ 1e-5. The prototype measured
  ~10−12 across all fixtures (10^7× headroom).
- **Band B — sparsity gain.** For `k_experts=2, n_experts ∈
  {4..16}`, the counted-decoder-row-touch ratio SHALL be within
  the band `[2/n_experts - 0.05, 2/n_experts + 0.05]`. The
  prototype measured ratios 0.385 (E=5) and 0.243 (E=9) on the
  K=211 fixture, both inside band.
- **Band D — round-trip stability.** `ForgedMoEConfig.to_dict()
  → from_dict()` reconstructs an equal config; the reconstructed
  module produces byte-identical reconstruction on the same input.

**Faithfulness band** (split by basis cluster structure):

- **Band C-strict (clusterable basis).** When the basis has
  natural cosine cluster structure (operational definition:
  intra-cluster cosine median > 0.5 after polygram clustering at
  threshold=0.3), routed reconstruction MSE vs flat-SAE
  reconstruction SHALL be ≤ 0.5× flat-SAE-vs-host MSE on a
  256-token calibration batch. The prototype's synthetic fixture
  measured 0.12× (4× headroom under the bound).
- **Band C-advisory (any basis).** Routed-vs-flat MSE SHALL be
  reported alongside the run output but does NOT gate acceptance
  on near-isotropic bases. The prototype measured 4.58× on the
  K=211 fixture at E=9 — usable, with the user accepting some
  faithfulness loss for the 75% sparsity gain. A clearly-
  documented basis-quality property, not a forge bug.

The smoke-results.md document records the per-fixture numbers and
the cosine-pair survey that demonstrates the basis-quality
dependence. v1 production code SHALL emit a `coherence_diagnostic`
field (max + median intra-cluster cosine) on every forge_to_moe
result so users see this signal up front.

## Capabilities

### Added Capabilities

- `sae-moe-forge` (new) — defines `forge_to_moe`, the
  `ForgedMoE` module, the `sub_dictionary` expert type, the
  `polygram_heuristic` router, and the acceptance contract.

### Modified Capabilities

- `subspace-projector` — `FeatureBasis` gains a
  `polygram_checkpoint_path: str | None` field surfaced by
  `from_polygram_checkpoint`, so `forge_to_moe` can reload the
  full polygram `Dictionary` for clustering without an extra
  path-passing kwarg. Backward-compatible (`None` for bases
  constructed via `__init__` directly).

## Impact

- **New module**: `saeforge/moe.py` — `ForgedMoEConfig`,
  `ForgedMoE`, `forge_to_moe`. Pure torch + numpy; lazy-imports
  torch.
- **New module**: `saeforge/_moe/sub_dictionary.py` —
  `SubDictionaryExpertSet` (the v1 expert implementation).
- **New module**: `saeforge/_moe/routers.py` —
  `PolygramHeuristicRouter` (the v1 router implementation).
- **Modified**:
  - `saeforge/basis.py` — `FeatureBasis.from_polygram_checkpoint`
    records the loaded checkpoint path for downstream MoE forging.
  - `saeforge/__init__.py` — export `ForgedMoE`, `forge_to_moe`,
    `ForgedMoEConfig`.
- **No breaking changes**: existing forge pipelines are unchanged.
- **Dependencies**: requires `polygram>=0.9.0` (already pinned by
  `pyproject.toml`). No new external dependencies.

## Risks

- **The heuristic router may not be a good enough baseline.** The
  polygram-side route function uses summed feature activations —
  intuitive but not necessarily aligned with reconstruction MSE.
  The acceptance gate's 5× tolerance reflects this; if the
  prototype shows the heuristic is dramatically worse than flat,
  the proposal will revise the gate or move to `add-moe-trained-
  router` before v1 lands. This is the same "validate before
  shipping" pattern that revised `add-host-wrapped-forge-fallback`'s
  gate after the prototype.
- **Source-checkpoint reload at forge-to-moe time.** The flow
  `FeatureBasis → load polygram Dictionary → cluster_experts`
  requires the original polygram checkpoint to be on disk at forge
  time. The proposal adds `polygram_checkpoint_path` to
  `FeatureBasis` to capture this. Callers who constructed a
  `FeatureBasis` programmatically (without loading from disk) must
  supply an `ExpertDictionary` directly to `forge_to_moe` — error
  surfaced clearly when neither is available.
- **Encoder side is unchanged.** v1 keeps the SAE's encoder
  (`W_enc`, `b_enc`) as the gateway to expert routing. This means
  the encoder still does an N-feature linear map per token; only
  the decoder side gets the sparsity gain. For SAEs where the
  encoder is the bottleneck, this gain is partial. Documented;
  full encoder-side MoE lands in `add-moe-encoder-side` after the
  decoder-side path is validated.
- **No fine-tune in v1.** ForgedMoE is inference-only. The path
  to making the heuristic router trainable (or adding a learned
  router on top) lives in `add-moe-trained-router`; v1 forge raises
  if a caller passes training-related kwargs.
- **Per-family rollout.** The expert assignment depends on the
  basis's `W_dec` shape and the polygram `Dictionary` schema —
  both family-agnostic. So unlike `add-host-wrapped-forge-fallback`,
  v1 here works across all bundled families out of the box (GPT-2,
  Llama, Gemma-2, Qwen, Whisper-encoder). No per-family stubs.
