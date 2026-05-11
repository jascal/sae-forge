# hybrid-bridge-forge Specification

## Purpose

The `hybrid-bridge-forge` capability defines an opt-in forge path
that uses three `FeatureBasis` instances anchored at the host
model's structural boundaries (embed / mid / lm-head) plus two
learnable bridge matrices on the forged module's forward pass.
The capability pins the routing contract (which host weight goes
through which basis), the bridge module's shape and forward
contract, the cross-basis shape-compatibility requirement, the
tied-embedding refusal, and the byte-equivalence-when-disabled
scenario.

This capability is opt-in (default `hybrid_bridge=False`); the v0
single-basis path remains the default and is preserved
byte-identically.

## ADDED Requirements

### Requirement: HybridBasisBundle requires shape-compatible bases

`saeforge.hybrid_basis.HybridBasisBundle.__post_init__` SHALL validate
that the three bases (`basis_embed`, `basis_mid`, `basis_lm_head`)
share identical `d_model` AND identical `n_features`. A mismatch in
either dimension SHALL raise `ValueError` whose message names both
the field that mismatched and the two conflicting values.

The bundle SHALL additionally validate `n_layer >= 3` so the
three-region split (`embed_layer=0`, `mid_layer ∈ [1, n_layer-2]`,
`lm_head_layer=n_layer-1`) is non-degenerate.

#### Scenario: d_model mismatch raises

- **GIVEN** `basis_embed.d_model=768`, `basis_mid.d_model=768`,
  `basis_lm_head.d_model=1024`
- **WHEN** `HybridBasisBundle(...)` is constructed
- **THEN** `ValueError` is raised
- **AND** the message contains `"d_model"` and both `768` and `1024`

#### Scenario: n_features mismatch raises

- **GIVEN** three bases with matching `d_model` but
  `basis_embed.n_features=128`, `basis_mid.n_features=256`
- **WHEN** `HybridBasisBundle(...)` is constructed
- **THEN** `ValueError` is raised
- **AND** the message contains `"n_features"`

#### Scenario: too-few-layers host rejected

- **GIVEN** three shape-compatible bases and `n_layer=2`
- **WHEN** `HybridBasisBundle(..., n_layer=2)` is constructed
- **THEN** `ValueError` is raised naming `"n_layer"` and the minimum `3`

### Requirement: basis_for_layer routes weights deterministically

`HybridBasisBundle.basis_for_layer(idx)` SHALL return `basis_embed`
when `idx == 0`, `basis_lm_head` when `idx == n_layer - 1`, and
`basis_mid` for every intermediate `idx ∈ [1, n_layer-2]`. Calling
with `idx < 0` or `idx >= n_layer` SHALL raise `IndexError`.

This routing SHALL be the sole authority on which basis projects
which weight. Adapter helpers, save-time fold logic, and any
follow-up `multi-anchor-forge` work SHALL consult
`basis_for_layer` rather than re-deriving the rule.

#### Scenario: GPT-2 routing matrix

- **GIVEN** `HybridBasisBundle(n_layer=12, ...)`
- **WHEN** `basis_for_layer(idx)` is called for `idx ∈ {0..11}`
- **THEN** `idx=0` returns `basis_embed`
- **AND** `idx ∈ {1..10}` returns `basis_mid`
- **AND** `idx=11` returns `basis_lm_head`

#### Scenario: out-of-range raises

- **GIVEN** `HybridBasisBundle(n_layer=12, ...)`
- **WHEN** `basis_for_layer(12)` or `basis_for_layer(-1)` is called
- **THEN** `IndexError` is raised

### Requirement: BridgeModule is a square parameter block

`saeforge.bridges.BridgeModule` SHALL be an `nn.Module` whose single
trainable parameter is a square `n_features × n_features` `Linear`
(no bias by default). The module SHALL apply, in order:

1. Optional `nn.LayerNorm(n_features)` when `pre_layernorm=True`.
2. The linear map.
3. Optional non-linearity (`nn.ReLU` or `nn.GELU`) when
   `nonlin != "none"`.

`BridgeModule.forward(x)` SHALL preserve `x.shape` and operate over
the last dimension (`n_features`). The module SHALL honor
`config.train` by setting `requires_grad=False` on its parameters
when `train=False`.

#### Scenario: shape preservation

- **GIVEN** `BridgeModule(n_features=128, BridgeConfig(...))`
- **WHEN** invoked on `x` of shape `(2, 7, 128)`
- **THEN** output shape is `(2, 7, 128)`

#### Scenario: identity init reproduces input

- **GIVEN** `BridgeModule(n_features=128, BridgeConfig(init="identity", nonlin="none", pre_layernorm=False))`
- **AND** input `x` of shape `(1, 1, 128)` with arbitrary values
- **WHEN** `bridge(x)` is called
- **THEN** the output equals `x` (within float-precision tolerance)

#### Scenario: train=False freezes parameters

- **GIVEN** `BridgeModule(n_features=128, BridgeConfig(..., train=False))`
- **WHEN** `bridge.parameters()` is iterated
- **THEN** every parameter has `requires_grad is False`

### Requirement: tied-embedding hosts are refused under hybrid_bridge

`ForgePipeline.__post_init__` SHALL raise `ValueError` when `hybrid_bridge=True` is combined with a host model whose `config.tie_word_embeddings is True`. The raised message SHALL:

1. States that the configuration is unsupported.
2. Names the host config flag (`tie_word_embeddings`).
3. References the workaround: either disable `hybrid_bridge` or
   re-load the host with `tie_word_embeddings=False`.
4. Points at the follow-up capability
   `hybrid-bridge-tied-embeddings`.

This is the v1 behavior; the follow-up will replace the refusal
with either basis-sharing or a constrained-bridge approach.

#### Scenario: GPT-2 with default tied embeddings refused

- **GIVEN** a default `gpt2` host (`tie_word_embeddings=True`)
- **WHEN** `ForgePipeline(hybrid_bridge=True, basis=..., basis_embed=..., basis_lm_head=..., host_model_id="gpt2")` is constructed
- **THEN** `ValueError` is raised
- **AND** the message contains `"tie_word_embeddings"` and the
  string `"hybrid-bridge-tied-embeddings"`

#### Scenario: untied GPT-2 accepted

- **GIVEN** a GPT-2 host loaded with `tie_word_embeddings=False`
- **WHEN** `ForgePipeline(hybrid_bridge=True, ...)` is constructed
  with shape-compatible bases
- **THEN** construction succeeds

### Requirement: hybrid_bridge=False is byte-identical to single-basis v0

`ForgePipeline.run()` SHALL leave every hybrid code path unreached when `hybrid_bridge=False` (the default). Specifically:

- `HybridBasisBundle` is not constructed.
- `BridgeModule` instances are not constructed.
- `ctx["hybrid_basis_bundle"]` is absent.
- `ctx["bridges"]` is absent.
- `SubspaceProjector.project_module` dispatches via its existing
  single-basis path.
- `NativeModel.forward` runs without bridge insertions.

The existing `test_imperative_and_fsm_byte_equivalent` test SHALL
pass without modification under this change.

#### Scenario: disabled toggle leaves forged weights byte-identical

- **GIVEN** two `ForgePipeline` instances with identical inputs:
  one with `hybrid_bridge=False` and the other with the same minimal
  config from before this change
- **WHEN** both are run end-to-end
- **THEN** the two `forged/model.safetensors` artifacts are byte-identical

#### Scenario: byte-equivalence gate continues to pass

- **GIVEN** the existing `test_imperative_and_fsm_byte_equivalent`
  test setup (no `hybrid_bridge` knobs set)
- **WHEN** the test runs against the post-this-change tree
- **THEN** the imperative-path and FSM-path forged weights are byte-identical
- **AND** the action sequence in `transitions_log` is unchanged from v0

### Requirement: hybrid_bridge=True wires bridges into the forged module

When `ForgePipeline.hybrid_bridge=True`, `ForgePipeline.run()` SHALL:

1. Construct a `HybridBasisBundle` from `basis_embed`, `basis`, and
   `basis_lm_head`.
2. Construct two `BridgeModule` instances per `bridge_config`:
   `bridge_emb_mid` and `bridge_mid_lm`.
3. Pass the bundle to `SubspaceProjector.project_module(host, hybrid=bundle)`.
4. Pass the bridges to `NativeModel.from_projected_weights(..., bridges={...})`.
5. Include the bridge parameters in the fine-tune stage's optimizer
   param group (when `bridge_config.train=True`).
6. Save the bridge state in the forged `safetensors` artifact under
   keys `bridges.emb_mid.*` and `bridges.mid_lm.*`.

The resulting `forged/model.safetensors` SHALL be loadable via
`NativeModel.from_safetensors` and reproduce the saved bridge
parameters byte-identically.

#### Scenario: bridges appear in forged state_dict

- **GIVEN** a successful hybrid forge run
- **WHEN** the forged `state_dict()` is inspected
- **THEN** it contains keys starting with `bridges.emb_mid.` and
  `bridges.mid_lm.`
- **AND** their tensors have shape `(n_features, n_features)`

#### Scenario: deterministic forge under hybrid_bridge

- **GIVEN** two hybrid `ForgePipeline` instances with identical
  seeds, configs, and inputs
- **WHEN** both are run end-to-end
- **THEN** the two `forged/model.safetensors` artifacts are
  byte-identical
- **AND** their per-step KL trajectories match

### Requirement: every projected host weight is attributed to exactly one basis

`SubspaceProjector.project_module(host, hybrid=bundle)` SHALL project every emitted weight through exactly one of the three bases in `bundle`. For block-indexed keys (containing `.h.<idx>.`), the basis is determined by `bundle.basis_for_layer(idx)`. For non-block keys, routing is fixed by the table: `wte.weight` and `wpe.weight` go through `basis_embed`; `ln_f.*` and `lm_head.weight` go through `basis_lm_head`.

No key SHALL be projected through more than one basis. No key SHALL
be missing from the output that the single-basis path would have
emitted for the same host.

#### Scenario: GPT-2 (untied) routes match the region table

- **GIVEN** a `gpt2` host with `tie_word_embeddings=False` and a
  3-basis bundle
- **WHEN** `project_module(host, hybrid=bundle)` is called
- **THEN** every `transformer.h.0.*` key was projected by `basis_embed`
- **AND** every `transformer.h.{1..10}.*` key was projected by `basis_mid`
- **AND** every `transformer.h.11.*` key, every `transformer.ln_f.*`
  key, and `lm_head.weight` were projected by `basis_lm_head`
- **AND** the set of returned keys equals the set returned by the
  single-basis path on the same host

### Requirement: validation tiering documents where defaults are decided

`docs/forge_layer_choice.md` SHALL document the capability's defaults (`bridge_init`, `bridge_nonlin`, `pre_layernorm`, `train`) as **provisional**. The doc SHALL include:

1. The shipping tier matrix (T0 `tiny_gpt2` CPU, T1 `gpt2` untied
   on Intel Mac, T2 `gpt2-medium/large`, T3 Gemma-2-2B on M4,
   T4 external NVIDIA/CUDA).
2. The current state of each tier (passed / pending / requested).
3. A direct link to the comparison harness output
   (`docs/hybrid_bridge_intel_gpt2.md`).
4. Instructions for external contributors to validate on T4
   hardware (`docs/hybrid_bridge_cuda_validation_request.md`).

A defaults change resulting from a tier passing or failing SHALL
be a follow-up patch with a CHANGELOG entry, not a re-spec of the
capability.

#### Scenario: tier matrix appears in documentation

- **GIVEN** the post-this-change repo state
- **WHEN** `docs/forge_layer_choice.md` is read
- **THEN** the "Hybrid / multi-basis forging" subsection contains a
  tier table with rows for T0, T1, T2, T3, T4
- **AND** each row identifies the host, hardware, owner, and status
- **AND** the subsection links to `docs/hybrid_bridge_intel_gpt2.md`
  and `docs/hybrid_bridge_cuda_validation_request.md`
