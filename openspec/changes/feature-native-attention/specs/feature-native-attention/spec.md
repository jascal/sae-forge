# feature-native-attention Specification

## Purpose

Defines the v0.2 opt-in mode that converges the forged model's
attention internal width onto `n_features` per
[`docs/algorithm.md`](../../../../docs/algorithm.md) §4. In `host`
mode (the v0.2 default), v0 behaviour is preserved exactly. In
`feature_native` mode, c_attn and c_proj are projected on both sides
so attention scores are computed in feature space.

## Requirements

### Requirement: NativeModelConfig.attention_width controls the internal width

`NativeModelConfig` SHALL expose an `attention_width: Literal["host",
"feature_native"]` field defaulting to `"host"`. When `"host"`,
`qkv_inner_size` is independent of `hidden_size` (set by the caller,
typically to the host's `n_embd`). When `"feature_native"`,
`qkv_inner_size` SHALL equal `hidden_size`.

#### Scenario: host mode preserves the v0 contract

- **GIVEN** `NativeModelConfig(attention_width="host", hidden_size=8,
  qkv_inner_size=16, num_heads=4, head_dim=4, ...)`
- **WHEN** the config is constructed
- **THEN** construction succeeds and `qkv_inner_size == 16`

#### Scenario: feature_native mode requires k-divisibility

- **GIVEN** `NativeModelConfig(attention_width="feature_native",
  hidden_size=8, num_heads=3, ...)` (8 % 3 != 0)
- **WHEN** the config is constructed
- **THEN** `ValueError` is raised whose message contains both
  `"hidden_size"` and `"num_heads"`

### Requirement: SubspaceProjector exposes the both-sides projection helpers

`SubspaceProjector` SHALL expose two new helpers:

- `project_residual_full(W: (d, d)) -> (k, k)` computing `D @ W @ E`.
- `project_qkv_full(W: (d, 3d)) -> (k, 3k)` splitting the input on
  the second axis into three `(d, d)` blocks, applying
  `project_residual_full` to each, and concatenating the results
  back along the second axis.

#### Scenario: project_residual_full on identity basis is the identity

- **GIVEN** a `FeatureBasis` with `W_dec = np.eye(d)` (so `k = d`)
- **WHEN** `project_residual_full(W)` is called for any `(d, d)` `W`
- **THEN** the result equals `W` within `1e-9` absolute tolerance

#### Scenario: project_qkv_full preserves block structure

- **GIVEN** a basis with `n_features=8, d_model=16` and a random `W`
  of shape `(16, 48)`
- **WHEN** `project_qkv_full(W)` is computed
- **THEN** the result has shape `(8, 24)`
- **AND** each of the three `(8, 8)` blocks (axis-1 splits) equals
  `project_residual_full` of the corresponding `(16, 16)` block of
  the input

### Requirement: project_module honours the attention_width kwarg

`SubspaceProjector.project_module(host_model, *, attention_width="host")`
SHALL accept an `attention_width` kwarg. When `"feature_native"`:

- `transformer.h.{i}.attn.c_attn.weight` SHALL have shape `(k, 3k)`
  via `project_qkv_full`.
- `transformer.h.{i}.attn.c_attn.bias` SHALL have shape `(3k,)`,
  computed by splitting the host bias into three `(d,)` blocks,
  projecting each via `project_residual_bias`, and concatenating.
- `transformer.h.{i}.attn.c_proj.weight` SHALL have shape `(k, k)`
  via `project_residual_full`.
- `transformer.h.{i}.attn.c_proj.bias` SHALL have shape `(k,)`
  (unchanged from v0).
- All other keys (embeddings, MLP, layer norms, lm_head) SHALL match
  v0's `host` mode exactly.

#### Scenario: feature_native shape contract

- **GIVEN** an HF GPT-2 host with `n_embd=16, n_layer=2, n_head=4` and
  a basis with `n_features=8`
- **WHEN** `project_module(host, attention_width="feature_native")`
  is called
- **THEN** every `transformer.h.{i}.attn.c_attn.weight` has shape
  `(8, 24)`
- **AND** every `transformer.h.{i}.attn.c_attn.bias` has shape `(24,)`
- **AND** every `transformer.h.{i}.attn.c_proj.weight` has shape
  `(8, 8)`
- **AND** all other keys match the v0 `host`-mode shapes

### Requirement: Identity basis preserves the host exactly under feature_native

When `W_dec = np.eye(d)` (so `k = d`, `D = I`, `E = I`),
`feature_native` mode SHALL produce a forged model whose forward pass
matches the host's forward pass exactly up to floating-point
arithmetic. The faithfulness KL SHALL be `< 1e-3` — the same
correctness signal v0's `host` mode satisfies.

This is the strongest spec-level test that the both-sides projection
algebra is right.

#### Scenario: identity-basis KL is sub-millibit under feature_native

- **GIVEN** a `FeatureBasis` with `W_dec = np.eye(d_model)` and a
  random-init tiny GPT-2 host
- **WHEN** `ForgePipeline(attention_width="feature_native",
  ...).run_synthetic(host, ..., eval_input_ids=...)` is called
- **THEN** the resulting `ForgeResult.faithfulness_kl` is below `1e-3`

### Requirement: feature_native and host produce different forged weights on non-trivial bases

When the basis is not the identity, `feature_native` and `host` modes
SHALL produce different forged-weight tensors (regression check that
the new code path actually runs). This SHALL be verified by
SHA-256-comparing the two `forged/model.safetensors` files.

#### Scenario: regression check on a random basis

- **GIVEN** a random non-identity `FeatureBasis` with `n_features=8,
  d_model=16` and a tiny GPT-2 host
- **WHEN** the same pipeline runs once with `attention_width="host"`
  and once with `attention_width="feature_native"`, both with the
  same RNG seed
- **THEN** `sha256(host_forged.safetensors) != sha256(fn_forged.safetensors)`

### Requirement: Default attention_width remains "host" in v0.2

`ForgePipeline.attention_width` SHALL default to `"host"`. This
preserves the v0.1 imperative/FSM byte-equivalence safety net and
gives users a deliberate opt-in for the new mode.

The v1.0 default flip is tracked by a separate OpenSpec change
(`feature-native-attention-default`) and SHALL NOT be part of v0.2's
acceptance criteria.

#### Scenario: missing kwarg gives v0 behaviour

- **GIVEN** `ForgePipeline(basis=..., projector=...)` with no
  `attention_width` argument
- **WHEN** `run_synthetic` is called
- **THEN** the resulting forged model's `c_attn.weight` shape matches
  v0's `(k, 3 * host n_embd)` exactly
- **AND** the SHA-256 of `forged/model.safetensors` matches a v0.1
  baseline run on the same inputs

### Requirement: CLI exposes the opt-in via --feature-native-attention

The `sae-forge forge` console command SHALL accept a
`--feature-native-attention` flag that sets `attention_width =
"feature_native"` on the constructed `ForgePipeline`. The default
(no flag) SHALL be `"host"`.

#### Scenario: flag presence

- **WHEN** `sae-forge forge --help` is invoked
- **THEN** the help text lists `--feature-native-attention` with a
  description naming the basis-width attention mode

### Requirement: docs/algorithm.md §10.2 is updated

`docs/algorithm.md` §10.2 SHALL be rewritten to describe v0.2's
opt-in behaviour (both modes ship; `host` is the default; v1.0 flips
the default; v1.1 removes `host`). The cross-reference to the
`subspace-projector` capability spec SHALL be preserved.
