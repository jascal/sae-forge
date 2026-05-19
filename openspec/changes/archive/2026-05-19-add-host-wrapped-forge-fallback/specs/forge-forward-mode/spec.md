# forge-forward-mode Specification

## Purpose

The `forge-forward-mode` capability defines two implementations of
the forged transformer's forward pass and the dispatch rule that
selects between them at `NativeModel` construction time. The
implementations share an identical interpretability contract — the
residual stream is in basis coordinates at every transformer block
boundary — but differ in *where* the per-block nonlinearities run.

`native_in_basis` runs every operation (LayerNorm, attention, MLP)
in the basis-space residual stream, using parameters that are
algebraically projected from the host. It is the existing v0.5.1
forward path. It is mathematically faithful when the basis is high-
fidelity (`quality_tier ∈ {good, saturated}`).

`host_wrapped` runs every operation in host's `d_model` coordinates
using the host's exact, unprojected weights, with `decode → host_op
→ encode` wrapping each block. It is structurally faithful at every
basis quality tier; forge KL is monotone in basis rank by
construction. Compute equals host inference plus per-block
decode/encode matmuls.

## Requirements

### Requirement: `NativeModelConfig.forward_mode` field

`NativeModelConfig` SHALL expose a `forward_mode: str = "auto"`
field accepting one of three values:

- `"auto"` — dispatch by basis quality tier at
  `NativeModel.from_host` time. SHALL select `"native_in_basis"`
  when `basis.quality_tier ∈ {good, saturated}` and
  `"host_wrapped"` when `basis.quality_tier ∈ {undersized,
  degenerate}`.
- `"native_in_basis"` — force the existing forward implementation.
- `"host_wrapped"` — force the new forward implementation.

`__post_init__` SHALL raise `ValueError` naming the legal values
when `forward_mode` is not in this set.

`NativeModelConfig.to_dict` / `from_dict` SHALL round-trip the
field. Configs serialised before this change land lack the key;
`from_dict` SHALL default to `"auto"` in that case (matching v0.5.1
behaviour on good/saturated bases via auto dispatch).

#### Scenario: explicit forward modes pass through unchanged

- **WHEN** a user constructs `NativeModelConfig(forward_mode=
  "native_in_basis", …)` against any basis
- **THEN** `NativeModel.from_host` SHALL use the existing forward
  path
- **AND** `NativeModel.resolved_forward_mode` SHALL equal
  `"native_in_basis"`

#### Scenario: auto-dispatch on a good-tier basis

- **WHEN** `forward_mode="auto"` and `basis.quality_tier` is `good`
  (basis_rank ≥ `0.5 * d_model`)
- **THEN** the resolved mode SHALL be `"native_in_basis"`
- **AND** the forward path, parameter set, and output SHALL be
  byte-identical to v0.5.1 for the same inputs.

#### Scenario: auto-dispatch on an undersized basis

- **WHEN** `forward_mode="auto"` and `basis.quality_tier` is
  `undersized` (basis_rank < `0.5 * d_model`)
- **THEN** the resolved mode SHALL be `"host_wrapped"`
- **AND** `NativeModel.resolved_forward_mode` SHALL equal
  `"host_wrapped"`.

#### Scenario: invalid `forward_mode`

- **WHEN** `NativeModelConfig(forward_mode="other", …)` is
  constructed
- **THEN** `__post_init__` SHALL raise `ValueError` naming the legal
  values `{"auto", "native_in_basis", "host_wrapped"}`.

### Requirement: `resolve_forward_mode` helper

`saeforge.forward_mode.resolve_forward_mode(basis, requested) ->
Literal["native_in_basis", "host_wrapped"]` SHALL:

- Compute `basis_rank` from `basis.W_dec` (via
  `forge_quality.compute_basis_rank`).
- Classify `quality_tier` via
  `forge_quality.classify_quality(basis_rank, basis.d_model)`.
- When `requested == "auto"`: return `"native_in_basis"` for tier
  in `{good, saturated}`; return `"host_wrapped"` for tier in
  `{undersized, degenerate}`.
- When `requested` is one of `"native_in_basis"` /
  `"host_wrapped"`: return it unchanged.
- Log the resolution at level `INFO` once per call when source was
  `"auto"`, naming the resolved mode and the quality tier that
  drove it.

The function SHALL NOT load or run the host model — it operates on
the basis alone.

#### Scenario: helper is a pure function of basis + request

- **WHEN** `resolve_forward_mode(basis, "auto")` is called twice
  for the same basis
- **THEN** the two calls SHALL return the same value.

### Requirement: host-wrapped forward contract

A host-wrapped native module for a causal-LM family SHALL implement
the forward pass:

```
x_host = host_wte(input_ids) + host_wpe(pos)
z = x_host @ pinv * scale_boost
for block in host_transformer.h:
    x_host = z @ W_dec
    x_host = block(x_host)
    z = x_host @ pinv * scale_boost
x_host = z @ W_dec
x_host = host_ln_f(x_host)
return host_lm_head(x_host)
```

where:

- `host_wte`, `host_wpe`, `host_transformer.h`, `host_ln_f`, and
  `host_lm_head` are frozen references to the loaded host model's
  modules.
- `W_dec` and `pinv` are registered as buffers (not parameters).
- `scale_boost` is a python float, not a parameter.

The module SHALL register no trainable parameters in v1. Calling
`run_finetune` against a host-wrapped module SHALL raise
`RuntimeError` naming the queued `add-host-wrapped-finetune-recipe`
follow-up.

The module SHALL expose `.config: NativeModelConfig` with
`forward_mode="host_wrapped"`, `.resolved_forward_mode` ==
`"host_wrapped"`, and `.forward(input_ids)` matching
`NativeModel.forward`'s signature.

For non-LM families (Whisper encoder), the host-wrapped module
SHALL mirror the family's encoder/decoder structure: decode at every
block boundary, host-native block, re-encode; entry and exit are
the family's native conv stem / positional embedding / output head
respectively.

#### Scenario: host-wrapped forward agrees with native-in-basis on a good-tier basis

- **GIVEN** a basis with `n_features = d_model` and orthonormal
  `W_dec`
- **WHEN** both `native_in_basis` and `host_wrapped` modules are
  constructed from the same host and the same basis
- **AND** both forward a common `input_ids` tensor
- **THEN** the per-token KL between the two output logits SHALL be
  ≤ 0.1 nats.

#### Scenario: host-wrapped fine-tune raises

- **WHEN** `run_finetune(host_wrapped_native_model, …)` is invoked
- **THEN** the call SHALL raise `RuntimeError`
- **AND** the error message SHALL name `add-host-wrapped-finetune-
  recipe` as the queued follow-up.

### Requirement: amplification-removal acceptance gate

On the GPT-2 layer-8 jbloom-sliced reference sweep (K ∈ {25, 103,
163, 211}, HEA_Rung2 n_qubits=10, scale_boost=1.0):

The `host_wrapped` arm SHALL:

- Produce `faithfulness_kl` strictly ≤ the matched-K
  `native_in_basis` KL at every K in the sweep.
- Produce `faithfulness_kl[K=211]` < 25.0 nats (vs the documented
  `native_in_basis` 86.39 nats — a ≥ 60-nat reduction). Prototype
  result: 15.4 nats.
- Exhibit no rank-dependent amplification — no adjacent K-pair
  ΔKL exceeds 10 nats in the `host_wrapped` arm. (Native's
  documented trajectory has ΔKL = +22.7, −4.0, +59.1 across the
  three pairs; host-wrapped's prototype trajectory is +0.1, +2.8,
  +2.9.)

The acceptance gate intentionally does NOT require monotone
non-increasing KL across the smoke series. The four smoke bases are
non-nested — different K targets pick different decoder subsets —
and host-wrapped KL is bounded by each basis's individual residual-
stream approximation quality, which is not monotone in K for these
particular bases. Host-wrapped removes the *amplification*, not the
*basis-approximation error*; the latter is a property of the bases
the user supplies.

`scripts/diagnose_layer_amplification.py` and
`scripts/prototype_host_wrapped_forward.py` are the canonical
reproducers; their outputs are committed to
`reports/layer_amplification/` and referenced from this change's
`smoke-results.md`.

#### Scenario: amplification removed on smoke regime

- **WHEN** `sweep-pareto --forward-mode host_wrapped` runs the four
  K targets
- **THEN** for every K, `faithfulness_kl` ≤ the matched-K
  `native_in_basis` KL
- **AND** `faithfulness_kl[K=211]` < 25.0
- **AND** no adjacent K-pair ΔKL exceeds 10 nats.

### Requirement: family rollout

The bundled adapter for the GPT-2 family SHALL ship a working
`host_wrapped_module` implementation in v1.

All other bundled adapters (Llama, Gemma-2, Qwen2, Qwen3,
Qwen3-MoE, Whisper-encoder) SHALL ship a `host_wrapped_module` stub
that raises `NotImplementedError` whose message:

- Names the family.
- Notes that v1 ships GPT-2 only.
- Points at follow-up proposals `add-host-wrapped-<family>` for
  other rollouts.

The `auto` dispatch SHALL NOT silently fall back to
`native_in_basis` for unsupported families — the error SHALL
surface to the caller so the limitation is observable.

#### Scenario: Llama host on undersized basis raises clear error

- **WHEN** `NativeModel.from_host(llama_host, undersized_basis,
  forward_mode="auto")` is called
- **THEN** the call SHALL raise `NotImplementedError`
- **AND** the message SHALL name "llama" and point at the queued
  follow-up.
