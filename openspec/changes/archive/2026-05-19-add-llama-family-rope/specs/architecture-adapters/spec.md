# architecture-adapters Specification — DELTA for add-llama-family-rope

This file describes the requirement deltas this change introduces on
the canonical `openspec/specs/architecture-adapters/spec.md`. The
deltas land in the canonical spec at archive time per the openspec
lifecycle (see PR #56 / `add-host-wrapped-forge-fallback` precedent).

## MODIFIED Requirement: ArchitectureAdapter contract

The `ArchitectureAdapter.build_native_config` method SHALL populate
the following fields on the returned `NativeModelConfig` for any
host whose family is in the Llama-family (`llama`, `gemma2`,
`qwen2`, `qwen3`, `qwen3_moe`):

- `rope_theta: float` — copied from `host.config.rope_theta`, or
  `10000.0` when absent.
- `rope_scaling: dict | None` — copied verbatim from
  `host.config.rope_scaling`, or `None` when absent.
- `partial_rotary_factor: float` — copied from
  `host.config.partial_rotary_factor` when the attribute exists
  (Qwen3 family); `1.0` otherwise.

These fields SHALL be ignored on `NativeModelConfig` instances for
non-Llama families (`gpt2`, `whisper_encoder`). The Llama-family
forward path reads them; the other families' forwards never
reference them.

## NEW Requirement: Llama-family attention applies RoPE

Every Llama-family forged module's attention block SHALL apply
rotary positional embedding to Q and K after the projection and
reshape, before the optional Q/K norm (Qwen3) and the scaled
dot-product. The rotation SHALL be parametrised by the host's
`rope_theta` and `partial_rotary_factor` per the
`NativeModelConfig` plumbing above.

When `cfg.rope_mode == "none"`, the rotation step SHALL be skipped
entirely; the attention forward path returns to the pre-fix
behaviour byte-identically. `cfg.rope_mode == "standard"` (the
default) applies rotation.

When `cfg.rope_scaling is not None and rope_scaling.get("type") not in
(None, "default")`, the forward SHALL raise `NotImplementedError`
naming `add-rope-scaling-types` as the queued follow-up that adds
support for `"linear"` / `"dynamic"` / `"yarn"` / `"longrope"`
types.

#### Scenario: Llama-family forge is position-sensitive at default

- **GIVEN** a tiny synthetic Llama host (2 layers, `hidden_size=64`,
  `n_heads=4`, `vocab=512`, `rope_theta=10000.0`) and a synthetic
  basis
- **WHEN** the forged module is built with default
  `rope_mode="standard"` and forwarded on token IDs `[1, 2, 3]` at
  positions `[0, 1, 2]`, then again on the reversed IDs
  `[3, 2, 1]` at the same positions
- **THEN** the last-token logits SHALL differ in L2 norm by at
  least `1e-3`. (Without positional info, attention is
  order-equivariant; this gate fails only when RoPE is absent.)

#### Scenario: Llama-family forge is position-invariant at rope_mode="none"

- **GIVEN** the same fixture as above
- **WHEN** the forged module is built with `rope_mode="none"`
  (the regression-diff arm) and the same two forward passes are
  run
- **THEN** the last-token logits SHALL differ in L2 norm by less
  than `1e-5`. (Pins the pre-fix behaviour; confirms the rotation
  step is the *only* source of position sensitivity.)

#### Scenario: rope_mode="none" emits a UserWarning on Llama-family configs

- **GIVEN** a `NativeModelConfig(family="llama", rope_mode="none", ...)`
- **WHEN** the config's `__post_init__` runs
- **THEN** a `UserWarning` SHALL be emitted naming
  `rope_mode="none"` as a regression-diff knob and pointing at
  the queued add-rope-scaling-types follow-up for users hitting a
  scaling-type they need

#### Scenario: invalid rope_mode

- **WHEN** `NativeModelConfig(rope_mode="garbage")` is constructed
- **THEN** `__post_init__` SHALL raise `ValueError` naming the
  legal values `{"standard", "none"}`.

#### Scenario: unsupported rope_scaling type raises from forward

- **GIVEN** a forged Llama-family module built with
  `cfg.rope_scaling = {"type": "linear", "factor": 2.0}` (or
  any non-default type)
- **WHEN** `forward(input_ids)` is called
- **THEN** the call SHALL raise `NotImplementedError`
- **AND** the message SHALL name `add-rope-scaling-types` as the
  follow-up that adds support for the requested type

## NEW Requirement: ForgeResult.positional_encoding diagnostic

`ForgeResult` SHALL gain a `positional_encoding: str | None = None`
field. The `ForgePipeline.run` implementation SHALL populate it
after model construction:

- `"absolute_projected"` for GPT-2-family forges (the `wpe`
  positional embedding projected through `pinv`).
- `"rotary"` for Llama-family forges with `rope_mode="standard"`
  (the default).
- `"none_skipped"` for Llama-family forges with `rope_mode="none"`
  (the regression-diff arm).
- `"sinusoidal"` for Whisper-encoder forges (the conv-stem
  positional embedding already wired by
  `forge-whisper-encoder`).

The field SHALL be present in `forge_result.json` and SHOULD be
surfaced in any consumer's run summary (e.g.
`examples/forge_gemma2_2b.py`'s `run_summary.json`).

The field's *purpose* is to surface silent skips. A Llama-family
forge reporting `"none_skipped"` in a production run summary is a
load-bearing signal that the user is in a known-buggy regime.

#### Scenario: GPT-2 forge reports absolute_projected

- **WHEN** a GPT-2 forge completes via `ForgePipeline.run`
- **THEN** the returned `ForgeResult.positional_encoding` SHALL
  equal `"absolute_projected"`.

#### Scenario: Llama-family forge at default reports rotary

- **WHEN** a Llama-family forge completes via `ForgePipeline.run`
  with default `forward_mode`/`rope_mode`
- **THEN** the returned `ForgeResult.positional_encoding` SHALL
  equal `"rotary"`.

#### Scenario: Llama-family forge with rope_mode="none" reports none_skipped

- **WHEN** a Llama-family forge completes via `ForgePipeline.run`
  with `rope_mode="none"` on the underlying `NativeModelConfig`
- **THEN** the returned `ForgeResult.positional_encoding` SHALL
  equal `"none_skipped"`.

## MODIFIED Requirement: Llama-3 adapter handles GQA and SwiGLU (extended)

The existing requirement for `LlamaAdapter` gains a positional-
encoding bullet:

- The adapter's `build_native_config` SHALL populate
  `rope_theta`, `rope_scaling`, and `partial_rotary_factor` as
  documented under "MODIFIED Requirement: ArchitectureAdapter
  contract" above.
- The forged module's attention SHALL apply RoPE per the "NEW
  Requirement: Llama-family attention applies RoPE" above.

## MODIFIED Requirement: Gemma-2 adapter shares Llama-family layout (extended)

Same positional-encoding bullet as Llama-3, inherited via
`Gemma2Adapter(LlamaAdapter)`. Gemma-2 does not introduce its
own positional-encoding deviation in v1.
