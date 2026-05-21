## Why

`ForgePipeline._run_real_fsm` and `_run_synthetic_fsm` both call `_write_basis_as_checkpoint` to persist the in-memory `FeatureBasis` to disk before invoking the FSM orchestrator. That synth-basis file is then handed to `compress_with_polygram` via `ctx["current_sae_path"]`.

Today `_write_basis_as_checkpoint` writes **only the `W_dec` key**. That works fine for two existing call paths:

- Forging a **pre-compressed** SAE (no validation report supplied → `compress_with_polygram` short-circuits to passthrough → only reads `W_dec` for projection).
- Round-tripping byte-identity tests where `FeatureBasis.from_polygram_checkpoint` reads back the same `W_dec` it wrote out.

It **breaks** the moment a user wires `ForgePipeline(validation_report_path=...)` and asks the FSM to run a real polygram compression in-band: `compress_with_polygram` calls polygram's `_load_sae_checkpoint(path, ["W_enc", "b_enc", "b_dec", "W_dec"])`, which fails because the synth-basis only has `W_dec`:

```
ForgeFailed: _load_sae_checkpoint: no key aliasing to 'W_enc' found in synth_basis.safetensors.
Tried aliases ['W_enc', 'encoder.weight']; file contains: ['W_dec']
```

Live evidence from the adaptive-regrow MBP smoke (2026-05-15): the cold-start path works fine, but composing **FSM orchestration + real polygram compression + adaptive-regrow** in one invocation fails at the compress step because the synth-basis is incomplete. This blocks the use case adaptive-regrow exists for — multi-shard continual-learning with per-shard re-validation.

This change makes the synth-basis a **complete SAE checkpoint** so any FSM action that loads it gets the keys it expects.

## What Changes

### `FeatureBasis` retains the full SAE state when loaded from a real checkpoint

`FeatureBasis.from_polygram_checkpoint` SHALL optionally retain `W_enc`, `b_enc`, and `b_dec` on the instance when the source checkpoint provides them. Three new optional fields on `FeatureBasis` (all default `None`):

- `W_enc: np.ndarray | None`
- `b_enc: np.ndarray | None`
- `b_dec: np.ndarray | None`

When `from_polygram_checkpoint` reads a file that contains those keys, the instance gets them populated. When the user constructs a `FeatureBasis` programmatically (the existing `tiny_synthetic_basis` test fixture pattern) or loads a SAE that only has `W_dec` (pre-compression intermediate state), the three fields stay `None`.

### `_write_basis_as_checkpoint` writes all available keys

The helper SHALL write whichever of `W_enc`, `b_enc`, `b_dec`, `W_dec` are non-`None` on the basis, in addition to the existing `W_dec` write. When a key is `None`, the helper SHALL synthesise a sensible placeholder so the written file is a valid SAE checkpoint:

- `W_enc` placeholder: `W_dec.T` (decoder transpose — a numerically-defensible default for "encode = decoder-pseudoinverse" style SAEs; not load-bearing for compression, which zeros rows of `W_enc` per the compression plan).
- `b_enc` placeholder: zeros of length `n_features`.
- `b_dec` placeholder: zeros of length `d_model`.

The placeholders are **not semantically correct encoder weights** — they are structural placeholders so polygram's loader doesn't fail on missing keys. The compress action then zeros the relevant rows / columns per its plan, and the resulting compressed checkpoint is what downstream actions consume. For forgings that round-trip the synth basis without compressing (the existing pre-compressed flow), the placeholders are written, read back, and never inspected — the byte-equivalence of the **projection step**'s output is preserved because projection only reads `W_dec`.

### Byte-identity guarantee for existing flows

When the user does NOT supply a `validation_report_path` (default), `compress_with_polygram` runs in passthrough mode and never reads the new keys. The projection step reads `W_dec` only. End-to-end output is byte-identical to today for every existing test.

When the user supplies a real `validation_report_path`, the FSM's compress action now works against the synth basis: it reads `W_enc` / `b_enc` / `b_dec` (placeholders or real), zeros the rows per the validation report's confirmed pairs, writes the compressed checkpoint. The basis-loop then continues normally.

### Out of scope, deliberately

- **`b_enc` / `b_dec` shapes for non-residual-stream hosts** — encoder-decoder architectures or multi-token-prediction heads have different bias shapes. The placeholder synthesis assumes the standard residual-stream LM shape (`b_enc: n_features`, `b_dec: d_model`). Hosts that deviate are out of scope; callers with non-standard biases pre-compress the SAE via the polygram CLI and feed the result to sae-forge.
- **Synthesising `W_enc` from scratch when `W_dec` is itself synthetic** — when both the basis AND a real validation report are wired in but the basis came from `tiny_synthetic_basis` (no source checkpoint), the placeholder synthesis still runs but the resulting compressed output has no behavioural meaning. Documented as a known limitation; the test fixture path is rarely combined with a real validation report.
- **Reusing the original source-checkpoint path** instead of re-writing — would require `FeatureBasis` to track its provenance path and `_write_basis_as_checkpoint` to copy that file into `output_dir`. Cleaner in principle but invasive (`FeatureBasis` is loaded from various sources today). Deferred to a follow-up if the placeholder-synthesis approach proves insufficient.
- **A test fixture that combines validation_report_path + adaptive_regrow** — the live MBP smoke proves the impl works; bundling that as a `tests/test_forge_pipeline_fsm_compose.py` integration test is a separate concern (real polygram + real forge + several minutes of CPU). Documented as follow-up work.

## Capabilities

### New Capabilities

- `forge-fsm-orchestrator`: documents the contract `_run_real_fsm` and `_run_synthetic_fsm` honour when writing the synth-basis checkpoint, and the requirement that the file be loadable by polygram's `_load_sae_checkpoint` with the full standard SAE key set.

### Modified Capabilities

- None. The existing capabilities (`pareto-sweep`, `polygram-tuning-passthrough`, etc.) don't touch the synth-basis write path.

## Impact

- **Modified**:
  - `saeforge/basis.py` — `FeatureBasis` gains 3 optional fields; `from_polygram_checkpoint` reads them when the source file has them; field validation extends to shape-check non-None values against `n_features` / `d_model`.
  - `saeforge/forge.py` — `_write_basis_as_checkpoint` writes all 4 keys (real or placeholder).
- **New**: a small integration-style test in `tests/test_forge_pipeline_fsm_compose.py` that constructs a `ForgePipeline(validation_report_path=...)`, runs `_run_real_fsm`, and asserts the FSM completes without the `_load_sae_checkpoint: no key aliasing to 'W_enc'` failure. Gated on `[torch]` + `[polygram]` extras.
- **No breaking changes**: existing call paths byte-identical. The synthetic test fixture (`tiny_synthetic_basis`) constructs `FeatureBasis` without the optional keys, and the placeholder-synthesis path produces a complete safetensors that polygram's loader accepts.
- **No new dependencies**.
