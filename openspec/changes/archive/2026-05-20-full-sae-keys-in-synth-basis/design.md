## Context

`ForgePipeline._run_real_fsm` (forge.py:672) and `_run_synthetic_fsm` (forge.py:820) write a `synth_basis.safetensors` from the in-memory `FeatureBasis` before invoking the FSM orchestrator. The FSM's first action (`compress_with_polygram`) is gated on `ctx["validation_report_path"]`:

- **No report** (today's default path): compress action passthrough. Only `W_dec` is read downstream by the projection step. The synth-basis-with-only-`W_dec` works fine.
- **With a report** (the unblocked path this change targets): compress action calls polygram's `Compressor.apply()`, which loads the SAE via `_load_sae_checkpoint(path, ["W_enc", "b_enc", "b_dec", "W_dec"])`. Polygram raises on missing keys.

The bug surfaced live during the adaptive-regrow MBP smoke (2026-05-15) when a user wired `ForgePipeline(validation_report_path=...)` to exercise the warm-cycle controller path. The cold-start path (no report) was unaffected.

The fix has two halves:

1. **`FeatureBasis` retains** `W_enc` / `b_enc` / `b_dec` when loaded from a real checkpoint. Today the loader pulls only `W_dec` (which is all the projector needs); the other keys are discarded. They should be retained as optional state.
2. **`_write_basis_as_checkpoint` writes all four keys.** When the basis has them (loaded from a real SAE), use the real values. When the basis is synthetic-from-scratch (test fixture path), synthesise placeholders so the written file is structurally valid for polygram's loader.

## Goals / Non-Goals

**Goals:**
- `synth_basis.safetensors` is a valid SAE checkpoint per polygram's `_load_sae_checkpoint` standard key set (`W_enc`, `b_enc`, `b_dec`, `W_dec`).
- Existing flows (no `validation_report_path`) stay byte-identical.
- FSM-orchestrated compress works against the synth basis when a real validation report is supplied.
- `FeatureBasis`'s additional optional fields are introspectable for debugging (the user can check whether the basis came from a real SAE or was synthesised).

**Non-Goals:**
- Numerically-correct encoder weights when synthesising from scratch. Placeholders are structural.
- Tracking the source-checkpoint path on `FeatureBasis` for file-copy round-trip. Deferred unless the placeholder approach proves insufficient.
- Supporting non-standard SAE layouts (encoder-decoder hosts, multi-head outputs). Out of scope; document the standard residual-stream LM assumption.

## Decisions

### Decision 1 — Three optional fields on `FeatureBasis`, all default `None`

Add `W_enc: np.ndarray | None = None`, `b_enc: np.ndarray | None = None`, `b_dec: np.ndarray | None = None` to the `FeatureBasis` dataclass. Default `None` keeps the existing test fixture (`tiny_synthetic_basis`) constructing without modification.

Shape validation in `__post_init__` fires only when the field is non-None: `W_enc.shape == (d_model, n_features)`, `b_enc.shape == (n_features,)`, `b_dec.shape == (d_model,)`. Existing dtype-preservation logic for `W_dec` extends to the new fields when present.

**Alternative considered**: bundle the four arrays into a single `SAEState` dataclass and have `FeatureBasis` hold an optional `sae_state: SAEState | None`. Rejected — flatter is more discoverable; the four arrays are conceptually one SAE, but at the `FeatureBasis` boundary the consumer asks "do you have W_enc?" not "do you have a full SAE state?"

### Decision 2 — `from_polygram_checkpoint` populates the optional fields when the source has them

The loader already reads `W_dec` via `safetensors.numpy.load_file`. Extend it to also read `W_enc`, `b_enc`, `b_dec` when present and to populate the corresponding `FeatureBasis` fields. Missing keys → `None` (not an error — pre-compression intermediate checkpoints may legitimately have only `W_dec`).

**Alternative considered**: a separate `FeatureBasis.from_full_sae(path)` classmethod that requires all four keys and raises on missing. Rejected — splits the loader API for no callsite benefit. The optional-fields-on-the-existing-loader path covers both regimes with one entry point.

### Decision 3 — Placeholder synthesis when fields are `None`

When `_write_basis_as_checkpoint` writes a basis whose `W_enc` is `None`, synthesise it as `W_dec.T` (decoder transpose). Defaults: `b_enc = zeros(n_features)`, `b_dec = zeros(d_model)`.

The synthesised `W_enc` is **not a semantically-correct encoder** (no pseudoinverse, no orthogonalisation). It is structurally valid (right shape, right dtype, contiguous) so polygram's loader and `Compressor.apply()` succeed. The compressor's role is to **zero out the rows/cols of non-representative features** per the validation report's confirmed pairs — it doesn't care whether the input rows were "real" or placeholder; it just writes its output.

For the round-trip path where no compression runs (the existing byte-equivalence flow), the placeholders are written, read back, and never inspected by downstream actions (projection reads only `W_dec`). Output is byte-identical.

**Risk**: an analyst who introspects `synth_basis.safetensors` (e.g., via `saeforge inspect`) and sees a `W_enc` populated may assume it's the real encoder. **Mitigation**: `_write_basis_as_checkpoint` writes a metadata key (`__synthesised_keys__: ["W_enc", "b_enc", "b_dec"]`) listing which keys were placeholder-synthesised vs. real. `saeforge inspect` surfaces this field. Documented in the spec.

**Alternative considered**: synthesise `W_enc` via the basis's pseudoinverse cache (already computed for projection). Rejected — pseudo-inverse-as-encoder is a defensible choice but not faster than `W_dec.T` (which is already contiguous in memory) and not load-bearing for the compress action. We're optimising for "polygram loader succeeds," not "encoder is meaningful."

**Alternative considered**: refuse to write the synth basis when `validation_report_path` is set AND `FeatureBasis.W_enc is None`. Forces the user to supply a real SAE via the source path. Rejected — pushes complexity onto the caller; the placeholder approach is invisible when it doesn't matter and only fails interestingly (no surprises) when introspection reveals it via the `__synthesised_keys__` metadata.

### Decision 4 — Document the contract as `forge-fsm-orchestrator` capability

A new capability spec defines what `_run_real_fsm` and `_run_synthetic_fsm` guarantee about the on-disk state they hand off to the FSM. Today this is implicit; making it explicit lets future changes (e.g., the deferred source-checkpoint-path reuse) modify the contract intentionally.

The capability covers:
- The synth-basis write contract (full standard SAE keys present, real-or-placeholder).
- The `__synthesised_keys__` metadata key.
- The interaction with `compress_with_polygram` under both passthrough and real-report modes.

**Alternative considered**: extend `subspace-projector` capability (which covers `FeatureBasis` loading). Rejected — the synth-basis-write path is an FSM-orchestrator concern, not a projector concern. New capability is cleaner.

## Risks / Trade-offs

- **Placeholder `W_enc` semantics are non-obvious.** A user staring at `synth_basis.safetensors` won't know `W_enc` is `W_dec.T` unless they read the docs or the `__synthesised_keys__` metadata. The metadata key is the mitigation; `saeforge inspect` should surface it in its summary output.

- **Disk cost increases.** A 4-key safetensors is roughly 4× larger than the current 1-key file. For a 24576-feature SAE at fp32, that's ~75 MB → ~300 MB per FSM run. Negligible for one-off forge calls; potentially relevant for sweeps that materialise many bases. Documented as a known cost; future optimisation (only write `W_dec` when no real-compress path will run) is possible but invasive.

- **Compress-action's compressed output now zeros placeholder rows.** When the user wires a real validation report against a synthetic basis (test-fixture path), the placeholder `W_enc` rows get zeroed per the compression plan. The resulting compressed checkpoint has "compressed-placeholder" rows that aren't behaviourally meaningful. This is documented as a known limitation of mixing real-validation-reports with synthetic bases; in practice the validation report itself is produced from a real SAE, so the basis being synthetic is the unusual case.

- **`__synthesised_keys__` metadata key.** Safetensors supports a metadata dict alongside tensor data. Adding a new metadata key is forward-compatible — readers that don't know about the key ignore it. polygram's loader doesn't read metadata today, so adding this is a no-op for the compressor's input path.
