## 1. `FeatureBasis` optional fields

- [ ] 1.1 Add three new optional fields to `FeatureBasis` (`saeforge/basis.py`): `W_enc: np.ndarray | None = None`, `b_enc: np.ndarray | None = None`, `b_dec: np.ndarray | None = None`.
- [ ] 1.2 Extend `__post_init__` shape validation to fire only on non-None fields: `W_enc.shape == (d_model, n_features)`, `b_enc.shape == (n_features,)`, `b_dec.shape == (d_model,)`. Mismatch raises `ValueError` naming the field + the expected shape.
- [ ] 1.3 Existing dtype-preservation logic for `W_dec` extends to the new fields when present (cast to the same dtype as `W_dec` if they differ).

## 2. `from_polygram_checkpoint` reads the optional fields

- [ ] 2.1 `FeatureBasis.from_polygram_checkpoint` SHALL read `W_enc`, `b_enc`, `b_dec` from the source safetensors when present and pass them into the dataclass constructor. Missing keys → `None` (not an error).
- [ ] 2.2 Round-trip preservation: `FeatureBasis.from_polygram_checkpoint(path)` for a path written by an unmodified polygram `compress --pareto-materialize` produces a basis whose three optional fields are populated and shape-correct.
- [ ] 2.3 Backwards compat: existing checkpoints with only `W_dec` (the synth-basis written by pre-change `_write_basis_as_checkpoint`) still load successfully — the three new fields stay `None`.

## 3. `_write_basis_as_checkpoint` writes full SAE keys

- [ ] 3.1 In `saeforge/forge.py::_write_basis_as_checkpoint`, write all four keys: `W_dec` (existing), `W_enc`, `b_enc`, `b_dec`. Use the real values from the basis when non-None; otherwise synthesise placeholders.
- [ ] 3.2 Placeholder synthesis: `W_enc = basis.W_dec.T`, `b_enc = np.zeros(n_features)`, `b_dec = np.zeros(d_model)`. All cast to `basis.W_dec.dtype`.
- [ ] 3.3 Write a `__synthesised_keys__` metadata entry (a comma-separated list of synthesised key names; empty string when all four are real) so introspection tools can distinguish real vs. placeholder. Use `safetensors.numpy.save_file`'s `metadata=` parameter.
- [ ] 3.4 The function signature is unchanged (still takes `basis` and `path`); no caller updates needed.

## 4. `saeforge inspect` surfaces the metadata

- [ ] 4.1 In `saeforge/cli.py::_cmd_inspect`, read the `__synthesised_keys__` metadata after loading the checkpoint and include it in the JSON summary as `synthesised_keys: list[str]`. Empty list when no placeholders.
- [ ] 4.2 The markdown report (`_render_inspect_markdown`) adds a "Synthesised keys" line when the list is non-empty, listing the placeholder keys with a one-line explanation.

## 5. Tests

### 5.1 `FeatureBasis` schema + loader

- [ ] 5.1.1 `tests/test_feature_basis.py::test_optional_keys_default_none` — `FeatureBasis(kept_ids=..., W_dec=...)` constructed without the optional fields has all three as `None`.
- [ ] 5.1.2 `test_optional_keys_shape_validation` — passing a wrong-shape `W_enc` raises `ValueError` naming the field.
- [ ] 5.1.3 `test_from_polygram_checkpoint_populates_optional_keys` — load a real polygram-output safetensors (the existing test fixture); assert all three optional fields are populated and shape-correct.
- [ ] 5.1.4 `test_from_polygram_checkpoint_handles_legacy_synth_basis` — load a `W_dec`-only safetensors; assert the three optional fields are `None`.

### 5.2 `_write_basis_as_checkpoint`

- [ ] 5.2.1 `tests/test_forge_pipeline.py::test_write_basis_writes_all_four_keys_from_real_basis` — basis with all four populated → on-disk safetensors has all four keys, no metadata `__synthesised_keys__` entry (or empty string).
- [ ] 5.2.2 `test_write_basis_synthesises_placeholders_for_synthetic_basis` — synthetic basis (only `W_dec`) → on-disk safetensors has all four keys, `__synthesised_keys__` metadata lists `W_enc,b_enc,b_dec`.
- [ ] 5.2.3 `test_write_basis_placeholder_shapes_correct` — `W_enc.shape == (d_model, n_features)`, `b_enc.shape == (n_features,)`, `b_dec.shape == (d_model,)`, all with the same dtype as `W_dec`.
- [ ] 5.2.4 `test_write_basis_round_trip_real` — write a real basis, read back via `from_polygram_checkpoint`, assert the round-trip preserves all four arrays bit-exactly.
- [ ] 5.2.5 `test_write_basis_round_trip_placeholder` — write a synthetic basis (placeholder synthesis), read back, assert `W_dec` is bit-identical and the other three match the synthesis formulas.

### 5.3 FSM compose integration

- [ ] 5.3.1 `tests/test_forge_pipeline_fsm_compose.py::test_fsm_with_validation_report_loads_synth_basis` — construct `ForgePipeline(validation_report_path=<real_path>)`, call `_run_real_fsm`, assert the FSM doesn't fail at the `_load_sae_checkpoint: no key aliasing to 'W_enc'` error. Gated on `[torch]` + `[polygram]` extras (`pytest.importorskip` both). End-to-end run can be a smoke (any non-zero finite KL is enough).

### 5.4 CLI inspect

- [ ] 5.4.1 `tests/test_cli.py::test_inspect_surfaces_synthesised_keys` — invoke `saeforge inspect <synth_basis_path>` via subprocess; assert stdout JSON contains `synthesised_keys` with the three placeholder names.
- [ ] 5.4.2 `test_inspect_no_synthesised_keys_on_real_sae` — invoke against a real polygram-output safetensors; assert `synthesised_keys` is an empty list.

### 5.5 Byte-equivalence regression

- [ ] 5.5.1 Run the full pre-change pytest suite against the change. Confirm 0 regressions on the pre-existing 488-test baseline. The pre-compressed flow (no validation report) is unaffected.

## 6. Spec

- [ ] 6.1 Author `openspec/changes/full-sae-keys-in-synth-basis/specs/forge-fsm-orchestrator/spec.md` (new capability) covering: synth-basis write contract; placeholder synthesis algorithm; `__synthesised_keys__` metadata; interaction with `compress_with_polygram` in both passthrough and real-compress modes.

## 7. Docs

- [ ] 7.1 Add a "Synth-basis composition" subsection to `docs/advanced-fsm-options.md` explaining when the synth basis carries placeholders, what `__synthesised_keys__` means, and what compositions are sound vs. unsound (real-validation-report + synthetic-basis is documented as unsound — the compressor zeros placeholder rows).
- [ ] 7.2 CHANGELOG entry under `[Unreleased]` → `### Added (full-sae-keys-in-synth-basis)`.

## 8. Validation

- [ ] 8.1 `openspec validate full-sae-keys-in-synth-basis --strict` is green.
- [ ] 8.2 Full `pytest` suite passes; new tests cover §5.
- [ ] 8.3 `ruff check` clean on touched files.
- [ ] 8.4 Re-run the adaptive-regrow MBP smoke (the one that surfaced this) with `validation_report_path=...` set; confirm the FSM now completes through the compress action without the `_load_sae_checkpoint` failure.
- [ ] 8.5 `openspec archive full-sae-keys-in-synth-basis` after merge.

## 9. What this change explicitly defers

- [ ] 9.1 Tracking the **source checkpoint path** on `FeatureBasis` for file-copy round-trip (instead of placeholder synthesis). Cleaner but invasive; revisit if the placeholder approach proves insufficient.
- [ ] 9.2 **Pseudoinverse-based `W_enc` synthesis** instead of `W_dec.T`. Marginal benefit; the compressor doesn't care.
- [ ] 9.3 Supporting **non-residual-stream SAE layouts** (encoder-decoder hosts, multi-token-prediction heads). Out of scope — assumes standard LM shape.
- [ ] 9.4 An **integration test that combines `validation_report_path` + `adaptive_regrow=True`** end-to-end. Bundled smoke is in §5.3; the live MBP smoke confirms it works post-change. Full automated coverage is a follow-up if real-host integration tests grow.
- [ ] 9.5 A **CLI flag** (`--source-sae-checkpoint`) for users who want the FSM to copy the original SAE file instead of relying on placeholder synthesis. Possible follow-up once we have evidence of placeholder-introspection confusion.
