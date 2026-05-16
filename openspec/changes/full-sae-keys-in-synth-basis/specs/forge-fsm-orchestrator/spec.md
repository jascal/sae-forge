# forge-fsm-orchestrator Specification

## Purpose

Documents the on-disk contract `ForgePipeline._run_real_fsm` and `_run_synthetic_fsm` honour when writing the **synth-basis** safetensors that the FSM orchestrator hands to its downstream actions. The contract is load-bearing for any FSM action that loads the file via polygram's `_load_sae_checkpoint` — specifically `compress_with_polygram` when a real `validation_report_path` is supplied.

Before this change, the synth-basis carried only `W_dec`, which worked for the passthrough-compress path (no validation report) but failed at `compress_with_polygram`'s `_load_sae_checkpoint(..., ["W_enc", "b_enc", "b_dec", "W_dec"])` call when a real validation report was wired in. This capability spec defines what the synth-basis SHALL contain so that path composes cleanly.

## ADDED Requirements

### Requirement: Synth-basis writes a complete SAE key set

`ForgePipeline._run_real_fsm` and `_run_synthetic_fsm` SHALL write a `synth_basis.safetensors` whose tensor keys are the full standard SAE key set: `W_enc`, `b_enc`, `b_dec`, `W_dec`. The write SHALL be performed by `saeforge.forge._write_basis_as_checkpoint(basis, path)`; the helper is the single point of contract for the synth-basis schema.

When `basis` (a `FeatureBasis` instance) has the corresponding optional fields populated (`basis.W_enc`, `basis.b_enc`, `basis.b_dec` are non-None), the real values SHALL be written. When any field is None, a **placeholder** SHALL be synthesised per the algorithm below.

#### Scenario: real-basis round-trip preserves all four keys

- **GIVEN** a `FeatureBasis` loaded from a polygram-compressed safetensors that contains `W_enc`, `b_enc`, `b_dec`, `W_dec`
- **WHEN** `_write_basis_as_checkpoint(basis, path)` writes the file
- **THEN** the resulting safetensors contains all four keys with bit-identical content to the source

#### Scenario: synthetic-basis writes all four keys via placeholder synthesis

- **GIVEN** a `FeatureBasis` constructed programmatically with only `kept_ids` + `W_dec` + `merged_norms` + `original_norms` (the existing `tiny_synthetic_basis` test-fixture pattern), with the three optional fields at their `None` defaults
- **WHEN** `_write_basis_as_checkpoint(basis, path)` writes the file
- **THEN** the resulting safetensors contains all four keys: real `W_dec`, plus `W_enc = W_dec.T`, `b_enc = zeros(n_features)`, `b_dec = zeros(d_model)`, all cast to `W_dec.dtype`

#### Scenario: written file is loadable by polygram's `_load_sae_checkpoint`

- **GIVEN** any `_write_basis_as_checkpoint` output (real or placeholder)
- **WHEN** `polygram.sae_import._load_sae_checkpoint(path, ["W_enc", "b_enc", "b_dec", "W_dec"])` is called
- **THEN** all four keys load successfully without `no key aliasing to ...` errors

### Requirement: `FeatureBasis` retains optional SAE state

`saeforge.basis.FeatureBasis` SHALL expose three optional fields, all defaulting to `None`:

- `W_enc: np.ndarray | None`
- `b_enc: np.ndarray | None`
- `b_dec: np.ndarray | None`

When non-None, shapes SHALL match the dataclass's existing `n_features` / `d_model` derivations: `W_enc.shape == (d_model, n_features)`, `b_enc.shape == (n_features,)`, `b_dec.shape == (d_model,)`. Shape mismatch SHALL raise `ValueError` in `__post_init__` naming the field and the expected shape.

`FeatureBasis.from_polygram_checkpoint` SHALL populate these fields when the source safetensors contains the corresponding keys. Missing keys in the source SHALL leave the fields at `None` — this is the legacy synth-basis-style file and is not an error.

#### Scenario: optional fields default to None

- **WHEN** `FeatureBasis(kept_ids=..., W_dec=..., merged_norms=..., original_norms=...)` is constructed without the new fields
- **THEN** the resulting instance has `W_enc is None`, `b_enc is None`, `b_dec is None`

#### Scenario: shape mismatch raises

- **WHEN** `FeatureBasis(..., W_enc=np.zeros((5, 7)))` is constructed where `n_features=8` and `d_model=16`
- **THEN** `__post_init__` raises `ValueError` whose message names `W_enc` and the expected shape `(16, 8)`

#### Scenario: from_polygram_checkpoint populates optional fields

- **GIVEN** a real polygram-output safetensors at `path` (e.g., produced by `polygram compress --pareto-materialize`)
- **WHEN** `FeatureBasis.from_polygram_checkpoint(path)` is called
- **THEN** the resulting basis has `W_enc`, `b_enc`, `b_dec` all non-None and shape-correct

#### Scenario: from_polygram_checkpoint tolerates legacy W_dec-only files

- **GIVEN** a safetensors at `path` containing only `W_dec` (the pre-change synth-basis layout)
- **WHEN** `FeatureBasis.from_polygram_checkpoint(path)` is called
- **THEN** the resulting basis has `W_enc is None`, `b_enc is None`, `b_dec is None`; no error is raised

### Requirement: Placeholder-synthesis algorithm

When `_write_basis_as_checkpoint` synthesises placeholders, it SHALL use the following deterministic formulas:

- `W_enc` placeholder = `numpy.ascontiguousarray(basis.W_dec.T)` (decoder transpose). Shape `(d_model, n_features)`.
- `b_enc` placeholder = `numpy.zeros(n_features, dtype=basis.W_dec.dtype)`.
- `b_dec` placeholder = `numpy.zeros(d_model, dtype=basis.W_dec.dtype)`.

The placeholders SHALL match `basis.W_dec.dtype` so the written file has a single coherent dtype. Placeholders SHALL be marked in the file's metadata via the `__synthesised_keys__` key (see next requirement).

#### Scenario: placeholder W_enc is the contiguous transpose of W_dec

- **GIVEN** a synthetic basis with `W_dec` of shape `(n_features, d_model)`
- **WHEN** the placeholder is synthesised
- **THEN** the written `W_enc` is `numpy.ascontiguousarray(W_dec.T)`, dtype-matched, shape `(d_model, n_features)`

#### Scenario: placeholder biases are zeros

- **WHEN** the placeholder bias is synthesised
- **THEN** the written `b_enc` is `zeros(n_features)` and `b_dec` is `zeros(d_model)`, both in `W_dec.dtype`

### Requirement: `__synthesised_keys__` metadata marks placeholder keys

`_write_basis_as_checkpoint` SHALL write a `__synthesised_keys__` metadata entry to the safetensors file via `safetensors.numpy.save_file`'s `metadata=` parameter. The value SHALL be a comma-separated string of key names that were synthesised (one of `""`, `"W_enc"`, `"b_enc"`, `"b_dec"`, or any subset, sorted lexicographically). An empty string means every key was loaded from a real source.

The metadata is purely informational — polygram's `_load_sae_checkpoint` does not read it, so the synth-basis remains a valid polygram input regardless of metadata. Sae-forge's `inspect` CLI surfaces the metadata so callers can distinguish "this SAE has a real encoder" from "this SAE has a structural placeholder."

#### Scenario: real basis writes empty synthesised_keys metadata

- **GIVEN** a basis with all four arrays loaded from a real polygram-output safetensors
- **WHEN** `_write_basis_as_checkpoint(basis, path)` writes the file
- **THEN** the safetensors metadata contains `"__synthesised_keys__": ""`

#### Scenario: synthetic basis writes all three placeholder keys in metadata

- **GIVEN** a basis with only `W_dec` (the synthetic test-fixture pattern)
- **WHEN** the helper writes the file
- **THEN** the safetensors metadata contains `"__synthesised_keys__": "W_enc,b_dec,b_enc"` (lexicographic order)

#### Scenario: partial-real basis lists only the truly-synthesised keys

- **GIVEN** a basis loaded from a file that had `W_dec` + `W_enc` but no biases (an unusual but possible mid-development SAE), so `basis.b_enc is None` and `basis.b_dec is None` while `basis.W_enc` is real
- **WHEN** the helper writes the file
- **THEN** the metadata contains `"__synthesised_keys__": "b_dec,b_enc"` and `W_enc` is bit-identical to the source

### Requirement: `compress_with_polygram` composes with the synth basis

When `_run_real_fsm` (or `_run_synthetic_fsm`) writes a synth basis and sets `ctx["validation_report_path"]` to a real `polygram.ValidationReport` JSON, the subsequent `compress_with_polygram` action SHALL load the synth basis via polygram's `_load_sae_checkpoint` and successfully invoke `Compressor.apply()`. The compressed output SHALL be a polygram-compatible safetensors with the standard key set.

#### Scenario: end-to-end FSM forge with real validation report succeeds

- **GIVEN** `ForgePipeline(..., orchestrator="fsm", validation_report_path=<path-to-real-report>)`
- **WHEN** `pipeline.run(...)` is called
- **THEN** the FSM completes through `compress_with_polygram` without `_load_sae_checkpoint: no key aliasing to 'W_enc'` errors; the run reaches `evaluate_faithfulness` and emits a finite `faithfulness_kl`

#### Scenario: compress action on synthetic-basis warns about placeholder semantics

- **GIVEN** a `FeatureBasis` whose `W_enc is None` (placeholder will be synthesised), AND a real `validation_report_path` so compress runs in non-passthrough mode
- **WHEN** the FSM dispatches `compress_with_polygram`
- **THEN** the action SHOULD emit a `UserWarning` naming the placeholder semantics (the compressed output's `W_enc` is derived from a placeholder, not a real encoder). The warning is informational; the FSM continues normally.

### Requirement: `saeforge inspect` surfaces the synthesised-keys metadata

The `saeforge inspect` CLI subcommand SHALL read the `__synthesised_keys__` metadata after loading a safetensors checkpoint and surface it in both the JSON summary (`synthesised_keys: list[str]`) and the optional markdown report (a "Synthesised keys" line when the list is non-empty).

#### Scenario: inspect on a real polygram SAE shows no synthesised keys

- **WHEN** `saeforge inspect <path-to-polygram-output>` is invoked
- **THEN** stdout JSON contains `"synthesised_keys": []`

#### Scenario: inspect on a synth basis lists the placeholder keys

- **WHEN** `saeforge inspect <path-to-synth-basis-with-placeholders>` is invoked
- **THEN** stdout JSON contains `"synthesised_keys": ["W_enc", "b_dec", "b_enc"]` (or a subset, depending on what was synthesised)
