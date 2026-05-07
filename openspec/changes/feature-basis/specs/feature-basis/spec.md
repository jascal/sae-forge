# feature-basis-loader Specification

## Purpose

Defines `FeatureBasis.from_polygram_checkpoint` — the canonical entry
point that turns a Polygram-compressed `.safetensors` artifact + its
companion `compression_report.json` into a fully-populated
`FeatureBasis`. The loader is pure-numpy and does not import the
`polygram` package; it reads the report's JSON schema directly so the
no-extras install path stays usable.

## Requirements

### Requirement: Kept ids are the complement of cluster `zeroed` lists

`from_polygram_checkpoint` SHALL set `basis.kept_ids` to the sorted
ascending list of row indices in `W_dec` that do NOT appear in the
union of every cluster's `zeroed` field. `basis.W_dec` SHALL be
`W_dec_full[basis.kept_ids]` and `basis.n_features` SHALL equal
`len(basis.kept_ids)`.

#### Scenario: 8-feature SAE, two clusters zero rows 5 and 7

- **GIVEN** a checkpoint of shape `(8, 16)` with cluster A zeroing
  feature 5 and cluster B zeroing feature 7
- **WHEN** `FeatureBasis.from_polygram_checkpoint(...)` is called
- **THEN** `basis.kept_ids.tolist()` equals `[0, 1, 2, 3, 4, 6]`
- **AND** `basis.W_dec.shape` equals `(6, 16)`

### Requirement: Merged norm picked from representative when present

For each kept feature `fid`, `basis.merged_norms[i]` SHALL equal the
cluster's `merged_norm` when `fid` is the cluster representative AND
`merged_norm` is non-null. Otherwise it SHALL fall back to
`np.linalg.norm(basis.W_dec[i])`.

#### Scenario: representative with merged_norm 1.5

- **GIVEN** the synthetic 8-feature checkpoint where cluster A has
  representative 2 and `merged_norm = 1.5`
- **WHEN** the basis is loaded
- **THEN** `basis.merged_norms[basis.kept_ids.tolist().index(2)]`
  equals 1.5 within floating-point tolerance

#### Scenario: representative with null merged_norm falls back to row norm

- **GIVEN** cluster B has representative 3 and `merged_norm = null`
  (zero strategy, no rescale applied)
- **WHEN** the basis is loaded
- **THEN** `basis.merged_norms[basis.kept_ids.tolist().index(3)]`
  equals `np.linalg.norm(W_dec_full[3])`

### Requirement: Scale compression ratio carried through

`basis.scale_compression_ratio` SHALL equal the report's
`scale_compression_ratio` field, or `1.0` when the field is absent.

#### Scenario: ratio 0.92 round-trips

- **GIVEN** a report with `scale_compression_ratio: 0.92`
- **WHEN** the basis is loaded
- **THEN** `basis.scale_compression_ratio` equals `0.92` exactly

### Requirement: Auto-locates companion report

When `report_path` is not supplied, the loader SHALL try the candidate
suffixes `_compression_report.json`, `.compression_report.json`,
`_report.json` against the checkpoint's stem in that order and pick
the first one that exists.

#### Scenario: default suffix matches

- **GIVEN** a checkpoint at `sae.compressed.safetensors` and a report
  at `sae.compressed_compression_report.json`
- **WHEN** the loader is called with no `report_path` argument
- **THEN** the report is found and parsed automatically

#### Scenario: explicit report path overrides discovery

- **GIVEN** a report file relocated to a non-default location
- **WHEN** the loader is called with `report_path=<that path>`
- **THEN** the loader uses that file regardless of any default-suffix
  match

### Requirement: Missing-report fallback yields full dictionary

When no `report_path` is supplied AND no candidate suffix matches on
disk, the loader SHALL return a basis covering every row of the
checkpoint (no rows zeroed), with `merged_norms == original_norms ==
row_l2_norms` and `scale_compression_ratio == 1.0`. This makes the
loader work on uncompressed SAEs as a graceful no-op compression
report.

When `report_path` IS supplied but the file is missing, the loader
SHALL raise `FileNotFoundError`.

#### Scenario: implicit fallback to full dictionary

- **GIVEN** a `.safetensors` checkpoint with shape `(8, 16)` and no
  companion report on disk
- **WHEN** the loader is called with no `report_path`
- **THEN** `basis.n_features` equals `8` and
  `basis.scale_compression_ratio` equals `1.0`

#### Scenario: explicit missing report raises

- **WHEN** the loader is called with a `report_path` that does not
  exist
- **THEN** `FileNotFoundError` is raised whose message contains
  "compression report"

### Requirement: Polygram is not imported

The loader SHALL NOT import the `polygram` package. The loader SHALL
work on a venv with only `numpy` and `safetensors` installed.

#### Scenario: loader runs without polygram installed

- **GIVEN** a venv with `numpy` and `safetensors` only (no `polygram`)
- **WHEN** `from_polygram_checkpoint` is called on a valid synthetic
  artifact
- **THEN** the call succeeds and `import polygram` is not visible in
  `sys.modules` after the call returns

### Requirement: Decoder-tensor key precedence and transpose

The loader SHALL try the keys `("W_dec", "decoder.weight", "dec")` in
order and use the first one present. When the matched key is
`decoder.weight` AND the matrix is non-square, the loader SHALL
transpose it (PyTorch `nn.Linear` `out × in` convention). When none of
the candidate keys is present, the loader SHALL raise `KeyError` whose
message lists both the candidate set and the keys actually present.

#### Scenario: decoder.weight non-square gets transposed

- **GIVEN** a checkpoint storing the decoder under key
  `decoder.weight` with shape `(16, 8)` (host out × in)
- **WHEN** the basis is loaded
- **THEN** `basis.W_dec.shape` equals `(8, 16)` (`n_features ×
  d_model`)

#### Scenario: missing decoder key raises with both lists

- **GIVEN** a checkpoint with only the key `some_other_key`
- **WHEN** the basis is loaded
- **THEN** `KeyError` is raised whose message contains the candidate
  set tuple AND `some_other_key`
