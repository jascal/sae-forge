## 1. Implement loader

- [x] 1.1 Implement `FeatureBasis.from_polygram_checkpoint` per the proposal: report autolocation, decoder-key resolution with `decoder.weight` transpose, `kept_ids` as the complement of the union of cluster `zeroed` lists, merged-norm by representative with row-norm fallback, `scale_compression_ratio` and `metadata` plumbed through
- [x] 1.2 Read the report JSON directly — do not import `polygram`. Keep `[polygram]` a true optional extra
- [x] 1.3 Raise `FileNotFoundError` on missing checkpoint, `FileNotFoundError` on explicitly-passed missing report, fall back to "every row is kept" when no report exists at all
- [x] 1.4 Raise `KeyError` listing candidate keys + present keys when no decoder tensor is found

## 2. Tests + fixtures

- [x] 2.1 Add `tests/conftest.py` with the `synthetic_compressed_sae` fixture producing an 8-feature, 16-dim, 2-cluster fake checkpoint + JSON report on a `tmp_path`
- [x] 2.2 Add `tests/test_feature_basis.py` covering kept-id resolution, merged-norm picking, fallback behaviour, explicit `report_path`, missing-report fallback, missing checkpoint raise, unknown-decoder-key raise, `decoder.weight` transpose, pseudoinverse caching, `to_summary` round-trip
- [x] 2.3 Update `bootstrap-package` spec scenario "FeatureBasis.from_polygram_checkpoint stub" to mark it superseded
- [x] 2.4 Update `tests/test_smoke.py` stub-points-to-change test to `test_from_polygram_checkpoint_missing_file_raises`

## 3. OpenSpec scaffolding

- [x] 3.1 `openspec/changes/feature-basis/proposal.md`
- [x] 3.2 `openspec/changes/feature-basis/tasks.md` (this file)
- [x] 3.3 `openspec/changes/feature-basis/specs/feature-basis/spec.md`
