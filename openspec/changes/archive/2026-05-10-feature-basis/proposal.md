## Why

`FeatureBasis.from_polygram_checkpoint` is the v0 entry point that turns
a Polygram-compressed `.safetensors` artifact into the surviving feature
basis sae-forge projects against. The bootstrap change shipped it as a
`NotImplementedError` stub. This change implements the loader: parse the
companion `compression_report.json`, resolve `kept_ids` from the union of
cluster `zeroed` lists, pull merged norms by representative, and return a
fully populated `FeatureBasis`.

Hard requirement: the loader must work without importing `polygram`
itself. sae-forge's `[polygram]` extra is for users who want Polygram's
upstream tooling (Compressor/Regrower), not a hard runtime dep of the
basis loader. The compression report's JSON schema is stable enough to
re-implement against — we read it directly.

## What Changes

- Implement `FeatureBasis.from_polygram_checkpoint(checkpoint_path,
  *, report_path=None)`:
  - Locate the companion report by trying `_compression_report.json`,
    `.compression_report.json`, `_report.json` suffixes on the
    checkpoint stem.
  - Open the `.safetensors` file via `safetensors.safe_open`,
    framework="numpy", and pull the decoder tensor under one of
    `("W_dec", "decoder.weight", "dec")`. Transpose when the matched
    key is `decoder.weight` and the matrix is non-square (PyTorch
    `nn.Linear` `out × in` convention).
  - Compute `kept_ids` as the row indices NOT in the union of all
    cluster `zeroed` arrays. Slice `W_dec` to those rows.
  - For each kept id, look up its merged norm: cluster representative
    with non-null `merged_norm` → use that; otherwise fall back to the
    row's L2 norm. `original_norms` always reflects the as-stored row
    L2 norms.
  - Carry `scale_compression_ratio` from the report; default to `1.0`
    when absent. Populate `metadata` with the source checkpoint hash,
    strategy, total feature count, kept count, cluster count.
- Behavior on missing report: when no `report_path` is provided and no
  candidate suffix is found on disk, treat the checkpoint as an
  uncompressed dictionary — every row is a kept feature, merged norms
  equal row norms, `scale_compression_ratio = 1.0`. When `report_path`
  is provided explicitly and missing, raise `FileNotFoundError`.
- Behavior on missing checkpoint: raise `FileNotFoundError` with the
  literal path.
- Behavior on missing decoder key: raise `KeyError` listing the
  candidate keys we tried and the keys present.

## Capabilities

### New Capabilities

- `feature-basis-loader`: Reads a Polygram-compressed `.safetensors` +
  companion `compression_report.json` and returns a fully populated
  `FeatureBasis`. Pure-numpy, no torch. Does not import the `polygram`
  package — parses the JSON schema directly so the loader stays usable
  on the no-extras install.

### Modified Capabilities

- `bootstrap`: The "FeatureBasis.from_polygram_checkpoint stub"
  scenario in `bootstrap` is superseded — the method now raises
  `FileNotFoundError` on missing inputs instead of
  `NotImplementedError`.

## Impact

- `saeforge/basis.py`: implementation lands in place. No public surface
  change beyond removing the `NotImplementedError` body.
- `tests/test_feature_basis.py`: 13 new tests covering kept-id
  resolution, merged-norm picking, fallback behaviour, missing-report
  paths, decoder-key transpose, summary round-trip.
- `tests/conftest.py`: new `synthetic_compressed_sae` fixture that
  builds a fake compressed checkpoint + JSON report end-to-end.
- One scenario in `bootstrap-package`'s spec is annotated as
  superseded.
- `tests/test_smoke.py`: the stub-points-to-change test rewrites to
  `test_from_polygram_checkpoint_missing_file_raises`.
