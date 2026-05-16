# polygram-tuning-passthrough Specification (delta)

## MODIFIED Requirements

### Requirement: ForgePipeline carries typed polygram tuning fields

`ForgePipeline` SHALL continue to expose three optional dataclass-typed fields (`compression`, `epoch_compression`, `regrow`) bound to polygram's `CompressionConfig`, `EpochCompressionConfig`, and `RegrowConfig`. With the polygram 0.4.0 dependency bump (see Impact below), `CompressionConfig` carries two additional fields — `target_n_features_kept: int | None = None` and `score_field: str = "polygram_overlap"` — and these SHALL flow through the existing ctx round-trip machinery (`_ConfigMixin.to_dict()` / `from_dict()`) with no sae-forge-side code change. The existing `compress_with_polygram` action SHALL reconstitute the full config including the new fields and pass it to `Compressor(..., config=...)`; polygram's planner SHALL dispatch to target-K mode when `target_n_features_kept is not None`.

When a field is `None`, sae-forge SHALL continue to call the corresponding polygram constructor without a `config=` argument so polygram's own defaults apply. The flat `compression_strategy` / `rep_selection` fields remain removed; callers use `compression=CompressionConfig(...)`.

#### Scenario: CompressionConfig with target_n_features_kept round-trips through ctx

- **GIVEN** `from polygram.config import CompressionConfig`
- **WHEN** `pipeline = ForgePipeline(host_model_id="gpt2", compression=CompressionConfig(target_n_features_kept=500, score_field="jaccard"))` is constructed and `ctx = pipeline._build_context(...)` runs
- **THEN** `ctx["compression"]` is a `dict` with `"target_n_features_kept": 500` and `"score_field": "jaccard"` present; `CompressionConfig.from_dict(ctx["compression"]).target_n_features_kept == 500`

#### Scenario: target-K mode is plumbed through compress_with_polygram

- **GIVEN** an FSM context with `ctx["compression"] = {"target_n_features_kept": 500, "score_field": "polygram_overlap", "strategy": "merge", "rep_selection": "scale_aware", "merge_mode": "freq_weighted"}`
- **WHEN** `compress_with_polygram(ctx, None)` runs
- **THEN** the action constructs `Compressor(..., config=CompressionConfig(target_n_features_kept=500, ...))`; polygram's planner dispatches to `plan_with_target` (not the threshold-mode `plan()`); the resulting `ctx["current_feature_count"]` is `<= 500`

#### Scenario: byte-identity preserved when target_n_features_kept is None

- **GIVEN** `pipeline = ForgePipeline(host_model_id="gpt2", compression=CompressionConfig(strategy="merge"))` (no target-K field set)
- **WHEN** the FSM runs through `compress_with_polygram` against the existing toy fixture
- **THEN** the output is byte-identical to the pre-polygram-0.4.0 reference

## ADDED Requirements

### Requirement: Polygram minimum version

The `polygram` dependency in `pyproject.toml` SHALL specify `polygram>=0.4.0` (lines 20, 66, 89). Earlier polygram versions SHALL NOT be supported; the new `CompressionConfig` fields are required by the `pareto-sweep` capability.

#### Scenario: pyproject pins polygram>=0.4.0

- **WHEN** the project metadata is parsed (`pip show polygram` or `tomllib.load(open("pyproject.toml", "rb"))`)
- **THEN** the resolved minimum version constraint on `polygram` is `>=0.4.0`
