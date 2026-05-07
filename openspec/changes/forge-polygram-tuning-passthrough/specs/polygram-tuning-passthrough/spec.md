## ADDED Requirements

### Requirement: ForgePipeline carries typed polygram tuning fields

`ForgePipeline` SHALL expose three optional dataclass-typed fields for polygram tuning:

- `compression: CompressionConfig | None = None` — passed to `polygram.Compressor` via its `config=` kwarg.
- `epoch_compression: EpochCompressionConfig | None = None` — passed to `polygram.EpochCompressor` via its `config=` kwarg.
- `regrow: RegrowConfig | None = None` — passed to `polygram.Regrower.from_compression_report` via its `config=` kwarg.

When a field is `None`, sae-forge SHALL call the corresponding polygram constructor without a `config=` argument so polygram's own defaults apply. The flat `compression_strategy` and `rep_selection` fields SHALL be removed; callers migrate to `compression=CompressionConfig(...)`.

#### Scenario: pipeline accepts the three config fields

- **GIVEN** `from polygram.config import CompressionConfig, EpochCompressionConfig`
- **WHEN** `ForgePipeline(host_model_id="gpt2", compression=CompressionConfig(strategy="merge"), epoch_compression=EpochCompressionConfig(coverage_target=0.6))` is constructed
- **THEN** the resulting instance has the supplied configs on the corresponding fields and `pipeline.regrow is None`

#### Scenario: legacy compression_strategy kwarg raises

- **WHEN** `ForgePipeline(host_model_id="gpt2", compression_strategy="merge")` is constructed
- **THEN** Python raises `TypeError` for the unexpected keyword argument `compression_strategy`

#### Scenario: pipeline with no polygram fields uses polygram defaults

- **GIVEN** `pipeline = ForgePipeline(host_model_id="gpt2")` with no compression/regrow fields set
- **WHEN** the FSM enters `compress_with_polygram` and constructs a `Compressor`
- **THEN** the call is `Compressor(validation_report=..., sae_checkpoint=...)` without a `config=` argument; polygram's own default `CompressionConfig` applies

### Requirement: regrow_count > 0 requires explicit RegrowConfig

`ForgePipeline.__post_init__` SHALL raise `ValueError` when `regrow_count > 0` is set together with `regrow is None`. The error message SHALL name the missing field and demonstrate the one-line fix (`regrow=RegrowConfig(model_name=..., layer=...)`). The previous fallbacks (`layer=10`, `model_name="gpt2"`) SHALL NOT be reinstated anywhere — neither in the pipeline nor in `perform_regrowth`.

#### Scenario: regrow_count without regrow raises at construction

- **WHEN** `ForgePipeline(host_model_id="pythia-160m", regrow_count=2)` is constructed without a `regrow` argument
- **THEN** `__post_init__` raises `ValueError` whose message names `regrow` and `RegrowConfig`

#### Scenario: regrow_count with explicit RegrowConfig is accepted

- **GIVEN** `cfg = RegrowConfig(model_name="pythia-160m", layer=4, strategy="residual_kmeans")`
- **WHEN** `ForgePipeline(host_model_id="pythia-160m", regrow_count=2, regrow=cfg)` is constructed
- **THEN** construction succeeds and `pipeline.regrow == cfg`

#### Scenario: perform_regrowth surfaces missing config when ctx lacks regrow

- **GIVEN** an FSM context dict whose `regrow_count > 0` but whose `regrow` key is absent (e.g. an externally-built ctx that bypassed `ForgePipeline`)
- **WHEN** `perform_regrowth(ctx, None)` is called
- **THEN** the action raises `ValueError` whose message names the missing `ctx["regrow"]` key and references `RegrowConfig`

### Requirement: FSM context round-trips polygram configs through dicts

`ForgePipeline._build_context` SHALL serialise each non-None polygram config field to a dict via `cfg.to_dict()` under the matching ctx key (`compression`, `epoch_compression`, `regrow`); a `None` field SHALL produce a missing key (not a `None` value) so the action layer's `ctx.get(key)` check is unambiguous. `compress_with_polygram` and `perform_regrowth` SHALL reconstitute via `<Config>.from_dict(ctx[key])` before calling polygram. The legacy per-field ctx keys (`compression_strategy`, `rep_selection`, `regrow_strategy`, `regrow_layer`, `regrow_seed`, `regrow_prompts`) SHALL be removed from both `_build_context` and the orca-lang machine context schema.

#### Scenario: configs serialise to JSON-compatible dicts

- **GIVEN** `pipeline = ForgePipeline(host_model_id="gpt2", compression=CompressionConfig(strategy="merge"))`
- **WHEN** `ctx = pipeline._build_context(...)` is called
- **THEN** `ctx["compression"]` is a `dict`, `json.dumps(ctx["compression"])` succeeds, and `CompressionConfig.from_dict(ctx["compression"]) == pipeline.compression`

#### Scenario: action reconstitutes config from ctx

- **GIVEN** an FSM context with `ctx["compression"] = {"strategy": "merge", "rep_selection": "scale_aware"}` and `ctx["epoch_compression"]` absent
- **WHEN** `compress_with_polygram(ctx, None)` runs
- **THEN** the action constructs `Compressor(..., config=CompressionConfig(strategy="merge", rep_selection="scale_aware"))` and does not pass an `epoch_compression`-derived argument

#### Scenario: legacy ctx keys are not consulted

- **GIVEN** an FSM context that contains both new (`compression`) and legacy (`compression_strategy`) keys
- **WHEN** `compress_with_polygram(ctx, None)` runs
- **THEN** the action reads `ctx["compression"]` and ignores `ctx["compression_strategy"]`; the legacy key has no effect on the resulting `Compressor` config

### Requirement: CLI exposes high-frequency polygram tuning flags

`saeforge.cli` SHALL accept the following flags and convert them into the matching polygram config dataclasses before constructing `ForgePipeline`:

- `--coverage-target FLOAT` → `epoch_compression.coverage_target`
- `--cosine-threshold FLOAT` → `epoch_compression.cosine_threshold`
- `--max-compress-iterations INT` → `epoch_compression.max_iterations`
- `--regrow-layer INT` → `regrow.layer` (required iff `--regrow-count > 0`)
- `--regrow-strategy STRING` → `regrow.strategy`

When any `--coverage-target` / `--cosine-threshold` / `--max-compress-iterations` flag is supplied, the CLI SHALL build an `EpochCompressionConfig` and pass it to `ForgePipeline(epoch_compression=...)`. Other tuning knobs SHALL remain accessible only via the Python API or a `--config-file` YAML/JSON path.

#### Scenario: CLI flags propagate into ForgePipeline.epoch_compression

- **WHEN** `python -m saeforge --coverage-target 0.6 --max-compress-iterations 2 ...other-required-args...` is run
- **THEN** the constructed `ForgePipeline` has `pipeline.epoch_compression.coverage_target == 0.6` and `pipeline.epoch_compression.max_iterations == 2`

#### Scenario: --regrow-layer required when --regrow-count > 0

- **WHEN** `python -m saeforge --regrow-count 2 ...other-required-args...` is run without `--regrow-layer`
- **THEN** the CLI exits with a nonzero status and prints an error naming both `--regrow-layer` and `--regrow-count`

### Requirement: ForgePipeline.from_dict loads the full surface from a mapping

`ForgePipeline` SHALL expose a classmethod `from_dict(cls, data: Mapping[str, Any]) -> ForgePipeline` that:

- Pops the keys `compression`, `epoch_compression`, `regrow` and feeds each (when present and non-None) through the matching polygram `<Config>.from_dict`.
- Passes remaining keys as keyword arguments to `ForgePipeline(...)`.
- Emits a `UserWarning` and ignores the value for any unknown top-level key (matching polygram's own forward-compat policy).

This SHALL allow callers to load a YAML file via `yaml.safe_load` and hand the resulting dict to `ForgePipeline.from_dict` without writing a per-field marshalling layer.

#### Scenario: from_dict reconstructs a configured pipeline

- **GIVEN** `data = {"host_model_id": "gpt2", "compression": {"strategy": "merge"}, "regrow_count": 2, "regrow": {"model_name": "gpt2", "layer": 10}}`
- **WHEN** `ForgePipeline.from_dict(data)` is called
- **THEN** the returned pipeline has `host_model_id == "gpt2"`, `compression == CompressionConfig(strategy="merge")`, `regrow_count == 2`, and `regrow == RegrowConfig(model_name="gpt2", layer=10)`

#### Scenario: unknown top-level key warns and is ignored

- **WHEN** `ForgePipeline.from_dict({"host_model_id": "gpt2", "futurefield": 42})` is called
- **THEN** a `UserWarning` is emitted naming `futurefield`, and the returned pipeline has `host_model_id == "gpt2"` and all other fields at their defaults
