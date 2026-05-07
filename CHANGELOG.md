# Changelog

All notable changes to sae-forge are tracked here. v0 entries land as
their corresponding OpenSpec change is archived.

## [0.1.0] — 2026-05-07

### Added (forge-polygram-tuning-passthrough)

- Three typed polygram-tuning fields on `ForgePipeline`:
  `compression: CompressionConfig | None`,
  `epoch_compression: EpochCompressionConfig | None`,
  `regrow: RegrowConfig | None`. Each round-trips through the FSM
  context as a JSON-friendly dict (`cfg.to_dict()` →
  `<Config>.from_dict(ctx[key])`).
- `ForgePipeline.from_dict(data)` classmethod for YAML/JSON config
  loading; emits `UserWarning` for unknown top-level keys.
- New CLI flags: `--coverage-target`, `--cosine-threshold`,
  `--max-compress-iterations`, `--regrow-count`, `--regrow-layer`,
  `--regrow-strategy`. Long-tail tuning lives behind
  `ForgePipeline.from_dict`.
- `docs/forge_config_example.yaml` showing the
  `ForgePipeline.from_dict(yaml.safe_load(...))` shape end-to-end.
- `tests/test_polygram_tuning_passthrough.py` (15 tests) and
  `tests/test_cli.py` (5 tests).

### Changed (Breaking — forge-polygram-tuning-passthrough)

- **Removed flat `compression_strategy` and `rep_selection` fields
  on `ForgePipeline`.** Passing either now raises `TypeError` at
  construction. Migrate to
  `compression=CompressionConfig(strategy=..., rep_selection=...)`.
- **`regrow_count > 0` requires explicit `regrow=RegrowConfig(...)`.**
  `__post_init__` raises `ValueError` otherwise.
- **`perform_regrowth` action requires `ctx["regrow"]`** when
  `regrow_count > 0`. The previous `ctx.get("regrow_layer", 10)` and
  `ctx.get("host_model_id") or "gpt2"` fallbacks were removed in
  lock-step with polygram 0.1.0 dropping the matching defaults from
  `Regrower.from_compression_report`.
- Pinned `polygram>=0.1.0` (was `>=0.0.1`).

### Migration

- Replace `ForgePipeline(compression_strategy="merge",
  rep_selection="scale_aware", ...)` with
  `ForgePipeline(compression=CompressionConfig(strategy="merge",
  rep_selection="scale_aware"), ...)`.
- Callers with `regrow_count > 0` now must pass
  `regrow=RegrowConfig(model_name=<host>, layer=<int>)`. Layer is
  host-specific and no longer has a GPT-2 default.

### Internal

- Two pre-existing CI tests (`test_forge_pipeline_run_requires_host_model_id`,
  `test_project_module_unsupported_arch_raises`) gated with
  `pytest.importorskip("torch")` so the no-extras CI install stays
  green.

### Added

- Repository scaffolding: `pyproject.toml`, `README.md`, `AGENTS.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, CI workflow,
  `saeforge/` package skeleton with stub `FeatureBasis`,
  `SubspaceProjector`, `NativeModel`, `ForgePipeline`, and `cli.main`.
- OpenSpec change `bootstrap-package` defining the v0 milestone.
