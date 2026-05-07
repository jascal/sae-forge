## 1. Dependency bump

- [x] 1.1 In `pyproject.toml`, bump the `polygram` minimum version to the
      release shipping `polygram-tuning-config`. Add a comment naming the
      change so future readers can grep.
- [x] 1.2 In `saeforge/actions/__init__.py`, wrap `from polygram import
      Compressor, Regrower, ...` with a friendlier `ImportError` that
      tells the user to upgrade polygram if the new symbols
      (`CompressionConfig`, `RegrowConfig`) aren't found.

## 2. ForgePipeline field surface

- [x] 2.1 Add `compression: CompressionConfig | None = None`,
      `epoch_compression: EpochCompressionConfig | None = None`,
      `regrow: RegrowConfig | None = None` fields to `ForgePipeline`
      (`saeforge/forge.py`).
- [x] 2.2 In `__post_init__`, raise `ValueError` when `regrow_count > 0`
      and `regrow is None`. Error message names the field and shows
      `regrow=RegrowConfig(model_name=..., layer=...)`.
- [x] 2.3 Implement `ForgePipeline.from_dict(cls, data)` — pops
      `compression` / `epoch_compression` / `regrow` keys through each
      `<Config>.from_dict`, warns on unknown top-level keys, passes
      the rest as kwargs.

## 3. FSM context wiring

- [x] 3.1 In `_build_context` (`saeforge/forge.py`), serialise each
      non-None polygram config field via `cfg.to_dict()` under the
      matching ctx key (`compression`, `epoch_compression`, `regrow`);
      omit the key entirely when the field is `None`.
- [x] 3.2 In `compress_with_polygram`
      (`saeforge/actions/__init__.py:45`): when `ctx.get("compression")`
      is present, reconstitute via `CompressionConfig.from_dict(...)`
      and pass `config=` to `Compressor`. Same for `epoch_compression`
      → `EpochCompressor`. When the key is absent, call without
      `config=`.
- [x] 3.3 In `perform_regrowth` (`saeforge/actions/__init__.py:103`):
      remove `ctx.get("regrow_layer", 10)` and `ctx.get("host_model_id")
      or "gpt2"` fallbacks. Require `ctx["regrow"]` whenever
      `regrow_count > 0`; reconstitute via `RegrowConfig.from_dict` and
      pass `config=`. Raise `ValueError` naming `ctx["regrow"]` and
      `RegrowConfig` when missing.

## 4. orca-lang machine schema

- [x] 4.1 Update `saeforge/machines/sae_forge.orca.md` context schema:
      add `compression`, `epoch_compression`, `regrow` (each `dict |
      None`); remove the per-field legacy keys
      (`compression_strategy`, `rep_selection`, `regrow_strategy`,
      `regrow_layer`, `regrow_seed`, `regrow_prompts`).
- [x] 4.2 Run `orca verify saeforge/machines/sae_forge.orca.md` and
      confirm zero exit.

## 5. CLI

- [x] 5.1 Add `--coverage-target`, `--cosine-threshold`,
      `--max-compress-iterations` flags to `saeforge/cli.py`. When any
      of the three is supplied, build an `EpochCompressionConfig` and
      pass it to `ForgePipeline(epoch_compression=...)`.
- [x] 5.2 Add `--regrow-layer`, `--regrow-strategy` flags. When either
      is supplied — or when `--regrow-count > 0` — build a
      `RegrowConfig` and pass to `ForgePipeline(regrow=...)`.
- [x] 5.3 In the CLI's argparse setup, mark `--regrow-layer` as
      required-conditional-on-`--regrow-count > 0`. Validate after
      parsing and exit with a clear message that names both flags.

## 6. Remove flat polygram fields

- [x] 6.1 Delete `compression_strategy: str = "merge"` and
      `rep_selection: str = "scale_aware"` from `ForgePipeline`
      (`saeforge/forge.py:55-56`).
- [x] 6.2 Delete the matching ctx-build lines (`saeforge/forge.py:247-248`).
- [x] 6.3 Audit any in-tree caller passing those fields:
      `examples/`, `tests/`, `scratch/`. Migrate each to
      `compression=CompressionConfig(...)`.

## 7. Examples

- [x] 7.1 In `examples/forge_gpt2_real_sae.py:125-138`, replace the
      `EpochCompressor(coverage_target=0.5, cosine_threshold=0.30,
      n_visits_per_feature=1, max_iterations=1, ...)` call with
      `EpochCompressor(config=EpochCompressionConfig(
      coverage_target=0.5, cosine_threshold=0.30,
      n_visits_per_feature=1, max_iterations=1))` — or, if polygram's
      iterative defaults match, just `EpochCompressor.fast()`.
- [x] 7.2 Verify the example runs end-to-end and produces a checkpoint
      whose first-batch eval roughly matches the pre-change baseline
      (within whatever tolerance the example asserts).

## 8. Tests

- [x] 8.1 Add `tests/test_polygram_tuning_passthrough.py` covering:
      pipeline-builds-context-dict, action-reconstitutes-config,
      legacy-ctx-keys-ignored, regrow-count-without-regrow-raises,
      from_dict round-trip, unknown-key-warns.
- [x] 8.2 Update `tests/test_forge_pipeline.py` for the renamed/added
      fields. Any test today passing
      `compression_strategy=` / `rep_selection=` migrates to
      `compression=CompressionConfig(...)`.
- [x] 8.3 Update `tests/test_actions_compress.py` and any test that
      builds an FSM context dict by hand: replace per-field legacy
      keys with the new dict-shaped ones.
- [x] 8.4 Add a CLI integration test (`tests/test_cli.py`) covering
      `--coverage-target`, `--regrow-layer`, and the
      `--regrow-count > 0 without --regrow-layer` error path.

## 9. Docs

- [x] 9.1 Update `README.md` with the new `compression` /
      `epoch_compression` / `regrow` fields and a one-paragraph
      explanation of the dict round-trip through FSM context.
- [x] 9.2 Add a CHANGELOG entry under "Breaking" listing:
      removed `compression_strategy` / `rep_selection` fields,
      removed `regrow_layer=10` / `model_name="gpt2"` fallbacks,
      new required `regrow=RegrowConfig(...)` for `regrow_count > 0`.
- [x] 9.3 Add a YAML example to `docs/` showing
      `ForgePipeline.from_dict(yaml.safe_load(...))` end-to-end.

## 10. Verification

- [x] 10.1 Run `pytest -q` and confirm all tests pass.
- [x] 10.2 Run `examples/forge_gpt2_real_sae.py` end-to-end on the
      bundled toy SAE and confirm it converges.
- [x] 10.3 Run `openspec validate forge-polygram-tuning-passthrough
      --strict` and confirm clean exit.
- [x] 10.4 Coordinate with polygram's `polygram-tuning-config` merge:
      this change MUST NOT land before the polygram release that
      ships the new dataclasses and `config=` kwargs.
