## Why

`ForgePipeline` partially threads polygram tuning knobs and partially hides them. The result is a tuning surface that's awkward in three places:

1. **EpochCompressor knobs are unreachable.** The biggest levers on iterative compression — `coverage_target`, `cosine_threshold`, `n_visits_per_feature`, `max_iterations`, `polygram_overlap_threshold`, `jaccard_threshold`, `min_both_fire`, `quality_delta_multiplier` — only appear hard-coded inside `examples/forge_gpt2_real_sae.py:125-138`. Any non-example caller cannot tune them at all.
2. **Regrowth knobs are FSM-context-only.** `regrow_strategy`, `regrow_layer`, `regrow_seed`, `regrow_prompts` are read from `ctx` in `saeforge/actions/__init__.py:118-124` but have no `ForgePipeline` field, no CLI flag, and no documented surface. To tune them a caller must construct the FSM context dict directly.
3. **GPT-2 defaults baked in.** `saeforge/actions/__init__.py:121` defaults `layer=10` and line 122 defaults `model_name="gpt2"`. Any non-GPT-2 host runs with silently wrong layer indices unless the caller knows to override via context. (The companion polygram change `polygram-tuning-config` removes those defaults at the polygram side, which means without this change sae-forge will start raising `TypeError` from `Regrower.from_compression_report`.)

This change makes the polygram tuning surface first-class on `ForgePipeline`: typed fields, CLI access for the high-frequency knobs, and a clean handoff into FSM context using polygram's new `CompressionConfig` / `EpochCompressionConfig` / `RegrowConfig` dataclasses.

## What Changes

- Add `compression: CompressionConfig | None = None` and `epoch_compression: EpochCompressionConfig | None = None` fields to `ForgePipeline` (replacing the flat `compression_strategy` and `rep_selection` fields, which are subsumed by `CompressionConfig`).
- Add `regrow: RegrowConfig | None = None` to `ForgePipeline`. When `regrow_count > 0` and `regrow is None`, raise at pipeline construction (no silent fallback).
- **BREAKING**: Remove the flat `ForgePipeline.compression_strategy` and `ForgePipeline.rep_selection` fields. Callers passing them get a `TypeError` from the dataclass; they migrate to `compression=CompressionConfig(strategy=..., rep_selection=...)`.
- **BREAKING**: Remove the `ctx.get("regrow_layer", 10)` and `ctx.get("host_model_id") or "gpt2"` fallbacks in `perform_regrowth` (`saeforge/actions/__init__.py:121-122`). When the FSM enters the regrow path, the action SHALL require `ctx["regrow"]` (a `RegrowConfig`) to be present; absent or partial config raises a clear error.
- Serialise the polygram configs onto the FSM context as plain dicts (via `cfg.to_dict()`), and reconstitute them inside the actions via `<Config>.from_dict(...)`. Round-trip preserves type but keeps ctx JSON-serialisable so the FSM's existing trace tooling keeps working.
- Add CLI flags for the small set of high-frequency knobs: `--coverage-target`, `--cosine-threshold`, `--max-compress-iterations`, `--regrow-layer`, `--regrow-strategy`. Everything else stays Python-API-only (the dataclass surface is enough; the CLI is for one-shot runs).
- Add a `ForgePipeline.from_dict(d)` classmethod so callers (and tests) can drive the pipeline from a YAML/JSON config without mirroring our dataclass shape by hand.
- Update `examples/forge_gpt2_real_sae.py` to use `EpochCompressionConfig` directly instead of the kwarg quartet.
- Document in README that polygram tuning configs round-trip through ctx unchanged, so downstream FSM observers can read them back.

## Capabilities

### New Capabilities

- `polygram-tuning-passthrough`: `ForgePipeline` exposes typed polygram-tuning fields (`compression`, `epoch_compression`, `regrow`); they round-trip through FSM context via `to_dict()` / `from_dict()`; FSM actions reconstitute the dataclass from ctx with no per-field fallback defaults; regrow path requires an explicit `RegrowConfig`; high-frequency knobs reachable from CLI.

### Modified Capabilities

<!-- None — sae-forge has no archived baseline specs (`openspec/specs/` is empty). All affected behaviour is still in unarchived changes (`forge-pipeline`, `forge-outer-loop-fsm`). The deltas there will be picked up when those changes archive; this change documents the new surface as its own capability. -->

## Impact

- `saeforge/forge.py` — replace flat fields with `compression`/`epoch_compression`/`regrow`; serialise via `to_dict()` into ctx; new `from_dict` classmethod.
- `saeforge/cli.py` — add the five new CLI flags; convert them into the matching config dataclasses before calling `ForgePipeline`.
- `saeforge/actions/__init__.py:45-100` (`compress_with_polygram`) — read `ctx["compression"]` and optional `ctx["epoch_compression"]`; pass into polygram via the `config=` kwarg added by `polygram-tuning-config`.
- `saeforge/actions/__init__.py:103-127` (`perform_regrowth`) — drop the `regrow_layer=10` / `host_model_id or "gpt2"` fallbacks; require `ctx["regrow"]`; pass via `config=`.
- `saeforge/machines/sae_forge.orca.md` — context schema documents the three new dict fields (`compression`, `epoch_compression`, `regrow`); the per-field legacy ctx keys (`compression_strategy`, `rep_selection`, `regrow_layer`, …) are removed.
- `examples/forge_gpt2_real_sae.py` — switch to `EpochCompressionConfig` (or `EpochCompressor.fast()`); drop the kwarg quartet at lines 125-138.
- `tests/` — new `tests/test_polygram_tuning_passthrough.py` covering pipeline → ctx → action round-trip; updates to `tests/test_forge_pipeline.py` and `tests/test_actions_compress.py` for the renamed/added fields.
- Depends on: `polygram-tuning-config` (this change cannot land before polygram exposes the dataclasses and accepts `config=` on `Compressor` / `Regrower`).
- Migration: callers using `ForgePipeline(compression_strategy="merge", rep_selection="scale_aware")` rewrite to `ForgePipeline(compression=CompressionConfig(strategy="merge", rep_selection="scale_aware"))`. Callers omitting compression entirely keep working — `compression=None` continues to use polygram's defaults. Callers with `regrow_count > 0` MUST add `regrow=RegrowConfig(model_name=..., layer=...)`; that's the loud-fail point that catches silent GPT-2 assumptions.
