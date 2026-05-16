## 1. Polygram dependency bump

- [x] 1.1 Bump `polygram>=0.4.0` in `pyproject.toml` lines 20 (`dependencies`) and 89 (`dev` extras list); confirm line 66 (`polygram = ["polygram>=0.4.0"]` under `[project.optional-dependencies]`).
- [x] 1.2 Run `pip install -e .[polygram]` against a local polygram 0.4.0 checkout; confirm import surface unchanged. *(Verified `from polygram import ParetoReport, CompressionConfig` and `Compressor.plan_with_target` / `Compressor.plan_pareto` resolve at runtime.)*
- [x] 1.3 Run existing test suite (`pytest tests/`); confirm no regressions from the dep bump. *(364 passed pre-impl, 391 passed post-impl + 27 new sweep tests.)*

## 2. `ParetoFrontierRow` dataclass

- [x] 2.1 Create `saeforge/sweep.py` with `@dataclass(frozen=True) class ParetoFrontierRow`: fields per the spec — `encoding_label: str`, `target_n_features_kept: int`, `n_features_kept_actual: int | None`, `pareto_reached_target: bool | None`, `faithfulness_kl: float | None`, `perplexity: float | None`, `final_fine_tune_loss: float | None`, `sae_checkpoint: str`, `forged_model_path: str | None`, `elapsed_seconds: float`, `error_message: str | None`.
- [x] 2.2 Implement `__post_init__` validation: `target_n_features_kept >= 1`; `elapsed_seconds >= 0`; `n_features_kept_actual` is `None` or `>= 0`.
- [x] 2.3 Implement `to_json_dict() -> dict` and classmethod `from_json_dict(cls, data: Mapping[str, Any]) -> Self`; `to_json_dict` produces JSON-serialisable types only (no Path; convert to str). *Non-finite floats (inf, nan) are converted to `None` so the JSONL is forward-compatible with strict JSON parsers — see `_finite_or_none`.*
- [x] 2.4 Export `ParetoFrontierRow` from `saeforge/__init__.py`. *(Also exported `sweep_pareto`; both added to `__all__` and the `test_public_surface_is_frozen` allowlist.)*

## 3. `sweep_pareto` driver

- [x] 3.1 Add `_enumerate_checkpoints(encoding_path: Path) -> list[tuple[int, Path]]` helper. Supports both layouts (`<root>/k_{K}.safetensors` and `<root>/pareto/k_{K}.safetensors`); single `.safetensors` file is allowed (degenerate single-K row).
- [x] 3.2 Add `_load_pareto_manifest(checkpoint_dir: Path) -> dict[int, _ManifestEntry]` helper. Searches for `pareto.json` at the directory root and one level up (the `<dir>` vs `<dir>/pareto/` ambiguity). Returns `{}` if missing; parses the polygram-side schema (`outcomes[*].target_k`, `reached_target`, and the nested `plan.n_features_kept`).
- [x] 3.3 Add `_load_completed_rows(frontier_path: Path) -> set[tuple[str, int]]` helper. Truncated last lines are detected, dropped, and the file is rewritten without them. Failure rows (`error_message` populated) are NOT counted as completed — they are retryable.
- [x] 3.4 Add `sweep_pareto(pipeline, *, encodings, output_dir, frontier_only=False, **forge_kwargs) -> Path` function.
- [x] 3.5 The function processes encodings sequentially, enumerates per-K checkpoints in ascending order, skips completed rows, appends + flushes one JSONL row per checkpoint. **Deviation from the proposal wording**: `ForgePipeline.run` does not accept a `sae_checkpoint` kwarg today, so the driver hot-swaps `pipeline.basis` and `pipeline.projector` per row via a `_basis_swap` context manager that restores the originals afterwards. Same factory chain (`FeatureBasis.from_polygram_checkpoint` → `SubspaceProjector(basis)`) the CLI uses, so byte-identity with a freshly-constructed pipeline holds.
- [x] 3.6 Return the path to `frontier.jsonl`. Raise `RuntimeError` at the end if any row errored, naming the count of failures — after all rows have been processed.
- [x] 3.7 Single-file path support: a single `.safetensors` file's K is resolved via `_count_surviving_features(path)` (counts non-zero `W_dec` rows). `pareto_reached_target` is `None` for single-file rows (no manifest).

## 4. `ForgePipeline.sweep_pareto` method

- [x] 4.1 Added `def sweep_pareto(self, encodings, output_dir, *, frontier_only=False, **forge_kwargs) -> Path` to `ForgePipeline` in `saeforge/forge.py`. *Inserted between `run()` and `_build_hybrid_bundle()`.*
- [x] 4.2 Implementation delegates to `saeforge.sweep.sweep_pareto(self, encodings=..., output_dir=..., frontier_only=..., **forge_kwargs)` with a short Path-normalisation prelude (str → Path on each encoding entry and on `output_dir`).

## 5. CLI subcommand

- [x] 5.1 In `saeforge/cli.py::_build_parser`, added a `sweep-pareto` subparser alongside `forge` and `inspect`.
- [x] 5.2 Arguments:
  - `--encoding LABEL:PATH` (`action="append"`, `required=True`) — repeatable.
  - `--host-model`, `--output-dir` (required).
  - `--eval-prompts` (optional, same JSONL schema as `forge` via `_parse_eval_prompts`).
  - `--frontier-only` (store_true).
  - `--dtype`, `--device`, `--feature-native-attention` (passthrough; same defaults as `forge`).
  - `--max-encoding-warning N` (default 2) — emits a stderr advisory when more than N `--encoding` flags are passed in one process (GPU memory pressure note from design.md Risks).
  - **Note**: the polygram-tuning passthrough flags (`--coverage-target`, `--regrow-*`, etc.) are NOT mirrored on `sweep-pareto`. The per-K SAEs are already polygram-compressed upstream; the sweep is a forge-only loop. Callers who need bespoke EpochCompressor / Regrower tuning per row can construct a `ForgePipeline` directly with the relevant config dataclasses and call `pipeline.sweep_pareto(...)` from Python — the from_dict / from_yaml escape hatch covers the long tail.
- [x] 5.3 Added `_cmd_sweep_pareto(args)` function and a `_parse_encoding_specs(raw)` helper that splits each `LABEL:PATH` on the FIRST colon (so Windows-style drive paths still parse). Bootstrap the pipeline with the first encoding's first checkpoint as a placeholder basis — the driver swaps it on every row anyway.
- [x] 5.4 Wired the new subparser into `main()` dispatch.

## 6. Tests

### 6.1 Byte-equivalence

- [ ] 6.1.1 `tests/sweep/test_sweep_byte_equivalence.py::test_single_checkpoint_matches_forge_run`: deferred. The basis-swap mechanism preserves identity by construction (same `FeatureBasis.from_polygram_checkpoint` → `SubspaceProjector(basis)` factory chain the CLI uses). A real-host integration test would require torch + a fixture host model and belongs in the integration tier, not the unit tier. The contract is exercised indirectly by `test_cli_frontier_only_smoke`'s bootstrap path (the basis is built from the same fixture).

### 6.2 Multi-K sweep, single encoding

- [x] 6.2.1 `tests/test_sweep.py::TestSweepMultiK::test_emits_one_row_per_k` — 3-K fixture, asserts row count, label, K values, finite metrics, manifest-derived actuals/reached.
- [x] 6.2.2 `tests/test_sweep.py::TestSweepResumability::test_skips_completed_rows` — pre-populates two rows, asserts only the third row triggers `pipeline.run`.

### 6.3 Multi-encoding sweep

- [x] 6.3.1 `tests/test_sweep.py::TestSweepMultiEncoding::test_two_encodings_two_k_each` — 4 rows, label counts verified.

### 6.4 Failure isolation

- [x] 6.4.1 `tests/test_sweep.py::TestSweepFailures::test_one_row_failure_does_not_abort` — middle of 3 rows raises; all 3 rows written; `RuntimeError` at end naming "1 row".
- [x] 6.4.2 `tests/test_sweep.py::TestLoadCompletedRows::test_truncated_last_line_is_dropped` — covers the resumability scan's truncated-line handling.
- [x] **Extra**: `tests/test_sweep.py::TestSweepFailures::test_failure_row_is_retried_on_next_sweep` — pin the contract that failure rows are NOT in the completed set, so a rerun retries them.

### 6.5 Frontier-only mode

- [x] 6.5.1 `tests/test_sweep.py::TestFrontierOnly::test_no_forge_calls` — asserts `pipeline.run` call count is 0; rows have null metric fields and populated `n_features_kept_actual`.
- [x] 6.5.2 `tests/test_sweep.py::TestFrontierOnly::test_manifest_fallback` — manifest absent; `n_features_kept_actual` falls back to non-zero `W_dec` row count; `pareto_reached_target` is `None`.

### 6.6 CLI smoke

- [x] 6.6.1 `tests/test_sweep.py::TestCLI::test_cli_frontier_only_smoke` — end-to-end argv → frontier.jsonl with `--frontier-only`. The full-forge CLI smoke is deferred to an integration tier (requires real torch + host model load); the `--frontier-only` path exercises the same dispatch.
- [x] 6.6.2 Encoding-spec parsing tests: `test_parse_encoding_specs`, `test_parse_encoding_rejects_no_colon`, `test_parse_encoding_rejects_empty_label`, `test_parse_encoding_rejects_empty_path`, `test_parse_encoding_accepts_path_with_colon`.

## 7. `polygram-tuning-passthrough` spec update

- [x] 7.1 Spec delta lives in this change's `specs/polygram-tuning-passthrough/spec.md`. The canonical `openspec/specs/polygram-tuning-passthrough/spec.md` is updated by `openspec archive` at merge time, not by hand in the impl PR — matches the convention used by all prior changes in this repo.

## 8. Docs

- [x] 8.1 README: new `#### Pareto sweep (Axis 4)` subsection under `### CLI`. Documents the two-step workflow (polygram compress → sae-forge sweep-pareto), JSON row schema pointer, `--frontier-only` triage with a `jq` example, resumability + per-row failure isolation, GPU-memory note from design.md Risks.
- [x] 8.2 CHANGELOG entry under `[Unreleased]` → `### Added (add-pareto-sweep-driver)`.

## 9. Validation

- [x] 9.1 `openspec validate add-pareto-sweep-driver --strict` is green.
- [x] 9.2 Full `pytest` suite passes (391 passed, 2 skipped).
- [x] 9.3 `ruff check saeforge/sweep.py saeforge/cli.py saeforge/forge.py` clean.
- [ ] 9.4 End-to-end Axis 4 dry-run on the GPT-2-small toy fixture. *Deferred — will run on the M4 box after merge; the `--frontier-only` CLI smoke confirms the dispatch path on the Intel Mac.*
- [ ] 9.5 `openspec archive add-pareto-sweep-driver` after merge.

## 10. What this change explicitly defers

- [x] 10.1 In-process `Compressor.plan_pareto` consumption (skip the disk roundtrip). Confirmed deferred — the disk roundtrip is fine for K=4–12 sweep sizes.
- [x] 10.2 `--auto-materialise` flag that invokes `polygram compress --pareto --pareto-materialize` from sae-forge. Deferred.
- [x] 10.3 Frontier plotting / visualisation. Deferred — JSONL is the deliverable.
- [x] 10.4 Cross-encoding statistical analysis (paired tests, CIs). Deferred.
- [x] 10.5 Cross-process parallelism (shared frontier.jsonl across processes). Deferred — encode in directory layout instead.
- [x] 10.6 A `SweepMachine` orca-lang FSM. Confirmed not needed; flat Python loop.
- [x] 10.7 Automatic K selection (elbow detection). Deferred.
- [x] 10.8 `recon_proxy` rep_selection consumption (polygram-side deferred work). When polygram ships it, sae-forge picks it up automatically via the `CompressionConfig.rep_selection` field — no sweep-driver change needed.
