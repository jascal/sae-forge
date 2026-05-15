## 1. Polygram dependency bump

- [ ] 1.1 Bump `polygram>=0.4.0` in `pyproject.toml` lines 20 (`dependencies`) and 89 (`dev` extras list); confirm line 66 (`polygram = ["polygram>=0.4.0"]` under `[project.optional-dependencies]`).
- [ ] 1.2 Run `pip install -e .[polygram]` against a local polygram 0.4.0 checkout; confirm import surface unchanged.
- [ ] 1.3 Run existing test suite (`pytest tests/`); confirm no regressions from the dep bump.

## 2. `ParetoFrontierRow` dataclass

- [ ] 2.1 Create `saeforge/sweep.py` with `@dataclass(frozen=True) class ParetoFrontierRow`: fields per the spec — `encoding_label: str`, `target_n_features_kept: int`, `n_features_kept_actual: int | None`, `pareto_reached_target: bool | None`, `faithfulness_kl: float | None`, `perplexity: float | None`, `final_fine_tune_loss: float | None`, `sae_checkpoint: str`, `forged_model_path: str | None`, `elapsed_seconds: float`, `error_message: str | None`.
- [ ] 2.2 Implement `__post_init__` validation: `target_n_features_kept >= 1`; `elapsed_seconds >= 0`; `n_features_kept_actual` is `None` or `>= 0`.
- [ ] 2.3 Implement `to_json_dict() -> dict` and classmethod `from_json_dict(cls, data: Mapping[str, Any]) -> Self`; `to_json_dict` produces JSON-serialisable types only (no Path; convert to str).
- [ ] 2.4 Export `ParetoFrontierRow` from `saeforge/__init__.py`.

## 3. `sweep_pareto` driver

- [ ] 3.1 Add `_enumerate_checkpoints(encoding_spec: str) -> list[tuple[int, Path]]` helper in `saeforge/sweep.py`. Accepts `LABEL:PATH`; splits on first `:`; if `PATH` is a directory, globs `k_{K}.safetensors` under it (or under a `pareto/` subdir produced by `polygram compress --pareto-materialize --out <dir>`) and parses K from each filename via regex `k_(\d+)\.safetensors`; if `PATH` is a single file, treats it as K=`None`-sentinel (the row's `target_n_features_kept` is read from SAE metadata, see §3.7).
- [ ] 3.2 Add `_load_pareto_manifest(checkpoint_dir: Path) -> dict[int, dict]` helper that reads `<dir>/pareto.json` if present; returns `{K: {"n_features_kept_actual": ..., "pareto_reached_target": ...}, ...}`. Returns `{}` if missing.
- [ ] 3.3 Add `_load_completed_rows(frontier_path: Path) -> set[tuple[str, int]]` helper that reads existing `frontier.jsonl` (if any) and returns the set of `(encoding_label, target_n_features_kept)` tuples already present. Handles a truncated last line by discarding it and rewriting the file without it.
- [ ] 3.4 Add `sweep_pareto(pipeline: ForgePipeline, *, encodings: list[tuple[str, Path]], output_dir: Path, frontier_only: bool = False, **forge_kwargs) -> Path` function.
- [ ] 3.5 The function SHALL:
  - Create `output_dir` if absent.
  - Load completed rows from `output_dir/frontier.jsonl`.
  - For each `(label, path)` in `encodings`, enumerate checkpoints, load the manifest, iterate K values ascending-by-K.
  - Skip rows already in completed-set.
  - If `frontier_only=True`: emit a row with metric fields `None`, populated only from manifest.
  - Otherwise: call `pipeline.run(sae_checkpoint=ckpt_path, output_dir=output_dir / label / f"k_{K}", **forge_kwargs)` inside a `try/except`; on success, populate metric fields from the returned `ForgeResult`; on exception, populate `error_message` with the exception's repr and leave metrics `None`.
  - Append the row to `frontier.jsonl` and `f.flush()` after each row.
- [ ] 3.6 Return the path to `frontier.jsonl`. Raise `RuntimeError` at the end if any row errored, naming the count of failures — but only AFTER all rows have been processed.
- [ ] 3.7 Single-file path support: if a `--encoding label:single.safetensors` is supplied, treat it as one row whose K is read from the SAE checkpoint metadata via `polygram.sae_import._load_sae_checkpoint(path, ["W_dec"])` and counting non-zero rows. This is the byte-equivalence path for §6.1.

## 4. `ForgePipeline.sweep_pareto` method

- [ ] 4.1 Add `def sweep_pareto(self, encodings: list[tuple[str, Path]], output_dir: Path, *, frontier_only: bool = False, **forge_kwargs) -> Path` to `ForgePipeline` in `saeforge/forge.py`.
- [ ] 4.2 Implementation is a one-line delegation: `return saeforge.sweep.sweep_pareto(self, encodings=encodings, output_dir=output_dir, frontier_only=frontier_only, **forge_kwargs)`.

## 5. CLI subcommand

- [ ] 5.1 In `saeforge/cli.py::_build_parser`, add a `sweep-pareto` subparser alongside `forge` and `inspect`.
- [ ] 5.2 Arguments:
  - `--encoding LABEL:PATH` (action="append", required=True, at least one) — repeatable.
  - `--host-model` (required) — same as `forge`.
  - `--output-dir` (required) — same as `forge`.
  - `--eval-prompts` (optional) — same JSONL schema as `forge` (commit e1f1246 dict-shorthand + raw text).
  - `--frontier-only` (action="store_true") — skip forge runs.
  - All standard `forge` passthrough knobs: `--dtype`, `--device`, `--finetune-steps`, `--seed`, `--finetune-distill-alpha`, `--finetune-distill-temperature`, `--compression-score-field`, etc. Mirror the existing `forge` parser; share where reasonable via a helper.
- [ ] 5.3 Add `_cmd_sweep_pareto(args)` function that:
  - Parses each `--encoding LABEL:PATH` string.
  - Constructs `ForgePipeline(host_model_id=args.host_model, ...)` with the same dtype/device/finetune-knob plumbing as `_cmd_forge`.
  - Calls `pipeline.sweep_pareto(encodings, output_dir, frontier_only=args.frontier_only, eval_prompts=..., ...)`.
  - Prints the path to `frontier.jsonl` on stdout and returns 0; non-zero if `sweep_pareto` raised at the end (i.e. some rows errored).
- [ ] 5.4 Wire the new subparser into `main()` dispatch.

## 6. Tests

### 6.1 Byte-equivalence

- [ ] 6.1.1 `tests/sweep/test_sweep_byte_equivalence.py::test_single_checkpoint_matches_forge_run`: build a single-K materialised SAE (or use the existing toy fixture), invoke `sweep_pareto([("toy", single_ckpt_path)], ...)`, parse the resulting JSONL row, and compare byte-for-byte to the artifacts from a vanilla `pipeline.run(single_ckpt_path, ...)`. Assert the forged model `state_dict()` is bit-equal.

### 6.2 Multi-K sweep, single encoding

- [ ] 6.2.1 `tests/sweep/test_sweep_multi_k.py::test_multi_k_emits_one_row_per_k`: fixture is a directory with 3 `k_{K}.safetensors` files + a `pareto.json` manifest. Invoke `sweep_pareto([("toy", dir)], ...)`. Assert `frontier.jsonl` has 3 rows, ascending K, each with finite metric fields and `error_message is None`.
- [ ] 6.2.2 `tests/sweep/test_sweep_multi_k.py::test_resumability`: pre-populate `frontier.jsonl` with rows for K=200 and K=500; invoke the sweep again with K=200,500,1000; assert only K=1000 was forged (mock `pipeline.run` to count calls).

### 6.3 Multi-encoding sweep

- [ ] 6.3.1 `tests/sweep/test_sweep_multi_encoding.py::test_two_encodings_two_k_each`: two encodings, two K values each; assert 4 rows in `frontier.jsonl`, both `encoding_label` values present, K values correct per label.

### 6.4 Failure isolation

- [ ] 6.4.1 `tests/sweep/test_sweep_failures.py::test_one_row_failure_does_not_abort`: mock `pipeline.run` to raise on the second of 3 rows. Assert all 3 rows are present in `frontier.jsonl`, the middle has `error_message` populated and metrics `None`, and `sweep_pareto` raises `RuntimeError` at the end naming 1 failure.
- [ ] 6.4.2 `tests/sweep/test_sweep_failures.py::test_truncated_jsonl_recovers`: write a `frontier.jsonl` with a truncated last line; invoke the sweep; assert the truncated line is dropped, the sweep continues, and the resulting file is parseable.

### 6.5 Frontier-only mode

- [ ] 6.5.1 `tests/sweep/test_sweep_frontier_only.py::test_frontier_only_no_forge_calls`: invoke with `frontier_only=True`; assert `pipeline.run` is never called; assert each JSONL row has `target_n_features_kept` and `n_features_kept_actual` populated and `faithfulness_kl is None`.
- [ ] 6.5.2 `tests/sweep/test_sweep_frontier_only.py::test_frontier_only_missing_manifest_falls_back`: remove the `pareto.json`; assert the driver counts surviving features from each `k_{K}.safetensors` directly and populates `n_features_kept_actual` (with `pareto_reached_target` as `None` — unknown without manifest).

### 6.6 CLI smoke

- [ ] 6.6.1 `tests/cli/test_sweep_pareto_cli.py::test_cli_smoke`: subprocess invocation of `saeforge sweep-pareto --encoding toy:<fixture-dir> --host-model gpt2 --output-dir <tmp> --finetune-steps 2`; assert exit 0 and that `<tmp>/frontier.jsonl` exists with the expected number of rows.
- [ ] 6.6.2 `tests/cli/test_sweep_pareto_cli.py::test_cli_frontier_only_smoke`: same but `--frontier-only`; assert exit 0 and rows present with manifest-only fields populated.

## 7. `polygram-tuning-passthrough` spec update

- [ ] 7.1 Update `openspec/specs/polygram-tuning-passthrough/spec.md` via the delta in this change's `specs/polygram-tuning-passthrough/spec.md`: pin `polygram>=0.4.0`, add a scenario covering `CompressionConfig(target_n_features_kept=K)` round-tripping through ctx.

## 8. Docs

- [ ] 8.1 New section in `README.md` (or `docs/sweep.md` if `docs/` is the convention) covering the two-step workflow: `polygram compress --pareto-materialize` → `saeforge sweep-pareto`. Include the Axis 4 framing.
- [ ] 8.2 `CHANGELOG.md` entry under unreleased: "**Pareto sweep driver** — new `saeforge sweep-pareto` CLI subcommand and `ForgePipeline.sweep_pareto()` method that runs the forge pipeline across a sequence of per-K materialised SAE checkpoints from `polygram compress --pareto-materialize`. Emits a JSONL frontier with `(encoding, K, n_features_kept, KL, perplexity)` per row. Resumable and per-row-failure-isolated. Requires `polygram>=0.4.0`."

## 9. Validation

- [ ] 9.1 `openspec validate add-pareto-sweep-driver --strict` is green.
- [ ] 9.2 Full `pytest` suite passes.
- [ ] 9.3 `ruff check saeforge/sweep.py saeforge/cli.py saeforge/forge.py` clean.
- [ ] 9.4 Run an end-to-end Axis 4 dry-run on the GPT-2-small toy fixture: 1 encoding × 3 K values; confirm `frontier.jsonl` has 3 finite rows.
- [ ] 9.5 `openspec archive add-pareto-sweep-driver` after merge.

## 10. What this change explicitly defers

- [ ] 10.1 In-process `Compressor.plan_pareto` consumption (skip the disk roundtrip). Defer until measurements show the disk cost is binding.
- [ ] 10.2 `--auto-materialise` flag that invokes `polygram compress --pareto --pareto-materialize` from sae-forge. Defer until the two-step friction is documented as a real complaint.
- [ ] 10.3 Frontier plotting / visualisation. JSONL is the deliverable; plots are an analyst concern.
- [ ] 10.4 Cross-encoding statistical analysis (paired tests, CIs). Emit the data; analyst computes the verdict.
- [ ] 10.5 Cross-process parallelism (shared frontier.jsonl across processes). Encode parallelism in directory layout instead.
- [ ] 10.6 A `SweepMachine` orca-lang FSM. The sweep is flat Python; an FSM would be ceremony without payoff.
- [ ] 10.7 Automatic K selection (elbow detection). Caller chooses K upstream in the polygram CLI.
- [ ] 10.8 `recon_proxy` rep_selection consumption (the polygram-side deferred work). When polygram ships it, sae-forge picks it up automatically via the `CompressionConfig.rep_selection` field — no sweep-driver change needed.
