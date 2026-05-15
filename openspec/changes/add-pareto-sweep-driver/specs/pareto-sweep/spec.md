# pareto-sweep Specification

## Purpose

Sweep the forge pipeline across a sequence of per-K materialised SAE checkpoints produced by `polygram compress --pareto-materialize`, optionally spanning multiple labelled encodings, and emit one JSONL row per `(encoding, target_n_features_kept)` capturing the kept-feature count, downstream KL, perplexity, and faithfulness. Enables Axis 4 of the polygram rung-viability methodology — cross-encoding Pareto-frontier comparison in forged-model output space — without bespoke scripting per analysis.

The sweep is sequential, resumable, and isolates per-row failures. The actual compression / Pareto planning happens upstream in polygram; sae-forge consumes the materialised artifacts.

## ADDED Requirements

### Requirement: ParetoFrontierRow dataclass

The `saeforge.sweep.ParetoFrontierRow` SHALL be a frozen dataclass with fields:

- `encoding_label: str` — caller-supplied label from `--encoding LABEL:PATH`.
- `target_n_features_kept: int` — the K requested upstream of polygram.
- `n_features_kept_actual: int | None` — the actual count from the polygram plan manifest (or counted from the SAE checkpoint if manifest absent); `None` if undetermined.
- `pareto_reached_target: bool | None` — from the polygram plan manifest; `None` if manifest absent.
- `faithfulness_kl: float | None` — post-forge KL(host ‖ forged); `None` on row failure or `--frontier-only`.
- `perplexity: float | None` — post-forge perplexity on the eval set; `None` on row failure or `--frontier-only`.
- `final_fine_tune_loss: float | None` — last training-loss value; `None` on row failure or `--frontier-only`.
- `sae_checkpoint: str` — absolute path to the per-K SAE the forge consumed.
- `forged_model_path: str | None` — absolute path to the forged-model output dir; `None` on row failure or `--frontier-only`.
- `elapsed_seconds: float` — wall-clock for the forge run; `0.0` for `--frontier-only` and failure-before-forge cases.
- `error_message: str | None` — the exception's `repr()` if the forge raised; `None` on success.

The class SHALL expose `.to_json_dict()` and classmethod `.from_json_dict(cls, data: Mapping[str, Any]) -> Self`. `__post_init__` SHALL validate `target_n_features_kept >= 1` and `elapsed_seconds >= 0`.

#### Scenario: ParetoFrontierRow is importable from saeforge

- **WHEN** `from saeforge import ParetoFrontierRow` is executed
- **THEN** the name resolves to the dataclass defined in `saeforge.sweep`

#### Scenario: ParetoFrontierRow rejects invalid target

- **WHEN** `ParetoFrontierRow(encoding_label="x", target_n_features_kept=0, ...)` is constructed
- **THEN** `__post_init__` raises `ValueError` naming `target_n_features_kept`

#### Scenario: ParetoFrontierRow round-trips through JSON

- **WHEN** a populated row is serialised via `.to_json_dict()`, passed through `json.dumps` / `json.loads`, and reconstructed via `from_json_dict(...)`
- **THEN** the reconstructed instance equals the original

### Requirement: ForgePipeline exposes sweep_pareto

`ForgePipeline` SHALL expose `sweep_pareto(self, encodings: list[tuple[str, Path]], output_dir: Path, *, frontier_only: bool = False, **forge_kwargs) -> Path`. The method SHALL delegate to a module-level `saeforge.sweep.sweep_pareto(pipeline, encodings, output_dir, frontier_only, **forge_kwargs)` function so the orchestration logic is testable without a fully-constructed pipeline.

The function SHALL:

1. Create `output_dir` if absent.
2. Load `(encoding_label, target_n_features_kept)` tuples from any existing `output_dir/frontier.jsonl`, treating a truncated last line as not-yet-recorded.
3. Enumerate each encoding's SAE checkpoints (single file or directory of `k_{K}.safetensors`), reading the optional `pareto.json` manifest for `n_features_kept_actual` and `pareto_reached_target`.
4. For each `(label, K, ckpt_path)`, skip if `(label, K)` is in the completed set.
5. If `frontier_only`: emit a row with metric fields `None` and metadata fields populated from manifest (or SAE checkpoint if manifest absent).
6. Otherwise: invoke `pipeline.run(sae_checkpoint=ckpt_path, output_dir=output_dir/label/f"k_{K}", **forge_kwargs)` inside a `try/except Exception`; on success, populate metric fields from the result; on exception, populate `error_message` and leave metrics `None`.
7. Append the row to `frontier.jsonl` and flush after each row.
8. Return the `Path` to `frontier.jsonl`.
9. Raise `RuntimeError` at the end if any row's `error_message is not None`, naming the failure count — but only after all rows have been processed.

#### Scenario: sweep_pareto byte-equivalence with single-checkpoint forge

- **GIVEN** `pipeline = ForgePipeline(host_model_id="gpt2", ...)` and a single-K SAE checkpoint produced by polygram
- **WHEN** `pipeline.sweep_pareto([("toy", single_ckpt_path)], output_dir=tmp, **forge_kwargs)` is invoked
- **THEN** the resulting `frontier.jsonl` contains exactly one row, and the forged-model `state_dict()` written under `tmp/toy/k_K/` is byte-identical to the output of `pipeline.run(single_ckpt_path, output_dir=..., **forge_kwargs)`

#### Scenario: sweep_pareto skips completed rows

- **GIVEN** `output_dir/frontier.jsonl` already contains rows for `(label="mps", K=200)` and `(label="mps", K=500)`
- **WHEN** `sweep_pareto(...)` is invoked with `encodings` covering K=200, 500, 1000 for the same label
- **THEN** only K=1000 triggers `pipeline.run`; the resulting JSONL contains all three rows (two skipped, one new)

#### Scenario: sweep_pareto isolates per-row failures

- **GIVEN** `pipeline.run` is configured to raise on the second of three rows
- **WHEN** `sweep_pareto(...)` runs three K values
- **THEN** the resulting `frontier.jsonl` contains three rows; the middle row has `error_message` populated and all metric fields `None`; the sweep function raises `RuntimeError` at the end naming "1 row failed"; the first and third rows have finite metric fields

#### Scenario: sweep_pareto recovers from a truncated last line

- **GIVEN** `output_dir/frontier.jsonl` whose last line is a partial JSON record (e.g. a crashed mid-write)
- **WHEN** `sweep_pareto(...)` is invoked
- **THEN** the truncated line is discarded from the resumability set and the file before appending; the sweep continues normally; the final file parses cleanly via `json.loads(line) for line in open(...)`

### Requirement: sweep_pareto supports frontier-only mode

When invoked with `frontier_only=True`, `sweep_pareto` SHALL enumerate the same checkpoints but SHALL NOT invoke `pipeline.run`. Each emitted row SHALL have `faithfulness_kl`, `perplexity`, `final_fine_tune_loss`, and `forged_model_path` set to `None`; `target_n_features_kept` and `n_features_kept_actual` SHALL be populated from the `pareto.json` manifest, or from the SAE checkpoint's surviving-feature count when the manifest is absent (in which case `pareto_reached_target` is `None`).

#### Scenario: frontier_only never calls pipeline.run

- **GIVEN** a sweep over 4 K values with `pipeline.run` instrumented to count calls
- **WHEN** `sweep_pareto(..., frontier_only=True)` runs
- **THEN** the call counter is zero, and the resulting `frontier.jsonl` has 4 rows with `faithfulness_kl is None` and `n_features_kept_actual` populated

#### Scenario: frontier_only falls back when manifest is missing

- **GIVEN** a checkpoint directory with `k_500.safetensors` and `k_1000.safetensors` but no `pareto.json`
- **WHEN** `sweep_pareto(..., frontier_only=True)` runs
- **THEN** each row's `n_features_kept_actual` is populated by counting non-zero feature rows in the corresponding SAE checkpoint; `pareto_reached_target` is `None`

### Requirement: CLI subcommand `saeforge sweep-pareto`

The `saeforge` CLI SHALL expose a `sweep-pareto` subcommand alongside the existing `forge` and `inspect` subcommands. Required arguments:

- `--encoding LABEL:PATH` — repeatable (`action="append"`); at least one required. `PATH` is a `.safetensors` file or a directory containing `k_{K}.safetensors` files (with an optional `pareto.json` manifest at the directory root or under a `pareto/` subdirectory).
- `--host-model ID` — host transformer id (same semantic as `forge`).
- `--output-dir DIR` — sweep output root.

Optional arguments include `--eval-prompts PATH`, `--frontier-only`, plus passthrough of the same fine-tune / dtype / device knobs as the `forge` subcommand.

#### Scenario: CLI parses repeatable --encoding

- **WHEN** `saeforge sweep-pareto --encoding mps:/path/to/mps --encoding rung4:/path/to/rung4 --host-model gpt2 --output-dir /tmp/out --frontier-only` is invoked
- **THEN** the parser produces a two-element encoding list `[("mps", Path("/path/to/mps")), ("rung4", Path("/path/to/rung4"))]` and the dispatch routes to the sweep handler

#### Scenario: CLI exits non-zero if any row errored

- **GIVEN** a sweep where one row's forge raises
- **WHEN** the CLI subcommand completes
- **THEN** the process exits with a non-zero status; stdout contains the path to `frontier.jsonl`; stderr names the failure count

#### Scenario: CLI emits frontier path on success

- **GIVEN** a sweep where all rows succeed
- **WHEN** the CLI subcommand completes
- **THEN** the process exits 0; stdout contains the absolute path of `frontier.jsonl`

### Requirement: Resumability via append-only JSONL

`sweep_pareto` SHALL write `frontier.jsonl` append-only, flushing after each row. Resumability SHALL be implemented by scanning the existing file for `(encoding_label, target_n_features_kept)` tuples; no separate lockfile, sentinel, or sidecar state. A truncated last line (one that fails `json.loads`) SHALL be discarded and rewritten out of the file before any new rows are appended, so the resulting file is always cleanly parseable.

#### Scenario: append-only writes survive interrupted runs

- **GIVEN** a sweep where the process is killed mid-row (after some rows have been written and flushed)
- **WHEN** the sweep is re-invoked with the same arguments
- **THEN** rows already in `frontier.jsonl` are skipped; new rows are appended in order; the final file's row count equals the number of unique `(label, K)` pairs in the input
