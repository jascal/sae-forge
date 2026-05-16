## 1. `saeforge/forge_quality.py` module

- [x] 1.1 Create `saeforge/forge_quality.py` with:
  - `@dataclass(frozen=True) class QualityThresholds`: fields `saturated: float = 1.0`, `good: float = 0.5`, `undersized: float = 0.0625`. `__post_init__` enforces `saturated > good > undersized >= 0`.
  - `class QualityTier(str, Enum)`: members `SATURATED = "saturated"`, `GOOD = "good"`, `UNDERSIZED = "undersized"`, `DEGENERATE = "degenerate"`.
- [x] 1.2 Add `compute_basis_rank(W_dec_kept: np.ndarray) -> int` — wraps `numpy.linalg.matrix_rank` with default tolerance; raises `ValueError` if input has 0 rows.
- [x] 1.3 Add `classify_quality(basis_rank: int, host_d_model: int, thresholds: QualityThresholds | None = None) -> tuple[float, QualityTier]` returning `(ratio, tier)`. Uses default `QualityThresholds()` when `thresholds is None`.
- [x] 1.4 Add `resolve_host_d_model(host_model_id: str) -> int | None` — lazy-imports `transformers.AutoConfig`, fetches `hidden_size`. Returns `None` on network failure or unsupported host (logged at stderr; not raised).
- [x] 1.5 Add `advise_sweep_quality(encodings, host_d_model, thresholds, *, manifest_loader, basis_rank_loader) -> str | None`. For each `(label, path)` in `encodings`, find the smallest K from the manifest, compute that K's basis rank, classify against thresholds. Return a formatted multi-line stderr-ready string when ANY encoding's smallest-K tier is `undersized` or `degenerate`; otherwise return `None`. The `manifest_loader` and `basis_rank_loader` arguments are injectable for testing (production wiring uses `_load_pareto_manifest` and a `safetensors`-backed loader).
- [x] 1.6 Export `QualityTier`, `QualityThresholds`, `compute_basis_rank`, `classify_quality` from `saeforge/__init__.py`.

## 2. `ParetoFrontierRow` schema extension

- [x] 2.1 Add four new fields to `ParetoFrontierRow` (`saeforge/sweep.py`), all defaulting to `None`: `host_d_model: int | None`, `basis_rank: int | None`, `quality_ratio: float | None`, `quality_tier: str | None`.
- [x] 2.2 Update `to_json_dict` / `from_json_dict` to round-trip the new fields. `from_json_dict` SHALL tolerate dicts missing the new keys.
- [x] 2.3 Update the row schema table in `openspec/changes/add-forge-quality-diagnostics/specs/pareto-sweep/spec.md` to include the new fields and their nullability per lifecycle state.
- [x] 2.4 `__post_init__` validation: when `quality_tier` is not None, it SHALL be one of the four `QualityTier` string values; when `quality_ratio` is not None, it SHALL be `>= 0`.

## 3. Sweep driver wiring

- [x] 3.1 Extend `sweep_pareto(...)` signature with `quality_floor: float | None = None`, `quality_thresholds: QualityThresholds | None = None`, `host_d_model_override: int | None = None`.
- [x] 3.2 At the top of `sweep_pareto`:
  - Resolve `host_d_model = host_d_model_override or resolve_host_d_model(pipeline.host_model_id)`. If `None`, skip the advisory and proceed (diagnostics fields will be `None` for every row).
  - When `host_d_model` is known: build the advisory via `advise_sweep_quality(...)`, print to stderr if non-None.
  - When `quality_floor is not None`: AFTER printing the advisory, check whether any encoding's smallest-K ratio falls below the floor; if so, raise `RuntimeError("sweep_pareto: quality_floor=N rejected ...")` BEFORE any forge runs.
- [x] 3.3 In `_process_row`, when `host_d_model` is known: after building the basis (via `_basis_swap` or the existing read path), compute `basis_rank` from the loaded `W_dec` and populate `quality_ratio` + `quality_tier`. Populate ALL FOUR diagnostic fields on the emitted row.
- [x] 3.4 The diagnostic fields are populated REGARDLESS of forge success/failure — diagnostics matter most for failure rows (telling the analyst whether the failure was structurally doomed).
- [x] 3.5 Frontier-only rows ALSO get diagnostic fields populated (the `n_features_kept_actual` count + the sweep's `host_d_model` are enough; basis_rank requires reading `W_dec`, which `--frontier-only` already does for its `_count_surviving_features` fallback path — extend it to also return rank).

## 4. CLI surface

- [x] 4.1 Add `--quality-floor FLOAT` to the `sweep-pareto` subparser. When set, parsed as a float in `[0, 1]`; refused if outside.
- [x] 4.2 Add `--quality-tier-thresholds STR` to the `sweep-pareto` subparser. Parsed as `name:value` comma-separated pairs (e.g., `saturated:1.0,good:0.5,undersized:0.0625`). The parser maps to a `QualityThresholds` instance with `__post_init__` validation.
- [x] 4.3 `_cmd_sweep_pareto` plumbs the new flags into `pipeline.sweep_pareto(...)` via the new kwargs.
- [x] 4.4 Update CLI help text to mention the advisory behaviour, the optional floor refusal, and a one-line note that `degenerate` describes rank ratio, not run validity.

## 5. `ForgePipeline.sweep_pareto` pass-through

- [x] 5.1 Extend `ForgePipeline.sweep_pareto(...)` with the same new kwargs; delegate unchanged.

## 6. Tests

### 6.1 `forge_quality` module

- [x] 6.1.1 `tests/test_forge_quality.py::test_compute_basis_rank_full_rank`: 8×64 random matrix → rank 8.
- [x] 6.1.2 `test_compute_basis_rank_with_zero_rows`: 8×64 matrix where 3 rows are zero → rank 5 (the implementation reads `W_dec_kept` already restricted to surviving rows, but defensive test).
- [x] 6.1.3 `test_compute_basis_rank_linearly_dependent`: 4 rows where one is a multiple of another → rank 3.
- [x] 6.1.4 `test_classify_quality_boundaries`: ratio of 1.0, 0.99, 0.5, 0.49, 0.0625, 0.06 → expected tier each. Use default thresholds.
- [x] 6.1.5 `test_classify_quality_custom_thresholds`: pass non-default `QualityThresholds(saturated=2.0, good=1.0, undersized=0.25)`; verify the boundaries shift.
- [x] 6.1.6 `test_quality_thresholds_validates_ordering`: `QualityThresholds(saturated=0.5, good=1.0)` raises `ValueError` (saturated must exceed good).
- [x] 6.1.7 `test_resolve_host_d_model_gpt2`: assert `resolve_host_d_model("gpt2") == 768` (gated on network — `pytest.importorskip("transformers")` and skip if offline).
- [x] 6.1.8 `test_resolve_host_d_model_returns_none_on_failure`: monkeypatch `AutoConfig.from_pretrained` to raise; assert `resolve_host_d_model("bogus")` returns `None` and stderr contains a warning.

### 6.2 Advisory

- [x] 6.2.1 `test_advise_sweep_quality_silent_on_good_setup`: one encoding whose smallest-K rank ≥ host_d_model/2 → returns `None` (no advisory).
- [x] 6.2.2 `test_advise_sweep_quality_warns_on_degenerate`: smallest-K rank << host_d_model/16 → returned string contains the encoding label, smallest K, computed ratio, and a suggested K floor.
- [x] 6.2.3 `test_advise_sweep_quality_suggested_floor_from_manifest`: with a 4-K manifest where K=16's rank ≥ host_d_model/2, the advisory's suggested floor is `K=16` (the smallest K above the good threshold).
- [x] 6.2.4 `test_advise_sweep_quality_multi_encoding`: advisory emitted only for the degenerate encodings; good encodings absent from the message.

### 6.3 Row schema

- [x] 6.3.1 `test_pareto_frontier_row_round_trip_with_diagnostics`: populated row with all four diagnostic fields round-trips via `to_json_dict` / `from_json_dict`.
- [x] 6.3.2 `test_pareto_frontier_row_missing_diagnostic_keys`: dict missing the four new keys → `from_json_dict` returns instance with `None` for those fields (backwards compat).
- [x] 6.3.3 `test_pareto_frontier_row_rejects_invalid_quality_tier`: `quality_tier="bogus"` → `ValueError`.

### 6.4 Sweep integration

- [x] 6.4.1 `test_sweep_populates_diagnostic_fields`: mocked pipeline.run with a known `W_dec` shape; assert resulting rows have `basis_rank`, `host_d_model`, `quality_ratio`, `quality_tier` populated and consistent.
- [x] 6.4.2 `test_sweep_skips_diagnostics_when_d_model_unresolvable`: monkeypatch `resolve_host_d_model` to return `None`; assert sweep proceeds, no advisory printed, rows have all four diagnostic fields as `None`.
- [x] 6.4.3 `test_sweep_quality_floor_refuses_degenerate_setup`: simulate degenerate smallest-K basis; invoke with `quality_floor=0.5`; assert `RuntimeError` is raised BEFORE any forge calls (mock pipeline.run; assert call count is 0).
- [x] 6.4.4 `test_sweep_quality_floor_accepts_good_setup`: with `quality_floor=0.5` and a basis whose smallest K is in `good`/`saturated`, sweep proceeds normally.
- [x] 6.4.5 `test_sweep_advisory_does_not_refuse_by_default`: degenerate setup without `--quality-floor` prints stderr advisory but sweep runs all rows to completion (mock pipeline.run; assert call count > 0).
- [x] 6.4.6 `test_failure_rows_still_carry_diagnostics`: pipeline.run raises on a row; the resulting JSONL row has `error_message` populated AND `basis_rank` / `quality_tier` populated.

### 6.5 CLI

- [x] 6.5.1 `test_cli_quality_floor_parses_float`: argv with `--quality-floor 0.5` parses to the kwarg correctly.
- [x] 6.5.2 `test_cli_quality_floor_rejects_out_of_range`: `--quality-floor 1.5` exits non-zero with an error message.
- [x] 6.5.3 `test_cli_quality_tier_thresholds_parses`: `--quality-tier-thresholds saturated:1.0,good:0.5,undersized:0.0625` parses to the right `QualityThresholds` instance.
- [x] 6.5.4 `test_cli_quality_tier_thresholds_malformed`: `--quality-tier-thresholds bogus` exits non-zero.

## 7. Spec delta

- [x] 7.1 Author `specs/pareto-sweep/spec.md` delta: MODIFIED `ParetoFrontierRow` requirement to include the four new diagnostic fields with nullability per lifecycle state; MODIFIED `CLI subcommand` requirement to include `--quality-floor` and `--quality-tier-thresholds` flags; ADDED new `Pre-flight quality advisory` requirement.

## 8. Docs

- [x] 8.1 Extend the `#### Pareto sweep (Axis 4)` README section with a short subsection on the quality diagnostics — what `quality_tier` means, the rule-of-thumb thresholds, the recommended workflow ("look at quality_tier before reading faithfulness_kl"). Include the `jq` filter idiom: `jq 'select(.quality_tier == "good" or .quality_tier == "saturated") | .faithfulness_kl' frontier.jsonl`.
- [x] 8.2 CHANGELOG entry under `[Unreleased]` → `### Added (add-forge-quality-diagnostics)`.

## 9. Validation

- [x] 9.1 `openspec validate add-forge-quality-diagnostics --strict` is green.
- [x] 9.2 Full `pytest` suite passes; new tests cover §6.1 through §6.5.
- [x] 9.3 `ruff check` clean on touched files.
- [x] 9.4 Live MBP smoke: re-run the existing N=32 Rung4 sweep with the new fields populated; confirm rows tag as `degenerate` (basis_rank=1, ratio=1/768 ≈ 0.0013) and the pre-flight advisory suggests a K floor.
- [x] 9.5 `openspec archive add-forge-quality-diagnostics` after merge.

## 10. What this change explicitly defers

- [x] 10.1 Predictive KL estimation from structural inputs. The diagnostic is a *structural* signal; correlating it with post-forge KL is a research project.
- [x] 10.2 Tier thresholds calibrated from real cross-host data. The defaults are GPT-2-era rule-of-thumb; revisit after a real Axis-4 dataset accumulates.
- [x] 10.3 SVD/condition-number-aware rank computation. `matrix_rank` default tolerance is adequate; revisit if measurements show otherwise.
- [x] 10.4 Eval-prompt-corpus quality diagnostics (warn when `--eval-prompts` has too few tokens to drive meaningful KL). Different axis; deserves its own proposal.
- [x] 10.5 Auto-rewriting `--pareto` lists to enforce the floor. The advisory prints a suggestion; rewriting is a UX foot-gun.
- [x] 10.6 Per-K advisories (current advisory only examines smallest K). Monotonic argument means it's sufficient; per-K extension is polish.
- [x] 10.7 Surfacing diagnostics in `saeforge inspect` for ad-hoc basis examination. Possible follow-up; out of scope here.
