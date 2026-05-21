## 1. `saeforge/polygram_diagnostics.py` module

- [ ] 1.1 Create `saeforge/polygram_diagnostics.py` with module-level docstring linking back to this proposal and to `AGENTS.md`'s polygram dependency contract.
- [ ] 1.2 Add `load_polygram_report(checkpoint_path: str | Path) -> dict | None`. Reuses the suffix-list constants from `saeforge.basis` (`_compression_report.json`, `.compression_report.json`, `_report.json`) — DO NOT duplicate the list; import or re-export. Returns the parsed JSON dict on success, `None` on any failure (missing report, decode error). Logs at INFO via `saeforge.utils.logging`.
- [ ] 1.3 Add `compute_redundancy_ratio(n_clusters: int | None, n_zeroed: int | None) -> float | None`. Returns `n_zeroed / (n_clusters + n_zeroed)` when both are non-None, non-negative, and their sum > 0. Returns `None` otherwise (including the both-zero case).
- [ ] 1.4 Add `resolve_encoding_capacity(encoding_spec: str) -> int | None`. Supports:
  - Bare `"rung3"` / `"Rung3"` → 16
  - Bare `"rung4"` / `"Rung4"` → 32
  - Bare `"rung5"` / `"Rung5"` → 128
  - Parametric `"hea_rung2(n_qubits=N)"` (any case) → `2 ** N`
  - Anything else → `None`
  Parsing SHALL tolerate whitespace, mixed case, and the `=` / `:` separator variants the existing sweep CLI accepts. Cross-check the regex against the encoding strings the sweep accepts (`saeforge/cli.py::_parse_encoding`).
- [ ] 1.5 Add `format_saturation_note(n_clusters: int, capacity: int, suggested_next_encoding: str) -> str`. Returns the literal `"Note: polygram_n_clusters ({n_clusters}) equals encoding capacity ({capacity}) — the encoding may be saturated. Consider re-running polygram compress with a larger encoding ({suggested_next_encoding}) to see whether additional concepts are present."` wording. The `suggested_next_encoding` argument is computed by the caller (e.g., Rung3 → `"Rung4"`, Rung5 → `"HEA_Rung2(n_qubits=8)"`); the module just formats.
- [ ] 1.6 Export `load_polygram_report`, `compute_redundancy_ratio`, `resolve_encoding_capacity` from `saeforge/__init__.py`. `format_saturation_note` is module-private (used only by `forge_quality.advise_sweep_quality`).

## 2. `ParetoFrontierRow` schema extension

- [ ] 2.1 Add four new fields to `ParetoFrontierRow` (`saeforge/sweep.py`), all defaulting to `None`:
  - `polygram_n_clusters: int | None`
  - `polygram_n_zeroed: int | None`
  - `polygram_redundancy_ratio: float | None`
  - `polygram_encoding_capacity: int | None`
- [ ] 2.2 Update `to_json_dict` / `from_json_dict` to round-trip the new fields. `from_json_dict` SHALL tolerate dicts missing the new keys.
- [ ] 2.3 `__post_init__` validation:
  - When non-None, each `_n_*` field SHALL be `>= 0`.
  - `polygram_redundancy_ratio` when non-None SHALL be in `[0.0, 1.0]`.
  - `polygram_encoding_capacity` when non-None SHALL be `>= 1`.

## 3. Sweep driver wiring

- [ ] 3.1 In `_process_row` (`saeforge/sweep.py`), after the basis is loaded:
  - Read `basis.metadata.get("n_clusters")` for `polygram_n_clusters`.
  - Call `load_polygram_report(basis.metadata.get("report_path"))` (or the original checkpoint path when `report_path` is None) and read `n_zeroed` from it; set `polygram_n_zeroed`.
  - Call `compute_redundancy_ratio(...)` for `polygram_redundancy_ratio`.
  - Call `resolve_encoding_capacity(row.encoding_label)` for `polygram_encoding_capacity`.
- [ ] 3.2 All four fields are populated REGARDLESS of forge success/failure (diagnostics matter most on failure rows). When the polygram report is missing or malformed, all four fields are `None` and the sweep proceeds normally.
- [ ] 3.3 Frontier-only rows also get the polygram diagnostic fields populated (same code path as `add-forge-quality-diagnostics` uses for its rank-side fields).

## 4. Pre-flight advisory extension

- [ ] 4.1 In `saeforge/forge_quality.py::advise_sweep_quality`, after the existing rank-tier advisory body is built, run a polygram-side check per encoding:
  - Resolve `capacity = resolve_encoding_capacity(encoding_label)`.
  - Load the report for the encoding's *largest*-K SAE (the manifest's max K is already iterated by the existing rank-tier loop).
  - Read `n_clusters` from that report.
  - If `capacity is not None and n_clusters == capacity`, compute `suggested_next_encoding` (Rung3→Rung4, Rung4→Rung5, Rung5→HEA_Rung2(n=ceil(log2(capacity*2))), HEA_Rung2(n=N)→HEA_Rung2(n=N+1)).
  - Append the `format_saturation_note(...)` line to the advisory string.
- [ ] 4.2 When NO rank-tier advisory was warranted but the saturation check fires, `advise_sweep_quality` SHALL return a single-line advisory containing just the saturation note (instead of returning `None`).
- [ ] 4.3 The saturation check NEVER triggers refusal — `--quality-floor` continues to react only to `quality_ratio`, never to `polygram_*` fields.

## 5. Tests

### 5.1 `polygram_diagnostics` module

- [ ] 5.1.1 `tests/test_polygram_diagnostics.py::test_load_polygram_report_finds_suffix_variants`: write a fake checkpoint + `_compression_report.json` to a tmpdir; assert the loader returns a dict with the expected keys.
- [ ] 5.1.2 `test_load_polygram_report_returns_none_on_missing`: assert `load_polygram_report(tmpdir/"missing.safetensors")` returns `None` and an INFO log line is emitted.
- [ ] 5.1.3 `test_load_polygram_report_returns_none_on_malformed_json`: write a `_compression_report.json` with `{` (invalid); assert `None`.
- [ ] 5.1.4 `test_compute_redundancy_ratio_basic`: `(6, 88) → 88/94 ≈ 0.936`, `(7, 62) → 62/69 ≈ 0.899`.
- [ ] 5.1.5 `test_compute_redundancy_ratio_handles_none_and_zero`: `(None, 5) → None`, `(0, 0) → None`, `(-1, 5) → None`.
- [ ] 5.1.6 `test_resolve_encoding_capacity_known_rungs`: `"rung3" → 16`, `"Rung4" → 32`, `"rung5" → 128`.
- [ ] 5.1.7 `test_resolve_encoding_capacity_parametric`: `"hea_rung2(n_qubits=6)" → 64`, `"HEA_Rung2(n_qubits=8)" → 256`.
- [ ] 5.1.8 `test_resolve_encoding_capacity_unknown`: `"bogus" → None`, `"rung99" → None`.
- [ ] 5.1.9 `test_format_saturation_note_includes_all_args`: assert the formatted string contains the cluster count, capacity, and suggested next encoding verbatim.

### 5.2 Row schema

- [ ] 5.2.1 `test_pareto_frontier_row_round_trips_polygram_fields`: populated row with all four polygram fields round-trips via `to_json_dict` / `from_json_dict`.
- [ ] 5.2.2 `test_pareto_frontier_row_tolerates_missing_polygram_keys`: dict missing the four new polygram keys → `from_json_dict` returns instance with `None` for those fields.
- [ ] 5.2.3 `test_pareto_frontier_row_rejects_negative_polygram_counts`: `polygram_n_clusters=-1` → `ValueError`.
- [ ] 5.2.4 `test_pareto_frontier_row_rejects_out_of_range_ratio`: `polygram_redundancy_ratio=1.5` → `ValueError`.

### 5.3 Sweep integration

- [ ] 5.3.1 `test_sweep_populates_polygram_fields`: mock the loaded basis to expose `metadata={"n_clusters": 6}` and a fake report with `{"n_zeroed": 88}`; assert the emitted row has `polygram_n_clusters=6`, `polygram_n_zeroed=88`, `polygram_redundancy_ratio≈0.936`, and a non-None `polygram_encoding_capacity`.
- [ ] 5.3.2 `test_sweep_polygram_fields_none_when_report_missing`: mock the basis with no report path; assert `polygram_n_clusters` is None (basis didn't carry it) and `polygram_n_zeroed` is None.
- [ ] 5.3.3 `test_sweep_polygram_fields_populated_on_failure_row`: pipeline.run raises; assert the emitted row has both `error_message` set and the polygram fields populated.

### 5.4 Advisory extension

- [ ] 5.4.1 `test_advisory_appends_saturation_note_when_clusters_equal_capacity`: report has `n_clusters=128`, encoding is `rung5` (cap 128); advisory string contains the saturation-note wording and "HEA_Rung2(n_qubits=8)" as the suggested next encoding.
- [ ] 5.4.2 `test_advisory_saturation_note_only_returns_alone`: rank-tier check is `good` (silent) but cluster saturation fires; assert `advise_sweep_quality` returns a non-None single-line advisory with only the saturation note.
- [ ] 5.4.3 `test_advisory_no_saturation_when_clusters_below_capacity`: report has `n_clusters=6`, encoding cap 128; saturation note NOT in the advisory.
- [ ] 5.4.4 `test_advisory_no_saturation_when_capacity_unknown`: encoding label is `"bogus_rung"`; saturation note NOT in the advisory.
- [ ] 5.4.5 `test_quality_floor_ignores_polygram_saturation`: contrived setup where `quality_ratio=0.9` (good tier) AND `n_clusters==capacity` (saturation note fires); invoke with `--quality-floor 0.5`; sweep proceeds, no refusal. Saturation is descriptive, not a gate.

## 6. Spec delta

- [ ] 6.1 Author `specs/pareto-sweep/spec.md` delta:
  - MODIFIED `ParetoFrontierRow` requirement: row gains four new optional polygram diagnostic fields with nullability per lifecycle state and `__post_init__` validation rules. Scenarios cover round-trip, missing-key tolerance, negative-count rejection, and out-of-range ratio rejection.
  - MODIFIED `Pre-flight quality advisory` requirement: advisory body MAY append a saturation note when `polygram_n_clusters == polygram_encoding_capacity` at the largest-K SAE in any encoding's manifest; the wording is fixed in the spec; the saturation check NEVER triggers `--quality-floor` refusal. Scenarios cover the append-on-saturation case, the no-rank-tier-but-saturation case, and the no-saturation-when-clusters-below-capacity case.
  - No ADDED requirements (capabilities-level): all changes are extensions of existing requirements.

## 7. Docs

- [ ] 7.1 Extend the `#### Pareto sweep (Axis 4)` README section with a short subsection "Polygram concept-structure diagnostics" — what `polygram_n_clusters` and `polygram_redundancy_ratio` mean, the rule-of-thumb that high redundancy means concentrated concepts (econ-sae phase 7.2 reference), and the recipe for the capacity sweep via `--encoding rung3,rung4,rung5`. Include the `jq` filter idiom: `jq 'select(.polygram_n_clusters != null) | {enc: .encoding_label, k: .n_features_kept_actual, clusters: .polygram_n_clusters, redundancy: .polygram_redundancy_ratio, kl: .faithfulness_kl}' frontier.jsonl`.
- [ ] 7.2 CHANGELOG entry under `[Unreleased]` → `### Added (add-polygram-cluster-diagnostics)`.

## 8. Validation

- [ ] 8.1 `openspec validate add-polygram-cluster-diagnostics --strict` is green.
- [ ] 8.2 Full `pytest` suite passes; new tests cover §5.1 through §5.4.
- [ ] 8.3 `ruff check` clean on touched files.
- [ ] 8.4 Live smoke (any polygram-compressed checkpoint with a colocated report): confirm row carries the new fields and the advisory note fires when manually contriving `n_clusters == capacity`.
- [ ] 8.5 `openspec archive add-polygram-cluster-diagnostics` after merge.

## 9. What this change explicitly defers

- [ ] 9.1 Recomputing `n_clusters` from `W_dec` when the report is missing. The polygram-side report is the source of truth; no recompute.
- [ ] 9.2 A `polygram_tier` categorical analogous to `quality_tier`. Cluster count is a small integer; analysts can filter directly without a tier vocabulary.
- [ ] 9.3 A `--polygram-capacity-sweep` CLI flag. The existing `--encoding rung3,rung4,rung5` plumbing already does this.
- [ ] 9.4 Correlating `polygram_n_clusters` or `polygram_redundancy_ratio` with `faithfulness_kl`. Research question; needs accumulated data from real forge runs.
- [ ] 9.5 Polygram-side report schema changes. Older reports without `n_zeroed` get `None` — no upstream PR required by this proposal.
- [ ] 9.6 Surfacing the polygram fields in `saeforge inspect`. Possible follow-up; out of scope here.
