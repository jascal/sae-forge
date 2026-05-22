# Implementation tasks

## 1. `saeforge/scaling.py` — new module

- [ ] 1.1 `ScalingTier` frozen dataclass: `max_n_proteins`, `inner_schedule`, `inner_history: ProgressiveHistory`, `wall_time_seconds`, `estimated_cost_usd: float | None`.
- [ ] 1.2 `ScalingRun` frozen container: `tiers: tuple[ScalingTier, ...]` + `to_csv(path)` + `to_json_dict()` + optional `plot()` method (lazy matplotlib import).
- [ ] 1.3 `run_scale_sweep(dataset_builder, *, protein_schedule, candidate_widths, output_root, inner_schedule_per_tier=None, dollars_per_gpu_hr=None, **progressive_kwargs) -> ScalingRun`. Validates: monotone increasing outer schedule, non-empty candidate_widths, dataset_builder callable.
- [ ] 1.4 Default `inner_schedule_per_tier(max_n) = [max(10, max_n // 10), max_n // 3, max_n]`.

## 2. CSV / JSON / manifest emitter

- [ ] 2.1 `ScalingRun.to_csv(path)`: header row + one data row per tier. Columns per the proposal's "scaling_summary.csv layout" section.
- [ ] 2.2 `ScalingRun.to_json_dict()`: structured payload carrying per-tier inner_history full dump (not just the summary).
- [ ] 2.3 `run_scale_sweep` writes `output_root/scaling_summary.csv` and `output_root/scaling_summary.json` at the end. Per-tier `tier_<max_n>/` subdirectories carry the inner sweep's `progressive_summary.json` + `frontier.jsonl`.

## 3. Dry-run cost estimator

- [ ] 3.1 `run_scale_sweep(..., dry_run=True)`: bench one cell at smallest outer tier × smallest inner stage × first candidate width. Multiply by total expected cell count; populate `ScalingTier.wall_time_seconds` + `estimated_cost_usd` for ALL tiers projectively. Returns a `ScalingRun` with `tiers` populated but `inner_history=None` per tier (signalling dry-run).
- [ ] 3.2 CLI `sae-forge scale-sweep --dry-run` emits the projection table to stdout instead of running the full sweep.

## 4. CLI subcommand `sae-forge scale-sweep`

- [ ] 4.1 New subparser. Flags:
  - `--dataset-config PATH` (YAML — same schema as sweep-capability).
  - `--host HOST_ID`.
  - `--candidate-widths W1,W2,...`.
  - `--protein-schedule N0,N1,N2,...` (outer-tier protein counts).
  - `--inner-schedule-per-tier N0,N1,... | auto` (default `auto` = geometric ladder).
  - `--output-root PATH`.
  - `--dollars-per-gpu-hr FLOAT` (optional; populates cost column).
  - `--retained-mauc-tolerance`, `--plateau-tolerance`, `--min-plateau-widths`, `--convergence-n-stages` (passthrough to inner progressive sweep).
  - `--plot` (optional matplotlib-emitting flag).
  - `--dry-run`.
  - `--no-host-cache`, `--no-forge-cache` (passthroughs assuming `add-forged-activations-cache` lands first; otherwise just `--no-host-cache`).
- [ ] 4.2 `_cmd_scale_sweep` dispatch function. Parses YAML, constructs a `dataset_builder = lambda n: CapabilityDataset.from_bio_sae(..., n_proteins=n, ...)`.
- [ ] 4.3 Exit codes: 0 all tiers converged + emitted; 1 some tier emitted with `converged=False`; 2 config error.

## 5. Optional matplotlib plot

- [ ] 5.1 `ScalingRun.plot(output_path=None) -> Figure`. Two subplots: (a) retained_mauc vs. max_n_proteins; (b) plateau_argmin_trajectory per tier (stacked). Lazy matplotlib import; raises clean `ImportError` with install hint.
- [ ] 5.2 CLI `--plot` invokes `.plot(output_root / "scaling_curve.png")`.

## 6. Unit tests

- [ ] 6.1 `tests/test_scaling.py`:
  - `test_run_scale_sweep_emits_csv_and_json`: synthetic 2-tier sweep against the ESM-fixture; assert CSV header + row count + JSON manifest shape.
  - `test_dry_run_returns_projection_without_running`: assert dry-run completes in ≪ full-sweep time; inner_history is None per tier; estimated_cost_usd populated.
  - `test_outer_schedule_validates_monotone`: non-monotone outer schedule → ValueError.
  - `test_default_inner_schedule_is_geometric`: smoke that the auto inner_schedule_per_tier produces a 3-stage ladder.
  - `test_plot_raises_clean_import_error_without_matplotlib`: monkeypatch import to fail; assert ImportError with install hint.
- [ ] 6.2 `tests/test_scaling_cli.py`:
  - Parser smoke (defaults match spec).
  - End-to-end CLI smoke against synthetic fixture (small protein_schedule, fast).
  - `--dry-run` exit code 0 + stdout projection.

## 7. Falsifiable acceptance gate (slow)

- [ ] 7.1 `tests/test_scaling_acceptance_gate.py::test_residue_scaling_curve` (slow, gated on bio-sae fixtures): run `[10, 50, 100]` outer schedule against residue fixture. Assert rec_n stable around 32-48 across tiers; retained_mauc monotone-nondecreasing.
- [ ] 7.2 `tests/test_scaling_acceptance_gate.py::test_pooled_scaling_curve` (slow): run `[200, 1000, 5000]` outer schedule against pooled fixture. Assert rec_n in [128, 512] across tiers; retained_mauc stable around 0.92-0.94.
- [ ] 7.3 `tests/test_scaling_acceptance_gate.py::test_dry_run_within_25pct` (slow): run dry-run against residue fixture, then run actual sweep; assert projected wall time within 25 % of actual.

## 8. Documentation

- [ ] 8.1 README: new "Scaling curves" section under "Capability-aware forge tuning", with the end-to-end CLI example + sample CSV output.
- [ ] 8.2 `docs/algorithm.md`: cross-reference from §5 to `run_scale_sweep` for users who want to characterise a substrate's scaling behaviour vs. just pick a width.
- [ ] 8.3 CHANGELOG entry.

## 9. Bio-sae-side adoption (post-merge)

- [ ] 9.1 `bio-sae/scripts/forge_capability_acceptance.py`: add a `--scale-sweep` mode that invokes `run_scale_sweep` with the bio-sae fixture configurations.
- [ ] 9.2 `bio-sae/docs/forge-capability-bottleneck.md`: replace §3's hand-curated tables with a `scaling_summary.csv` reference; commit the CSV under `bio-sae/runs/forge/scale_sweep_uniref50/`.
