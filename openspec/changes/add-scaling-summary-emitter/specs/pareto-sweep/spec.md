# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `ScalingTier` + `ScalingRun` dataclasses

`saeforge.ScalingTier` SHALL be a frozen dataclass with:

- `max_n_proteins: int` — this tier's maximum protein count.
- `inner_schedule: tuple[int, ...]` — the inner progressive schedule used at this tier.
- `inner_history: ProgressiveHistory | None` — populated by full-runs; `None` under `dry_run=True`.
- `wall_time_seconds: float`.
- `estimated_cost_usd: float | None` — populated when `dollars_per_gpu_hr` is supplied.

`saeforge.ScalingRun` SHALL be a frozen container with:

- `tiers: tuple[ScalingTier, ...]`.
- `to_csv(path: Path) -> None`.
- `to_json_dict() -> dict[str, Any]`.
- `from_json_dict(data: dict) -> ScalingRun` classmethod (round-trip with `to_json_dict()`).
- `plot(output_path: Path | None = None) -> "matplotlib.figure.Figure"` — lazy matplotlib import; raises clean `ImportError` with install hint when matplotlib absent.

### Requirement: `run_scale_sweep` entry point

```
saeforge.run_scale_sweep(
    dataset_builder: Callable[[int], CapabilityDataset],
    *,
    protein_schedule: list[int],
    candidate_widths: list[int],
    output_root: str | Path,
    inner_schedule_per_tier: Callable[[int], list[int]] | None = None,
    dollars_per_gpu_hr: float | None = None,
    dry_run: bool = False,
    **progressive_kwargs,
) -> ScalingRun
```

SHALL:

1. Validate inputs: `protein_schedule` monotone non-decreasing + non-empty; `candidate_widths` non-empty; `dataset_builder` callable.
2. For each outer tier `max_n` in `protein_schedule`:
   - Materialise `dataset = dataset_builder(max_n)`.
   - Compute inner schedule: `inner_schedule_per_tier(max_n)` if provided, else the default geometric ladder `[max(10, max_n // 10), max_n // 3, max_n]`.
   - Call `sweep_pareto_capability_progressive` with the inner schedule + `**progressive_kwargs`. Per-tier output directory: `output_root/tier_<max_n>/`.
   - Stamp `wall_time_seconds` + optional `estimated_cost_usd`.
3. Write `output_root/scaling_summary.csv` + `output_root/scaling_summary.json` carrying the full `ScalingRun`.

Under `dry_run=True`: bench one cell at the smallest tier's smallest inner stage, project all tiers' wall_time + cost projectively, return a `ScalingRun` with `inner_history=None` per tier.

### Requirement: `scaling_summary.csv` schema

The emitted CSV SHALL carry these columns in this order, with one row per outer tier:

```
max_n_proteins, rec_n, retained_mauc, host_mauc, converged,
inner_stages_run, wall_time_seconds, estimated_cost_usd,
plateau_argmin_trajectory
```

- `rec_n` is `tier.inner_history.recommendation.target_n_features_kept`.
- `retained_mauc` is `tier.inner_history.recommendation.retained_mauc_vs_host`.
- `host_mauc` is the last-inner-stage's host_baseline_mauc.
- `converged` is `tier.inner_history.recommendation.converged`.
- `inner_stages_run` is `len(tier.inner_history.stages)`.
- `plateau_argmin_trajectory` is a `|`-separated string of per-inner-stage argmin_plateau_width values.

For dry-run rows, `rec_n` / `retained_mauc` / `host_mauc` / `converged` / `inner_stages_run` / `plateau_argmin_trajectory` are empty; only `wall_time_seconds` (projected) and `estimated_cost_usd` (projected) populate.

### Requirement: `sae-forge scale-sweep` CLI subcommand

The CLI SHALL ship a new `sae-forge scale-sweep` subcommand with these flags:

- `--dataset-config PATH` (required; same YAML schema as `sweep-capability`).
- `--host HOST_ID` (required).
- `--candidate-widths W1,W2,...` (required).
- `--protein-schedule N0,N1,...` (required; monotone non-decreasing).
- `--inner-schedule-per-tier N0,N1,...` (optional; defaults to the geometric ladder).
- `--output-root PATH` (required).
- `--dollars-per-gpu-hr FLOAT` (optional; informational only).
- `--retained-mauc-tolerance`, `--plateau-tolerance`, `--min-plateau-widths`, `--convergence-n-stages` (passthrough to inner progressive sweep).
- `--plot` (optional; emits `scaling_curve.png` via the matplotlib plot helper).
- `--dry-run` (project cost + wall time without running the sweep).
- `--no-host-cache` (passthrough to inner sweep).
- `--device DEV`.

Exit codes:
- `0` — all tiers ran + emitted; the inner recommendation converged at every tier.
- `1` — some tier emitted with `converged=False`. Caller decides whether to ship.
- `2` — config error.

Dry-run emits the projection table to stdout (one row per tier) and exits 0.

### Requirement: Plot helper is optional

`ScalingRun.plot()` SHALL be invokable as a method but SHALL raise `ImportError("matplotlib is required for ScalingRun.plot(); install via `pip install matplotlib`")` when matplotlib isn't on the path. Other reporting (`to_csv`, `to_json_dict`) SHALL work without matplotlib.

### Requirement: Falsifiable acceptance gate

The change SHALL include slow integration tests against bio-sae's existing fixtures:

1. `test_residue_scaling_curve` (`@pytest.mark.slow`): `[10, 50, 100]` outer schedule against the residue fixture. Assert `rec_n` stable in [32, 48] across tiers; `retained_mauc` monotone-nondecreasing.
2. `test_pooled_scaling_curve` (`@pytest.mark.slow`): `[200, 1000, 5000]` outer schedule against the pooled fixture. Assert `rec_n` in [128, 512] across all tiers; `retained_mauc` variance across tiers ≤ 0.05 (validates the writeup's "uniform tax" framing across data scales).
3. `test_dry_run_within_25pct` (`@pytest.mark.slow`): run dry-run against residue fixture; run actual sweep; assert projected wall time within 25 % of actual.
