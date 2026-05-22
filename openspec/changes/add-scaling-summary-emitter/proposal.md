# Scaling Summary Emitter — cross-run reporting for capability scaling laws

Add a thin outer-loop wrapper that drives `sweep_pareto_capability_progressive` across an *increasing-protein outer schedule* (e.g. `[10k, 50k, 200k, 1M]`) and emits a `scaling_summary.csv` + JSON manifest tying each outer-tier's recommendation, retained_mauc, plateau argmin, wall time, and per-stage convergence narrative together. Pure reporting layer — does not change the sweep wrapper's behaviour, just bundles cross-run outputs in a paper-ready shape.

## Why

The current `sweep_pareto_capability_progressive` answers "what's the smallest stable n on this fixture, at this maximum protein count?" Each invocation produces a `progressive_summary.json` for one (fixture, schedule) tuple.

What the current architecture doesn't give you (without manual scripting): **a scaling curve**. Plot retained_mauc vs. protein-count, with per-tier plateau-argmin trajectories overlaid, so a single command produces the figure that would go in a paper or capability-roadmap doc.

This proposal lands the thin wrapper that does that — explicitly separating "the sweep mechanism" (already shipped) from "the reporting story" (new). Bio-sae's empirical work has been hitting this gap manually: the writeup at `bio-sae/docs/forge-capability-bottleneck.md` carries hand-curated tables from separate sweep runs at n=10, n=100, n=500, n=1000, n=5000 because the sweep wrapper has no built-in cross-tier aggregation. **An automated emitter turns the writeup into a `--output-root` + `python -m saeforge.scripts.run_scale_sweep` invocation.**

## What

### `saeforge.ScalingRun` + `saeforge.run_scale_sweep(...)`

```python
@dataclass(frozen=True)
class ScalingTier:
    """One outer-tier's outcome inside a scaling sweep."""
    max_n_proteins: int
    inner_schedule: tuple[int, ...]   # the progressive sub-schedule
    inner_history: ProgressiveHistory
    wall_time_seconds: float
    estimated_cost_usd: float | None  # populated if --dollars-per-gpu-hr given

@dataclass(frozen=True)
class ScalingRun:
    tiers: tuple[ScalingTier, ...]

    def to_csv(self, path: Path) -> None: ...
    def to_json_dict(self) -> dict[str, Any]: ...

def run_scale_sweep(
    dataset_builder: Callable[[int], CapabilityDataset],  # n -> dataset
    *,
    protein_schedule: list[int],         # outer-tier protein counts
    inner_schedule_per_tier: Callable[[int], list[int]] | None = None,
    candidate_widths: list[int],
    output_root: str | Path,
    dollars_per_gpu_hr: float | None = None,
    **progressive_kwargs,
) -> ScalingRun: ...
```

Per outer tier:

1. Call `dataset_builder(max_n_proteins)` to materialise the appropriate `CapabilityDataset` slice.
2. Default `inner_schedule_per_tier(max_n)` returns a 3-stage geometric ladder from `max(10, max_n // 10)` to `max_n`. Users can override.
3. Run `sweep_pareto_capability_progressive(...)` with the inner schedule, accumulating `ProgressiveHistory`.
4. Stamp wall time + optional cost ($/GPU-hr × elapsed_GPU_hours).
5. Append `ScalingTier` to the `ScalingRun`.

### `scaling_summary.csv` layout

| max_n_proteins | rec_n | retained_mauc | host_mauc | converged | inner_stages_run | wall_time_seconds | estimated_cost_usd | plateau_argmin_trajectory |
|---|---|---|---|---|---|---|---|---|
| 10000 | 48 | 0.9876 | 0.9012 | True | 3 | 412.3 | 0.34 | 48,48,48 |
| 50000 | 48 | 0.9921 | 0.9134 | True | 3 | 1820.7 | 1.52 | 64,48,48 |
| 200000 | 32 | 0.9943 | 0.9201 | True | 3 | 6402.1 | 5.33 | 64,48,32 |
| 1000000 | 32 | 0.9961 | 0.9215 | True | 4 | 24180.4 | 20.15 | 64,48,32,32 |

One row per outer tier. `plateau_argmin_trajectory` carries the per-inner-stage argmin sequence so analysts can see whether the recommendation converged within the inner schedule.

### `sae-forge scale-sweep` CLI

Mirrors the `sweep-capability-progressive` schema with outer-loop additions:

```bash
sae-forge scale-sweep \
    --dataset-config bio-residue.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --candidate-widths 4,8,16,32,64,128,256 \
    --protein-schedule 10000,50000,200000,1000000 \
    --output-root sweeps/bio_sae_may2026/ \
    --dollars-per-gpu-hr 3.0
```

Output: `output-root/scaling_summary.csv` + `scaling_summary.json` + per-tier `tier_<max_n>/progressive_summary.json` + per-tier `tier_<max_n>/frontier.jsonl`.

### Optional: matplotlib plot helper

```python
ScalingRun.plot(output_path: Path | None = None) -> "matplotlib.figure.Figure"
```

Two subplots: (a) `retained_mauc vs. max_n_proteins` with the host baseline overlaid; (b) `plateau_argmin_trajectory` per tier (stacked). Optional dependency; raises a clean `ImportError` with install hint if matplotlib not present.

### Dry-run cost estimator

```bash
sae-forge scale-sweep --dry-run [...]
```

Benchmarks one cell at the smallest outer-tier's smallest schedule entry, projects total wall time + cost across the full schedule. Lets you estimate a 1M-protein run's cost in ~30 seconds before committing to ~12 hours.

## Falsifiable acceptance gate

Three predictions against bio-sae's existing fixtures:

| schedule | predicted scaling-curve shape | falsifies if |
|---|---|---|
| `[10, 50, 100]` against residue fixture | rec_n stable around 32-48 across all three tiers; retained_mauc monotone-nondecreasing | rec_n shifts > 1 plateau bucket between tiers, OR retained_mauc decreases as data scale grows |
| `[200, 1000, 5000]` against pooled fixture | rec_n in [128, 512] across all three tiers; retained_mauc stable around 0.92-0.94 | rec_n outside the predicted range, OR retained_mauc varies by more than 0.05 across tiers (would falsify the writeup §3.2 "uniform tax" framing) |
| dry-run estimator | projected wall time within 25 % of actual on the residue regime | estimator's projection drifts by more than 25 % (would mean the per-cell-benchmark step isn't representative) |

## Scope (v1)

- **In:** outer-loop wrapper, CSV / JSON / matplotlib emitters, CLI subcommand, dry-run cost estimator.
- **Out:**
  - Multi-host / multi-feed within one scaling run (each outer tier is one (host, feed, dataset_builder) tuple; comparing across hosts is a separate `compare-scaling-sweep` future).
  - Distributed execution (the inner sweep is already CPU-bound; users can shard outer tiers via separate invocations + manual aggregation).
  - Cross-run aggregation (combining multiple `ScalingRun`s into one figure) — deferred until the single-run case is shaken out.

## Why this is NOT the "ProgressiveForgeSweep" the original proposal asked for

The original proposal (pasted 2026-05-22) suggested a `ProgressiveForgeSweep` that "runs each tier with warm-start from previous best" — the warm-start being model-state transfer between tiers. As `add-forged-activations-cache`'s proposal documents, **the progressive sweep doesn't train models per tier**, so there's no model state to warm-start. What the original proposal correctly identified is the **reporting gap** — the lack of cross-tier aggregation. This openspec ships that piece in isolation, without forcing the (separately-questionable) warm-start coupling.

If a future change adds optional fine-tuning to the progressive cell, the `ScalingRun` aggregator emits the fine-tune metrics in additional CSV columns automatically — the reporting layer is forward-compatible with that work.

## Related

- The sweep mechanism this composes: `saeforge.sweep_pareto_capability_progressive` (PR #82).
- Companion proposal addressing the orthogonal "compute waste" half of the original proposal: `add-forged-activations-cache`.
- Bio-sae writeup that motivates the reporting story: `bio-sae/docs/forge-capability-bottleneck.md` §3 (currently hand-curated; this proposal automates it).
