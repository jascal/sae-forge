## Context

`sweep_pareto_capability_progressive` answers "what's the smallest stable n at this protein-count maximum?" The natural follow-up — "how does that recommendation evolve as I increase the maximum?" — has no first-class API today. Users run the progressive sweep multiple times at increasing protein counts and aggregate the `progressive_summary.json` files by hand.

Bio-sae's `forge-capability-bottleneck.md` is the canonical example: §3 carries hand-curated tables comparing rec_n / retained_mauc at n=10, n=100, n=500, n=1000, n=5000. Each row required a separate sweep run, a separate progressive_summary.json read, and manual paste-into-markdown. The shape of the data wants a CSV + a single CLI invocation.

The original warm-start proposal (2026-05-22 chat) framed this as `ProgressiveForgeSweep` with model-state chaining between tiers. The chaining is not currently meaningful (no model state to chain), but the *outer-loop reporting* IS valuable on its own. This openspec ships exactly that piece.

## Goals / Non-Goals

**Goals:**
- A first-class outer-loop API: `run_scale_sweep(protein_schedule=[...], ...) -> ScalingRun`.
- Paper-ready CSV + JSON manifest output.
- Optional matplotlib plotting (a `ScalingRun.plot()` method gated on matplotlib being installed).
- Dry-run cost estimator that benchmarks one cell at the smallest tier and projects the full sweep's wall time + dollar cost.
- CLI subcommand `sae-forge scale-sweep`.

**Non-Goals:**
- Multi-host / multi-feed within one scaling run. Each outer tier is one `(host, feed, dataset_builder)` tuple; comparing across hosts is a separate future.
- Distributed / multi-process execution. The inner sweep is CPU-forward-pass-bound; users shard outer tiers via separate invocations.
- Model-state chaining between tiers. The progressive sweep is closed-form; there's no state to chain. If a future change adds fine-tuning, `ScalingRun`'s reporting layer accepts the additional metrics in extra CSV columns automatically — no architecture change needed here.
- Cross-run aggregation. v1 reports on one `ScalingRun`; aggregating multiple is a downstream concern (the JSON manifest is structured for downstream merging).

## Decisions

### Decision 1 — `dataset_builder` callable, not a single `dataset`

```python
def run_scale_sweep(
    dataset_builder: Callable[[int], CapabilityDataset], ...
)
```

Each outer tier needs a dataset materialised at that tier's protein count. Two options:

(a) Pass one `CapabilityDataset` covering the largest tier's protein count; each tier subsamples via `dataset.sequences[:max_n]`. **Rejected** because residue feed needs `metadata['residues_per_protein']` aligned to the subsampled protein count (PR #85's plumbing); the largest-tier dataset's metadata doesn't correctly describe the smaller-tier subset.

(b) Pass a `dataset_builder(max_n) -> CapabilityDataset` that materialises per-tier. **Chosen.** Each tier gets a freshly-built dataset with correct metadata. The builder is typically a one-line lambda wrapping `CapabilityDataset.from_bio_sae(...)`.

The CLI's `--dataset-config` flag handles the typical case (bio-sae fixture) by constructing the builder internally; advanced users invoke the Python API directly.

### Decision 2 — Default inner schedule is geometric

When the user supplies only the *outer* schedule (e.g. `[10000, 50000, 200000, 1000000]`), the default inner schedule for each outer tier is a 3-stage geometric ladder:

```python
def _default_inner(max_n: int) -> list[int]:
    floor = max(10, max_n // 10)
    return [floor, max_n // 3, max_n]
```

At outer tier `max_n = 50000`, that gives `[5000, 16666, 50000]`. The user can override with `inner_schedule_per_tier=...`.

### Decision 3 — Dry-run benchmarks at the SMALLEST outer tier

`--dry-run` runs ONE cell of the smallest outer tier's smallest inner stage (typically `max_n // 10` proteins × 1 candidate width). Multiplies that elapsed time by the full sweep's cell count to project total wall time + dollar cost. This trades estimator accuracy for cheapness: the projection is within ~25 % on commodity CPUs (per the falsifiable gate in `proposal.md`), which is enough to decide "yes/no this is affordable" before committing.

### Decision 4 — CSV is the primary reporting format; JSON is the audit trail

CSV columns are paper-ready: `max_n_proteins`, `rec_n`, `retained_mauc`, `host_mauc`, `converged`, `inner_stages_run`, `wall_time_seconds`, `estimated_cost_usd`, `plateau_argmin_trajectory`.

The companion `scaling_summary.json` carries the same data + the full per-tier `ProgressiveHistory.to_json_dict()` payload (for trajectory inspection without re-reading per-tier `progressive_summary.json` files). Choice: CSV for humans/papers, JSON for downstream analysis.

### Decision 5 — Matplotlib is an optional dependency

`ScalingRun.plot()` imports `matplotlib.pyplot` lazily. If matplotlib isn't installed, the call raises a clean `ImportError` with `pip install matplotlib` hint. Other reporting paths (CSV, JSON) work without matplotlib.

CLI `sae-forge scale-sweep` does NOT automatically generate plots — users opt in via `--plot` or call `.plot()` on the returned `ScalingRun`.

### Decision 6 — Cost estimator is informational, not blocking

`--dollars-per-gpu-hr` populates `estimated_cost_usd` in the CSV; absent, the column is empty. The CLI does NOT refuse to run on a high projected cost (that's a workflow concern, not a sweep concern).

## Risks / Trade-offs

- **Outer-tier serialisation.** Stages within an outer tier run sequentially (progressive's design). Outer tiers also run sequentially in v1. A multi-tier run on a single machine is `sum(per_tier_wall_time)`; users wanting parallelism shard across machines manually.
- **Dataset-builder side effects.** A `dataset_builder` that calls `CapabilityDataset.from_bio_sae(...)` re-reads the bundle + parquet each tier. Cheap relative to the sweep itself but worth noting. Mitigation: users can memoize the builder externally.
- **`plateau_argmin_trajectory` truncation in CSV.** The trajectory is a list; embedded as a comma-separated string in the CSV cell. For long inner schedules this becomes hard to read in a spreadsheet; the JSON manifest carries the structured form.
- **Dry-run accuracy on the spread regime.** At the smallest outer tier, the spread regime's per-cell forward cost may dominate over CPU thermal variation in ways the projection assumes is linear. The 25 % accuracy bound is calibrated on the residue regime; the gate test asserts the same bound on residue. Pooled-regime accuracy may be looser; not a hard contract in v1.
- **Plot styling.** `ScalingRun.plot()`'s matplotlib output is utilitarian (two stacked subplots, default styling). Paper-quality plots require user customisation; the openspec ships the data, not the typography.
