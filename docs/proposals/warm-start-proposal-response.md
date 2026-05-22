# Response: Warm-Start + Outer Scaling Loop Proposal (2026-05-22)

**Status:** Counter-shape drafted as two separate openspecs:
- `openspec/changes/add-forged-activations-cache/` — the architecture-aware "warm-start"
- `openspec/changes/add-scaling-summary-emitter/` — the cross-tier reporting layer

## TL;DR

The proposal's stated goals (cheaper 1M-protein sweeps; structured scaling-curve reporting) are good and worth shipping. The proposed *mechanism* — model-state warm-start chained across stages, cosine-restart LR scaling, optimizer checkpoint resume — assumes an architecture sae-forge doesn't have. Specifically: **the progressive sweep doesn't train models per stage**. There's no model state to warm-start.

The two openspecs above pick up the proposal's correct intuitions and ship them in shapes that fit the current architecture.

## What the proposal got right

1. **"Earlier converged models are excellent initializations for larger data regimes."** Yes — but the analog in this codebase isn't model weights, it's **redundant forward passes through the closed-form forge across overlapping sweep cells**. Stage K+1's cumulative subsample re-extracts forge activations for cells that already ran at stage K. `add-forged-activations-cache` ships per-cell caching to cut this redundancy. Same mechanism the host-extraction cache uses today (PR #77).

2. **"Turn one-off runs into reproducible scaling curves."** Yes — the cross-tier aggregation step is missing. `add-scaling-summary-emitter` ships a thin outer-loop wrapper (`run_scale_sweep(protein_schedule=...)`) that drives `sweep_pareto_capability_progressive` across an increasing-protein outer schedule and emits a `scaling_summary.csv` + JSON manifest. This is the "ProgressiveForgeSweep" the proposal asked for, minus the model-chaining (because there's no model to chain).

3. **"Dry-run mode to estimate total cost/time before launching."** Yes — `add-scaling-summary-emitter` includes a dry-run cost estimator that benchmarks one cell and projects the full sweep's wall time + dollar cost. ~30 seconds to decide "yes / no this is affordable" before committing 12 hours.

4. **Cost-per-run as a load-bearing concern.** Yes — at 1M proteins, every percentage of wall-time saved is real money. The forged-activations cache + the dry-run estimator together address this directly.

## What the proposal got wrong

### "Load SAE / forged model from previous stage."

`ParetoFrontierRow.forged_model_path` is `None` for every progressive sweep cell because the forge is reconstructed deterministically from `(SAE_checkpoint, host, basis_config)` via closed-form linear algebra — no gradient steps, no model artifact persisted per cell. The `forged_model_path` slot exists for a separate (currently-unused-in-progressive) optional fine-tune step.

### "Reset or scale learning rate (default: cosine restart at 30-50% of previous peak LR)."

There's no learning rate in the progressive sweep because there's no training. The closed-form forge has no gradient, no optimizer, no LR schedule. The LR-scaling logic in the proposal applies if and only if a future change adds fine-tuning to the progressive cell — which would be its own openspec (`add-progressive-finetune`, deferred).

### Cost estimate of `$120-160 → $70-110 on 2× H200s` for 1M-protein runs

The proposal's cost numbers imply a training-and-inference compute model. The current architecture is inference-only at the progressive layer:

| step | per-cell cost @ 1M proteins on H200 |
|---|---|
| Host extraction (cached after first cell) | ~80 min once, then 0 |
| Forge construction (closed-form) | ~1 sec |
| Forge extraction | ~80 min per cell |
| AUC scoring | ~5 sec |

For 8 widths × 4 stages with the existing host cache + the proposed forge cache: ~10-15 GPU-hours, $30-50 on 2× H200s. The proposal's $120-160 estimate maps cleanly to a *fine-tuning* compute model (~40 GPU-hours), which isn't what the progressive sweep currently does.

If fine-tuning IS the goal, the cost numbers + warm-start mechanism + LR scaling make sense — but that's a different feature than what the proposal frames as "warm-start for the progressive sweep."

### "Better final models (less risk of catastrophic forgetting)."

There's no training, so there's no forgetting. The closed-form forge produces identical outputs at identical inputs regardless of "warm-start" state.

## What ships in the counter-shape (two openspecs)

### `add-forged-activations-cache` (small, fast to land)

- `ForgeExtractionCache` mirroring `HostExtractionCache`'s contract.
- Cache key: `(host_model_id, sequences_hash, basis_config_hash, feed, max_seq_len)`.
- Default-on; opt-out via `cache_forged=False` / `--no-forge-cache`.
- Falsifiable gate: warm/cold wall-time ratio ≤ 0.65 on residue, ≤ 0.70 on pooled.
- ~1-2 day implementation; ~5-10× ROI on 100k+-protein sweeps.

### `add-scaling-summary-emitter` (medium scope; pure reporting)

- `ScalingTier` + `ScalingRun` dataclasses.
- `run_scale_sweep(protein_schedule=[10k, 50k, 200k, 1M], ...) -> ScalingRun`.
- `scaling_summary.csv` + JSON manifest + optional matplotlib plot.
- Dry-run cost estimator (one-cell benchmark × cell count).
- `sae-forge scale-sweep` CLI.
- Falsifiable gates: residue scaling curve has `rec_n` stable across tiers; pooled has `retained_mauc` stable across tiers; dry-run within 25 % of actual.
- ~3-5 day implementation; turns the bio-sae writeup's hand-curated §3 tables into a single CLI invocation.

### `add-progressive-finetune` (deferred; cleanly the original proposal's home)

- Adds an optional fine-tuning step to each progressive cell.
- Then warm-start across stages becomes meaningful (model weights to chain).
- LR scaling on resume, optimizer state, all the proposed mechanics apply.
- Out of scope until the closed-form forge's limitations on larger substrates demand it. Current evidence (bio-sae's residue regime retaining 103-104.5 % of host capability without any training) suggests fine-tuning isn't load-bearing for the bundled substrates.

## What this means for the 1M-protein scaling-laboratory framing

The proposal's broader framing — "make 1M-protein runs feel routine rather than heroic" — is exactly right. Two ways to get there:

1. **Make each cell cheaper.** This is `add-forged-activations-cache`'s job. Cuts wall time 30-50 % on progressive sweeps via cell-level memoization.
2. **Make cross-tier orchestration first-class.** This is `add-scaling-summary-emitter`'s job. Replaces manual aggregation with one CLI invocation + a paper-ready CSV.

Together these address ~80 % of the proposal's value with ~50 % of the implementation cost, and they don't presuppose a fine-tuning architecture that hasn't been built. The remaining 20 % (true model-state warm-start with LR scaling) lives in a future `add-progressive-finetune` openspec when there's empirical evidence the closed-form forge needs gradient updates at scale.

## Decision needed

If you (proposal authors) agree with this framing:

- I'll land `add-forged-activations-cache` next (~1 PR, ~1-2 day).
- Then `add-scaling-summary-emitter` (~2-3 PRs).
- `add-progressive-finetune` waits until there's a falsifiable substrate where the closed-form forge underperforms a fine-tuned one.

If you disagree — specifically, if you've been assuming a different progressive-sweep architecture than what shipped — let's clarify. The conversation that produced this response is in the chat log; happy to walk through the closed-form construction step-by-step.
