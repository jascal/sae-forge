# Response: Progressive Encoding Sweep Proposal (2026-05-22)

**Status:** Counter-shape drafted as `openspec/changes/add-encoding-early-stopping/`.

## TL;DR

The proposal's substantive idea — **cheap-first encoding ordering + skip-when-satisfied** — is architecturally compatible and worth shipping. The counter-shape is the openspec linked above.

The proposal's *framing* repeats the same architectural confusion documented in `docs/proposals/warm-start-proposal-response.md`: it pitches "warm-start across encoding tiers" as a compute optimization, but the closed-form forge has no model state to warm-start across encoding choices. The "thousands → hundreds GPU-hours" cost estimate inherits that framing's miscalibration.

The honest version of the proposal:

- Cheap-first ordering: **already supported** (users supply encodings in priority order via `--encoding LABEL:PATH` or `encodings=[...]`).
- Skip-when-satisfied: **new feature, captured in `add-encoding-early-stopping`.**
- Warm-start across tiers: **not applicable** (no model state to inherit).
- Per-protein adaptive routing: **separate larger openspec** (`add-per-protein-adaptive-routing`), deferred.

## What the proposal gets right

1. **Cost varies across encodings.** MPSRung1 at low bond dim is cheaper to materialize than higher-bond-dim or richer-partition variants. Ordering them cheap-first IS sensible. The current `--encoding LABEL:PATH` flag accepts user-supplied order, but no machinery uses it for compute-saving today.

2. **Early-stopping when sufficient is real compute saving.** If a cheap encoding clears the user's quality bar at stage 0, evaluating expensive encodings at stage 1+ is wasted budget. The progressive wrapper has the per-stage hook to make this decision cleanly.

3. **The "long-tail" framing has substance.** Sequence-level heterogeneity is real — common motif proteins are easier substrates than rare-family ones. There's a meaningful research question hidden here about per-protein adaptive routing.

## What it gets wrong (recurring architectural confusion)

> "warm-start from the previous results"

Same confusion as the warm-start proposal (PR #86 counter-shape) and the mixed-χ proposal (PR #90 counter-shape). **No model state to warm-start.** The progressive sweep is closed-form. Each encoding's forge is reconstructed deterministically from `(SAE_checkpoint, host, basis_config)`; encoding A's earlier evaluation doesn't "inherit" anything into encoding B's basis.

What can be reused across encodings (already exists):

- Host activations — shared cache via `HostExtractionCache` (PR #77).
- Per-protein labels — same labels apply to every encoding.

What can't be reused:

- Per-encoding forge activations (each encoding's W_dec slice → different forge).
- Per-encoding plateau (each encoding's row-norm distribution → different plateau).

The "warm-start" framing doesn't mean what the proposal implies. It already means "the host cache is already warm" — which the existing slice-1 multi-encoding wrapper already gives you.

## What it gets wrong (cost estimate)

> "potentially bringing a million-protein run down from thousands of GPU-hours to just a few hundred"

A 1M-protein single-encoding progressive sweep on H200 with host cache is ~8-12 GPU-hours. K=3 multi-encoding with shared host cache is ~24-30 GPU-hours. Not 1000+.

The proposal's cost model assumes per-encoding training, which isn't what happens. Honest savings of cheap-first-with-early-stop: **~20-40 % of multi-encoding wall time** on substrates where the cheapest encoding satisfies the user's threshold, **~0 % savings** on substrates where it doesn't (correct — the user gets the full sweep when their threshold isn't met).

The "save the easy half; don't break the hard half" pitch is honest. The "thousands → hundreds" pitch isn't.

## What ships in the counter-shape

**`add-encoding-early-stopping` openspec** — sized for a ~1-2 day implementation that lands as a patch release (v0.10.1) on top of `add-multi-encoding-capability-sweep`'s v0.10.0 ship:

### Core surface

```python
@dataclass(frozen=True)
class StoppingCriterion:
    min_retained_mauc: float = 0.95
    require_converged: bool = True
    max_n_features_kept: int | None = None
    mode: Literal["early_exit", "early_skip"] = "early_skip"

    def satisfies(self, rec: ProgressiveRecommendation) -> bool: ...

history = sweep_pareto_capability_progressive(
    encodings=[("raw_slice", p1), ("partition_q4", p2), ("partition_q8", p3)],
    stopping_criterion=StoppingCriterion(min_retained_mauc=0.95),
    ...
)
print(history.recommendation.stopped_early)  # True if cheap-enough encoding sufficed
```

Two modes:

- **`early_skip`** (default): an encoding that satisfies the criterion stops sweeping further cells but stays in the recommendation pool. Other encodings continue running. The cross-encoding tiebreaker (smallest-stable-n) still applies.
- **`early_exit`**: first encoding to satisfy → done. Faster, but loses the Occam's-razor smallest-stable-n contract. Opt-in.

### CLI

```bash
sae-forge sweep-capability-progressive \
    --encoding raw_slice:p1 --encoding partition_q4:p2 --encoding partition_q8:p3 \
    --stop-when "retained-mauc>=0.95,converged,n<=512" \
    --stop-mode early_skip \
    ...
```

### Falsifiable gates

Two slow tests against bio-sae fixtures:

1. **Concentrated regime triggers early-stop**: residue fixture under `feed='residue'` with 3 encodings + the criterion above. Wall time ≤ 70 % of unstoppped sweep; first-position encoding satisfied.
2. **Spread regime completes full schedule**: pooled fixture under `feed='pooled'` with same criterion. `stopped_early=False`; sweep ran to completion (per the partition-validation evidence, the spread regime's retained_mauc doesn't clear 0.95).

The pair tests both directions: early-stop fires where it should + doesn't fire where it shouldn't.

## What about per-protein adaptive routing?

The proposal also implied per-protein routing ("most proteins get handled quickly by the simple encodings; you only pay the full cost for the hard long-tail cases"). Currently the forge is evaluated population-wide, not per-protein. Per-protein routing requires:

- Per-protein per-encoding scoring (much more memory + compute).
- A routing decision function (heuristic or learned).
- Bookkeeping for which encoding handled which protein.

This is genuinely interesting research but it's outside the current sae-forge architecture and would need its own openspec (`add-per-protein-adaptive-routing`). Deferred until there's empirical motivation that population-wide encoding choice is leaving compute on the table.

## Decision needed

If the framing here lands:

- I'll land `add-encoding-early-stopping` as a ~1-2 day impl after the multi-encoding shipping (slices 4 + 5 of `add-multi-encoding-capability-sweep`).
- `add-per-protein-adaptive-routing` waits for empirical motivation.

If you disagree — specifically, if you've been assuming a per-encoding training step that hasn't been built — let's clarify. The closed-form forge construction step is at `saeforge/sweep_capability.py::_run_capability_cell` lines ~280-310; happy to walk through it.
