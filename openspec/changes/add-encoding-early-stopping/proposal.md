# Encoding Early-Stopping — cheap-first multi-encoding sweeps

Add an optional `StoppingCriterion` to `sweep_pareto_capability_progressive` that short-circuits the multi-encoding sweep once any encoding satisfies user-specified retention thresholds. Encodings are evaluated in the user's CLI-flag / `encodings`-list order; cheaper encodings get a chance to "win" before more expensive ones consume compute. **The architecture-aware version of the 2026-05-22 progressive-encoding-sweep proposal** that asked for "cheap-first ordering + warm-start", reduced to the part that fits the closed-form forge (no warm-start; cheap-first ordering + skip-when-satisfied).

## Why

The multi-encoding sweep (PRs #90-95) lets users compare encodings on the same fixture. Today's behaviour sweeps ALL encodings to convergence on EVERY stage, even when an earlier (and cheaper) encoding has already cleared the user's quality bar. That's wasted compute on the dominant cost — per-cell forge extraction — when the user only needed the recommendation, not the full per-cell frontier across all encodings.

The proposal that motivated this openspec (2026-05-22 chat) framed this as "warm-start across encoding tiers", which doesn't fit the closed-form forge (`docs/proposals/progressive-encoding-sweep-proposal-response.md` for the full mismatch analysis). The substantive idea — **don't run encoding B if encoding A already satisfies my threshold** — IS architecturally compatible and lands here as a small additive change.

Bio-sae's empirical baseline (the in-flight slice 4 acceptance gate at `tests/test_multi_encoding_acceptance_gate.py`): 3 encodings × 8 widths × 2 stages = 48 cells, ~6 hours wall on CPU at n=5000. If the cheapest encoding (`raw_slice`) satisfies a `retained_mauc >= 0.95` criterion at stage 0, we save the ~5.5 hours of stage 1 + the other encodings' cells.

The real savings depend on the substrate. Concentrated regimes where one encoding clearly wins → big savings. Spread regimes where multiple encodings tie close → smaller savings (sweep continues past the threshold to disambiguate). Empirically: bio-sae's pooled regime, where the partition-validation showed PARTIAL_WIN at stage 1 with retained_mauc ≈ 0.92 (UNDER 0.95), would NOT trigger early-stop at the 0.95 threshold — sweep runs to completion. Bio-sae's residue regime, where retained_mauc clears 0.95 at small n, WOULD trigger early-stop.

The contract is that early-stopping is **opt-in** (default `None` → current behaviour) and **honest about what it does** — it doesn't change the recommendation contract, it just lets you skip exploring encodings you don't need.

## What

### `StoppingCriterion` frozen dataclass

```python
@dataclass(frozen=True)
class StoppingCriterion:
    """Per-encoding sufficiency criteria for early-stopping in
    multi-encoding sweeps.

    Encoding E "satisfies" this criterion when:
      - E's per-encoding recommendation has retained_mauc >=
        min_retained_mauc, AND
      - E's recommendation has converged (if require_converged), AND
      - E's recommendation's target_n_features_kept <=
        max_n_features_kept (if set).
    """
    min_retained_mauc: float = 0.95
    require_converged: bool = True
    max_n_features_kept: int | None = None
    mode: Literal["early_exit", "early_skip"] = "early_skip"
```

Two modes:

- **`early_exit`**: As soon as ANY encoding satisfies, the sweep terminates. The satisfying encoding becomes the recommendation. Encodings later in the list don't run any further cells; the top-level `winning_encoding` is the first-to-satisfy.
- **`early_skip`** (default): A satisfying encoding stays in the recommendation pool but stops sweeping new cells (no further plateau refinement, no further data-scale stages for that encoding). Other encodings keep running. After all encodings either satisfy or exhaust the schedule, the winner is picked per the existing tiebreaker. **This is the recommended production setting** because it lets a cheaper-but-also-satisfying encoding win the tiebreaker even when an expensive earlier encoding also clears the bar.

### `sweep_pareto_capability_progressive(stopping_criterion=...)` kwarg

New optional parameter. Default `None` → current behaviour byte-equivalently preserved. When set:

- At the end of each stage's per-encoding plateau computation, check satisfaction per encoding.
- `early_exit` + any satisfies → break the stage loop immediately + emit recommendation.
- `early_skip` + encoding E satisfies → mark E "done" (skip further cells for E in subsequent stages); continue with remaining encodings.
- All encodings done (either satisfied or schedule-exhausted) → emit recommendation.

### `ProgressiveRecommendation.stopped_early` field (optional)

New `bool` field on `ProgressiveRecommendation`. `True` when the sweep terminated because `stopping_criterion` was satisfied (vs naturally exhausting the schedule). `False` (default) otherwise. JSON-serialised only when `True` for back-compat.

Plus a small `early_stopping_metadata: dict` field (optional) capturing which stage triggered the stop, which encodings satisfied vs exhausted, etc. Audit trail for "was my sweep cut short?" questions.

### `sae-forge sweep-capability-progressive --stop-when EXPR`

CLI flag parsing into `StoppingCriterion`. Format: comma-separated predicates like `retained-mauc>=0.95,converged,n<=512`. Predicate parser reuses the existing `_parse_recommend_predicate` infrastructure where possible.

Also `--stop-mode {early_exit, early_skip}` to choose the mode; default `early_skip`.

## Falsifiable acceptance gate

Two predictions tested as slow integration tests:

| prediction | falsifies if |
|---|---|
| On bio-sae's residue fixture under `feed='residue'` at progressive `[10, 50, 100]` with 3 encodings + `--stop-when retained-mauc>=0.95,converged`: wall time ≤ 70 % of the non-stopping equivalent | wall time is ≥ 70 % of no-stopping (early-stopping didn't actually save compute) |
| On bio-sae's pooled fixture under `feed='pooled'` at `[1000, 5000]` with 3 encodings + `--stop-when retained-mauc>=0.95,converged`: the run completes the full schedule (no early-stop fires) AND `stopped_early=False` | early-stop fires unexpectedly (would mean the criterion is too generous OR the spread regime now clears 0.95, which would be a substrate-change finding) |

The pair tests the two regimes: concentrated (where early-stop SHOULD fire) + spread (where it SHOULD NOT, because the data-scale tax keeps retained_mauc below the threshold).

## Compute savings — honest estimate

| substrate | K=3 sweep at [1000, 5000] | with early-skip @ retained>=0.95 |
|---|---|---|
| concentrated (residue, n_proteins=100) | ~1 hour | ~30-40 min (~40 % saved; first encoding satisfies at stage 1) |
| spread (pooled, n_proteins=5000) | ~6 hours | ~6 hours (no encoding clears 0.95; sweep completes) |

**~20-40 % wall-time reduction on substrates where early-stop fires**; zero savings on substrates where it doesn't (which is correct — the user gets the full sweep when their threshold isn't met). This is NOT "thousands → hundreds GPU-hours" — that figure from the original proposal was uncalibrated. The honest pitch: "save the easy half; don't break the hard half."

## Why this is NOT the cheap-first-warm-start proposal

The original chat proposal (2026-05-22) framed cheap-first ordering as enabling warm-start of model state across encoding tiers, which doesn't fit the closed-form forge (no model state). This openspec retains the cheap-first ordering idea + adds the sufficiency-stopping piece — without the warm-start framing that doesn't fit.

The proposal also implied **per-protein adaptive routing** ("most proteins get handled by simple encodings"). That's a *different and larger* architectural change (per-protein per-encoding scoring + a routing decision function); deferred to a separate `add-per-protein-adaptive-routing` openspec when there's empirical motivation.

## Scope (v1)

- **In:** `StoppingCriterion` dataclass + `stopping_criterion` kwarg on `sweep_pareto_capability_progressive` + `--stop-when` / `--stop-mode` CLI flags + 2 slow falsifiable gates + docs.
- **Out:**
  - Per-protein adaptive routing (separate openspec).
  - Stopping criteria on `sweep_pareto_capability` (the non-progressive single-shot version). Single-shot sweeps don't have the per-stage hook the stopping logic needs.
  - Adaptive ordering of encodings (re-sorting encodings based on early-stage signal). v1 honors the user's input order; adaptive ordering is a future refinement.
  - Cost-based ordering hints (telling the user "you probably want raw_slice first"). The CLI accepts the order as supplied.

## Related

- The multi-encoding sweep this layers on top of: `add-multi-encoding-capability-sweep` (PR #90, impl PRs #92-#95).
- The 2026-05-22 chat proposal this counter-shapes: `docs/proposals/progressive-encoding-sweep-proposal-response.md` (this commit).
- Companion warm-start counter-shapes that motivated documenting these in `docs/proposals/`: `warm-start-proposal-response.md` (PR #86).
- Future related work: `add-per-protein-adaptive-routing` (deferred; per-protein per-encoding scoring).
