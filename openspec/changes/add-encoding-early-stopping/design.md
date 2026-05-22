## Context

Multi-encoding capability sweep (PRs #92-#95) ships a per-stage per-encoding plateau identification + per-encoding `ProgressiveRecommendation` + cross-encoding winner pick. Today's wrapper sweeps EVERY encoding to convergence on EVERY stage, even when an earlier (cheaper) encoding has already cleared the user's threshold. That's wasted forge-extraction compute when the user's stopping criterion is satisfiable by a subset of encodings.

The 2026-05-22 chat proposal pitched this as "cheap-first ordering + warm-start across encoding tiers". The cheap-first ordering idea IS architecturally compatible (encodings are already user-supplied in priority order via `encodings=[(label, path), ...]` / `--encoding LABEL:PATH` flag order); the "warm-start" piece doesn't fit (closed-form forge has no model state to inherit). This openspec ships the cheap-first sufficiency-stopping piece + flags the warm-start framing as not-applicable per the prior counter-shape pattern.

The cost model: a K=3-encoding sweep at `[1000, 5000]` on bio-sae's pooled fixture is ~6 hours warm-cache. If the cheapest encoding satisfies the user's threshold at stage 0, we save ~75 % of that work. If it satisfies only at stage 1 (the larger-data stage), we save the *other* encodings' stage 1 cells (~50 %). If no encoding clears the threshold, we save nothing (and the user gets the full sweep — correct behaviour).

The contract is honest: opt-in; default `None` → current behaviour byte-equivalently preserved; the recommendation contract (smallest-stable-n / Occam's razor) doesn't change.

## Goals / Non-Goals

**Goals:**
- `StoppingCriterion` frozen dataclass with retention/convergence/width threshold fields + `mode` (early_exit vs early_skip).
- `sweep_pareto_capability_progressive(stopping_criterion=...)` optional kwarg.
- `ProgressiveRecommendation.stopped_early: bool` field + optional `early_stopping_metadata: dict` audit trail.
- `sae-forge sweep-capability-progressive --stop-when EXPR --stop-mode {early_exit, early_skip}` CLI surface.
- Falsifiable wall-time gate on both regimes (concentrated: early-stop fires; spread: doesn't).

**Non-Goals:**
- Per-protein adaptive routing. The proposal implied routing different proteins through different encodings; that's a separate larger openspec (`add-per-protein-adaptive-routing`), deferred.
- Adaptive re-ordering of encodings based on early-stage signal. v1 honors the user's input order; adaptive ordering is a future refinement.
- Stopping criteria on `sweep_pareto_capability` (the single-shot non-progressive version). Single-shot doesn't have the per-stage hook the stopping logic needs.
- Cost-based ordering hints. Users supply the order; the wrapper doesn't second-guess.

## Decisions

### Decision 1 — Two stopping modes: `early_exit` and `early_skip`

The proposal pitched what's effectively `early_exit` — first encoding that satisfies → done. But that misses the recommendation contract's *smallest-stable-n* / Occam's-razor framing: if the first encoding satisfies but a LATER (more expensive) encoding might satisfy at SMALLER n, the cheaper one should still win the recommendation. The `early_exit` mode loses that.

`early_skip` (default) is more honest:
- An encoding that satisfies stops sweeping new cells (no further plateau refinement; no further data-scale stages for that encoding).
- The encoding STAYS in the recommendation pool — its current per-encoding recommendation is the "winning" candidate for that encoding.
- Other encodings continue running.
- After all encodings either satisfy or schedule-exhaust, the cross-encoding tiebreaker picks the winner (smallest n at retained_mauc >= median, etc).

`early_exit` is exposed for users who genuinely want "first to satisfy wins; no further work":
- Faster than `early_skip` (no further work after the FIRST encoding satisfies).
- Doesn't let a cheaper encoding catch up with a more permissive criterion.

Both modes are spec'd; `early_skip` is the recommended production setting.

### Decision 2 — Per-stage check, NOT per-cell

The stopping criterion is checked at the END of each progressive stage (after per-encoding plateaus + per-encoding convergence have been computed). NOT per-cell.

Rationale:
- Per-cell checking adds bookkeeping complexity (which encoding's cells have run? when do we re-check?).
- Per-stage checking aligns naturally with the existing per-stage convergence-detection hook.
- Cost: per-stage granularity means we may run a few "wasted" cells of an encoding that ultimately satisfies — but the wasted cells are bounded by per-stage cells (~8 widths × scale_boosts), which is tractable.

Per-cell mode could be a future enhancement if the per-stage waste is empirically significant.

### Decision 3 — `StoppingCriterion` defaults are sane production values

The dataclass defaults to:
- `min_retained_mauc=0.95` — the "ship-able" threshold from bio-sae's writeup. Substrate-dependent; users tune.
- `require_converged=True` — recommendation must be data-scale-robust per the progressive wrapper's contract.
- `max_n_features_kept=None` — no width upper bound by default. Users add this when they have a parameter-cost budget.
- `mode="early_skip"` — the safer default that respects the recommendation contract.

A user who wants aggressive early-exit at any quality bar can override: `StoppingCriterion(min_retained_mauc=0.80, require_converged=False, mode="early_exit")`. The default is calibrated for production deployment ("I want a recommendation that survives my downstream task"); the override surface accommodates exploratory probes.

### Decision 4 — `ProgressiveRecommendation.stopped_early` is the audit trail

When the sweep terminates because the stopping criterion was satisfied (vs naturally exhausting the schedule), `recommendation.stopped_early=True`. JSON serialisation omits this field when `False` for back-compat with v0.9.x readers.

A companion field `recommendation.early_stopping_metadata: dict | None` (optional) captures:
- Which stage triggered the stop.
- Which encodings satisfied vs which exhausted the schedule.
- The criterion that fired (specific retained_mauc value, etc).

This lets downstream consumers (`sae-forge recommend`, future `scaling-summary-emitter`) distinguish "the sweep ran to completion and converged" from "the sweep stopped early because we asked it to" — different recommendations with different deployment semantics.

### Decision 5 — CLI predicate parser shared with `recommend`

`--stop-when retained-mauc>=0.95,converged,n<=512` reuses the predicate-parsing machinery from `_parse_recommend_predicate`. The split-on-comma + per-predicate parsing pattern is identical; the only addition is the bare `converged` token (no operator) mapping to `require_converged=True`.

Wired through a small helper `_parse_stop_when(expr)` that returns a `StoppingCriterion` instance.

### Decision 6 — Stopping criterion is BOUNDED by the schedule

Even with aggressive `early_exit` settings, the sweep ALWAYS runs at least stage 0 fully (all encodings at stage 0). The criterion can't trigger before stage 0 completes because per-encoding recommendations don't exist until at least one stage's plateau has been computed. Stage 0's full sweep is the floor.

This means the BEST-CASE savings are ~`(T-1) / T` of total wall time — for a 2-stage schedule, that's ~50 %; for a 4-stage schedule, ~75 %. The 80-90 % figures in the proposal were uncalibrated to this floor.

## Risks / Trade-offs

- **Threshold calibration is substrate-dependent.** A user setting `--stop-when retained-mauc>=0.95` on a substrate where 0.95 is unreachable (bio-sae's spread regime) gets the full sweep — correct behaviour but the user may not realize the threshold is the constraint. Mitigation: the CLI summary names the criterion + reports whether early-stopping fired.
- **`early_exit` mode can pick a suboptimal encoding.** If encoding A satisfies the criterion at stage 0 with n=512, but encoding B would satisfy at stage 1 with n=64 (smaller stable n is the recommendation contract's tiebreaker), `early_exit` returns A instead of B. The `--help` text + docs explicitly call this out: "early_exit prioritises wall-time savings over the Occam's-razor smallest-stable-n picking; use early_skip if you want the smallest n."
- **Bookkeeping for "encoding is done" state.** Per-encoding `done` flag has to interact correctly with the per-encoding active-width set + the existing convergence detection. Tests cover the matrix of (encoding-done-vs-active × convergence-state × stopping-criterion).
- **Mid-stage termination semantics.** When the criterion fires at the end of stage K, do encodings finish stage K's remaining widths or stop mid-stage? **v1 decision: finish the current stage's per-encoding cells, then exit.** Partial-stage data complicates the JSON manifest; finishing the stage gives clean per-stage records.
- **Downstream consumer compatibility.** The new `stopped_early` field is back-compat additive. `sae-forge recommend` should ALSO surface the `stopped_early` flag in its output (so users see "this recommendation came from an early-stopped sweep") — covered in task §3.5.
