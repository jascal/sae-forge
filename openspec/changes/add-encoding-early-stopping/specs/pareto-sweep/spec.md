# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `StoppingCriterion` frozen dataclass

`saeforge.StoppingCriterion` SHALL be a frozen dataclass with:

```
@dataclass(frozen=True)
class StoppingCriterion:
    min_retained_mauc: float = 0.95
    require_converged: bool = True
    max_n_features_kept: int | None = None
    mode: Literal["early_exit", "early_skip"] = "early_skip"

    def satisfies(self, rec: ProgressiveRecommendation) -> bool: ...
```

The `satisfies(rec)` method SHALL return True iff:

- `rec.retained_mauc_vs_host >= self.min_retained_mauc`, AND
- `(not self.require_converged) or rec.converged`, AND
- `(self.max_n_features_kept is None) or
   rec.target_n_features_kept <= self.max_n_features_kept`.

Otherwise False.

The dataclass SHALL be importable as `saeforge.StoppingCriterion` and
added to `saeforge.__all__`.

### Requirement: `sweep_pareto_capability_progressive(stopping_criterion=...)` parameter

The progressive wrapper SHALL accept an optional `stopping_criterion:
StoppingCriterion | None = None` keyword argument. Default `None`
preserves byte-equivalent v0.9.x behaviour (no early-stopping).

When `stopping_criterion` is set:

1. **Per-stage check.** At the end of each progressive stage (after
   per-encoding plateau identification + per-encoding convergence
   detection), the wrapper SHALL compute a per-encoding snapshot
   `ProgressiveRecommendation` for THIS STAGE and call
   `stopping_criterion.satisfies(snapshot)` for each encoding.

2. **`mode="early_exit"` semantics.** If ANY encoding satisfies the
   criterion at the end of any stage, the stage loop SHALL break
   (after finishing this stage's remaining cells). The
   `ProgressiveRecommendation` emitted at exit SHALL have:
   - `stopped_early = True`.
   - `winning_encoding` = the first-satisfying encoding in CLI-flag /
     encodings-list order.
   - `rationale` mentioning the criterion that fired.

3. **`mode="early_skip"` semantics.** Encodings that satisfy the
   criterion at any stage SHALL be marked "done" — future stages
   SHALL NOT sweep new cells for those encodings. Other encodings
   continue running until they either satisfy OR exhaust the
   schedule. The cross-encoding tiebreaker chain (per
   `add-multi-encoding-capability-sweep` Decision 4) picks the
   winner.

4. **Per-encoding state tracking.** The wrapper SHALL track per-
   encoding "done" state via the existing `per_encoding_converged`
   dict's semantics extended to include "criterion-satisfied even if
   not formally convergence-converged".

5. **Stage 0 is the minimum floor.** Even with aggressive
   `early_exit`, stage 0 SHALL run all encodings fully. The
   per-encoding `ProgressiveRecommendation` requires at least one
   stage's plateau on record.

### Requirement: `ProgressiveRecommendation.stopped_early`

`saeforge.ProgressiveRecommendation` SHALL gain two new optional
fields (back-compat: defaults preserve v0.9.x JSON byte-equivalence
when False / None):

- `stopped_early: bool = False`. `True` when the sweep terminated
  because `stopping_criterion` was satisfied; `False` when the
  schedule naturally exhausted.
- `early_stopping_metadata: dict[str, Any] | None = None`. When
  non-None, carries:
  - `triggering_stage: int` — which stage triggered the stop.
  - `triggering_encoding: str` — the encoding that satisfied the
    criterion (first-satisfier under `early_exit`; per-encoding
    under `early_skip`).
  - `criterion_summary: str` — human-readable criterion description.
  - `skipped_encodings: list[str]` — encodings that were stopped
    early (under early_skip, these are the encodings that satisfied;
    under early_exit, these are encodings that were never given a
    chance).

`to_json_dict()` SHALL omit both fields when `False` / `None` so
v0.9.x consumers see no schema change on un-early-stopped progressive
frontiers.

### Requirement: `sae-forge sweep-capability-progressive --stop-when EXPR`

The CLI subcommand SHALL accept new optional flags:

- `--stop-when EXPR` — predicate string parsed into a
  `StoppingCriterion`. Format: comma-separated predicates of the form
  `FIELD<OP>VALUE`, plus bare tokens. Recognized:
  - `retained-mauc>=FLOAT` → `min_retained_mauc=FLOAT`.
  - `n<=INT` → `max_n_features_kept=INT`.
  - `converged` (bare token) → `require_converged=True`.
  Example: `--stop-when "retained-mauc>=0.95,converged,n<=512"`.

- `--stop-mode {early_exit, early_skip}` — defaults to `early_skip`.

Predicate parsing SHALL reuse the existing `_parse_recommend_predicate`
infrastructure where possible (kebab/snake field name conversion).

When `--stop-when` fires during a sweep, the CLI SHALL emit a stdout
note after the recommendation summary:

```
sae-forge sweep-capability-progressive: early-stopped at stage 1.
  Triggering encoding: partition_q4 (satisfies retained-mauc>=0.95, converged)
  Skipped: partition_q8, mps_rung1_x16
  Estimated savings: ~3 hours wall time vs unstoppped sweep
```

### Requirement: `sae-forge recommend` flags early-stopped frontiers

When `sae-forge recommend` reads a `progressive_summary.json` whose
`recommendation.stopped_early=True`, the subcommand SHALL emit a
one-line note alongside the recommendation:

```
Note: this recommendation came from an EARLY-STOPPED sweep
(triggered at stage 1 via encoding partition_q4). For exhaustive
recommendation, re-run sweep-capability-progressive without
--stop-when.
```

Users see that the recommendation is sufficient-not-exhaustive +
have the recipe to upgrade if they want broader frontier data.

### Requirement: Falsifiable acceptance gate

The change SHALL include two slow integration tests:

1. `test_residue_regime_triggers_early_stop` (`@pytest.mark.slow`):
   bio-sae residue fixture under `feed='residue'` at `[10, 50, 100]`
   schedule with 3 encodings + `--stop-when retained-mauc>=0.95,converged`.
   Asserts:
   - `recommendation.stopped_early=True`.
   - Wall time <= 70 % of an equivalent `stopping_criterion=None` run.
   - First-position (cheapest) encoding triggered the stop.

2. `test_pooled_regime_completes_full_schedule` (`@pytest.mark.slow`):
   bio-sae pooled fixture under `feed='pooled'` at `[1000, 5000]`
   schedule with 3 encodings + same `--stop-when`. Asserts:
   - `recommendation.stopped_early=False` (no encoding clears 0.95
     on the spread regime per the partition-validation evidence).
   - All encodings ran to the full schedule.

The pair tests the two regimes: concentrated (where early-stop
SHOULD fire) + spread (where it SHOULD NOT, because the data-scale
tax keeps retained_mauc below the threshold).

### Requirement: NOT `sweep_pareto_capability` (single-shot)

The single-shot `sweep_pareto_capability` SHALL NOT accept a
`stopping_criterion` argument. Single-shot doesn't have the
per-stage hook the stopping logic needs; users wanting early-stopping
use the progressive variant. CLI: `--stop-when` is rejected on
`sweep-capability` with a clear stderr message pointing at
`sweep-capability-progressive`.

### Requirement: Public API surface

`saeforge.__all__` SHALL gain `StoppingCriterion` as an exported
symbol. `saeforge.__init__` SHALL re-export it from
`saeforge.sweep_capability_progressive`.

`test_public_surface_is_frozen` in `tests/test_smoke.py` SHALL be
updated to include the new symbol.
