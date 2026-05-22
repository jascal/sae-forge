# Implementation tasks

## 0. Pre-locks (blocking)

- [ ] 0.1 Wait for `add-multi-encoding-capability-sweep` slices 4 (gate result) + 5 (docs + v0.10.0) to land. This openspec layers on top of #92-95; cleanest to build against a stable multi-encoding base.
- [ ] 0.2 Confirm the bio-sae acceptance gate's empirical numbers (retained_mauc per encoding at n=5000) inform the `StoppingCriterion` default values. If concentrated regime clears 0.95 cleanly and spread doesn't, defaults are well-calibrated; otherwise tune.

## 1. `saeforge/sweep_capability_progressive.py` — StoppingCriterion + early-stopping logic

- [ ] 1.1 New `StoppingCriterion` frozen dataclass with fields per design.md Decision 3: `min_retained_mauc=0.95`, `require_converged=True`, `max_n_features_kept=None`, `mode: Literal["early_exit", "early_skip"]="early_skip"`. Plus a `satisfies(rec: ProgressiveRecommendation) -> bool` method that evaluates an encoding's recommendation against the criterion.
- [ ] 1.2 `sweep_pareto_capability_progressive(...)` signature: new `stopping_criterion: StoppingCriterion | None = None` kwarg. Default `None` → byte-equivalent v0.9.x behaviour preserved.
- [ ] 1.3 Per-stage early-stopping logic inside the existing main loop:
  - At end of each stage (after per-encoding plateau identification + convergence detection): if `stopping_criterion` is set, build a per-encoding `ProgressiveRecommendation` snapshot for THIS STAGE and call `stopping_criterion.satisfies(snapshot)` for each encoding.
  - `early_exit` mode: ANY encoding satisfies → break the stage loop (after finishing this stage's remaining cells).
  - `early_skip` mode: An encoding that satisfies is marked done; future stages skip it.
- [ ] 1.4 Per-encoding "done" state tracked via the existing `per_encoding_converged` dict — extend its semantics to also flag "criterion-satisfied even if not convergence-converged".
- [ ] 1.5 `ProgressiveRecommendation` gains `stopped_early: bool = False` + `early_stopping_metadata: dict[str, Any] | None = None`. `to_json_dict()` omits both fields when False/None for back-compat.
- [ ] 1.6 Top-level rationale string mentions early-stopping when fired: "Early-stopped at stage K because encoding L satisfied criterion (retained_mauc=0.953 >= 0.95, converged=True). Other encodings: skipped further work."

## 2. Tests

- [ ] 2.1 `tests/test_sweep_progressive.py` "Suite 5: encoding early-stopping" (+4 tests):
  - `test_stopping_criterion_satisfies_smoke`: pure-helper test; build a synthetic ProgressiveRecommendation; assert `criterion.satisfies(rec)` returns expected True/False across permutations.
  - `test_early_exit_terminates_on_first_satisfier`: 3-encoding 3-stage sweep where encoding 0 satisfies at stage 1; assert `len(history.stages) == 2`, `recommendation.stopped_early=True`, `winning_encoding == "encoding_0"`.
  - `test_early_skip_finishes_other_encodings`: same fixture; `mode="early_skip"`; assert encoding 0 stops sweeping at stage 1 but encodings 1 + 2 continue; total stages run = full schedule; encoding 0's recommendation still uses stage 1's plateau.
  - `test_no_criterion_preserves_v09_behaviour`: explicit `stopping_criterion=None`; output is byte-equivalent to the same sweep without the kwarg.
  - `test_no_encoding_satisfies_completes_schedule`: criterion threshold too tight for any encoding → sweep runs to natural completion; `stopped_early=False`.
- [ ] 2.2 `tests/test_sweep_pareto_capability.py`: no changes needed (the single-shot wrapper doesn't accept stopping_criterion per Decision Out-of-Scope).

## 3. CLI surface

- [ ] 3.1 `sae-forge sweep-capability-progressive --stop-when EXPR --stop-mode {early_exit, early_skip}`. Parses EXPR into `StoppingCriterion`.
- [ ] 3.2 `_parse_stop_when(expr: str) -> StoppingCriterion` helper. Comma-separated predicates; reuses `_parse_recommend_predicate` for the field>value forms; adds bare-token `converged` → `require_converged=True`.
- [ ] 3.3 CLI emits to stdout when early-stopping fires: "Early-stopped at stage K via encoding L. Saved approximately N cells." After the recommendation summary.
- [ ] 3.4 `--help` text on `--stop-when` documents the predicate format + the early_exit-vs-early_skip mode distinction + the "early_exit prioritises wall time over smallest-stable-n" caveat.
- [ ] 3.5 `sae-forge recommend` extension: when the progressive_summary.json's `recommendation.stopped_early=True`, the recommend output emits a one-line note: "Note: this recommendation came from an EARLY-STOPPED sweep (criterion fired at stage K). For exhaustive recommendation, re-run without --stop-when."
- [ ] 3.6 New CLI tests in `tests/test_progressive_cli.py`:
  - `test_cli_stop_when_parses`: flag accumulates into a StoppingCriterion.
  - `test_cli_stop_when_invalid_predicate`: bad EXPR → exit 2 with actionable error.
  - `test_cli_recommend_flags_early_stopped`: synthetic frontier + summary with `stopped_early=True` → recommend output mentions early-stop.

## 4. Falsifiable acceptance gate

- [ ] 4.1 `tests/test_encoding_early_stopping_gate.py::test_residue_regime_triggers_early_stop` (slow): bio-sae residue fixture under `feed='residue'` at `[10, 50, 100]` with 3 encodings + `--stop-when retained-mauc>=0.95,converged`. Asserts:
  - `recommendation.stopped_early=True`.
  - Wall time <= 70 % of an equivalent run with `stopping_criterion=None`.
  - The encoding that triggered the stop is the cheapest (first-position) one that satisfies.
- [ ] 4.2 `tests/test_encoding_early_stopping_gate.py::test_pooled_regime_completes_full_schedule` (slow): bio-sae pooled fixture under `feed='pooled'` at `[1000, 5000]` with 3 encodings + same `--stop-when`. Asserts:
  - `recommendation.stopped_early=False` (no encoding clears retained_mauc >= 0.95 on this substrate).
  - All encodings ran to the full schedule.
  - The output's stopped_early field correctly indicates the criterion did NOT fire.

## 5. Documentation

- [ ] 5.1 README: extend the "Multi-encoding capability sweep" section (added by `add-multi-encoding-capability-sweep` slice 5) with an "Early-stopping" subsection. CLI example + recommended `mode` setting + substrate-dependent threshold guidance.
- [ ] 5.2 `docs/algorithm.md` §5 cross-reference: when downstream-task fidelity is critical AND multiple encodings are being compared, early-stopping is a compute-saving opt-in.
- [ ] 5.3 CHANGELOG entry under `[Unreleased]`.

## 6. Release

- [ ] 6.1 Bump `__version__` (likely v0.10.1, patch release on top of the v0.10.0 multi-encoding ship). Pure additive surface; no breaking changes.
- [ ] 6.2 Tag the release.
