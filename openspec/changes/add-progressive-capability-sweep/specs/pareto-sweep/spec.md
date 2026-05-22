# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `sweep_pareto_capability_progressive` entry point

`saeforge.sweep_pareto_capability_progressive(sae_checkpoint,
host_model_id, dataset, *, candidate_widths, n_proteins_schedule,
scale_boosts, encodings, retained_mauc_tolerance, plateau_tolerance,
min_plateau_widths, convergence_n_stages, output_dir, **sweep_kwargs)`
SHALL drive an outer loop around `sweep_pareto_capability` that
returns a *stable* recommendation across data-scale stages, not an
argmax-on-one-sample.

Per-stage protocol:

1. **Subsample.** Take the first `n_proteins_schedule[stage]`
   sequences + corresponding labels from `dataset`. Cumulative:
   stage K+1 SHALL be a strict superset of stage K's subsample. The
   subsampling SHALL be order-preserving (no shuffle) so the host-
   extraction cache from prior stages remains addressable.

2. **Sweep.** Call `sweep_pareto_capability(...)` with the current
   active width set. The sub-sweep's `output_dir` SHALL be a stage-
   indexed subdirectory of the progressive wrapper's `output_dir`
   (e.g. `<output>/stage_0/`, `<output>/stage_1/`).

3. **Plateau identification.** Identify the *plateau* of widths
   whose `retained_mauc_vs_host` is within `plateau_tolerance` of
   the stage's peak (the global maximum retained_mauc across that
   stage's cells). The plateau set SHALL have at least
   `min_plateau_widths` members; if `plateau_tolerance` would yield
   fewer, widen the effective tolerance to include the
   `min_plateau_widths` highest-retained_mauc widths.

4. **Neighbour expansion.** Build the next stage's active width set
   as: plateau ∪ {immediate `candidate_widths` neighbours of each
   plateau member}. Neighbours are pulled from the user-supplied
   `candidate_widths` list — the wrapper SHALL NOT invent widths the
   user didn't request.

5. **Convergence check.** Convergence fires when:
   (a) The argmin of the current stage's plateau (smallest n in the
       plateau) equals the argmin of the previous stage's plateau,
       AND
   (b) `|current_argmin_retained_mauc - previous_argmin_retained_mauc|
       < retained_mauc_tolerance`,
   for `convergence_n_stages` consecutive stages.

   On convergence, the loop SHALL break and emit the recommendation.

6. **Schedule exhaustion.** If the schedule is exhausted without
   convergence, the wrapper SHALL still return a recommendation
   (the last stage's argmin-plateau-member) but with
   `converged=False` and a rationale naming the unstable transition.

The output `output_dir/frontier.jsonl` SHALL contain every cell from
every stage, with `stage` populated on each row.
`output_dir/progressive_summary.json` SHALL carry the
`ProgressiveHistory.to_json_dict()` payload (per-stage results +
recommendation + convergence narrative).

### Requirement: `ProgressiveStageResult` + `ProgressiveRecommendation` + `ProgressiveHistory` dataclasses

`saeforge.ProgressiveStageResult` SHALL be a frozen dataclass with:

- `stage: int` — 0-indexed stage number.
- `n_proteins: int` — protein count this stage ran at.
- `active_widths: tuple[int, ...]` — widths swept at this stage.
- `rows: tuple[ParetoFrontierRow, ...]` — the per-cell results.
- `plateau_widths: tuple[int, ...]` — widths within plateau_tolerance
  of this stage's peak (always ≥ `min_plateau_widths`).
- `peak_n: int` — `argmax(retained_mauc_vs_host)` on this stage.
- `peak_retained_mauc: float`.

`saeforge.ProgressiveRecommendation` SHALL be a frozen dataclass with:

- `target_n_features_kept: int` — the smallest stable-plateau width.
- `retained_mauc_vs_host: float` — the converged width's retained_mauc.
- `stages_converged: int` — how many consecutive stages this width
  has been a plateau member.
- `converged: bool` — whether convergence_n_stages was reached.
- `rationale: str` — human-readable explanation (e.g. "Smallest
  plateau-member n=48 stable across stages 1, 2, 3; retained_mauc
  variance 0.003 within tolerance 0.005.").

`saeforge.ProgressiveHistory` SHALL be a container with:

- `stages: tuple[ProgressiveStageResult, ...]`.
- `recommendation: ProgressiveRecommendation`.
- `to_json_dict() -> dict[str, Any]`.
- `from_json_dict(...) -> ProgressiveHistory` classmethod (round-trip
  with `to_json_dict()`).

### Requirement: `ParetoFrontierRow.stage` field

`saeforge.sweep.ParetoFrontierRow` SHALL accept an optional `stage:
int | None = None` field. Default `None` preserves byte-equivalence
with pre-change frontier files for single-shot rows.

- `__post_init__` validation: `stage is None or stage >= 0`.
- `to_json_dict()` SHALL omit the `stage` key when `stage is None`.
- `from_json_dict()` SHALL read `data.get("stage")` (None when
  absent).

### Requirement: `sae-forge sweep-capability-progressive` CLI subcommand

The CLI SHALL ship a new `sae-forge sweep-capability-progressive`
subcommand mirroring the `sweep-capability` schema with these
additional flags:

- `--candidate-widths W1,W2,...` (required, same role as
  `--widths` on `sweep-capability`).
- `--schedule N0,N1,...` (required; protein count per stage).
- `--retained-mauc-tolerance FLOAT` (default 0.005).
- `--plateau-tolerance FLOAT` (default 0.01).
- `--min-plateau-widths INT` (default 3).
- `--convergence-n-stages INT` (default 2).

Exit codes:

- `0`: converged; recommendation in `progressive_summary.json` is
  trustworthy.
- `1`: schedule exhausted without convergence; recommendation
  emitted with `converged=False`. Caller decides whether to ship.
- `2`: config error (missing required flag, bad YAML, schedule not
  monotone, etc.).

### Requirement: `sae-forge recommend` rejects un-converged progressive frontiers

When `sae-forge recommend` is invoked against a frontier whose rows
carry the `stage` field (a progressive frontier), the subcommand
SHALL check the companion `progressive_summary.json` for
`recommendation.converged`. If `False`, the subcommand SHALL refuse
to emit a recommendation unless `--accept-unconverged` is passed.
Refusal SHALL exit non-zero with an error message naming the
unconverged stage(s) and pointing to the `--accept-unconverged`
flag.

### Requirement: Falsifiable acceptance gate

The `add-progressive-capability-sweep` change SHALL include
integration tests that:

1. Run `sweep_pareto_capability_progressive` against bio-sae's
   `runs/uniref50_small/residue` fixture under `feed="residue"`
   with schedule `[10, 50, 200]`. Assert:
   - Recommendation converges within 3 stages.
   - `recommendation.target_n_features_kept` ∈ [12, 64].
   - `recommendation.retained_mauc_vs_host` ≥ 0.98.

2. Run against `runs/uniref50_n5000/pooled_w1024_k64` under
   `feed="pooled"` with schedule `[200, 500, 1000]`. Assert:
   - Recommendation converges in **1 stage** (single-shot is already
     stable on spread substrates per Bio-sae writeup §3.2).
   - `recommendation.target_n_features_kept` = 512 ± 1 plateau
     bucket.

Both tests `@pytest.mark.slow` (cumulative protein extraction
exceeds 1 minute CPU). The bio-sae fixture's presence gates the
test via `pytest.importorskip` + filesystem check.
