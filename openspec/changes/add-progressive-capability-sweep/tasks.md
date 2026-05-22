# Implementation tasks

## 0. Design pre-locks (blocking)

- [ ] 0.1 Confirm `ParetoFrontierRow.stage` is the only schema addition needed (vs adding a separate `ProgressiveStageRow`). The progressive wrapper emits the existing row type with the new optional field; downstream consumers (`sae-forge recommend`) handle it via the existing partition-by-field pattern.
- [ ] 0.2 Lock the convergence contract: argmin of the last-stage plateau, unchanged across `convergence_n_stages` consecutive stages, retained_mauc within tolerance. See design.md Decision 3.

## 1. `saeforge/sweep_capability_progressive.py` — wrapper module

- [ ] 1.1 New module. Lazy-imports torch via the existing `require_extra` path (no torch at module-import time).
- [ ] 1.2 `ProgressiveStageResult` dataclass (frozen): `stage`, `n_proteins`, `active_widths`, `rows`, `plateau_widths`, `peak_n`, `peak_retained_mauc`.
- [ ] 1.3 `ProgressiveRecommendation` dataclass (frozen): `target_n_features_kept`, `retained_mauc_vs_host`, `stages_converged`, `converged`, `rationale`.
- [ ] 1.4 `ProgressiveHistory` class (or list-like) carrying `stages: list[ProgressiveStageResult]` + `recommendation: ProgressiveRecommendation` + a `to_json_dict()` method for the summary JSON file.
- [ ] 1.5 `sweep_pareto_capability_progressive(...)` function:
  - Validate inputs: schedule monotone increasing, schedule[-1] ≤ len(dataset.sequences), candidate_widths non-empty.
  - For each stage in schedule:
    - Subsample dataset (cumulative): take first `n_proteins_schedule[stage]` sequences + corresponding labels.
    - Call `sweep_pareto_capability` with the current active width set.
    - Identify plateau (widths within `plateau_tolerance` of stage's peak; honour `min_plateau_widths`).
    - Compute neighbour expansion: plateau + immediate `candidate_widths` neighbours.
    - Update history.
    - Check convergence: argmin-of-plateau stable for `convergence_n_stages` consecutive stages, retained_mauc within `retained_mauc_tolerance`.
    - Break on convergence.
  - Build `ProgressiveRecommendation`: argmin of the last-stage plateau; converged flag; rationale string.
- [ ] 1.6 Tests at `tests/test_sweep_progressive.py`:
  - Plateau identification: known-shape rows → known plateau (tolerance edges).
  - Neighbour expansion: plateau {16, 32, 64} on candidates {4, 8, 16, 32, 64, 128, 256} → next-stage actives {8, 16, 32, 64, 128}.
  - Convergence detector: synthesized history where argmin plateau-member shifts at stage 2 → converged=False after stage 1 only.
  - End-to-end smoke (against the synthetic ESM fixture from `test_sweep_pareto_capability.py`): one full progressive run, 2-stage schedule, asserts convergence + recommendation populated.

## 2. `ParetoFrontierRow.stage` field

- [ ] 2.1 Add optional `stage: int | None = None` to `ParetoFrontierRow`. Default `None` so v0.8.x rows lacking the field stay byte-equivalent.
- [ ] 2.2 Validation: `stage >= 0 or None`. Cell error.
- [ ] 2.3 `to_json_dict()`: include `stage` only when populated (v0.8.x rows omit it; back-compat with the capability-fields omit-when-None pattern).
- [ ] 2.4 `from_json_dict()`: read with `data.get("stage")` (default None when absent).
- [ ] 2.5 Test in `tests/test_sweep.py` or `tests/test_sweep_pareto_capability.py`: row construction + round-trip with `stage` populated and omitted.

## 3. `sae-forge sweep-capability-progressive` CLI

- [ ] 3.1 New subcommand `sweep-capability-progressive`. Flags:
  - `--dataset-config PATH` (YAML — same schema as `sweep-capability`).
  - `--host HOST_ID`.
  - `--candidate-widths W1,W2,...`.
  - `--schedule N0,N1,N2,...` (protein count per stage).
  - `--scale-boosts F1,F2,... | auto` (default `1.0`).
  - `--encodings E1,E2,...` (default `raw_slice`).
  - `--retained-mauc-tolerance FLOAT` (default 0.005).
  - `--plateau-tolerance FLOAT` (default 0.01).
  - `--min-plateau-widths INT` (default 3).
  - `--convergence-n-stages INT` (default 2).
  - `--output-dir PATH`.
  - `--no-host-cache` (passes through).
  - `--max-seq-len INT` (default 512).
  - `--device DEV`.
- [ ] 3.2 Output: `frontier.jsonl` carrying every cell across every stage (with `stage` populated), plus `progressive_summary.json` carrying the recommendation + per-stage convergence state.
- [ ] 3.3 Exit code: 0 if converged; 1 if not converged but recommendation emitted (so the script's caller can branch on the flag); 2 on config error.
- [ ] 3.4 `sae-forge recommend` extension: when consuming a progressive frontier (detected via `stage` field on any row), default to refusing to recommend on un-converged data unless `--accept-unconverged` is passed.
- [ ] 3.5 CLI tests in `tests/test_cli.py` (or new `tests/test_progressive_cli.py`): subcommand smoke + accept-unconverged guard.

## 4. Falsifiable acceptance gate

- [ ] 4.1 Integration test (slow; `@pytest.mark.slow`): against bio-sae's `runs/uniref50_small/residue` fixture under `feed="residue"`, assert:
  - Recommendation converges in ≤ 3 stages.
  - Recommendation's `target_n_features_kept` ∈ [12, 64].
  - Recommendation's `retained_mauc_vs_host` ≥ 0.98.
- [ ] 4.2 Integration test (slow; `@pytest.mark.slow`): against `runs/uniref50_n5000/pooled_w1024_k64` under `feed="pooled"`, assert:
  - Recommendation converges in **1 stage** (single-shot is already stable on the spread regime).
  - Recommendation's `target_n_features_kept` = 512 ± 1 plateau bucket.

## 5. Documentation

- [ ] 5.1 README: new "Progressive capability sweep" section under the existing "Capability-aware forge tuning" section, with the end-to-end CLI example.
- [ ] 5.2 `docs/algorithm.md`: cross-reference from the capability target's §5 footer to the progressive wrapper. Note the recommendation contract: smallest stable n, not argmax.
- [ ] 5.3 CHANGELOG: `[Unreleased]` entry under `add-progressive-capability-sweep`.

## 6. Bio-sae-side adoption (post-merge, in bio-sae's repo)

- [ ] 6.1 Bump sae-forge pin to the new tag.
- [ ] 6.2 Update `scripts/forge_capability_acceptance.py` with a `--progressive` flag that switches to the new wrapper.
- [ ] 6.3 Update `tests/test_forge_capability_acceptance.py` with a progressive variant that asserts the convergence contract (smallest-stable-n) directly.
- [ ] 6.4 Update `docs/forge-capability-bottleneck.md` §3 with the progressive measurements + the cross-stage-stability rationale.
