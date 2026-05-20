# Tasks — `fix-scale-boost-calibration` (shipped as diagnostics-only)

Pre-merge: an earlier draft of this change proposed a
`scale_boost="calibrate"` auto-picking mode. The 2026-05-16 smoke gate
falsified the mechanism — three successive proxies for forge KL all
picked the wrong `scale_boost`. The shipped change is
**diagnostics-only**; the calibrate-mode tasks were implemented and
then removed when the mechanism failed. The current tasks reflect the
landing scope.

## 1. `saeforge/calibration.py` module

- [x] 1.1 New module with: built-in calibration corpus loader
      (`load_calibration_corpus(host_model_id, layer, n_tokens=1024,
      prompts_path=None) -> np.ndarray`); curated
      `ANOMALOUS_TOKEN_IDS: dict[str, frozenset[int]]` keyed by
      tokenizer name (GPT-2 starter set: SolidGoldMagikarp family).
- [x] 1.2 `compute_host_logit_std(host_acts, host_unembed) -> float` —
      per-position std on host logits.
- [x] 1.3 `compute_forged_logit_std(host_acts, projector, host_unembed) -> float`
      — same shape, post-`encode`/`decode` round-trip.
- [x] 1.4 `top1_is_anomalous(host_acts, projector, host_unembed, anomalous_set) -> bool`
      — mode top-1 check against the curated set.
- [x] 1.5 `load_host_unembed(host_model_id) -> np.ndarray`.
- [x] 1.6 Export the public surface from `saeforge/__init__.py`.

## 2. `ParetoFrontierRow` schema extension

- [x] 2.1 Add two new fields to `ParetoFrontierRow`, both defaulting
      to `None`: `logit_std_ratio: float | None`,
      `top1_anomalous: bool | None`.
- [x] 2.2 Update `to_json_dict` / `from_json_dict` to round-trip;
      `from_json_dict` tolerates dicts missing the new keys.
- [x] 2.3 `__post_init__` validation: when `logit_std_ratio` is not
      None, `>= 0`.
- [x] 2.4 Update the row schema table in `specs/pareto-sweep/spec.md`.

## 3. Sweep driver wiring

- [x] 3.1 Extend `sweep_pareto(...)` signature with
      `magnitude_diagnostics: Path | int | None = None` and
      `rank_monotonicity_check: bool = False`.
- [x] 3.2 When `magnitude_diagnostics` is set: load the calibration
      corpus + unembed once at sweep entry; thread through to every
      row via `_process_row`.
- [x] 3.3 In `_process_row`: when diagnostics_payload is set,
      compute `logit_std_ratio` and `top1_anomalous`; populate the
      two new row fields.
- [x] 3.4 When `rank_monotonicity_check=True`: post-sweep, group rows
      by encoding label, sort by `n_features_kept_actual` ascending,
      flag any adjacent pair with `KL[high] - KL[low] > 0.1`; print a
      stderr advisory listing violations. Advisory only.

## 4. CLI surface

- [x] 4.1 Add `--magnitude-diagnostics VALUE` to the `sweep-pareto`
      subparser. Accepts `tokens:N` or `prompts:PATH`.
- [x] 4.2 Add `--rank-monotonicity-check` (boolean flag).
- [x] 4.3 `_cmd_sweep_pareto` parses both flags and plumbs into
      `pipeline.sweep_pareto(...)`. `--magnitude-diagnostics`
      requires `--layer`.
- [x] 4.4 Update CLI help text.

## 5. `ForgePipeline.sweep_pareto` pass-through

- [x] 5.1 Extend `ForgePipeline.sweep_pareto(...)` signature with the
      two new kwargs; delegate unchanged.

## 6. `forge_quality.advise_magnitude_diagnostics`

- [x] 6.1 New post-sweep advisory listing per-row `logit_std_ratio`
      and emitting `[!] anomalous-token canary fired` lines for
      rows with `top1_anomalous=True`.
- [x] 6.2 Sweep driver calls the advisory after the row loop;
      printed alongside the existing `advise_sweep_quality`
      pre-flight advisory.

## 7. Tests

- [x] 7.1 `tests/test_calibration.py`: anomalous set seeded, logit
      std shape, identity-basis equality, top-1 anomalous +/-,
      shape mismatches.
- [x] 7.2 `tests/test_sweep.py::TestParetoFrontierRow`: round-trip
      with diagnostic fields, legacy-row backward compat,
      negative-ratio rejected.
- [x] 7.3 `tests/test_sweep.py::TestRankMonotonicityAdvisory`:
      flags violations, silent on monotone, silent within tolerance,
      per-label grouping, error rows skipped.
- [x] 7.4 `tests/test_sweep.py::TestMagnitudeDiagnosticsAdvisory`:
      None on no-diagnostics, lists ratios + canary lines.
- [x] 7.5 `tests/test_sweep.py::TestAutoMaterialiseCLIValidation`
      additions: bogus format, no-colon, missing prompts file,
      negative tokens, rank-monotonicity flag parses.

## 8. Spec delta

- [x] 8.1 `specs/pareto-sweep/spec.md` delta: MODIFIED
      `ParetoFrontierRow` to include the two new diagnostic fields;
      MODIFIED CLI subcommand to include the new flags; ADDED new
      `Post-sweep magnitude-diagnostics advisory` requirement.

## 9. Docs

- [x] 9.1 README section explaining `scale_boost` modes (literal /
      auto only; calibrate dropped) and the new diagnostic flags.
- [x] 9.2 CHANGELOG `[Unreleased]` entry under `### Fixed
      (fix-scale-boost-calibration)`.

## 10. Validation

- [x] 10.1 `openspec validate fix-scale-boost-calibration --strict`
      green.
- [x] 10.2 Full `pytest` suite passes; 527+ tests including the new
      diagnostic / advisory tests.
- [x] 10.3 `ruff check` clean on touched files.
- [x] 10.4 **Live MBP smoke** documented in `smoke-results.md` in
      this change dir. Falsifies the auto-calibration premise (kept
      as the audit trail).
- [ ] 10.5 `openspec archive fix-scale-boost-calibration` after merge.

## 11. What this change explicitly defers

- [x] 11.1 Auto-picking `scale_boost`. Falsified by the smoke gate;
      see `design.md` Decision 1. Any future revival needs a real
      forge-level calibration (~5× forge per row) — a separate
      proposal.
- [x] 11.2 The structural fix for the documented KL blow-up. The
      blow-up is in the projected NativeModel's stacked-layer
      compounding, not in scale_boost magnitude — a different
      proposal.
- [x] 11.3 Per-feature, per-block, or fine-tune-time `scale_boost`
      adjustment.
- [x] 11.4 Expanding the anomalous-token set beyond the GPT-2 starter.
