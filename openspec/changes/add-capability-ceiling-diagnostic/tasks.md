# Implementation tasks

## 0. Design pre-locks (blocking)

- [ ] 0.1 Lock the **activation-level** oracle: the ceiling is a trained rank-`N` projection fit on host
  activations against the capability target (R2-tied style: init readout-aligned SVD, readout/decode tied to
  the task encoder), held-out + `overfit_flag`-guarded — NOT trained through the full forge. Confirm it is
  measured at the same level as `retained_mauc_pinv` (activation-level decode∘encode) so the gap is
  apples-to-apples.
- [ ] 0.2 Lock that the trained subspace is **never** returned/persisted as a forge basis — oracle only. No
  code path lets `project_module` consume it.

## 1. The ceiling oracle — `saeforge/training` (reuse the X2 machinery)

- [ ] 1.1 `train_subspace_ceiling(host_acts, host_encoder, labels, n_features, init_svd, *, steps, seed)`:
  fit a rank-`N` projection `B` (init readout-aligned SVD of the host decision geometry; fall back to top-`N`
  SAE-atom span when no unembed), tied readout, on the held-out capability target; return
  `retained_mauc_ceiling` + `overfit_flag`. Reuse `train_encoder`'s split / early-stop / scoring-only-AUC
  discipline.
- [ ] 1.2 `retained_mauc_svd`: project host activations onto the top-`N` readout-aligned SVD subspace
  (`_readout_aligned_order` already computes the geometry) and score retained-mAUC.

## 2. `sweep_pareto_capability` — opt-in diagnostic

- [ ] 2.1 `compute_capability_ceiling: bool = False`. When True, `_run_capability_cell` also computes
  `retained_mauc_svd`, `retained_mauc_ceiling`, and the derived `interpretability_tax = ceiling − pinv` and
  `irreducible_floor_gap = host − ceiling`. Default False ⇒ byte-identical to today.
- [ ] 2.2 Surface the five fields on `ParetoFrontierRow` (+ `to_json_dict`); back-compat: omitted when the
  flag is off (mirror the existing optional-capability-field pattern).

## 3. Tests

- [ ] 3.1 Synthetic fixture where the SAE atoms are a *bad* rank-`N` subspace → ceiling > pinv (a real
  interpretability tax); and a fixture where SAE atoms ARE the optimal subspace → ceiling ≈ pinv. Assert the
  decomposition fields populate and `interpretability_tax`/`irreducible_floor_gap` have the right signs.
- [ ] 3.2 `compute_capability_ceiling=False` leaves rows byte-identical (the optional fields absent).

## 4. Acceptance gate (descriptive — the decomposition IS the result)

- [ ] 4.1 `scripts/capability_ceiling_gate.py`: report the three retained-mAUC quantities + the two gaps on
  **GPT-2 + Pythia-70m** at compressed widths, multi-seed (reuse the gpt2 / pythia ladder fixtures). Route the
  decomposition into the proposal "Gate RESULT". No pass/fail; no "irreducible"/"closes the tax" language.
