# Implementation tasks

## Implementation status — DONE (2026-06-13)

`saeforge/capability_ceiling.py` (`capability_ceiling_decomposition` — random/activation-PCA `svd`/`pinv`/
capability-supervised `best_atoms`/trained `ceiling` + the three gaps, all **encoder-side** per PR #122),
`scripts/capability_ceiling_gate.py` (+ the matched-SAE control: GPT-2 ReLU vs TopK at `hidden_states[8]`),
`tests/test_capability_ceiling.py` (3). **Gate RESULT** in `proposal.md`: encoder-side `ceiling ≈ activation-PCA`
(training the subspace is decode-specific — the mirror of R2); matched-SAE control shows the SAE-type signal is
in `selection_gap` (ReLU + / TopK −), not gross `pinv` conditioning. *Note:* shipped as a standalone library +
gate (the science); wiring `compute_capability_ceiling` into `sweep_pareto_capability`'s `ParetoFrontierRow` is
a thin follow-up.

## 0. Design pre-locks (blocking)

- [ ] 0.1 Lock the **activation-level** measurement: every quantity (`svd` / `pinv` / `best_atoms` / `ceiling` /
  `random`) is a rank-`N` projection of host activations scored through the SAE encoder against the same
  labels on the same held-out items — apples-to-apples. NOT through the full forge (that tax sits on top).
- [ ] 0.2 Lock the **ceiling recipe** (per review — the tax is only as good as this): a single linear rank-`N`
  projection `B`, init **activation-PCA** (encoder-side; NOT readout SVD — `retained_mauc` is encoder-side and
  readout-alignment is decode-specific / harmful here, per polygram `add-readout-aligned-geometry-profile`),
  readout **tied** to the task encoder, Adam on the held-out capability target, `train_encoder`-style split /
  early-stop / scoring-only-AUC / `overfit_flag`. Empirical ceiling = **lower bound** on the intrinsic cost.
- [ ] 0.3 Lock the **circularity guard**: the label-defining SAE features are **held out** of the ceiling
  oracle's training target (train on complement features / independent signal); scoring still uses the labels.
- [ ] 0.4 Lock that the trained subspace is **never** returned/persisted/forged — oracle only.

## 1. The reference quantities — `saeforge/training` (reuse X2 machinery)

- [ ] 1.1 `retained_mauc_svd`: top-`N` **activation-PCA** subspace projection (encoder-side frozen-linear
  reference) → retained-mAUC.
- [ ] 1.2 `retained_mauc_best_atoms`: `pinv` of the best-`N` SAE atoms by **capability-supervised selection**
  (rank atoms by how much they preserve the downstream features) — distinct from `pinv`(top-`N`-by-norm) which
  is `retained_mauc_pinv`. (NOT readout-aligned — encoder-side metric.)
- [ ] 1.3 `train_subspace_ceiling(...)`: the ceiling recipe from 0.2 → `retained_mauc_ceiling` + `overfit_flag`.
- [ ] 1.4 `retained_mauc_random`: mean over `k` random rank-`N` projections (floor) + multi-init ceiling spread.

## 2. `sweep_pareto_capability` — opt-in diagnostic

- [ ] 2.1 `compute_capability_ceiling: bool = False`. When True, `_run_capability_cell` computes 1.1–1.4 and the
  derived `selection_gap = best_atoms − pinv`, `interpretability_tax = ceiling − best_atoms`,
  `ceiling_gap = 1.0 − ceiling`. Default False ⇒ byte-identical to today.
- [ ] 2.2 Surface the fields on `ParetoFrontierRow` (+ `to_json_dict`); omitted when the flag is off.

## 3. Tests

- [ ] 3.1 **Selection-fixable fixture:** a *different* `N` atoms span a much better subspace than top-`N`-by-norm,
  free directions add little → `selection_gap` dominant, `interpretability_tax` ≈ 0.
- [ ] 3.2 **Intrinsic fixture:** no `N`-atom selection matches free directions → `interpretability_tax` dominant,
  `selection_gap` small.
- [ ] 3.3 `ceiling > random` always; `compute_capability_ceiling=False` byte-identical (fields absent).

## 4. Acceptance gate (descriptive — the decomposition IS the result)

- [ ] 4.1 `scripts/capability_ceiling_gate.py`: report the four retained-mAUC quantities + three gaps + random
  floor + multi-init spread on **GPT-2 + Pythia-70m** at compressed widths, multi-seed (reuse the gpt2 / pythia
  fixtures). Route the decomposition into the proposal "Gate RESULT".
- [ ] 4.2 **Descriptive verdict:** which gap dominates (`selection_gap` ⇒ chase capability-supervised atom
  selection; `interpretability_tax` ⇒ a
  real interpretability tradeoff) per host/width. No pass/fail; no "irreducible"/"closes the tax" language.
