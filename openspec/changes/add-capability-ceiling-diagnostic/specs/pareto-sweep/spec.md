# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: Capability-ceiling diagnostic (opt-in)

`sweep_pareto_capability` SHALL accept `compute_capability_ceiling: bool = False`. When True, each cell SHALL
additionally report, **at the activation level** (the same decode∘encode level as the `pinv` basis, so all
gaps are apples-to-apples — same labels, same held-out items):

- `retained_mauc_svd` — top-`N` readout-aligned **SVD** subspace of the host activations.
- `retained_mauc_best_atoms` — `pinv` of the best-`N` SAE atoms by **readout-aligned selection** (the X1
  ordering), i.e. the best *interpretable* basis.
- `retained_mauc_ceiling` — a **trained single linear rank-`N` projection** (init readout-aligned SVD, readout
  **tied** to the task encoder, fit on the held-out capability target, `overfit_flag`-guarded) — the
  **empirical oracle ceiling** (a lower bound on the intrinsic cost, not a proven optimum).
- `retained_mauc_random` — mean retained-mAUC over a few random rank-`N` projections (a floor that bounds the
  ceiling recipe's quality from below).

and the derived gaps (note the split at `best_atoms`):

- `selection_gap = retained_mauc_best_atoms − retained_mauc_pinv` — *fixable by atom selection (X1);
  interpretability preserved.*
- `interpretability_tax = retained_mauc_ceiling − retained_mauc_best_atoms` — *intrinsic cost of an SAE-feature
  basis; not selection-fixable.*
- `ceiling_gap = (host retained-mAUC = 1.0) − retained_mauc_ceiling` — *measured gap at rank `N`, achievability
  OPEN.*

Constraints:

- `compute_capability_ceiling=False` (default) SHALL leave every emitted row **byte-identical** to current
  behaviour (the new fields absent from `to_json_dict`, mirroring the optional-capability-field pattern).
- The trained subspace SHALL be used **only** to compute `retained_mauc_ceiling`. It SHALL NOT be returned,
  persisted, or consumed by `project_module` / any forge path — oracle, never a shipped basis.
- **Circularity guard:** the labels (SAE prevalence-band features) used to *score* SHALL be **held out** of the
  ceiling oracle's *training target* (the oracle trains on the complement features / an independent signal),
  so the ceiling does not trivially chase a self-referential objective.
- The verdict SHALL be the **decomposition** (four retained-mAUC values + three gaps + the random floor), not a
  pass/fail. `ceiling_gap` SHALL be described as a *measured gap at rank `N`*, achievability **open**, and the
  ceiling as **empirical** (no "irreducible" / "closes the tax" language).

#### Scenario: a selection-fixable gap surfaces as `selection_gap`, not `interpretability_tax`

- **GIVEN** host activations where a *different* `N` SAE atoms span a much better subspace than the top-`N`-by-
  norm ones, but free directions add little beyond that
- **WHEN** `sweep_pareto_capability(compute_capability_ceiling=True, ...)` runs a cell
- **THEN** `selection_gap` (`best_atoms − pinv`) SHALL be the dominant positive gap and `interpretability_tax`
  (`ceiling − best_atoms`) SHALL be ≈ 0 — i.e. the diagnostic attributes the gap to **atom selection (X1)**,
  not to the interpretability constraint

#### Scenario: an intrinsic gap surfaces as `interpretability_tax`, not `selection_gap`

- **GIVEN** host activations where *no* selection of `N` SAE atoms spans the task subspace as well as free
  directions do
- **WHEN** the diagnostic runs
- **THEN** `interpretability_tax` (`ceiling − best_atoms`) SHALL be the dominant positive gap and
  `selection_gap` SHALL be small — i.e. the cost is intrinsic to insisting on SAE atoms, **not** selection-
  fixable

#### Scenario: ceiling sits above the random-subspace floor

- **GIVEN** any cell with the diagnostic on
- **THEN** `retained_mauc_ceiling` SHALL exceed `retained_mauc_random` (the recipe finds a non-trivial
  subspace), establishing the ceiling as a meaningful — if empirical — upper reference

#### Scenario: diagnostic off is byte-identical

- **GIVEN** `compute_capability_ceiling=False` (the default)
- **WHEN** the sweep emits rows
- **THEN** the rows SHALL be byte-identical to the pre-change behaviour (no ceiling/tax fields present)
