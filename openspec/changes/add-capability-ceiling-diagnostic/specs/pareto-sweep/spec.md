# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: Capability-ceiling diagnostic (opt-in)

`sweep_pareto_capability` SHALL accept `compute_capability_ceiling: bool = False`. When True, each cell SHALL
additionally report, **at the activation level** (the same decode∘encode level as the `pinv` basis, so the gap
is apples-to-apples):

- `retained_mauc_svd` — retained-mAUC of the top-`N` readout-aligned **SVD** subspace of the host activations.
- `retained_mauc_ceiling` — retained-mAUC of a **trained** rank-`N` subspace (initialised at the readout-aligned
  SVD, fit on the held-out capability target, `overfit_flag`-guarded) — the **oracle ceiling**.

and the derived gaps:

- `interpretability_tax = retained_mauc_ceiling − retained_mauc_pinv`.
- `irreducible_floor_gap = (host retained-mAUC = 1.0) − retained_mauc_ceiling`.

Constraints:

- `compute_capability_ceiling=False` (default) SHALL leave every emitted row **byte-identical** to the current
  behaviour (the new fields absent from `to_json_dict`, mirroring the existing optional-capability-field
  pattern).
- The trained subspace SHALL be used **only** to compute `retained_mauc_ceiling`. It SHALL NOT be returned,
  persisted, or consumed by `project_module` / any forge path — it is an oracle, never a shipped basis.
- The verdict SHALL be the **decomposition** (the three retained-mAUC values + the two gaps), not a pass/fail.
  `irreducible_floor_gap` SHALL be described as a *measured gap at rank `N`*, with achievability **open** (no
  "irreducible" / "closes the tax" language).

#### Scenario: ceiling exceeds the interpretable basis when the SAE atoms are a poor subspace

- **GIVEN** host activations whose top-`N` SAE atoms span a sub-optimal rank-`N` subspace for the task
- **WHEN** `sweep_pareto_capability(compute_capability_ceiling=True, ...)` runs a cell
- **THEN** `retained_mauc_ceiling` SHALL exceed `retained_mauc_pinv`, and `interpretability_tax` SHALL be
  positive (a measured cost of using the interpretable basis)

#### Scenario: ceiling collapses to the interpretable basis when the SAE atoms ARE near-optimal

- **GIVEN** host activations whose top-`N` SAE atoms already span the near-optimal rank-`N` subspace
- **WHEN** the diagnostic runs
- **THEN** `retained_mauc_ceiling` SHALL be ≈ `retained_mauc_pinv` and `interpretability_tax` ≈ 0 (no
  interpretability cost to pay at this width)

#### Scenario: diagnostic off is byte-identical

- **GIVEN** `compute_capability_ceiling=False` (the default)
- **WHEN** the sweep emits rows
- **THEN** the rows SHALL be byte-identical to the pre-change behaviour (no ceiling/tax fields present)
