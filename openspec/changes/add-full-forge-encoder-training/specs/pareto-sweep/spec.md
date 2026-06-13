# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `train_encoder` full-forge objective `forge_distill`

`saeforge.training.train_encoder` SHALL accept `objective="forge_distill"` in addition to the
`add-capability-trained-encoder` objectives (`distill`, `supervised`). Under `forge_distill` the
loss SHALL be computed through the **full forge** rather than the activation proxy:

```
target     = host_encoder(host_X)                                      # fixed, host's own latents
prediction = host_encoder( differentiable_forge_h(host, basis, E, seqs_minibatch) )
loss       = dist(prediction, target)                                  # cosine default; standardized MSE optional
```

Constraints:

- `forge_distill` SHALL require a forge context (host, sequences, `feed`, `aggregator`) sufficient
  to call `differentiable_forge_h`; absent it, construction SHALL raise `ValueError`.
- It SHALL train **only** `E` (matched capacity, the tied design) and SHALL keep the held-out
  split, `overfit_flag`, early-stop, and **scoring-only** rank-AUC from the parent change — the
  AUC SHALL NOT be the loss.
- `dist` SHALL reuse the parent change's `loss` parameter: **`loss="cosine"`** (cosine distance,
  default) or **`loss="mse"`** (standardized MSE) — same names and default as
  `add-capability-trained-encoder`.
- It SHALL minibatch the sequences per step using a **seeded** RNG derived from the call's `seed`
  (so a fit is reproducible per seed; the gate's multi-seed run varies the minibatch draws with the
  init/data-order together) and SHALL precompute the `E`-independent host-latent target once (cost
  mitigation), without changing the held-out gate semantics.

#### Scenario: forge_distill trains E through the full forge

- **GIVEN** a tiny `esm2` forge context (host + sequences + encoder + labels)
- **WHEN** `train_encoder(objective="forge_distill", ...)` runs
- **THEN** it SHALL return `(E, EncoderCalibrationReport)` with `E` updated from the `pinv` init and
  the held-out `retained_mauc_trained` / `retained_mauc_pinv_baseline` scored through the full forge

#### Scenario: forge_distill without a forge context is rejected

- **WHEN** `objective="forge_distill"` is requested without the host/sequences forge context
- **THEN** construction SHALL raise `ValueError` naming the missing forge context

### Requirement: `sweep_pareto_capability` `train_objective` selector

`saeforge.sweep_pareto_capability` SHALL accept `train_objective: Literal["proxy", "full_forge"] =
"proxy"`. With `train_objective="proxy"` (default) behaviour SHALL be byte-identical to
`add-capability-trained-encoder`. With `train_objective="full_forge"` and `train_encoder=True`, each
cell SHALL fit `E` via the `forge_distill` objective (passing the host + sequences + feed) and report
the trained held-out retained-mAUC against the always-computed `pinv` baseline, exactly as the proxy
path does — only the training objective differs.

#### Scenario: default train_objective preserves the proxy behaviour

- **WHEN** `sweep_pareto_capability(train_encoder=True)` is called with the default
  `train_objective="proxy"`
- **THEN** the produced rows SHALL match the `add-capability-trained-encoder` proxy-path rows

#### Scenario: full_forge objective re-scores the same metric the proxy was null on

- **WHEN** `sweep_pareto_capability(train_encoder=True, train_objective="full_forge")` runs on an
  `esm2` host
- **THEN** each row's `retained_mauc_trained` SHALL be the full-forge held-out retained-mAUC of the
  forge-trained `E`, and `retained_mauc_pinv_baseline` SHALL equal the `train_encoder=False` baseline
  at the same cell (within tolerance)
