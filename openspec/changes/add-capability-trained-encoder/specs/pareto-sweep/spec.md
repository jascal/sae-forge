# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `train_encoder` fits a matched-capacity encoder, gated on a held-out split

`saeforge.training.train_encoder` SHALL fit an encoder `E` of shape
`(d_model, n_features)` — the same shape as `pinv(W_dec)` — from the initialization
`E0 = pinv(W_dec) * scale_boost`, and SHALL return `(E, EncoderCalibrationReport)`.

Constructor / call signature:

```python
train_encoder(
    *,
    basis: FeatureBasis,
    dataset: CapabilityDataset,
    objective: Literal["distill", "supervised"] = "distill",
    init: Literal["pinv"] = "pinv",
    steps: int = 300,
    lr: float = 1e-3,
    holdout_frac: float = 0.3,
    host_encoder: Callable | None = None,   # required for objective="distill"
    seed: int = 0,
) -> tuple[np.ndarray, EncoderCalibrationReport]
```

Constraints:

- The fit SHALL carve a **disjoint** fit / held-out split over `dataset` items, seeded
  by `seed`; `holdout_frac` SHALL be in `(0, 1)`.
- The trained `E` SHALL have shape exactly `(d_model, n_features)` (matched capacity —
  no additional degrees of freedom over the `pinv` it replaces).
- `objective="distill"` (default) SHALL be **label-free**: the loss SHALL match the
  forged→decoded latents to the **host** encoder's latents (MSE or cosine on host
  feature activations) and SHALL require `host_encoder`. `objective="supervised"` SHALL
  be a differentiable BCE of the decoded latents against `dataset.labels`.
- The rank-AUC retained-mAUC metric SHALL be used for **scoring only** and SHALL NEVER
  be the training loss (the gameable-excess-metric guard; see `design.md` Decision 2).
- The returned `EncoderCalibrationReport` SHALL carry, all measured on the **held-out**
  split: `retained_mauc_trained`, `retained_mauc_pinv_baseline` (the `pinv(W_dec)`
  baseline scored on the *same* held-out items), and `delta_heldout =
  retained_mauc_trained - retained_mauc_pinv_baseline`; plus `retained_mauc_trained_fit`
  (fit-split score), `objective`, `steps`, `lr`, `holdout_frac`, `n_fit`, `n_heldout`,
  and `overfit_flag` (True iff the fit-split score improves over baseline while the
  held-out score regresses below baseline).
- `train_encoder` SHALL NOT mutate `basis` or `dataset`.

#### Scenario: identity host-encoder on a saturated fixture yields no spurious gain

- **GIVEN** an identity `host_encoder`, an identity-like basis, and `objective="distill"`
- **WHEN** `train_encoder` runs on a fixture already near retained-mAUC ceiling
- **THEN** the trained `E` SHALL stay close to `pinv` and `retained_mauc_trained` SHALL be
  within tolerance of `retained_mauc_pinv_baseline` on the held-out split (no spurious win)

#### Scenario: a planted label-aligned direction the pinv misses is recovered

- **GIVEN** a synthetic fixture with a label-discriminative direction outside the
  Frobenius-optimal subspace
- **WHEN** `train_encoder` runs with that direction's labels/host latents
- **THEN** `retained_mauc_trained > retained_mauc_pinv_baseline` on the held-out split
  **AND** `overfit_flag` SHALL be `False`

#### Scenario: over-capacity / over-training trips the overfit flag

- **WHEN** `train_encoder` is run with too many `steps` on a tiny fixture such that the
  fit-split score rises above baseline but the held-out score falls below baseline
- **THEN** `overfit_flag` SHALL be `True` **AND** the report SHALL still carry both the
  fit and held-out numbers (the failure is surfaced, not hidden)

### Requirement: `sweep_pareto_capability` opt-in trained encoder and readout-aligned ordering

`saeforge.sweep_pareto_capability` SHALL accept `train_encoder: bool = False`,
`basis_order: Literal["row_norm", "readout_aligned"] = "row_norm"`, and an optional
`host_encoder`. The defaults SHALL reproduce the current sweep exactly.

- When `train_encoder=True`, the sweep SHALL fit a trained `E` per
  `(encoding, width, scale_boost)` cell via `train_encoder(...)` and SHALL report the
  trained held-out retained-mAUC **alongside** the `pinv` baseline (both on the held-out
  split). The trained `E` SHALL be applied via `SubspaceProjector(encoder_override=E)`.
- When `basis_order="readout_aligned"`, the width slice SHALL be ordered by a
  readout-aligned score (projection onto the top-competitor `gain⊙U` SVD subspace) when a
  readout geometry (`u_matrix` + `gain`) is available. When it is not available
  (encoder-only families: esm2, whisper_encoder), the sweep SHALL either raise a clear
  error naming the missing `u_matrix` OR fall back to the downstream encoder's decode
  geometry — the chosen behavior SHALL be documented per family and SHALL NOT silently
  reorder by a different criterion.
- The per-cell row schema SHALL gain optional fields `retained_mauc_trained`,
  `retained_mauc_pinv_baseline`, `encoder_trained` (bool), and `basis_order`, all
  defaulting to `None`/`False`, preserving serialization back-compat with the existing
  row schema.

#### Scenario: default sweep is unchanged

- **WHEN** `sweep_pareto_capability` is called with `train_encoder=False` and
  `basis_order="row_norm"` (the defaults)
- **THEN** the produced frontier rows SHALL match the pre-change sweep, and the new fields
  SHALL be `None`/`False`

#### Scenario: trained-encoder sweep reports both columns at a cell

- **WHEN** `sweep_pareto_capability(train_encoder=True, host_encoder=…)` runs
- **THEN** each row SHALL carry both `retained_mauc_trained` and
  `retained_mauc_pinv_baseline`, **AND** the `retained_mauc_pinv_baseline` column SHALL
  equal the `retained_mauc` a `train_encoder=False` run produces at the same cell (within
  tolerance)

#### Scenario: readout-aligned ordering on an encoder-only family is explicit

- **WHEN** `basis_order="readout_aligned"` is requested for an `esm2` / `whisper_encoder`
  forge with no `u_matrix` supplied
- **THEN** the sweep SHALL either raise a clear error naming the missing readout geometry
  OR use the documented downstream-decode-geometry fallback — never silently fall back to
  `row_norm`
