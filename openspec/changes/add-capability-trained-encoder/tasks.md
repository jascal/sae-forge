# Implementation tasks

## 0. Design pre-locks (blocking)

- [ ] 0.1 Confirm `SubspaceProjector.encode` is the single read-path for the basis projection
  (`saeforge/projector.py:145`, `return x @ self.basis.pseudoinverse() * self.scale_boost`) and that adding
  an `encoder_override` branch there covers every caller (no other code multiplies by `pinv(W_dec)` directly).
- [ ] 0.2 Confirm the `encoder_override` is the **full** map (absorbs `scale_boost`): `encode` MUST NOT
  re-apply `scale_boost` when the override is set, else a `train_encoder` round-trip (init `pinv·scale`)
  double-applies scale. Lock this in `__post_init__` + `encode`.
- [ ] 0.3 Lock matched capacity: `encoder_override.shape == (d_model, n_features)` — identical DOF to the
  `pinv` it replaces (the "tied" design, Decision 1). Construction raises `ValueError` on any other shape.
- [ ] 0.4 Lock the held-out gate convention: `train_encoder` carves a disjoint split, reports BOTH the
  trained-E and the `pinv` baseline retained-mAUC on the *same* held-out split; the fit-split numbers are
  reported but never gate (Decision 3).
- [ ] 0.5 Confirm `objective="distill"` (label-free, host-latent matching) is the default and the rank-AUC
  metric is **scoring-only, never the loss** (Decision 2) — re-affirm against
  `saeforge/eval/targets/downstream_capability.py`'s AUC matmul (reused for scoring only).

## 1. `saeforge/projector.py` — optional trained encoder

- [ ] 1.1 Add field `encoder_override: Optional[np.ndarray] = None` to `SubspaceProjector`.
- [ ] 1.2 `__post_init__`: when `encoder_override` is not None, validate `ndim == 2` and
  `shape == (basis.d_model, basis.n_features)`; cast to `W_dec`'s dtype; raise `ValueError` otherwise.
  The existing `scale_boost` validation/auto path runs unchanged and is **ignored for `encode`** when the
  override is set (document: the override already absorbs scale).
- [ ] 1.3 `encode(x)`: `return x @ self.encoder_override` when set; else the existing
  `x @ pinv(W_dec) * scale_boost`. `decode` unchanged.
- [ ] 1.4 Unit tests `tests/test_projector_encoder_override.py`:
  - `encoder_override=None` ⇒ `encode` byte-identical to current behavior (regression guard) on `tiny_gpt2`.
  - `encoder_override = pinv(W_dec)*scale_boost` reproduces the default `encode` exactly (round-trip identity).
  - bad shape / ndim raises `ValueError` with an actionable message.

## 2. `saeforge/training/encoder.py` — `train_encoder` (matched-capacity fit)

- [ ] 2.1 New module. Lazy torch via `require_extra` (no torch import at module scope); reuses the
  `CapabilityDataset` from `add-downstream-capability-target` and that target's Mann-Whitney AUC matmul for
  **scoring only**.
- [ ] 2.2 `EncoderCalibrationReport` dataclass: `retained_mauc_trained`, `retained_mauc_pinv_baseline`,
  `delta_heldout`, `retained_mauc_trained_fit`, `objective`, `steps`, `lr`, `holdout_frac`, `n_fit`,
  `n_heldout`, `overfit_flag` (True iff fit improves but held-out regresses vs baseline).
- [ ] 2.3 `train_encoder(*, basis, dataset, objective="distill", init="pinv", steps=300, lr=1e-3,
  holdout_frac=0.3, host_encoder=None, seed=0)`:
  - Carve a disjoint fit/held-out split (seeded) over `dataset` items.
  - `E = nn.Parameter(pinv(W_dec)*scale_boost)`; Adam(lr). Loss per `objective`:
    - `"distill"`: MSE (or cosine) between `host_encoder(decode(forged_latents))` and the **host's** encoded
      latents on the fit split (label-free). Requires `host_encoder`.
    - `"supervised"`: BCE of decoded latents vs `dataset.labels` on the fit split.
  - After training: score retained-mAUC of the trained `E` AND of `pinv` on the held-out split (same items);
    fill the report; set `overfit_flag`.
  - Return `(E.detach().cpu().numpy(), report)`.
- [ ] 2.4 Pin `lr`/`steps` defaults from the gate reproduction; expose both fit and held-out curves so the
  under-/over-fit regime is visible (R2 saw both — Decision 1 / open question).
- [ ] 2.5 Unit tests `tests/test_train_encoder.py`:
  - Identity host-encoder + identity basis + `objective="distill"`: trained `E` stays ≈ `pinv` and held-out
    retained-mAUC ≈ baseline (no spurious gain on a saturated fixture).
  - A synthetic fixture where a non-Frobenius `E` is provably better (planted label-aligned direction the
    pinv misses): trained `E` held-out > baseline, `overfit_flag=False`.
  - An over-capacity / too-many-steps run on a tiny fixture trips `overfit_flag=True` (fit up, held-out down).
  - `objective="supervised"` path runs and returns a valid report.

## 3. `saeforge/sweep_capability.py` — opt-in trained encoder + readout-aligned ordering

- [ ] 3.1 `_BasisCube`: add a `readout_aligned` ordering helper (project rows onto the top-competitor
  `gain⊙U` SVD subspace) gated on a supplied `u_matrix`/`gain`; default ordering stays `row_norms`.
- [ ] 3.2 `sweep_pareto_capability(..., train_encoder=False, basis_order="row_norm", host_encoder=None)`:
  - `basis_order="readout_aligned"` selects the readout-aligned width slice **when a readout geometry is
    available**; otherwise raises a clear error (encoder-only family without `u_matrix`) OR falls back to the
    downstream encoder's decode geometry per family (documented).
  - `train_encoder=True` calls `train_encoder(...)` per cell, caches `E`, and reports the trained retained-mAUC
    alongside the pinv baseline.
- [ ] 3.3 Extend the per-cell row schema with optional `retained_mauc_trained`,
  `retained_mauc_pinv_baseline`, `encoder_trained` (bool), `basis_order` — all default `None`/`False`,
  serialization back-compat with the existing row schema.
- [ ] 3.4 Tests `tests/test_sweep_capability_trained.py`: a `train_encoder=True` sweep on a tiny fixture
  populates the new fields and the pinv-baseline column matches a `train_encoder=False` run at the same cell.

## 4. `saeforge/cli.py` — flags

- [ ] 4.1 `sae-forge sweep capability` gains `--train-encoder` and `--basis-order {row_norm,readout_aligned}`.
- [ ] 4.2 `sae-forge recommend` prefers a trained-encoder row when its **held-out** retained-mAUC clears the
  pinv baseline by the reported margin; ties keep the simpler (pinv) forge.
- [ ] 4.3 CLI help text documents the encoder-only-family readout caveat and the held-out gate.

## 5. Exports + docs

- [ ] 5.1 Export `train_encoder`, `EncoderCalibrationReport` from `saeforge.training.__init__` and
  `saeforge.__init__`; export the `encoder_override` field is already on the public `SubspaceProjector`.
- [ ] 5.2 Route the gate results into `FORGE_TAX_TRACK.md` (sae-forge) — trained-E vs pinv retained-mAUC on
  the two bio fixtures, held-out, with the overfit-mode note. No new doc page.

## 6. Acceptance gate (blocking merge)

- [ ] 6.1 Compression-controlled, held-out: trained-E retained-mAUC **≥ pinv baseline** on bio-sae's spread
  (`uniref50_n5000` pooled, n=512) and concentrated (`uniref50_small` residue, n=16) fixtures. A tie is a pass
  (reported descriptively); trained-E < baseline on held-out is the documented overfit failure (must surface,
  not hide). No "closes the tax" / "irreducible" language either way (`no-necessity-claims`).
