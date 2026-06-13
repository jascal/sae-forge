# Implementation tasks

## Implementation status — de-risk milestone (2026-06-13)

Landed the load-bearing core (tasks 0–2, 4.4, 5.1) + tests; the de-risk gate **passes** on a controlled
synthetic fixture (bottleneck + nonlinear ReLU host-encoder): held-out retained-mAUC **pinv 0.713 →
trained 0.839 (Δ +0.126)**, `overfit_flag=False`, early-stopped; the saturated control ties (Δ +0.000, no
spurious gain). 13 new tests green; existing projector suite unaffected.

Two implementation notes for the follow-up (sweep/CLI, tasks 3–4):
- **`train_encoder` takes the decomposed pieces** (`host_acts`, `host_encoder`, `labels`) rather than a
  `CapabilityDataset`, because `add-downstream-capability-target` is **not yet implemented** (0/36). A thin
  `dataset=` wrapper adapts onto this core when that change lands (`CapabilityDataset.encoder` IS the
  `host_encoder`). Task 2.3's signature is updated accordingly.
- The capability infra **already on the public API** (`recipe_auc_matrix`, `load_host_unembed`,
  `sweep_pareto_capability`) should be **reused** by task 3: `recipe_auc_matrix` for held-out scoring (replace
  `encoder.py`'s local `_auc_matrix`) and `load_host_unembed` as the `u_matrix` source for `readout_aligned`.

## 0. Design pre-locks (blocking)

- [x] 0.1 Confirm `SubspaceProjector.encode` is the single read-path for the basis projection
  (`saeforge/projector.py:145`, `return x @ self.basis.pseudoinverse() * self.scale_boost`) and that adding
  an `encoder_override` branch there covers every caller (no other code multiplies by `pinv(W_dec)` directly).
- [x] 0.2 Confirm the `encoder_override` is the **full** map (absorbs `scale_boost`): `encode` MUST NOT
  re-apply `scale_boost` when the override is set, else a `train_encoder` round-trip (init `pinv·scale`)
  double-applies scale. Lock this in `__post_init__` + `encode`.
- [x] 0.3 Lock matched capacity: `encoder_override.shape == (d_model, n_features)` — identical DOF to the
  `pinv` it replaces (the "tied" design, Decision 1). Construction raises `ValueError` on any other shape.
- [x] 0.4 Lock the held-out gate convention: `train_encoder` carves a disjoint split, reports BOTH the
  trained-E and the `pinv` baseline retained-mAUC on the *same* held-out split; the fit-split numbers are
  reported but never gate (Decision 3).
- [x] 0.5 Confirm `objective="distill"` (label-free, host-latent matching) is the default and the rank-AUC
  metric is **scoring-only, never the loss** (Decision 2) — re-affirm against
  `saeforge/eval/targets/downstream_capability.py`'s AUC matmul (reused for scoring only).

## 1. `saeforge/projector.py` — optional trained encoder

- [x] 1.1 Add field `encoder_override: Optional[np.ndarray] = None` to `SubspaceProjector`.
- [x] 1.2 `__post_init__`: when `encoder_override` is not None, validate `ndim == 2` and
  `shape == (basis.d_model, basis.n_features)`; cast to `W_dec`'s dtype; raise `ValueError` otherwise.
  The existing `scale_boost` validation/auto path runs unchanged and is **ignored for `encode`** when the
  override is set (document: the override already absorbs scale).
- [x] 1.3 `encode(x)`: `return x @ self.encoder_override` when set; else the existing
  `x @ pinv(W_dec) * scale_boost`. `decode` unchanged.
- [x] 1.4 Unit tests `tests/test_projector_encoder_override.py`:
  - `encoder_override=None` ⇒ `encode` byte-identical to current behavior (regression guard) on `tiny_gpt2`.
  - `encoder_override = pinv(W_dec)*scale_boost` reproduces the default `encode` exactly (round-trip identity).
  - bad shape / ndim raises `ValueError` with an actionable message.

## 2. `saeforge/training/encoder.py` — `train_encoder` (matched-capacity fit)

- [x] 2.1 New module. Lazy torch via `require_extra` (no torch import at module scope); reuses the
  `CapabilityDataset` from `add-downstream-capability-target` and that target's Mann-Whitney AUC matmul for
  **scoring only**.
- [x] 2.2 `EncoderCalibrationReport` dataclass: `retained_mauc_trained`, `retained_mauc_pinv_baseline`,
  `delta_heldout`, `retained_mauc_trained_fit`, `objective`, `steps`, `lr`, `holdout_frac`, `n_fit`,
  `n_heldout`, `overfit_flag` (True iff fit improves but held-out regresses vs baseline).
- [x] 2.3 `train_encoder(*, basis, host_acts, host_encoder, labels, objective="distill", loss="cosine",
  init="pinv", scale_boost=1.0, steps=300, lr=1e-3, holdout_frac=0.3, patience=30, eval_every=5, seed=0)`
  — decomposed inputs pending `CapabilityDataset` (status note); `host_encoder` == `CapabilityDataset.encoder`:
  - Carve a disjoint fit/held-out split (seeded) over items.
  - `E = nn.Parameter(pinv(W_dec)*scale_boost)`; Adam(lr). Loss per `objective` (design.md Decision 2,
    precise definition — `host_encoder` is the `CapabilityDataset.encoder`, NOT the host transformer):
    - `"distill"` (label-free): `dist( host_encoder((x @ E) @ W_dec), host_encoder(x) )` on the fit split,
      where `dist` is **cosine distance** (`loss="cosine"`, default) or standardized MSE (`loss="mse"`).
      Requires `host_encoder`.
    - `"supervised"`: BCE-with-logits of decoded latents vs `dataset.labels` on the fit split.
  - **Early-stop** on held-out-score plateau (`patience` steps with no held-out improvement; Decision 6).
  - After training: score retained-mAUC of the trained `E` AND of `pinv` on the held-out split (same items);
    fill the report (incl. both fit and held-out curves); set `overfit_flag`.
  - Return `(E.detach().cpu().numpy(), report)`.
- [x] 2.4 Pin `lr`/`steps` defaults from the gate reproduction; expose both fit and held-out curves so the
  under-/over-fit regime is visible (R2 saw both — Decision 1 / open question).
- [x] 2.5 Unit tests `tests/test_train_encoder.py`:
  - Identity host-encoder + identity basis + `objective="distill"`: trained `E` stays ≈ `pinv` and held-out
    retained-mAUC ≈ baseline (no spurious gain on a saturated fixture).
  - A synthetic fixture where a non-Frobenius `E` is provably better (planted label-aligned direction the
    pinv misses): trained `E` held-out > baseline, `overfit_flag=False`.
  - An over-capacity / too-many-steps run on a tiny fixture trips `overfit_flag=True` (fit up, held-out down).
  - `objective="supervised"` path runs and returns a valid report.

## 3. `saeforge/sweep_capability.py` — opt-in trained encoder + readout-aligned ordering

- [ ] 3.1 `_BasisCube`: add a `readout_aligned` ordering helper (project rows onto the top-competitor
  `gain⊙U` SVD subspace) gated on a supplied/resolvable `u_matrix`/`gain`; default ordering stays `row_norms`.
  Detection: LM-family adapters expose the unembed; encoder-only adapters return `None` (Decision 5).
- [ ] 3.2 `sweep_pareto_capability(..., train_encoder=False, basis_order="row_norm",
  readout_fallback=None, host_encoder=None)`:
  - `basis_order="readout_aligned"`: order by the readout-aligned slice when a readout geometry is available;
    when absent, **raise `ValueError`** naming the missing `u_matrix` + family, **unless**
    `readout_fallback="downstream_decode"` (then warn once + use the downstream encoder's decode geometry).
    Never silently revert to `row_norm` (Decision 5).
  - `train_encoder=True`: call `train_encoder(...)` per cell; **cache** `E` keyed by `(basis hash, width,
    encoding, scale_boost, objective, seed)`; **always** compute + store the `pinv` baseline on the *same*
    held-out split for every cell (apples-to-apples, Decision 6); apply the trained `E` via
    `SubspaceProjector(encoder_override=E)`.
- [ ] 3.3 Extend the per-cell row schema with optional `retained_mauc_trained`,
  `retained_mauc_pinv_baseline`, `delta_heldout`, `encoder_trained` (bool), `overfit_flag` (bool),
  `basis_order`, and `encoder_artifact_path` — all default `None`/`False`, serialization back-compat. The
  trained `E` matrix is saved as a **sidecar** (`<cell>.encoder.npy` / safetensors entry beside the basis)
  and referenced by `encoder_artifact_path`, keeping rows lightweight + the encoder reproducible (Decision 8).
- [ ] 3.4 Tests `tests/test_sweep_capability_trained.py`: a `train_encoder=True` sweep on a tiny fixture
  populates the new fields, the pinv-baseline column matches a `train_encoder=False` run at the same cell, the
  sidecar `E` round-trips, and `basis_order="readout_aligned"` without `u_matrix` raises (and the
  `readout_fallback` opt-in warns instead).

## 4. `saeforge/cli.py` — flags

- [ ] 4.1 `sae-forge sweep capability` gains `--train-encoder` and `--basis-order {row_norm,readout_aligned}`.
- [ ] 4.2 `sae-forge recommend` prefers a trained-encoder row only when `delta_heldout > 0.02`
  (held-out retained-mAUC margin over the pinv baseline) **and** `overfit_flag is False`; otherwise it keeps
  the simpler (pinv) forge. The `0.02` threshold is a flag (`--trained-margin`); ties default to pinv.
- [ ] 4.3 CLI help text documents the encoder-only-family readout caveat and the held-out gate.

- [x] 4.4 **De-risk script (run early, before the full sweep wiring):**
  `scripts/forge_trained_encoder_gate.py` — load one bio fixture, fit `train_encoder` at the single gate
  width, print trained vs pinv held-out retained-mAUC + `overfit_flag`. This exercises the core claim
  (tasks 1–2) on real data before the sweep/CLI surface exists, per the reviewer's de-risk suggestion.

## 5. Exports + docs

- [x] 5.1 Export `train_encoder`, `EncoderCalibrationReport` from `saeforge.training.__init__` and
  `saeforge.__init__`; export the `encoder_override` field is already on the public `SubspaceProjector`.
- [ ] 5.2 Route the gate results into `FORGE_TAX_TRACK.md` (sae-forge) — trained-E vs pinv retained-mAUC on
  the two bio fixtures, held-out, with the overfit-mode note. No new doc page.

## 6. Acceptance gate (blocking merge)

- [ ] 6.1 Compression-controlled, held-out: trained-E retained-mAUC **≥ pinv baseline**. Spread
  (`uniref50_n5000` pooled, n=512): single seed, report held-out n. Concentrated (`uniref50_small` residue,
  n=16): **multi-seed** `{0,1,2,3,4}` — require **mean** trained-E ≥ mean pinv baseline within tolerance,
  report mean ± std (Decision 7; ~5 held-out items is noisy). A tie is a pass (reported descriptively);
  trained-E < baseline on held-out is the documented overfit failure (must surface, not hide). No "closes the
  tax" / "irreducible" language either way (`no-necessity-claims`).
