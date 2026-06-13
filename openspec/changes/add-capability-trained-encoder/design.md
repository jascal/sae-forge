# Design — capability-trained encoder

## Decision 1: Train the encoder `E`, at matched capacity — not a new subspace, not the whole model

Three places could "see capability": (a) the kept-row **selection** (upstream in Polygram's compression),
(b) the **encoder** `E` (sae-forge's `SubspaceProjector`), (c) the full **fine-tune** of the `NativeModel`.

We choose **(b)**, and we train `E` *at the same shape as `pinv(W_dec)`* — `(d_model, n_features)`. Rationale:

- (a) is Polygram's job (FABLE X1/X4); sae-forge consumes the kept basis.
- (c) (the v0.3 `forge-finetune-recipe`) is heavier, retrains everything, and confounds "did the basis get
  better?" with "did the weights heal?". Training only `E` isolates the **basis-quality** question R2 asks.
- **Matched capacity is the load-bearing methodological choice.** R2's free head (train the full `r×V`
  readout, more capacity than the frozen lens) **overfit catastrophically** (GPT-2 open-class R@32 79%→25%);
  the matched-capacity *tied* head (train only the rank-`r` subspace, same DOF as the frozen lens) **won**
  (+13pp, converged). `E` with shape `(d_model, n_features)` has exactly the DOF of the `pinv` it replaces —
  it is the "tied" design. This is also the direct fix for the retracted `U_C` overfit (writer-OV ≈
  random-OV): same capacity, real objective, held-out gate.

## Decision 2: Default objective is label-free distillation, not label-supervised

`objective="distill"` (default): match the **host's** decoded-then-encoded latents — label-free,
differentiable, and the analogue of R2's self-distillation to the model's own output + bio-sae's
representation-distillation (Reckoning #5, "closes the mAUC half"). `objective="supervised"` (opt-in): a
differentiable BCE surrogate predicting the binary labels.

**Precise definition (locks the implementation — review Q1/Q2).** `host_encoder` is the **same downstream
task encoder** that `DownstreamCapabilityTarget` uses (the SAE / probe that maps `d_model → latent_width`) —
**not** the host transformer. It is the `encoder` field of the `CapabilityDataset`. Let `x` be a host
hidden state (`d_model`), `E` the encoder being trained (`d_model × n_features`), `W_dec` the basis decoder.
The two latent vectors compared are:

```
target     = host_encoder(x)                          # the host's own latents (the cov95 baseline path)
prediction = host_encoder( (x @ E) @ W_dec )          # forged: encode→decode (E(x) then decode), then encode
```

i.e. the trained `E` makes the **basis-projected** residual `P_E·x = (x @ E) @ W_dec` read the *same*
downstream features the host's un-projected residual does. This is exactly `DownstreamCapabilityTarget`'s
`decode_via_basis=True` pipeline, but as a differentiable latent-matching loss instead of the rank-AUC score.
(With `E = pinv(W_dec)`, `P_E·x` is the Frobenius projection of `x` onto `row(W_dec)` — the baseline; training
`E` learns a better-for-capability projection at the same DOF.)

**Loss (Q2).** Default is **cosine distance** `1 − cos(prediction, target)` per item, mean-reduced — chosen
over plain MSE because SAE latents are sparse and magnitude-skewed, so MSE is dominated by the few
high-magnitude features; cosine targets the *direction* (which the AUC reads). `loss="mse"` is available
(standardized per-feature) for callers who want magnitude-faithful matching. `objective="supervised"` uses
BCE-with-logits of the decoded latents against `dataset.labels` (the labels enter only here, never in
`distill`).

Why distill is the default: it needs no labels (works wherever a host encoder exists), it can't game the
held-out AUC metric (the metric is never the loss), and it matches the manifesto's already-measured
label-free result. The rank-AUC metric stays **scoring-only**, never the loss — this is what keeps the gate
honest (a supervised loss *on the same labels* the AUC scores would be the gameable-excess-metric trap that
sank `U_C`).

## Decision 3: Held-out split is mandatory and is the gate

`train_encoder` SHALL carve a disjoint held-out split (`holdout_frac`, default 0.3) and report retained-mAUC
on it, **and** the `pinv` baseline on the *same* held-out split. The acceptance gate is the held-out
comparison. Fit-split improvement is reported but never gates. This directly encodes the R2 / `U_C` lesson:
the failure mode is overfitting, and only a held-out compression-controlled comparison detects it.

## Decision 4: `encoder_override` is `None`-default and behavior-preserving

The projector change is purely additive: `encoder_override=None` is byte-identical to today. The override is
the **full** map (it absorbs `scale_boost`), so `encode` does not re-multiply scale when the override is set —
otherwise a round-trip through `train_encoder` (which starts at `pinv·scale`) would double-apply scale.

## Decision 5: `readout_aligned` ordering degrades gracefully on encoder-only families

The width-slice ordering by readout-aligned score needs a vocabulary unembed `U` (the top-competitor
`gain⊙U` SVD subspace). LM families (gpt2/llama/gemma2/qwen) have `U`. Encoder-only families
(esm2/whisper_encoder) have **no** vocabulary readout.

**Policy (locks the implementation — review Q5).** *Raise by default; fall back only on explicit opt-in.*
When `basis_order="readout_aligned"` is requested, the sweep detects a readout geometry by whether a
`u_matrix` (+ `gain`) was supplied OR resolvable from the host adapter (LM-family adapters expose the
unembed; encoder-only adapters return `None`). If present → readout-aligned ordering. If absent → **raise a
clear `ValueError`** naming the missing `u_matrix` and the family, **unless** the caller also passes
`readout_fallback="downstream_decode"`, in which case the sweep emits a one-shot `UserWarning` and orders by
the **downstream encoder's** own decode geometry (the SAE's `W_dec`-aligned directions) — explicitly opt-in,
never silent. Default `basis_order="row_norm"` so no family regresses by default.

## Decision 6: Per-cell encoder caching + always-on baseline (review Q3)

Sweeps fan out over many `(encoding, width, scale_boost)` cells. `train_encoder` SHALL be cheap per cell
(one linear map, ≈300 steps), but to keep the aggregate bounded: (a) trained `E` matrices SHALL be **cached**
keyed by `(basis identity hash, width, encoding, scale_boost, objective, seed)`, so re-runs and adjacent
cells that share a basis reuse the fit; (b) training SHALL **early-stop** on a held-out-score plateau
(patience, default 30 steps with no held-out improvement) so easy cells finish fast. When `train_encoder=True`
the sweep SHALL **always** compute and store the `pinv` baseline on the *same* held-out split for every cell
(apples-to-apples), not only on demand — the comparison is the whole point.

## Decision 7: Small-fixture variance (review Q4)

`holdout_frac=0.3` on the concentrated `n=16` fixture leaves ~5 held-out items — retained-mAUC there is
noisy. The acceptance gate on that fixture SHALL therefore be **multi-seed**: run `seed ∈ {0,1,2,3,4}` and
require the **mean** trained-`E` held-out retained-mAUC `≥` the mean `pinv` baseline within a stated
tolerance (report mean ± std). The spread `n=512` fixture is large enough for a single seed but reports the
held-out n for context. This keeps the "≥ baseline" bar honest under small-sample noise rather than reading a
single noisy draw as a win or a regression.

## Decision 8: Persistence + serialization (review Q6 + minor suggestion)

The `EncoderCalibrationReport` scalar fields SHALL live as **optional columns on the sweep result rows**
(`retained_mauc_trained`, `retained_mauc_pinv_baseline`, `delta_heldout`, `encoder_trained`, `overfit_flag`,
`basis_order`; all default `None`/`False`, back-compat). The trained `E` **matrix** SHALL be saved as a
**sidecar artifact** (`<cell>.encoder.npy` or a safetensors entry beside the basis) and referenced by path on
the row — keeping rows lightweight while making the trained encoder reproducible and reusable downstream
(e.g. polygram experiments). `recommend` / Pareto plotting read the new columns with `None`-safe defaults.

## Alternatives considered

- **Train the selection, not the encoder.** Rejected: selection is Polygram's contract (X1/X4); sae-forge
  shouldn't re-derive it.
- **Use the AUC metric as the loss (straight-through / soft-rank).** Rejected: that is the gameable-excess
  trap (`U_C`); keep the metric scoring-only.
- **Free encoder (more DOF than `pinv`).** Rejected on R2 evidence: the free head overfits; matched capacity
  is the fair, generalizing design.

## Open questions

- Does `objective="distill"` (host-latent matching) transfer to the **supervised** retained-mAUC as well as
  it did for fieldrun's self-distillation? Measured by the gate; if distill helps held-out AUC, it confirms
  the host latents carry the capability signal.
- Optimal `holdout_frac` / `steps` on the small bio fixtures (≈600 items) — pinned by the acceptance gate's
  reproduction, with the overfit mode explicitly watched (R2 saw under-fitting at high rank / few steps and
  over-fitting at high capacity — `train_encoder` reports both the fit and held-out curves so the regime is
  visible).
