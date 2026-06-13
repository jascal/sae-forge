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

`objective="distill"` (default): match the **host's** decoded-then-encoded latents (the host SAE's feature
activations) — label-free, differentiable, and the analogue of R2's self-distillation to the model's own
output + bio-sae's representation-distillation (Reckoning #5, "closes the mAUC half"). `objective="supervised"`
(opt-in): a differentiable BCE surrogate predicting the binary labels.

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
(esm2/whisper_encoder) have **no** vocabulary readout — there the ordering falls back to the **downstream
encoder's** decode geometry (the SAE's own `W_dec`-aligned directions), which is *not* guaranteed to beat
row-norm and is gated separately per family. Default `basis_order="row_norm"` so no family regresses
silently.

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
