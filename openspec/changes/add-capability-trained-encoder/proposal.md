# Capability-Trained Encoder — the "supervised forge" deferred by `add-downstream-capability-target`

Add an **optional trained encoder** to `SubspaceProjector`: instead of always reading the forge through
the Frobenius-optimal `E = pinv(W_dec) · scale_boost`, allow `E` to be *fine-tuned at matched capacity*
against a capability objective (host-feature representation distillation by default; label-supervised
optionally), and surface it as an opt-in in the capability sweep. Also add an opt-in **readout-aligned
ordering** for the width slice, replacing row-norm magnitude where a readout geometry is available.

## Why

`add-downstream-capability-target` made the forge tax **legible** — it measures retained-mAUC against the
downstream task instead of residual cosine — and then explicitly drew the line this change crosses:

> *"This is a **capability-aware metric**, not a capability-aware *forge algorithm*. The forge itself
> doesn't see the labels. Future work could feed labels into the projection (a 'supervised forge'), but
> that's a different proposal."* — `add-downstream-capability-target/proposal.md`, "What this does NOT solve"

This is that proposal. Today both halves of the basis are **reconstruction objects that never see the
metric the sweep optimizes**:

- the **encoder** is `E = pinv(W_dec) · scale_boost` — Moore-Penrose, i.e. Frobenius/L2-optimal
  reconstruction (`saeforge/projector.py:145`, `SubspaceProjector.encode`);
- the **width slice** orders kept rows by `row_norms` magnitude (`saeforge/sweep_capability.py`,
  `_BasisCube.order`).

The sibling decompilation track just measured the gap this leaves on the table. In `fieldrun` PR #33 (R2,
`lo3a/tau_star_trained.py`): at **matched rank and matched capacity**, a projection whose `r×d` subspace is
*trained* to a decode objective beats the frozen Frobenius/SVD projection — GPT-2 open-class R@32
**15% → 28% (+13pp)**, converged by ~150 steps, on a held-out split. The lesson is the one this repo
already discovered from the metric side (cosine ≠ capability): **the basis-selection criterion should match
the evaluation criterion.** A Frobenius encoder minimizes the wrong norm.

**Honest precedent (why "trained", not "a new subspace").** sae-forge's earlier writer-output `U_C` attempt
(`saeforge/composition_subspace.py`, `circuit_heads.py`; the "−111% induction-tax fix") was **retracted**
under compression-controlled re-validation (writer-OV ≈ random-OV, 0/6 wins — a gameable excess-metric
artifact). The lesson there is the same as R2's free-vs-tied result (the over-capacity free head overfit;
the matched-capacity tied head won): **fix the objective, at matched capacity, and gate on a held-out
compression-controlled comparison — do not graft a heuristic subspace.** This change is built around that
discipline.

## What

### 1. `SubspaceProjector.encoder_override` — an optional trained encoder

`SubspaceProjector` gains an optional `encoder_override: np.ndarray | None` field (shape
`(d_model, n_features)`, the same shape as `pinv(W_dec)`). Default `None` ⇒ **today's behavior is
byte-identical**. When set, `encode(x)` returns `x @ encoder_override` (the override already absorbs any
scale), and `decode` is unchanged (`z @ W_dec`). Matched capacity by construction — the override has the
same shape as the pinv it replaces; this is the "tied" (not "free") design from R2.

```python
proj = SubspaceProjector(basis=basis, scale_boost="auto")          # unchanged: E = pinv(W_dec)*scale
proj = SubspaceProjector(basis=basis, encoder_override=E_trained)   # NEW: E = the trained matrix
```

### 2. `saeforge.training.encoder.train_encoder(...)` — fit `E` at matched capacity

A new routine (torch-gated via `require_extra`, living beside the existing `saeforge/training/`) that
initializes `E0 = pinv(W_dec) · scale_boost` and gradient-descends `E` against a **capability objective**,
on a **fit split**, then returns `(E, EncoderCalibrationReport)` measured on a disjoint **held-out split**:

```python
from saeforge.training import train_encoder

E, report = train_encoder(
    basis=basis,
    dataset=capability_dataset,        # the same CapabilityDataset add-downstream-capability-target defined
    objective="distill",               # "distill" (label-free, default) | "supervised" (BCE to labels)
    init="pinv",                       # E0 = pinv(W_dec)*scale_boost  (the frozen baseline)
    steps=300, lr=1e-3, holdout_frac=0.3,
)
# report.retained_mauc_trained, report.retained_mauc_pinv_baseline, report.delta_heldout, ...
```

- **`objective="distill"` (default, label-free).** Train `E` so the forged→decoded→host-encoded latents
  match the **host's own** encoded latents (MSE/cosine on the host SAE's feature activations). This is the
  direct analogue of R2's self-distillation-to-the-model's-own-output and of bio-sae's label-free
  representation-distillation (manifesto Reckoning #5, "closes the mAUC half"); it needs no labels.
- **`objective="supervised"` (opt-in).** A differentiable BCE surrogate of the rank-AUC metric: the decoded
  latents predict the binary labels. Used where labels are cheap and the user wants to target them directly.

The AUC metric itself stays non-differentiable and is used **only for held-out scoring**, never as the loss.

### 3. `sweep_pareto_capability(..., train_encoder=False, basis_order="row_norm")`

Two opt-in knobs on the existing capability sweep (`saeforge/sweep_capability.py`):

- `train_encoder=True` — fit + cache a trained `E` per `(encoding, width, scale_boost)` cell and report
  `retained_mauc` **with the trained `E`** alongside the pinv baseline (both on the held-out split).
- `basis_order="readout_aligned"` — order the width slice by a readout-aligned score (projection onto the
  top-competitor `gain⊙U` SVD subspace) instead of `row_norms`, **when a readout geometry is available**
  (LM families: the host unembed `U`; encoder-only families ESM-2/whisper have no vocabulary unembed → this
  degrades to the downstream encoder's own decode geometry, documented per family). Default `"row_norm"`
  keeps today's behavior.

### 4. CLI

`sae-forge sweep capability` gains `--train-encoder` and `--basis-order {row_norm,readout_aligned}`;
`sae-forge recommend` learns to prefer a trained-encoder row when its held-out retained-mAUC clears the
pinv baseline by a reported margin.

## How (sketch)

- `saeforge/projector.py` — add `encoder_override: Optional[np.ndarray] = None`; `__post_init__` validates
  shape `(d_model, n_features)` and dtype; `encode` branches to the override (no `scale_boost` re-multiply —
  the override is the full map). All existing call sites unchanged at `encoder_override=None`.
- `saeforge/training/encoder.py` — new. `train_encoder(...)`, `EncoderCalibrationReport`. Torch via
  `require_extra`; matched-capacity gradient fit; held-out scoring reuses
  `DownstreamCapabilityTarget`'s Mann-Whitney AUC matmul.
- `saeforge/sweep_capability.py` — `_BasisCube` gains a `readout_aligned` ordering helper; `sweep_pareto_
  capability` gains `train_encoder` + `basis_order`; `_CapabilityCell`/row schema gains
  `retained_mauc_trained`, `retained_mauc_pinv_baseline`, `encoder_trained` (all optional, back-compat).
- `saeforge/cli.py` — the two flags.

## Falsifiable acceptance gate

Compression-controlled (same width, same kept rows, **only `E` differs**), scored on a **held-out** split:

| fixture | baseline (pinv) | gate on the trained `E` | falsified if … |
|---|---|---|---|
| bio-sae spread (`uniref50_n5000` pooled, n=512) | retained-mAUC ≈ 0.93 | trained-E held-out retained-mAUC **≥ pinv baseline** | trained-E < pinv on held-out (overfit — the U_C cautionary tale) |
| bio-sae concentrated (`uniref50_small` residue, n=16) | retained-mAUC ≈ 1.03 | trained-E ≥ pinv (no regression on the already-saturated regime) | trained-E degrades the saturated regime |

The headline success is a **win at matched capacity on held-out data**; a tie (≥ baseline, not better) is a
legitimate descriptive outcome ("Frobenius is already near-optimal on this substrate"), **not** a failure,
and is reported as such. A trained-E that beats on the fit split but loses on held-out is the explicit
overfit-failure mode (mirrors R2's free head and the retracted U_C) and **must** be reported, not hidden.

## What this does NOT solve

- **The structural forge tax is not eliminated.** R2 showed a trained linear projection *dents but does not
  close* the tail (best open-class R@32 ~57–62% vs ~95% closed). The cov95 sharp-feature floor on spread
  substrates (LayerNorm non-commutation + TopK rank-shuffle, Reckoning #5) is **structural, not
  Frobenius-suboptimality**; this change targets the gradient-correctable half. No "irreducible" or
  "closes the tax" claim is made — achievability of the residual stays OPEN.
- **`basis_order="readout_aligned"` needs a vocabulary unembed.** LM families have it; encoder-only families
  (ESM-2, whisper) do not — there the ordering degrades to the downstream encoder's decode geometry, which is
  documented and gated separately (it may not beat row-norm there).
- This is **one host at a time.** Cross-host transfer of the trained `E` is X5 (a separate change).

## Related

- `add-downstream-capability-target` — this change is its explicitly-deferred "supervised forge."
- `fieldrun` PR #33 R2 (`lo3a/tau_star_trained.py`, `tau_star_budget.py`): the matched-capacity
  trained-projection-beats-frozen-Frobenius result (+13pp, converged, held-out) this generalizes.
- The retracted writer-OV `U_C` (`saeforge/composition_subspace.py`, `circuit_heads.py`;
  `FORGE_TAX_TRACK.md` item 4): the cautionary precedent — fix the objective at matched capacity, gate on
  held-out compression-controlled comparison.
- `FABLE_DIRECTIONS.md` Round 3 X2 (the cross-pollination direction this implements).
- existing primitives this composes: `saeforge.SubspaceProjector`, `saeforge.sweep_pareto_capability`,
  `saeforge.eval.targets.DownstreamCapabilityTarget`, `saeforge.training.*`.
