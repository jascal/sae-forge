# Design — full-forge encoder training

## Decision 1: A dedicated differentiable forge forward, not a retrofit of `project_module`

`project_module` is numpy-by-contract (`_to_numpy` → float64) and feeds `NativeModel.from_projected_weights`,
which builds detached `nn.Parameter`s. Retrofitting grad-through-`E` into that path would entangle the
inference forge with a training-only concern. Instead we add a **separate** `saeforge.forge_diff
.differentiable_forge_h` that re-expresses the *same* projection algebra (`D@W`, `W@E`, `D@W@E`) as torch ops
on a grad-enabled `E` and runs the forged forward differentiably. The numpy `project_module` is untouched —
inference forges and the existing sweep stay byte-identical.

## Decision 2: `E`-only, matched capacity (inherited from the parent change)

Only `E` (shape `(d_model, n_features)`) is trainable; host weights, `W_dec`, and the downstream encoder are
fixed. This is the X2 "tied" design — it isolates the **projection-geometry** question from a full
fine-tune, and keeps the result interpretable: if `E`-only full-forge training plateaus, the residual tax is
*not* a projection problem (it's the LayerNorm/TopK structure, per bio-sae Reckoning #5).

## Decision 3: v1 scopes the differentiable forward to one host family (`esm2`)

A differentiable forge forward must mirror each adapter's architecture. Writing all of
gpt2/llama/gemma2/whisper differentiably is a large surface and not needed to settle the question — the gate
host is ESM-2. v1 implements `esm2` and **raises `NotImplementedError`** for other families (no silent
fallback to the proxy). The verdict on ESM-2 is what flips (or doesn't) the X2 null.

## Decision 4: the objective is the eval path — and the metric stays scoring-only

The loss distills the **full-forge** forged latents to the host's own latents
(`host_encoder(differentiable_forge_h(E)) ≈ host_encoder(host_X)`), label-free by default (supervised BCE
optional). This is the literal eval path, so the proxy/metric mismatch that sank X2 cannot recur. The
rank-AUC retained-mAUC stays **scoring-only on a held-out split** — never the loss (the gameable-excess guard
from X2 Decision 2 carries over).

## Decision 5: the gate is multi-seed, and BOTH outcomes are first-class

The X2 null taught us the spread deltas live inside ±1.7pp noise, so a single seed cannot settle it. The gate
runs ≥3 seeds and reports mean ± std. **Success** (mean Δ clears noise, > 0) validates the thesis;
**also-plateaus** (Δ ≈ 0 even through the full forge) is the deeper, equally-publishable finding that the
spread tax is structural beyond `E` (Reckoning #5). We pre-commit to reporting whichever happens — no tuning
until positive, no "irreducible" claim if negative (`no-necessity-claims`).

## Decision 6: cost mitigations make the full forge affordable on the gate host

The full forge forward per step is the cost. Mitigations: (a) **minibatch the sequences** per step — a
**seeded** random subset of the dataset proteins (the per-step subset is drawn from `train_encoder`'s
`seed`, so a fit is fully reproducible per seed; the gate's multi-seed run then varies *both* the
init/data-order and the minibatch draws together); (b) **precompute the host-latent target** once (it's
`E`-independent); (c) modest `steps` with early-stop on the held-out plateau. The tiny ESM-2-8M gate host
makes this tractable
on CPU; large hosts are out of scope (Decision 3).

## Alternatives considered

- **Retrofit grad into `project_module`.** Rejected (Decision 1): entangles inference with training.
- **Full `NativeModel` fine-tune.** Rejected for v1: that's the v0.3 `forge-finetune-recipe`; training `E`
  alone is the controlled experiment that answers "is the projection the problem?".
- **Straight-through / soft-rank AUC as the loss.** Rejected: the gameable-excess trap; keep AUC scoring-only.

## Open questions

- Does `esm2`'s forward go fully differentiable through the adapter's wrapped module, or does a layer need a
  torch-native re-expression? (Pre-lock task 0.x: confirm autograd reaches `E` end-to-end on a tiny ESM-2.)
- If ESM-2 full-forge training **succeeds**, does the win transfer to an LM family (next change, Decision 3)?
- If it **plateaus**, is the residual the cov95 (sharp-feature) half specifically, matching Reckoning #5's
  "distillation recovers mAUC not cov95"?
