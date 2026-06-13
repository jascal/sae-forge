# Full-Forge Encoder Training — fix the proxy/metric mismatch the capability-trained-encoder gate exposed

Train the encoder `E` against the **full forge path** (backprop through the forged `NativeModel` forward),
not the cheap activation proxy. This is the follow-up the `add-capability-trained-encoder` formal bio gate
explicitly pointed to after returning a **null**.

## Why

`add-capability-trained-encoder` shipped the trained-encoder surface (`SubspaceProjector.encoder_override`,
`train_encoder`, the sweep integration, the CLI) and ran a formal acceptance gate on bio-sae's real ESM-2
fixtures. **The gate returned a null** — the trained encoder did not systematically beat the Frobenius
`pinv` (spread deltas ±1.7pp, sign-inconsistent across widths, mean ≈ 0; concentrated ties) — and it
**diagnosed why**:

> `train_encoder` optimizes an **activation proxy** — `host_encoder((x @ E) @ W_dec) ≈ host_encoder(x)` —
> but the sweep scores the **full forge**: `E` is applied to the host *weights* (`project_module`), the
> forged `NativeModel` runs its forward, and the forged hidden state `forged_h` is **not** `host_X @ E`. The
> training objective is therefore mismatched to the metric it is evaluated on — this repo's
> cosine-vs-capability lesson, one level up. *(See `add-capability-trained-encoder/proposal.md` "Acceptance
> gate — RESULT" and `design.md` Decision 9.)*

The de-risk validated the *mechanism* (a trained `E` beats `pinv` when the training objective **is** the
eval path). This change makes the objective the eval path: train `E` so the **forged** latents — produced by
the full forge with `E`-projected weights — read the downstream features, scored on a held-out split.

**The one architectural blocker, named up front.** `SubspaceProjector.project_module` returns **numpy**
weights (`_to_numpy` detaches everything to float64), so the current forge is **not differentiable w.r.t.
`E`**. The core of this change is a **torch-native differentiable forge forward** where `E` is a grad-enabled
parameter flowing through the weight projection *and* the `NativeModel` forward.

## What

### 1. A differentiable forge forward — `saeforge.forge_diff.differentiable_forge_h`

```python
forged_h = differentiable_forge_h(host, basis, E, input_ids, aggregator, feed)  # torch, grad flows to E
```

Mirrors the numpy `project_module` → `NativeModel` → forward pipeline, but with **`E` a torch tensor**: the
weight projections (`D @ W`, `W @ E`, `D @ W @ E` — all differentiable matmuls) build the forged weights as
functions of `E`, and the forged forward runs differentiably to produce `forged_h(E)`. Only `E` carries
grad; host weights, `W_dec`, and the downstream encoder are fixed buffers. **v1 scopes the differentiable
forward to the gate's host family (`esm2`)**; other families (`gpt2`/`llama`/`gemma2`/`whisper`) raise a
clear `NotImplementedError` and are a follow-up (no silent fallback).

### 2. `train_encoder(..., objective="forge_distill")` — fit `E` through the full forge

Extend `saeforge.training.train_encoder` (or add `train_encoder_full_forge`) with the full-forge objective:

```
target     = host_encoder(host_X)                                  # the host's own latents (fixed)
prediction = host_encoder( differentiable_forge_h(host, basis, E, seqs_minibatch) )   # the FULL forge, diff'able
loss       = cosine_distance(prediction, target)                    # (or supervised BCE)
```

Matched capacity (only `E`, the tied design from X2), held-out compression-controlled gate, `overfit_flag`,
early-stop — all inherited from `add-capability-trained-encoder`. Cost mitigations: **minibatch the
sequences** per step (the full forge forward is the cost), precompute + cache the host-latent target, and
keep `steps` modest. The rank-AUC metric stays scoring-only.

### 3. Sweep + CLI

`sweep_pareto_capability(train_encoder=True, train_objective="proxy"|"full_forge")` selects which objective
fits `E` per cell; `full_forge` re-scores the same full-forge metric the proxy was null on. CLI:
`sae-forge sweep-capability --train-encoder --train-objective full_forge`.

## How (sketch)

- `saeforge/forge_diff.py` — new. `differentiable_forge_h(...)`: build the `esm2`-family forged forward in
  torch with `E`-projected weights (grad to `E`); reuse the adapter's layer structure. Lazy torch.
- `saeforge/training/encoder.py` — add `objective="forge_distill"`; route the loss through
  `differentiable_forge_h`; everything else (split, held-out scoring, `overfit_flag`) unchanged.
- `saeforge/sweep_capability.py` — `train_objective` param; when `full_forge`, `_run_capability_cell` fits
  `E` through the differentiable forge instead of the proxy.
- `saeforge/cli.py` — `--train-objective`.

## Falsifiable acceptance gate (the gate the proxy failed)

Re-run `scripts/forge_trained_encoder_bio_gate.py` with `--train-objective full_forge` on bio-sae's spread
(n ∈ {64, 128, 256}) and concentrated fixtures, compression-controlled, held-out. **Two honest outcomes,
both reported descriptively:**

- **Success** — full-forge-trained `E` held-out retained-mAUC **> pinv baseline** by a margin that *clears
  noise* (multi-seed mean, not a single draw), on at least the spread mid-widths. This validates the
  trained-encoder thesis on the real forge and turns the X2 null into a win.
- **Also-plateaus** — full-forge training **also** fails to beat `pinv` (within noise). This is the deeper
  finding: the spread forge tax is **structural beyond the encoder** — consistent with bio-sae's Reckoning
  #5 (a label-free representation-distillation fine-tune recovers the mAUC half but not the cov95 half). It
  would say the *projection geometry itself* is near-optimal and the tax lives in LayerNorm
  non-commutation / TopK rank-shuffle, not in `E`.

Either way the verdict is **descriptive**, multi-seed, and carries **no "irreducible" / "closes the tax"
claim** (`no-necessity-claims`). Falsified-as-a-feature: if full-forge training *also* plateaus, we say so.

### Gate RESULT (real bio-sae ESM-2, 2026-06-13) — the ALSO-PLATEAUS outcome

Implemented (tasks 1–5) and run on bio-sae's real spread fixture (`scripts/forge_trained_encoder_bio_gate.py
--train-objective full_forge`, multi-seed). **Full-forge training also does NOT beat `pinv` — it is
consistently *slightly worse*:**

| width | seeds | pinv | trained (full_forge) | Δ mean ± std | overfit |
|---:|---:|---:|---:|---:|:--:|
| 128 | 0,1,2 | 1.0053 | 0.9909 | **−0.0144 ± 0.0052** | no |

The negative is **consistent across seeds (~2.8σ), not noise**, and `overfit_flag` is False. So we hit the
**also-plateaus** branch — and it is the deeper, equally-valuable finding the proposal pre-committed to:
**even E-only training through the *correct* objective (autograd verified to reach `E` end-to-end by the
task 0.1 spike) does not beat the Frobenius `pinv`.** That isolates the result cleanly — the **basis
*projection* (`pinv`) is near-optimal for this substrate**, and the spread forge tax lives **structurally
beyond `E`** (LayerNorm non-commutation / TopK rank-shuffle in the downstream encoder), not in the projection
geometry. This confirms bio-sae's **Reckoning #5** (a representation-distillation fine-tune recovers the mAUC
half but not the cov95 half) from a new, controlled angle.

**Honest caveats (no-necessity-claims).** (i) *Methodological:* the sweep row's Δ is compression-controlled
over all proteins (the trained `E` saw the fit subset — an *optimistic* scoring), yet it still loses;
`train_encoder`'s internal held-out `E` is early-stop-protected. (ii) *Budget:* 80 steps, 120 proteins, the
label-free distill objective, n=128 — **achievability via more steps/data or a supervised-through-forge
objective stays OPEN**; we report the null rather than tune until positive. (iii) The **mechanism** (autograd
through the forge) is *validated* (the spike); this negative is about whether E-only training *helps*, not
whether it's *possible*. **The change ships the validated machinery; the science is the also-plateaus
finding.**

**(iv) HOST-CLASS caveat — the big one (do NOT over-generalize this null).** The trained-encoder hypothesis
was *motivated* by `lm-sae`'s R2 result (a trained rank-`r` projection beats frozen SVD, +13pp) — measured on
**causal autoregressive LMs** (GPT-2, SmolLM, Pythia), where the forge tax = the **open-class lexical / Zipf
tail of the decode distribution**. But this gate ran on **ESM-2, which is NOT a causal model** — it is a
*bidirectional masked encoder* over a 20-letter amino-acid alphabet, with **no autoregressive decode, no
readout-aligned decision geometry, and no open-class-lexis heavy tail**. So the very structure R2's trained
lens exploited *plausibly does not exist in ESM-2*. The honest reading of this null is therefore **host-class-
specific**: "on a *non-causal* protein encoder, trained-`E` does not beat `pinv`" — **not** a universal
"projection is near-optimal." The decisive, still-open test is the trained encoder on a **causal-LM forge**
(GPT-2/Llama host + an LM SAE), where R2's structure is actually present; the X2 *proxy* path already supports
LM families, so it is testable now. Until that runs, every "trained-`E` doesn't help" statement carries the
qualifier *"on non-causal hosts."* (Tracked as the follow-up causal-LM-forge experiment.)

## What this does NOT solve

- **Not a full fine-tune.** Only `E` is trained; the host weights and the `NativeModel` are otherwise fixed.
  This isolates "is the *basis projection* near-optimal?" from "does the *whole model* heal?" (the v0.3
  `forge-finetune-recipe`). If `E`-only training plateaus, the residual is not a projection-geometry problem.
- **One host family in v1** (`esm2`). The differentiable forward for LM families (gpt2/llama/gemma2) and
  whisper is deferred; they raise `NotImplementedError`, no silent fallback.
- **Cost.** The full forge forward per step is far heavier than the proxy. v1 targets the tiny ESM-2-8M gate
  host; scaling the differentiable forge to large hosts is out of scope.

## Related

- `add-capability-trained-encoder` — this change's parent; its gate null + Decision 9 motivate this work.
- bio-sae Reckoning #5 / `docs/forge-capability-bottleneck.md` — the structural-tax prior the "also-plateaus"
  outcome would confirm.
- v0.3 `forge-finetune-recipe` — the full-model fine-tune this `E`-only training is the controlled subset of.
