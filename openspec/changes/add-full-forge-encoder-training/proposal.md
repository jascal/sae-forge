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
--train-objective full_forge`, multi-seed). **Full-forge training does NOT reliably beat `pinv` — at the
cleanest (non-overfit) width it is *slightly worse*, and its one positive width is overfit-tainted:**

| width | seeds | pinv | trained (full_forge) | Δ mean ± std | overfit |
|---:|---:|---:|---:|---:|:--:|
| 128 | 0,1,2 | 1.0053 | 0.9909 | **−0.0144 ± 0.0052** | no |
| 256 | 0,1,2 | 0.9612 | 1.0006 | +0.0395 ± 0.0222 | **yes** |

At **n=128** the negative is consistent across seeds (~2.8σ), `overfit_flag` False — a clean small loss. At
**n=256** the Δ flips positive (+0.0395) but `overfit_flag` is **True** (the trained `E` beat `pinv` on the
fit subset while regressing on the internal held-out — the +0.0395 is the *optimistic* all-protein row delta,
not a trustworthy win). The activation-**proxy** width sweep is likewise sign-inconsistent (single-seed:
n=16 −0.017, n=64 +0.0065, n=128 −0.0003, n=256 +0.0133). **Net: no reliable E-only win on this substrate** —
the **also-plateaus** branch the proposal pre-committed to. Even E-only training through the *correct*
objective (autograd verified to reach `E` end-to-end by the task 0.1 spike) does not reliably beat the
Frobenius `pinv`, so for **this substrate** the basis *projection* (`pinv`) is **near-optimal** and the spread
forge tax lives structurally beyond `E` (LayerNorm non-commutation / TopK rank-shuffle), not in the projection
geometry — consistent with bio-sae's **Reckoning #5** (a representation-distillation fine-tune recovers the
mAUC half but not the cov95 half). **But see the host-class caveat (iv) and its now-resolved causal control
below: this near-optimal reading is ESM-2-(non-causal)-specific, not universal.**

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

### Causal-LM control RESULT (2026-06-13) — the host-class caveat is CONFIRMED: the null is non-causal-specific

The caveat's decisive test, run (`scripts/causal_lm_forge_gate.py`; results in
`scripts/causal_lm_forge_gate_results.json`). **Scope honesty first:** the *exact* analog — the full
multi-layer `NativeModel` forge on GPT-2 — is **blocked**, not optimistic-supported: the cached jbloom GPT-2
SAEs live on a *mid-layer* residual (`blocks.8.hook_resid_pre`), but `ForgedGPT2.forward` emits only
*final-layer logits* with no intermediate-hidden-state extraction, and the sweep's `_extract_*` helpers are
ESM-shaped (`host.esm` / `last_hidden_state` / `[1:-1]`). Generalising that is a separate plumbing change (see
"What this does NOT solve"). So the control runs the question **R2 is actually about** — the *projection* at
the **activation level**: at matched compressed rank `N`, does a *trained* `E` beat the closed-form `pinv`
through the decode∘encode bottleneck? `train_encoder(objective="distill")` runs exactly that, host-agnostic,
held-out, compression-controlled. We ran the **identical** gate on a **causal** host (GPT-2 layer-8 residual +
jbloom ReLU/L1 SAE) and the **non-causal** control (ESM-2 + bio TopK SAE), matched N=1400, 3 seeds, same
SAE-self-label protocol:

| width | **GPT-2 (causal)** Δ | **ESM-2 (non-causal)** Δ |
|---:|---|---|
| 64  | +0.0154 ± 0.011 (overfit) | +0.0081 ± 0.001 (overfit) |
| 128 | **+0.0312 ± 0.0020** (clean) | +0.0016 ± 0.0013 |
| 256 | +0.0230 ± 0.003 (overfit) | 0.0000 (tie) |
| 512 | **+0.0702 ± 0.003** (clean) | 0.0000 (tie) |

**On the causal host, trained-`E` beats `pinv` by a clean, multi-seed, noise-clearing margin at every width**
(+0.031 to +0.070; the cleanest cell, n=128 non-overfit, is ~15σ); **on the non-causal host, identical
protocol and matched N, it barely moves** (+0.0016 at n=128, exactly 0 at n=256/512 where the bottleneck is
near/over `d_model`=320). A GPT-2 **layer sweep** (n=128, 3 seeds) confirms the causal win is **not a
single-hook fluke** — positive at every layer {1: +0.099, 4: +0.024, 8: +0.031, 11: +0.061}, U-shaped (largest
near the lexical surface: embeddings-adjacent layer 1 and readout-adjacent layer 11), consistent with R2's
open-class/Zipf-tail reading.

**Confounds, assessed honestly (no over-claim of "causality" in isolation):**
- **Compression-regime / `d_model`** (768 vs 320) — **ruled out**: GPT-2 wins *more* when *less* compressed
  (n=512 = 67% of `d` → +0.070) than ESM at 40% of `d` (+0.0016); the opposite of what this confound predicts.
- **Layer / hook point** — **ruled out**: win robust across all four GPT-2 layers.
- **SAE activation type** (GPT-2 ReLU/L1 *dense* vs ESM TopK *sparse* → better-conditioned dictionary → `pinv`
  nearer-optimal) — **STANDING**. This is the one confound the control can't isolate offline; a matched-SAE
  test (a TopK GPT-2 SAE or a ReLU non-causal SAE) is the decisive next step.

**Verdict (descriptive).** The ESM null is **host-class-specific**, exactly as caveat (iv) anticipated: the
trained-encoder learning that motivated this whole line (R2, on causal LMs) **does** reproduce on a causal
host at the projection level, and is **absent** on the non-causal protein encoder the X2/full-forge nulls were
measured on. The "`pinv` is near-optimal" reading is therefore **ESM-2-specific, not universal**. What stays
open: (1) isolating SAE-type from host-class via a matched-SAE control; (2) whether the causal projection win
*survives the full multi-layer forge* (the LayerNorm/TopK tax that erased ESM's tiny activation-level gain) —
which needs the blocked mid-layer-hidden-state forge plumbing. No "irreducible" / "closes the tax" language
(`no-necessity-claims`).

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
