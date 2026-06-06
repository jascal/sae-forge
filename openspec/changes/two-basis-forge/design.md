# Design — two-basis forge

## The two-kinds-of-content claim, made algebraic

The forge projects host weights through a kept residual subspace
`S = rowspace(W_dec)` (`n_features` directions). For a residual-input
weight `W` (e.g. `W_Q`, which reads the residual), the projected forged
weight is `D @ W` where `D` (encode) and `E = W_dec` (decode) satisfy
`E @ D ≈ P_S` (the projector onto `S`); for residual-output weights it
is `W @ E`. The forged forward (`native_in_basis`) therefore runs the
host computation **as if the residual were replaced by `P_S(r)`** and
everything off `S` were discarded.

The forge tax follows from *what* `S` is and *what* the host actually
puts in the residual:

- **Assertions** are directions `d_X`. They are reconstructed well in
  aggregate (`mAUC` survives) but Polygram **merges** atoms to compress,
  so individual forged latents stop being single-feature detectors
  (`cov95` collapses).
- **Computation** is the pair of bilinear forms the host applies to the
  residual: `M_h = W_Q^h W_K^hᵀ` (QK) and `OV_h = W_V^h W_O^h`. A rule
  such as induction is a property of `M_h` and `OV_h`, i.e. of *pairs*
  of residual directions. Projecting onto `S` only preserves
  `d_Xᵀ M_h d_Y` for the `(X,Y)` whose directions both lie in `S`. The
  directions attention actually reads/writes (e.g. the predecessor-
  identity that a prev-token head writes for a downstream induction
  head to key on) are **not** the assertion atoms, and are not
  guaranteed to be in `S`. When they fall outside `S`, the forged QK/OV
  no longer reproduces the host's match/move and the circuit breaks —
  silently, because global KL barely moves while the rare
  circuit-driven tokens degrade.

Two preserved subspaces fix the two costs:

```
U_A  = decoder directions of the top-K_A sharpest monosemantic atoms   (assertions, cov95)
U_C  = principal subspace of the host attention read/write geometry     (computation, circuits)
S'   = orthonormalise( [ U_A ; U_C ; W_dec_remainder ] )                (augmented kept subspace)
```

Projecting through `S'` and writing the `U_A ∪ U_C` rows **verbatim**
(not Polygram-merged) makes the forged weights reproduce the host
exactly on those directions: assertions stay monosemantic, and
`d_Xᵀ M_h^forged d_Y = d_Xᵀ M_h^host d_Y` for all directions in `U_C`,
so the QK/OV macros — and the multi-head idioms composed from them —
are faithful.

## Extracting `U_C` from host weights

Per capture layer `ℓ`, the residual directions that affect attention
are exactly the row spaces of the query/key/value projections:

- **Read (QK) geometry.** A change to the residual `Δr` changes
  pre-softmax scores only through `Δr W_Q^h` and `Δr W_K^h`. Stack the
  per-head query and key projections column-wise,
  `R_ℓ = [ W_Q^1 | W_K^1 | … | W_Q^H | W_K^H ]` (shape `d_model × 2·H·head_dim`),
  and take the top-`r` left-singular vectors of `R_ℓ`. These are the
  residual directions attention *reads*.
- **Write (OV) geometry.** Attention *adds* `Σ_h (attn_h · (r W_V^h)) W_O^h`
  to the residual; the column space of `W_V^h W_O^h` is what it can
  write. Take the top-`r` left-singular vectors of
  `[ W_V^1 W_O^1 | … | W_V^H W_O^H ]`.
- `U_C^ℓ = orthonormalise( read_dirs ∪ write_dirs )`, optionally
  restricted to a supplied `composition_heads` list (e.g. only the
  prev-token / induction / copy heads identified by an analysis pass)
  to shrink the budget.

Rank `r` defaults to a per-layer singular-value-knee (the
`scale_boost="auto"`-style heuristic, logged), bounded by
`composition_rank` when supplied. The whole `U_C^ℓ` is small — at most
a few multiples of `head_dim` — so the preserved-dimension budget
(reported as a fraction of `d_model`) stays well under the basis size.

## Why read directly from weights, not learn it

`U_C` is a property of the *host*, fixed before any fine-tuning. It is
the macro basis from the `lm-sae` analysis (`B_h`, `V_h` are these same
`M_h`, `OV_h` read in feature coordinates). Reading it by SVD keeps the
mechanism a pure projection-time change with no new optimiser, and
keeps the algebraic guarantee exact (verbatim preserve), rather than
approximate (learned bridge). The learnable-bridge direction is
`hybrid-bridge-forge`; the two are composable and deliberately kept
separate here.

## Circuit-faithfulness metric

Global `KL(host ‖ forged)` is dominated by the common, assertion-driven
next-token mass and is nearly blind to circuit breakage (induction-
predictable tokens are ~6% of tokens in the `lm-sae` shakespeare slice).
The new metric restricts KL to a boolean token mask:

- `induction_predictable[t]` — the next token equals what followed the
  current token's previous occurrence (the `lm-sae` rung-3 label).
- `in_context_repeat[t]` — the current token recurs in-context.

`circuit_kl` reports `KL` on the mask and its complement separately. The
shipping invariant is **induction-predictable KL(two-basis) ≤
induction-predictable KL(single-basis)** on matched bases/seed, with no
global-KL regression beyond tolerance. `assertion_cov95` re-probes the
forged residual with the host oracle and reports the monosemantic-
detector fraction (the `lm-sae` `cov95`).

## Risks / open questions

- **Budget vs. fidelity.** Preserving `U_C ∪ U_A` spends kept-subspace
  capacity that the Polygram basis would otherwise use for residual
  reconstruction. If the budget is too large the global KL regresses;
  the comparison harness sweeps `composition_rank` / `assertion_k` to
  find the knee. Conservative defaults: small `r`, circuit-head-
  restricted `U_C`.
- **LayerNorm.** The host applies `ln_1` before QK/OV; the read
  geometry is strictly the row space of the projections *after* the LN
  gain fold. v1 folds `ln_1.weight` into `R_ℓ` before the SVD (the same
  fold the `lm-sae` probes use); the LN mean-subtraction is treated as
  approximately rank-1 and not removed. This approximation is **logged
  explicitly per layer** — `extract_composition_subspace` records an
  `ln_meansub_approx` flag (and the dropped-rank magnitude) in
  `CompositionSubspace.metadata`, surfaced in the run report, so a
  consumer can see the approximation is in force rather than inferring it.
- **`U_A` scope: global vs per-layer.** v1 keeps `U_A` **global** — the
  sharp atoms of the single capture-layer SAE, applied at every layer —
  because the v0 basis is itself a single-capture-layer object and `U_A`
  must live in the same coordinate frame to stack with it in
  `kept_subspace`. `U_C` is genuinely **per-layer** (attention geometry
  is per-block). Per-layer assertion atoms — re-deriving a sharp-atom set
  at each block — are a plausible refinement but require per-layer SAEs
  (the `hybrid-bridge` multi-anchor direction) and are deferred to
  `per-layer-assertion-atoms`.
- **Tied embeddings.** Unlike `hybrid-bridge-forge`, two-basis forge has
  no embed/lm-head basis split, so tied embeddings are *not* a problem
  — `U_C`/`U_A` are per-capture-layer subspaces of a single basis. No
  refusal needed.
- **Does it replicate at scale.** The motivating numbers are laptop
  `lm-sae` (GPT-2, self-trained SAE, single seed). The Polygram-
  compressed production SAE may already place much of `U_C` inside `S`
  (in which case two-basis ≈ single-basis and the toggle correctly does
  nothing). The harness measures exactly this overlap
  (`dim(U_C ∩ S) / dim(U_C)`) and reports it; a high overlap is itself
  an informative result about the production basis.
