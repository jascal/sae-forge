# Design — writer-output U_C

## The algebra: preserve what the circuit writes, not what it reads

A forge replaces a layer's residual with its basis reconstruction. A
circuit breaks when that reconstruction smears the **specific signal the
circuit reads** — for induction, the predecessor-identity that the
prev-token head wrote. That signal is the prev-token head's **OV output**.

Head `A`'s output map is `OV_A = W_V^A W_O^A` (d×d, rank ≤ head_dim). The
set of residual vectors it can write is `image(src ↦ src · OV_A) =
rowspace(OV_A)` — the span of `OV_A`'s rows, equivalently
`rowspace(W_O^A[head_slice, :])`, dimension ≤ head_dim. The corrected
`U_C` is the orthonormalised union of the writer heads' OV-output row
spaces:

```
U_C = top-r right-singular-vectors of  [ OV_{A1} ; OV_{A2} ; … ]   (writers A1, A2, …)
```

Preserving `U_C` verbatim in the projection (the existing augmented-basis
displacement mechanism, unchanged) keeps the predecessor-write the SAE
would otherwise smear, so the downstream circuit still reads it.

The **shipped `U_C` is the reader-layer geometry** — the SVD of
`[W_Q^h | W_K^h]` and `W_V^h W_O^h` over the *capture/reader* layers. For
the induction circuit those are layers 5–7; the writer (`4.11`) is at
layer 4 and is **not in that subspace at all**, which is precisely why it
does nothing (−6% in the alive forge).

## Identifying the writer heads

`composition_heads` selects the writers. Two forms:

- **Explicit** — a list of `(layer, head)`. The caller already knows the
  circuit (from analysis / an idiom library).
- **Preset** — `"prev-token"` / `"duplicate-token"`. `circuit_heads`
  runs one forward pass over the eval corpus
  (`attn_implementation="eager"`) and returns the top-`k` heads by Δ=1
  attention (prev-token movers, the induction feeders) or same-token
  earlier attention (duplicate-token). This is the minimal idiom
  detector; the writer heads are a *per-model, per-circuit* behavioral
  fact, not a constant.

`"all"` keeps the legacy reader-geometry path for ablation, documented as
the falsified-weaker option.

## Why there is no functional shortcut

The obvious label-free alternative — preserve the top directions of
`∂(circuit-loss)/∂(residual)` — was tested and **falsified**. Its overlap
with the writer subspace is **0.05** (orthogonal), and it does not
protect the circuit (excess +0.733 vs single +0.643). It does give the
*best global KL* (2.02), because the gradient points at the directions
the *output* is most sensitive to. Those are the high-magnitude
prediction directions, not the small-magnitude, mechanistically-specific
predecessor-write. The general lesson:

> **Loss-sensitivity ≠ circuit-mechanism.** Where the output is sensitive
> to the residual is a different subspace than where the circuit's signal
> lives. Circuit preservation requires *mechanistic* identification of
> the writer heads; it cannot be recovered by differentiating the loss.

This is why the behavioral writer-detection (`circuit_heads`) is
load-bearing rather than a convenience.

## The honest cost — a real trade, not a free lunch

Writer-output preserve costs **~+0.7 global KL** (3.30 → 4.02 on the
alive GPT-2 forge): the verbatim writer dimensions spend
SAE-reconstruction budget. And because writer-output (circuit-best) and
attribution (global-best) subspaces are nearly orthogonal (overlap 0.05),
**one small preserved subspace cannot deliver both** circuit fidelity and
global fidelity. The mechanism is therefore the right lever *for
circuit-faithful forging specifically*; a forge optimising aggregate KL
should not enable it. The run report surfaces both numbers so the
trade is explicit, not hidden.

## Scope of the evidence

The result is a single-layer **alive** forge (layer-5 residual of GPT-2,
self-trained TopK SAE, blocks otherwise host) of the induction circuit.
Single-layer is used because the whole-model single-basis forge of GPT-2
collapses to uniform output at every scale (a single basis cannot
reconstruct twelve residual streams — a separate, real finding). The
mechanism — preserve the circuit writers' OV output — is general; the
*specific* writers and the whole-model setting (which needs multi-layer
`hybrid-bridge` bases) are per-circuit / follow-up. The validated claim is
narrow and honest: **on an alive forge, preserving the circuit writers'
OV-output protects the circuit where aggregate geometry and gradient
attribution do not.**
