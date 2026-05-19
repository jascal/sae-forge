# Design — `add-host-wrapped-forge-fallback`

## Context

`SubspaceProjector.project_module` and the per-family native modules
produce a "small forged transformer" whose residual stream is in
basis coordinates. Every weight matrix is projected: residual-read
parameters via `W_dec @ host_W` (decode-then-host-multiply, exact),
residual-write parameters via `host_W @ pinv` (host-multiply-then-
encode, exact), residual-aligned parameters (LayerNorm γ/β, biases
added to the residual) via `γ @ pinv` (encode-as-vector — *category
error*).

The 2026-05-18 layer-amplification diagnostic
(`scripts/diagnose_layer_amplification.py`) confirmed the category
error is the dominant source of forge KL blow-up in under-complete
regimes:

- Forged residual norms post-block-0 scale with `||pinv||_2`
  (1.71× → 9.30× across K=25 → 211).
- The amplification jumps in a single block (block 0) and then
  saturates — subsequent blocks neither amplify further nor recover.
- `via_host_kl` (per-layer KL via host's ln_f + lm_head applied to
  decoded forged residual) saturates at ~10–15 nats from layer 1
  onward, matching the documented forge output KL (89.9 at K=211).

A direct fix attempt (`scripts/validate_host_native_ln.py`) replaced
the projected LayerNorms with `decode → LN_host → encode`. Result:
non-uniform improvement (K=211 KL 89.9 → 64.4, K=163 KL 29.2 → 50.8).
The decode/LN-host arm normalizes over a rank-deficient projection of
the residual whose variance estimate diverges from a true host
residual's; the `c_proj` and bias paths still inherit
`||pinv||_2`-driven amplification. **The pathology is not a single
op — it's the basis-space-vs-host-space architectural choice.**

## Goal

Add a second forward implementation that runs every operation
host-native, preserving the residual stream in basis coordinates at
every block boundary. Default to this implementation when the basis
is under-complete; preserve the existing implementation when the
basis is high-fidelity (where it produces correct math). Dispatch by
`basis_quality_tier` (already computed in `forge_quality`).

## Two implementations, one residual contract

The interpretability contract `saeforge` ships is "residual stream is
in basis coordinates." Both forward implementations preserve that
contract at the per-block-boundary granularity.

### `native_in_basis` (the existing path)

Residual stream `z ∈ R^(n_features)` flows through every block. Every
op is the projected version of host's op. Mathematically equivalent
to host computation *only when the basis is closed under host
operations* — i.e., when `decode(LN_basis(z)) ≈ LN_host(decode(z))`,
and similarly for every other host op. This holds (approximately) for
basis quality tier `good`/`saturated` and breaks for `undersized`/
`degenerate`.

Parameter count: `n_features`-dependent, smaller than host.

### `host_wrapped` (the new path)

Residual stream `z ∈ R^(n_features)` only at block boundaries;
internally each block runs in host's `d_model`-space using host's
exact weights. Forward at block i:

```
x_host = z_i @ W_dec                   # decode (loses info iff basis is rank-deficient)
x_host_new = host_block_i(x_host)      # host-native: host γ, β, host softmax, host activation
z_{i+1} = x_host_new @ pinv            # encode
```

Mathematically equivalent to "run host transformer, project the
residual stream at every block boundary." Forge KL is bounded by the
projection error `||encode(decode(x)) - x||` accumulated across N
boundaries — for any basis this is monotone in basis rank. As rank →
d_model the bound → 0 and the two implementations converge.

Parameter count: equal to host (host weights are not projected; the
basis matrices are added).

## Dispatch

`NativeModelConfig.forward_mode`:

- `"auto"` (default): map `basis.quality_tier` to mode.
  - `saturated` / `good` → `native_in_basis` (where the existing
    math is valid).
  - `undersized` / `degenerate` → `host_wrapped`.
- `"native_in_basis"`: force existing path. Allows reproducing the
  documented blow-up for regression testing.
- `"host_wrapped"`: force fallback. Allows testing the fallback on a
  good-tier basis (should agree with `native_in_basis` within 0.1
  nats).

The default is `auto` because the *behaviour the user gets* should
respect their basis's regime. If they want the explicit math at any
tier, they pass the explicit value.

`quality_tier` is the natural switch because it's already computed,
already understood, and already documents the "is the basis large
enough to span the residual stream" question that the dispatch
hinges on. The boundary between `undersized` and `good` is `0.5 *
d_model` in basis rank, which empirically aligns with where the
native-in-basis math starts producing usable KL. The thresholds
themselves are not a new design decision — they're inherited from
`add-forge-quality-diagnostics`.

## Why not modify `native_in_basis` math?

Three options were considered for fixing native-in-basis:

1. **Replace LayerNorm with decode/LN_host/encode.** Validated
   experimentally; non-uniform improvement (worse at low K, better at
   high K). Decode/LN_host operates on a rank-deficient projection of
   the host residual; the variance estimate diverges from a true host
   residual's. Insufficient.
2. **Tikhonov-regularized pseudoinverse.** Caps `||pinv||_2` at a
   chosen threshold, which would shrink the encoded parameter
   magnitudes. Would help marginal cases but doesn't address the
   category error (per-coord γ projected as a vector). Deferred to a
   follow-up proposal that can be evaluated independently.
3. **Re-fit projected parameters via small calibration.** Run a
   calibration batch through the host, capture per-block residuals,
   solve a per-block LS problem for the projected γ/β/c_proj that
   best matches the host's per-block residual under the basis. More
   expressive than (1) or (2) but introduces a training step at
   forge-time and complicates the "forge is a pure projection"
   story. Also deferred.

The host-wrapped path is preferred over (1)–(3) for the
under-complete regime because it sidesteps the architectural
mismatch entirely: there's no "right way" to map host's per-coord
nonlinearities to a non-orthonormal basis. Running them in host
space is the honest answer.

For the good-tier regime the existing math is correct (within
projection error that vanishes as rank → d_model), so `native_in_
basis` remains the highlighted path. The flagship demo lands on
clustered polygram SAEs, which compress at quality_tier `good` or
`saturated` by design.

## Cost analysis

Per-block extra cost in host-wrapped mode (compared to host inference):

- 1 decode: `(B, T, n_features) @ (n_features, d_model) = (B, T,
  d_model)`. Flops: `2 * B * T * n_features * d_model`.
- 1 encode: `(B, T, d_model) @ (d_model, n_features) = (B, T,
  n_features)`. Same flop count.

For GPT-2 layer-8 K=211 (n_features=523, d_model=768): per block ≈
2 * 1 * 256 * 523 * 768 * 2 ≈ 400M flops. 12 blocks ≈ 4.8G flops.
Host's GPT-2 attn+MLP per block is ≈ 4 * 256 * 768² + 8 * 256 * 768
* 3072 ≈ 5.2G flops. So overhead per block ≈ 8% of host block; total
host-wrapped forward ≈ 1.08× host inference.

For Gemma-2-2B (d_model=2304, hypothetical n_features=2048): per
block decode+encode ≈ 2 * 256 * 2048 * 2304 * 2 ≈ 4.8G. Block ≈ 50G.
Overhead ≈ 10%. Within "small constant factor" budget.

Memory: host_wrapped mode stores the host transformer in addition to
the basis matrices. At GPT-2-small this is +124M params (host) vs
the ~7M params a forged `native_in_basis` model has. At Gemma-2-2B
it's +2.6B params. Not a free fallback. Documented.

## Alternatives considered

### A. Quality-tier-gated refusal

Refuse to forge when `quality_tier ∈ {undersized, degenerate}`,
documenting the limitation. Simpler one-PR change. Rejected because
research users actively want to explore under-complete bases (the
smoke regime exists for exactly this reason) — refusing would close
off the experimental surface that diagnosed this issue in the first
place.

### B. Single implementation: host-wrapped everywhere

Drop `native_in_basis` and ship only `host_wrapped`. Simpler to
maintain. Rejected because:

1. Loses the "small forged transformer" parameter-count story for
   the regime where it's valid.
2. The native-in-basis math IS correct on good/saturated tiers; we
   shouldn't replace it with strictly more compute when it works.
3. Existing tests, examples, and documentation all assume the
   native-in-basis math. Single-implementation pivot is a bigger
   change than warranted.

### C. Runtime-mixed: some blocks native, some host-wrapped

Have each block independently decide. Rejected: the dispatch decision
hinges on the basis, not the block — and the documented amplification
is in *block 0*, not a specific layer-internal phenomenon. Per-block
heterogeneity adds complexity without addressing the cause.

## Open questions deferred to follow-up

- **Whether `host_wrapped` should support fine-tune.** Adding
  fine-tune means training host weights (since those are the
  trainable parameters in this mode). That's a different objective
  than the current "fine-tune to match host" recipe — it's
  effectively continued pre-training. Out of v1 scope; queued as
  `add-host-wrapped-finetune-recipe`.

- **Whether `host_wrapped` should support `bridges=True`.** Hybrid
  bridges insert learnable layers between projected blocks. Host-
  wrapped has no projected blocks, so bridges have no anchor points.
  v1 raises a clear error. A separate proposal could redefine
  bridges as "between host blocks" but that's a substantial scope
  change.

- **Whether `quality_tier == "saturated"` should also go to
  `host_wrapped`.** Saturated means `basis_rank > d_model` —
  over-complete. The existing math is correct in this regime; native
  is preferred. Sticking with current threshold (saturated → native).

- **Multi-architecture rollout cadence.** v1 ships GPT-2 only. Other
  families raise `NotImplementedError`. The follow-up `add-host-
  wrapped-{llama,gemma2,…}` proposals are bounded mechanical work but
  not v1 critical-path.
