# Smoke gate results — `add-host-wrapped-forge-fallback`

Run 2026-05-18 on Intel 16GB MBP (Python 3.11 / torch 2.2.2 /
saeforge 0.5.1). Target: jbloom GPT-2 layer-8 SAE sliced to
top-1024-by-norm (same artifact used by
`fix-scale-boost-calibration`), HEA_Rung2 n_qubits=10,
K ∈ {25, 103, 163, 211}, scale_boost=1.0. Calibration corpus =
`saeforge.calibration._BUILTIN_CALIBRATION_TEXT` truncated to 256
tokens.

Reproducers:

- `scripts/diagnose_layer_amplification.py` — per-layer residual
  divergence, identified block 0 as the sole amplifier and pinpointed
  `||pinv||_2` as the driver.
- `scripts/diagnose_projection_magnitudes.py` — confirmed projected
  residual-aligned parameter magnitudes track `||pinv||_2`.
- `scripts/validate_host_native_ln.py` — falsified the cheaper
  alternative (decode-LN_host-encode) that fixes only the LayerNorm
  pathology.
- `scripts/prototype_host_wrapped_forward.py` — the prototype this
  proposal lands.

Per-layer trajectory artifacts in `reports/layer_amplification/`:
`summary.csv`, per-K `k_{K}.json`, and post-fix
`k_{K}_host_native_ln.json` for the falsified alternative.

## Headline

The host-wrapped forward path **removes the rank-dependent
amplification** documented by `fix-scale-boost-calibration`'s smoke
gate. K=211 forge KL drops 89.9 → 15.4 nats (5.8× reduction).
Native-in-basis remains correct on good/saturated tiers — the
fallback is additive, not a replacement.

## Diagnosis

The 2026-05-16 smoke gate left the structural fix deferred with the
note "characterise which projected layer(s) drive the amplification."
The per-layer instrumentation does so.

**Block 0 is the sole amplifier.** Forged residual norms post-
block-0 jump from 0.40× to 52.3× host norms across K=25 → 211. All
12 subsequent blocks roughly preserve that magnitude. `via_host_kl`
(per-layer KL via host's ln_f + lm_head) saturates at ~10–15 nats
from layer 1 onward.

**The driver is `||pseudoinverse(W_dec)||_2`.** Forged-to-host
parameter-norm ratio for every projected residual-aligned parameter
tracks the pseudoinverse's max singular value:

| K   | n_feat | pinv_max | ln_1.w ratio | c_proj.w ratio | block-0 norm-ratio |
|-----|--------|----------|--------------|-----------------|---------------------|
| 25  | 34     | 1.71     | 0.20         | 0.27            | 0.40                |
| 103 | 177    | 3.92     | 0.52         | 0.68            | 2.92                |
| 163 | 287    | 4.56     | 0.82         | 1.00            | 8.39                |
| 211 | 523    | 9.30     | **2.09**     | **2.08**        | **52.34**           |

**Root cause: LayerNorm doesn't commute with basis change.**
`LayerNorm.weight` and `LayerNorm.bias` are per-coordinate gains, not
vectors in residual space. The current code projects them via
`encode(γ) = γ @ pinv` — a category error. Per-coord gains in host
space have no isomorphism to per-coord gains in basis space when the
basis is non-orthonormal.

## Falsified alternative

A direct fix attempt replaced the projected LayerNorms with
`decode → LN_host → encode`. Non-uniform improvement:

| K   | KL native | KL host-LN | Δ KL              |
|-----|-----------|------------|-------------------|
| 25  | 9.65      | 13.75      | **+4.10 worse**   |
| 103 | 37.91     | 42.39      | **+4.48 worse**   |
| 163 | 29.23     | 50.76      | **+21.52 worse**  |
| 211 | 89.91     | 64.44      | −25.46 better     |

The decode/LN-host arm normalizes over a rank-deficient projection
`decode(z) = z @ W_dec` whose variance estimate diverges from a true
host residual's. The `c_proj` and bias paths still inherit
`||pinv||_2`-driven amplification. **Fixing LayerNorm alone is
insufficient — the pathology is the basis-space-vs-host-space
architectural choice, not a single op.**

This rules out cheaper "patch a single op" fixes and motivates the
host-wrapped fallback.

## Host-wrapped acceptance arm

`HostWrappedGPT2` prototype: decode at every block boundary, host-
native block (host weights, host nonlinearity), re-encode. Entry =
host wte+wpe; exit = host ln_f + lm_head.

| K   | n_feat | KL native (smoke) | KL host_wrapped | reduction       |
|-----|--------|-------------------|------------------|-----------------|
| 25  | 34     | 9.65              | 9.58             | ~0%             |
| 103 | 177    | 37.91             | 9.65             | **3.93×**       |
| 163 | 287    | 29.23             | 12.48            | **2.34×**       |
| 211 | 523    | 89.91             | 15.42            | **5.83×**       |

Adjacent-pair ΔKL in host-wrapped: +0.06, +2.84, +2.94. No pair
exceeds 10 nats. The documented amplification (+22.7, −4.0, +59.1
in native mode) is **gone**.

### Non-monotonicity is a property of the bases, not the forward

Host-wrapped KL is not monotone in K on this smoke. That's expected:
the four smoke bases are non-nested — each K target picks a
different decoder subset, so basis approximation quality varies
independently of K. Host-wrapped KL is bounded by per-basis
approximation error to host's residual stream, accumulated over 12
encode/decode round-trips. That bound is irreducible by any forge-
side change.

The proposal's acceptance gate reflects this: ΔKL ≤ 10 nats and
host_wrapped ≤ native at every K, but **not** monotone in K.

## Good-tier sanity

Synthetic basis: `n_features = d_model = 768`, orthonormal `W_dec`
(QR decomposition of an iid Gaussian). On this basis the encode/
decode round-trip is exact (`pinv @ W_dec = I`), so host-wrapped
should equal host computation byte-identically.

Prototype result: **KL ≈ 0** (`-0.0000` to float precision). PASSED.

This is the test that distinguishes the host-wrapped contract from a
heuristic: at zero approximation error (orthonormal full-rank
basis), forge KL is exactly host KL = 0.

## What this change does NOT fix

- **Forge quality on under-complete bases.** Host-wrapped removes
  amplification, not basis-approximation error. K=25 KL in
  host-wrapped (9.58 nats) is still high because a 34-feature basis
  cannot meaningfully approximate a 768-dim residual stream.
  Improving this requires improvements to the basis itself (more
  features, better polygram compression) — out of scope.
- **The `native_in_basis` math.** Host-wrapped is a *dispatch
  target* under `auto` mode; the existing forward path is unchanged.
  On good/saturated bases (where the existing math is valid), auto
  dispatches to `native_in_basis` and behaviour is byte-identical to
  v0.5.1.
- **Fine-tune support for host-wrapped mode.** v1 raises clearly.
  Deferred to `add-host-wrapped-finetune-recipe`.
- **Non-GPT-2 family rollout.** v1 GPT-2 only. Other families raise
  `NotImplementedError` with follow-up pointers.
