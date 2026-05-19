# Smoke gate results — `add-llama-family-rope`

Run 2026-05-19 on Intel 16GB MBP (Python 3.11 / torch 2.2.2 /
transformers 4.49.0 / saeforge 0.6.0). Fixture: a tiny
random-initialised `LlamaForCausalLM` (vocab=512, hidden=64,
n_heads=4, 2 layers, rope_theta=10000.0) forged against an
**identity basis** (`W_dec = I_d`) so the projection is exactly
identity and the only mathematical deviation between forge and
host is the missing RoPE step.

Reproducer: `scripts/prototype_llama_rope.py`. Outputs at
`reports/llama_rope/summary.json`.

## Headline

The Llama-family RoPE bug is **mechanically confirmed**; the proposed
fix (insert `apply_rotary_pos_emb(q, k, cos, sin)` between Q/K
projection-and-reshape and the SDPA) **exactly recovers host
behaviour** on the identity-basis fixture (gap 7.5e-7 vs no-RoPE
gap 1.71e-2 — a **22,671× improvement**).

The Gemma-2-2B at-scale gate (KL 13.19 → target <6.0) is M4-only
and fills in post-impl. The mechanical math of the fix is validated.

## Gates

| Gate | Threshold | Measured | Status |
|---|---|---|---|
| 1. No-RoPE forge differs from host above float noise | > 1e-4 | 1.71e-02 | **PASS** |
| 2. RoPE-patched forge matches host on identity basis | < 1e-4 | 7.52e-07 | **PASS** |
| 3. Fix improves forge-vs-host fidelity by ≥ 100× | ≥ 100× | 22,671× | **PASS** |
| 4. NativeModelConfig round-trips through to_dict/from_dict | byte-identical | identical | **PASS** |

Overall: **PASS**.

## Design corrections made mid-prototype

The proposal's original gate design was wrong on two axes; both got
caught at the prototype phase, which is exactly the smoke-gate
cadence's job. The proposal's `## Falsifiable acceptance gate`
section now reflects the corrected design.

### Correction 1: "Position-invariance" framing was wrong

The original proposal asserted that the no-RoPE forge would be
*position-invariant* on inputs that differ only in prefix order
(e.g., `[1,2,3,7]` vs `[3,2,1,7]`). The first prototype run measured
L2 = 1.23 — clearly not invariant.

Why the original framing was wrong: causal-masked attention computes
intermediate hidden states differently when the prefix tokens are
reordered, *even without RoPE*. Token at position 0 in input A
(value=1) produces a different layer-1 hidden state than token at
position 0 in input B (value=3); those propagate through subsequent
layers' K and V matrices. The last-token output ends up different
between A and B regardless of RoPE.

The bug is NOT "no-RoPE forge is order-invariant." It's "no-RoPE
forge is order-sensitive in a way that doesn't match host." The
correct gate measures **forge-vs-host distance**, not
**forge-A-vs-forge-B distance**.

### Correction 2: Random orthonormal basis introduces drift that swamps RoPE signal

The original prototype used a random orthonormal basis (`W_dec`
from QR of an iid Gaussian) on the theory that the projection
would be near-identity. Empirically the no-RoPE forge differed
from host by L2 = 4.94, and the RoPE-patched forge differed by
the *same* L2 = 4.94. The RoPE step's signal was completely
obscured.

Root cause: even an orthonormal `W_dec` rotates the host's
per-coordinate `LayerNorm` γ/β into new coordinates. Per-coord
gains don't commute with basis change (the same "category error"
that motivated the host-wrapped fallback in `forge-forward-mode`).
With random orthonormal basis, the LN drift is ~5 L2 — vastly
larger than the RoPE effect this prototype is measuring.

**Fix**: use `W_dec = I_d` (literal identity). Encode/decode is
exactly identity, the LN parameters round-trip unchanged, and the
only forge-vs-host gap is whatever RoPE-related math is missing.

This is a prototype-only fixture choice. Production users always
have a non-identity basis (they're forging a *compressed* SAE), so
the production faithfulness regime is "RoPE math correct + small
basis-projection error from the actual basis quality." The
identity-basis gate validates the RoPE math; the M4 Gemma run
validates the production regime.

## What the gates DON'T validate

- **Scale of the bug at production size.** The 2-layer/4-token
  fixture under-represents the compounding effect of RoPE absence
  across 25 layers and longer sequences. The 1.7e-2 no-RoPE gap
  here corresponds to a 13.19 forge KL at Gemma-2-2B scale; the
  prototype can't measure that on Intel. **Gate 5** (M4 Gemma KL
  drop) is the at-scale validation.

- **Per-family correctness.** The prototype exercises generic
  `LlamaForCausalLM` only. Gemma-2 (4 norms per block), Qwen3
  (Q/K-norm), and Qwen3-MoE (router + experts) inherit from
  `LlamaAdapter` and share the same `LlamaSelfAttention` forward,
  so the RoPE fix lands once and benefits all four. Per-family
  assertion tests (task §8.1 in `tasks.md`) cover each in the impl
  PR.

- **`rope_scaling` types beyond default.** Linear / dynamic / yarn
  / longrope scaling are explicitly out of v1 scope. The prototype's
  fixture uses the default-no-scaling regime.

## The inline RoPE implementation in the prototype

`scripts/prototype_llama_rope.py` carries a self-contained
~20-line RoPE implementation (`compute_rope_cache`, `rotate_half`,
`apply_rotary_pos_emb`) that mirrors HF's reference math. When
`patch_forged_with_rope` mounts a replacement `forward` on each
`LlamaSelfAttention` instance, the modified forward path is:

```python
q, k, v = q_proj/k_proj/v_proj followed by reshape   # unchanged
cos, sin = compute_rope_cache(seq_len, head_dim, rope_theta)
q, k = apply_rotary_pos_emb(q, k, cos, sin)          # NEW
# ... Q/K norm (Qwen3 only), GQA expansion, SDPA ... # unchanged
```

This is the exact shape the production code will take. The impl PR
moves the helper to `saeforge/_positional/rope.py` and the patched
forward into `saeforge/adapters/llama.py` (replacing the existing
`LlamaSelfAttention.forward`), gated by `cfg.rope_mode`.

## Files changed by this prototype

- `scripts/prototype_llama_rope.py` — the prototype itself
  (~330 lines including the inline RoPE math and gate machinery).
- `reports/llama_rope/summary.json` — measurements as structured
  data.
- `openspec/changes/add-llama-family-rope/proposal.md` — `##
  Falsifiable acceptance gate` revised: original "position-
  invariance" framing replaced with "forge-vs-host gap on identity
  basis." Same revision pattern as
  `add-host-wrapped-forge-fallback`'s Band C strict/advisory split
  after its prototype.
