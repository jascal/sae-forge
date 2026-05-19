# Design — `add-llama-family-rope`

## Context

The Llama-family forged attention module
(`saeforge/adapters/llama.py:264-338`) has shipped without rotary
positional encoding since the family was added. The five adapters
inheriting from `LlamaAdapter` (Llama, Gemma-2, Qwen2, Qwen3,
Qwen3-MoE) all produce forged models that compute attention as a
pure bag-of-tokens — Q · Kᵀ with no rotation. For an autoregressive
LM whose host model is RoPE-based, this means the forge has been
producing models that fundamentally cannot represent the host's
ordering-sensitive computation.

The bug was masked by:

1. **Short eval prompts.** The four prompts in
   `examples/forge_gemma2_2b.py`'s `EVAL_PROMPTS` are 8-12 tokens
   each. Bag-of-tokens models do meaningfully worse than RoPE-based
   ones on longer prompts; the short suite under-detected the gap.
2. **GPT-2 reference smoke.** The host-wrapped fix
   (`add-host-wrapped-forge-fallback`) was validated on GPT-2,
   which has working absolute positional embeddings via `wpe`. So
   the smoke gate's headline KL reduction (89.9 → 15.4) did not
   exercise the same defect that Llama-family forges have been
   carrying.
3. **Confusing docs.** Three adapter docstrings claimed "GQA, RoPE"
   in their summary lines; `docs/algorithm.md` claimed positional
   handling was "identical to the host." Both are wrong for
   Llama-family hosts.

The 2026-05-19 at-scale Gemma-2-2B run (KL = 13.19, only 0.74 nats
above `ln(V_gemma)`) is what surfaced the gap empirically.

## Goal

Add RoPE to the Llama-family forge with a config-gated `rope_mode`
field so the fix is also a regression-diff knob. Default
(`"standard"`) reproduces the host's behaviour; `"none"` exactly
reproduces the pre-fix buggy behaviour for ablation. Surface the
resolved positional mode on `ForgeResult` so future silent skips
are immediately visible.

## Three implementations considered

### A. Re-implement RoPE manually in the forge (chosen)

A ~20-line `apply_rotary_pos_emb` helper in
`saeforge/_positional/rope.py`, called from
`LlamaSelfAttention.forward` after Q/K projection-and-reshape.

Pros:
- Self-contained. No new external dependency.
- Matches the existing pattern: the forge re-implements transformer
  primitives in `nn.Module` form (LlamaSelfAttention reimplements
  HF's attention; SwiGLU_MLP reimplements HF's MLP; etc.).
- Mechanical correctness is straightforward to verify: rope_mode=
  "none" arm produces byte-identical output to pre-fix; rope_mode=
  "standard" produces position-sensitive output.

Cons:
- We own the RoPE math. If HF changes their algorithm subtly
  (e.g., introduces a new rope_scaling type) we need to track it.

### B. Use `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`

Import HF's reference impl and call it from our forward.

Pros:
- Free upgrades when HF improves their RoPE.
- Strongly typed inputs/outputs.

Cons:
- Adds a transformers import to the forged-module hot path. The
  current forge is deliberately structured so the forged modules
  don't reach into transformers at forward time — only at
  weight-walk time.
- HF's `LlamaRotaryEmbedding` interface has changed several times
  in recent transformers releases; pinning to a stable surface
  costs more in maintenance than re-implementing.
- Doesn't generalize cleanly to Qwen3's `partial_rotary_factor`
  without a manual override anyway.

**Rejected** in favor of A.

### C. Don't implement RoPE; route Llama-family forges to host_wrapped by default

The `add-host-wrapped-forge-fallback` capability already lets a
host run with its own attention (RoPE included) by wrapping the
host blocks. Could we just default Llama-family forges to
`forward_mode="host_wrapped"` and skip implementing RoPE in the
native path?

Pros:
- Zero new code.
- Cooperates with the existing structural-fix surface.

Cons:
- `host_wrapped` is GPT-2 only in v1. Routing Llama-family forges
  to it just changes the failure from "silent RoPE skip" to
  "explicit NotImplementedError from Llama adapter." Doesn't fix
  the bug; just relocates it.
- Even if per-family `host_wrapped` rollouts ship later, they cost
  ~equal compute to host inference per-block (not the "small native
  transformer" story). Wrong default for the common case where the
  basis is good-tier and `native_in_basis` math is valid.
- The native-in-basis math is *correct* for RoPE — q and k come
  out of the projected Q/K matrices, get rotated, then dot-product.
  No category error analogous to the LayerNorm-pinv one. The fix
  here is "implement the missing op," not "route around an
  architectural mismatch."

**Rejected** in favor of A.

## Two regimes the fix touches

### Regime 1: Good/saturated basis on Llama-family host

Native-in-basis forge with RoPE. The chosen impl. This is the
hot path: Gemma Scope SAEs typically land at good or saturated
quality tier after compression, so auto-dispatch picks
`native_in_basis`. Pre-fix this regime was silently broken.

### Regime 2: Under-complete basis on Llama-family host

Auto-dispatch resolves to `host_wrapped`, which currently raises
`NotImplementedError` for Llama-family (per the
`forge-forward-mode` spec — only GPT-2 ships in v1). This proposal
does NOT change that behaviour. After this lands, the queued
`add-host-wrapped-{llama,gemma2,…}` follow-ups need to be
reprioritized based on the post-RoPE measurement: if KL drops
substantially in native_in_basis, the per-family host-wrapped
rollout is much less load-bearing than it was before today.

## Why a `rope_mode = "none"` knob

Three reasons to keep the buggy behaviour reachable as an opt-in:

1. **Regression diffing in the impl PR.** The first commit on the
   impl branch should re-run pre-fix tests with `rope_mode="none"`
   and demonstrate byte-identical output to main. Without the knob,
   we can only check "old test passes; new test passes" — we can't
   isolate "*this* change is the only source of behaviour
   difference."

2. **Future bisection.** If a Llama-family forge produces a
   surprising result post-fix, being able to flip `rope_mode="none"`
   and recover the pre-fix numbers is the cheapest debug step.

3. **The diagnostic surface needs a name for it.** The proposed
   `ForgeResult.positional_encoding = "none_skipped"` value is
   what a reader of the run summary sees when someone explicitly
   ran with `rope_mode="none"`. Without the field, the user
   wouldn't know they were in the regression-diff arm.

The `"none"` arm emits a `UserWarning` at config construction so
nobody accidentally ships a no-RoPE production run.

## Why the diagnostic surface is part of *this* proposal

Two reasons:

1. **The bug took ~6 hours to detect from a 13.19 KL value.** A
   `ForgeResult.positional_encoding` field showing `"none_skipped"`
   in the run summary would have made it a 60-second diagnosis. The
   marginal cost of the field is ~10 lines of code; the marginal
   value is "the next silent-skip bug surfaces immediately."

2. **The field can't be added later without breaking the canonical
   forge result schema.** `ForgeResult` is already exposed at the
   public surface (`saeforge.ForgeResult`). Adding the field here
   means it ships alongside the fix; adding it later means another
   schema migration.

Lumping vs splitting: I considered making the positional diagnostic
its own follow-up. Decided against — it's tightly coupled to the
fix (the failure mode it surfaces is exactly the bug this proposal
fixes), and splitting it would mean either shipping the fix without
visibility into similar future regressions, or doing two
schema-migration cycles.

## Cost analysis

Per-forward additional cost in the Llama-family hot path:

- `compute_rope_cache(seq_len, head_dim, theta, partial_factor)`:
  one-time per forward. Builds `(cos, sin)` of shape `(seq_len,
  head_dim)` from `theta ** (-2k/d)`. For Gemma-2-2B's
  `head_dim=256, seq_len=512`: ~1 MB of float32, ~250k flops to
  build.
- `apply_rotary_pos_emb(q, k, cos, sin)`: per-block, two small
  elementwise ops over Q and K tensors. For Gemma-2-2B at
  `(B=1, T=512, n_heads=32, head_dim=256)`: ~10 MFLOPs per block,
  ~250 MFLOPs over the 25 blocks. Negligible against the host's
  attention compute (~25 GFLOPs total).

Memory: the `(cos, sin)` cache is freed after the forward; no
persistent buffer.

Total overhead: <1% of host inference cost at the target
configuration. Within the "small constant factor" budget the host-
wrapped fallback also operates in.

## Alternatives considered

### A1. Pre-compute the RoPE cache once in `__init__`

Store `(cos, sin)` as buffers on `LlamaTransformer` indexed by the
maximum supported seq_len. Per-forward, slice to the actual seq_len.

Saves the per-forward cache compute (~250k flops, negligible) at
the cost of a fixed-size buffer (~1 MB for max_position_embeddings).
Not worth the API surface complexity. Deferred to a perf-tuning
follow-up if benchmarks show the per-forward cache compute matters.

### A2. Add a `BiasModule` instead of touching `LlamaSelfAttention.forward`

Insert a hook between Q/K projection and the dot-product that
applies rotation. More modular, but adds a new module type and
indirection layer for a one-line forward change.

Rejected. The Llama-family attention layout is small and stable;
modifying its forward in place is clearer than introducing a
generic hook surface for one use case.

### B1. Skip the regression-diff knob; just fix forward

Land RoPE in the forward unconditionally, no `rope_mode` field.

Simpler diff, but loses the regression-diff arm and the diagnostic
field's `"none_skipped"` value. Also doesn't preserve any path back
to the pre-fix behaviour for bisection.

Rejected for the three reasons in "Why a rope_mode = 'none' knob"
above.

## Open questions deferred to follow-up

- **Should the diagnostic field also surface on
  `ParetoFrontierRow`?** Probably yes, in the same follow-up that
  adds `forward_mode_resolved` to the row schema (queued from
  `add-host-wrapped-forge-fallback`). Out of v1 scope for this
  proposal.

- **What's the right re-measurement protocol for Gemma post-RoPE?**
  Re-run `examples/forge_gemma2_2b.py` on M4 with the same SAE
  layer / L0 / n-features as today's 13.19 baseline, compare KL.
  Goes into the post-impl smoke results (in this change's
  smoke-results.md) when M4 is available. The flagship-demo runbook
  has a slot for the number.

- **Do per-family host-wrapped rollouts still make sense?** After
  the re-measurement: if KL drops to within ~1 nat of an
  acceptable noise floor, the per-family host-wrapped rollouts are
  much lower priority. If KL is still high, the LayerNorm-pinv
  pathology in those families is the dominant remaining headroom
  and host-wrapped becomes the next chunk of work. Decision lives
  in the post-impl re-evaluation, not in this proposal.
