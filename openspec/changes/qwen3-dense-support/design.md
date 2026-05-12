# Design: qwen3-dense-support

## The minimal diff (concretely)

This change is small enough to describe by example. Three coordinated
edits + one new file.

## Edit 1: `NativeModelConfig` — one new field

```diff
 # Qwen2-specific (Llama-shaped but with Q/K/V biases). False for Llama
 # and Gemma-2 (no biases on attention projections). The o_proj remains
 # bias-free across all families.
 qkv_bias: bool = False
+# Qwen3-specific. When True, the attention block applies RMSNorm(head_dim)
+# on Q and K per head between projection-and-reshape and the scaled dot-
+# product. Llama / Gemma-2 / Qwen2 default to False (no q_norm/k_norm).
+qk_norm: bool = False
```

`_SUPPORTED_FAMILIES` extends to `("gpt2", "llama", "gemma2", "qwen2", "qwen3")`
and `_build_torch_module`'s Llama-family dispatch picks up `"qwen3"`.

## Edit 2: `LlamaSelfAttention` — conditional q_norm/k_norm

Today's attention block (in the closure inside `_get_forged_llama_class`)
looks roughly like:

```python
class LlamaSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...
        qkv_bias = getattr(cfg, "qkv_bias", False)
        self.q_proj = nn.Linear(...)
        ...

    def forward(self, x):
        q = self.q_proj(x).view(..., self.num_heads, self.head_dim).transpose(-3, -2)
        k = self.k_proj(x).view(..., self.n_kv_heads, self.head_dim).transpose(-3, -2)
        v = self.v_proj(x).view(..., self.n_kv_heads, self.head_dim).transpose(-3, -2)
        # RoPE applied here in real models; v0 forge skips rotary (ε_rope)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        ...
```

After the change:

```python
class LlamaSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...
        qkv_bias = getattr(cfg, "qkv_bias", False)
        self.q_proj = nn.Linear(...)
        ...
        # Qwen3 inserts an RMSNorm(head_dim) on Q and K AFTER per-head
        # reshape and BEFORE SDPA. Llama / Gemma-2 / Qwen2 set
        # cfg.qk_norm=False and these stay as None.
        if getattr(cfg, "qk_norm", False):
            self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps or 1e-6)
            self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps or 1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x):
        q = self.q_proj(x).view(..., self.num_heads, self.head_dim).transpose(-3, -2)
        k = self.k_proj(x).view(..., self.n_kv_heads, self.head_dim).transpose(-3, -2)
        v = self.v_proj(x).view(..., self.n_kv_heads, self.head_dim).transpose(-3, -2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        ...
```

Two new submodules + one short conditional in forward. The `RMSNorm`
class is already in the same factory closure (used by `input_layernorm`
and friends), so this is a reuse, not a new dependency.

## Edit 3: walker emits q_norm/k_norm pass-through

Qwen3's HF model has these as direct submodule attributes:

- `model.layers[i].self_attn.q_norm.weight` (shape `(head_dim,)`)
- `model.layers[i].self_attn.k_norm.weight` (shape `(head_dim,)`)

These are NOT residual-aligned. They live in the head_dim subspace,
so the projector leaves them untouched — same pattern as Qwen2's
Q/K/V biases:

```python
for qkv in ("q_proj", "k_proj", "v_proj"):
    b = getattr(block.self_attn, qkv).bias
    if b is not None:
        out[f"{prefix}.self_attn.{qkv}.bias"] = to_numpy(b)
# NEW: Qwen3 q_norm / k_norm pass through unprojected.
for qk in ("q_norm", "k_norm"):
    norm = getattr(block.self_attn, qk, None)
    if norm is not None:
        out[f"{prefix}.self_attn.{qk}.weight"] = to_numpy(norm.weight)
```

This block lives inside `LlamaAdapter.walk` (the shared walker, also
inherited by Qwen2Adapter). Adding it there means *all* Llama-family
adapters get q_norm/k_norm pass-through whenever the host has those
attributes — which only Qwen3-and-later have. Llama / Gemma-2 / Qwen2
hosts have no `q_norm` attribute, so the `if norm is not None` guard
no-ops them. Backward-compat preserved.

## Edit 4: new `Qwen3Adapter`

```python
# saeforge/adapters/qwen3.py (new, ~60 LOC)

from saeforge.adapters.llama import LlamaAdapter

class Qwen3Adapter(LlamaAdapter):
    """Adapter for HF Qwen3ForCausalLM.

    Qwen3 dense is Llama-shaped (SwiGLU MLP, RMSNorm, GQA, RoPE) with
    one structural addition: RMSNorm(head_dim) on Q and K per head
    between projection-and-reshape and SDPA. No Q/K/V biases (vs
    Qwen2). The Llama walker emits q_norm/k_norm pass-through when
    those attributes exist on the host; the Llama factory builds the
    forward-side RMSNorm modules when cfg.qk_norm is True.
    """

    family = "qwen3"

    def build_native_config(self, host, n_features, *, attention_width="host"):
        from dataclasses import replace

        base = super().build_native_config(
            host, n_features, attention_width=attention_width
        )
        qk_norm = (
            len(host.model.layers) > 0
            and getattr(host.model.layers[0].self_attn, "q_norm", None) is not None
        )
        return replace(base, family=self.family, qk_norm=qk_norm)


try:
    from transformers import Qwen3ForCausalLM
    from saeforge.adapters import register_adapter
    register_adapter(Qwen3ForCausalLM, Qwen3Adapter())
except ImportError:  # transformers < 4.51, or Qwen3 not built in
    pass
```

That's the whole adapter. It inherits `walk` (which now emits
q_norm/k_norm pass-through for any host that has them) and overrides
only the family tag + `qk_norm` detection.

The Qwen2 adapter is similarly thin — this is the deliberate pattern
of "Llama is the canonical Llama-family walker; sub-families override
only what truly differs."

## Why detect `qk_norm` from the host instead of the config

HF `Qwen3Config` does expose attributes that could be checked
(`use_qk_norm`-style flags), but those flags' names have varied across
the model family's history and across HF transformers versions. The
robust check is "does the host's first attention block have a
`q_norm` submodule," which is true iff the HF code path constructed
one — i.e. iff the feature is actually in use. Same pattern as the
qkv_bias auto-detection.

## Validation tier matrix (recap)

| Tier | Where Qwen3 tests run |
|---|---|
| T0 GitHub Actions CI | Skip (no transformers in `[dev]`) |
| T1 Intel Mac `[intel]` | Skip (transformers <4.50, no Qwen3) |
| T1.5 Intel Mac `[torch]` | Possible if user upgrades torch off the Intel pin |
| T2 M4 Apple Silicon | Real — `Qwen3-0.6B` / `Qwen3-1.7B` smoke on M4 |
| T3 NVIDIA/CUDA | Larger Qwen3 dense (4B, 8B) — community follow-up |

The shipping gate is T0 + T1 green (regression check on all *non*-Qwen3
surfaces) plus a one-time T2 smoke run before merge. This is the same
gating Qwen2 had, except T1's role flips from "validates Qwen2
end-to-end" to "regression-checks everything *but* Qwen3."

## Risks

### Risk: RMSNorm placement is wrong

Qwen3's actual HF code applies q_norm/k_norm directly on the
per-head reshaped tensor — shape `(..., num_heads_or_kv_heads, head_dim)`.
RMSNorm operates on the last dim by default, which is `head_dim`. The
proposed forward matches this layout. Confirmed by inspection of
`transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward`
(transformers ≥ 4.51). Worth pinning via a numeric correctness test
on a small synthetic host (see tasks.md §4.4).

### Risk: q_norm/k_norm interaction with RoPE

In the HF reference, q_norm/k_norm is applied *before* RoPE. The v0
forge skips RoPE entirely (accepts `ε_rope` per `docs/algorithm.md`).
Skipping RoPE doesn't change where q_norm sits — it still applies
between projection-and-reshape and the scaled dot-product. No
interaction concern.

### Risk: Walker emits q_norm/k_norm for Llama hosts (forward-compat)

The walker change is gated by `getattr(block.self_attn, qk, None) is not None`.
Llama / Gemma-2 / Qwen2 hosts return `None`, so the emit is skipped.
If a *future* Llama derivative grows q_norm/k_norm submodules, the
walker would silently emit those weights into the projected dict — but
the forged module's `cfg.qk_norm` would still be False (its
auto-detection lives in `LlamaAdapter.build_native_config`, which
doesn't peek at q_norm), and `from_projected_weights` would raise
`KeyError("projected key ... has no slot")` because the native module
wouldn't expect those keys. Tight failure mode, not a silent drift.
Acceptable; documented.

### Risk: Tests pass on M4 but I can't run them locally

The user explicitly cannot run Qwen3 tests on their `[intel]` install
(transformers < 4.50, no Qwen3 import). The pre-merge T2 M4 smoke is
load-bearing — without it, the only validation is "no other tests
regressed." Plan: the user runs the new tests once on M4 before merge
and pastes the output into the PR. If M4 access is delayed, the
proposal can ship behind a feature flag that defaults the Qwen3
adapter to "registered but not first-class," but that adds churn for
no real benefit; better to wait for the M4 confirmation.

## Why not collapse the walker change into `Qwen3Adapter.walk`

A cleaner-feeling alternative is to override the walker in
`Qwen3Adapter` rather than touch the shared `LlamaAdapter.walk`. Two
reasons not to:

1. **The guard is host-attribute-driven, not adapter-driven.** A
   future Qwen-family variant that has q_norm should emit it
   automatically; making `Qwen3Adapter.walk` the only emitter would
   force every such variant to repeat the override. Putting it in the
   shared walker, gated by host-attribute presence, is the more
   forward-compatible structure.
2. **No regression risk.** The `getattr(..., None) is not None` guard
   makes the emission inert for Llama / Gemma-2 / Qwen2 hosts.
   Existing test fixtures and integration paths are unaffected.

The pattern matches how Qwen2's Q/K/V bias emission was integrated —
inside the shared walker, gated by host-attribute presence — rather
than as a Qwen2-only override.
