## Why

The forge ships adapters for GPT-2, Llama-3, Gemma-2, and Qwen2. The
queued roadmap (`Qwen3 + Qwen3-MoE forge support` discussion) starts
with **Qwen3 dense** — a structurally Llama-shaped family with two
small differences that the current Qwen2 adapter mis-handles:

1. **Q/K per-head normalization (`q_norm`, `k_norm`).** Qwen3 inserts
   an `RMSNorm(head_dim)` on Q and K *after* the per-head reshape and
   *before* the scaled dot-product. These two small parameter tensors
   (one per attention block, shape `(head_dim,)`) are head-dim aligned
   — they live in attention-internal space, not residual space, so
   they pass through the projector unchanged. Today's Qwen2 adapter
   does not emit them; pointed at a Qwen3 host, it silently produces a
   forged model whose attention scores diverge from host.
2. **No Q/K/V biases.** Qwen2 has them; Qwen3 dense doesn't. This is
   *already* handled by the auto-detection logic in
   `LlamaAdapter.build_native_config` (`host.model.layers[0].self_attn.q_proj.bias is None`
   → `qkv_bias=False`), so Qwen3 dense doesn't need new bias-handling
   code — it picks up `qkv_bias=False` for free. Worth noting because
   it makes the patch smaller than Qwen2 was.

Everything else (SwiGLU MLP, two-norm RMSNorm structure, GQA, RoPE,
tied-embedding refusal, hybrid-bridge integration via the now-shipped
`build_llama_family_module` factory) is identical to Qwen2's path.

This change is the prerequisite for the larger `qwen3-moe-support`
follow-up: MoE shares the dense Q/K-norm requirement and adds its own
MLP routing logic on top. Landing dense first separates the
attention-correctness concern from the routing-correctness concern.

## What Changes

### Scope

Add `Qwen3Adapter` (`saeforge/adapters/qwen3.py`) inheriting from
`Qwen2Adapter` with two responsibilities:

1. Override `walk()` to emit `model.layers.{i}.self_attn.q_norm.weight`
   and `model.layers.{i}.self_attn.k_norm.weight` as pass-through
   tensors (not projected — they're `(head_dim,)`-shaped, not
   residual-aligned).
2. Override `build_native_config()` to set
   `qk_norm=True` when the host's first attention block exposes a
   `q_norm` attribute. Family stamped as `"qwen3"`.

Extend the Llama-family factory (`build_llama_family_module` in
`saeforge/adapters/llama.py`) so that when `cfg.qk_norm=True`, the
attention block:

- Constructs `RMSNorm(head_dim)` modules at `self.q_norm` and
  `self.k_norm` in `__init__`.
- Applies them between Q/K projection-and-reshape and the scaled
  dot-product in `forward`.

Add a `qk_norm: bool = False` field to `NativeModelConfig`. Default
preserves every existing family's behavior (Llama, Gemma-2, Qwen2 all
remain at `qk_norm=False`). Add `"qwen3"` to `_SUPPORTED_FAMILIES` and
to the `_build_torch_module` dispatch (routes to
`build_llama_family_module` — same factory as Llama / Gemma-2 / Qwen2).

When `cfg.qk_norm=False` (the default), the attention block constructs
no q_norm/k_norm submodules and the forward path is byte-identical to
today. This is the load-bearing backward-compat invariant — Llama,
Gemma-2, and Qwen2 single-basis + hybrid forges continue producing
identical safetensors.

### New artifacts

- **`saeforge/adapters/qwen3.py`** (new, ~60 LOC). `Qwen3Adapter`
  class inheriting from `Qwen2Adapter`. Imports `Qwen3ForCausalLM`
  inside a `try` block so old transformers installs (anything with
  the `[intel]` extra, capped at `<4.50`) silently skip registration
  — same pattern as every other adapter in the package.
- **`tests/integration/test_hybrid_bridge_qwen3.py`** (new). Mirrors
  `test_hybrid_bridge_qwen2.py`. Smoke / round-trip / disabled
  equivalence. Plus a Q/K-norm-correctness test asserting that
  `q_norm.weight` and `k_norm.weight` appear in the forged state_dict
  and round-trip cleanly.
- **`tests/test_qwen3_adapter.py`** (new). Unit tests for adapter
  dispatch, walker key-set, `qk_norm` auto-detection. Skipped when
  `Qwen3ForCausalLM` is unavailable in the installed transformers.

### Modified artifacts

- **`saeforge/model.py`** — `NativeModelConfig` gains
  `qk_norm: bool = False`. `_SUPPORTED_FAMILIES` and the
  `_build_torch_module` dispatch include `"qwen3"`.
- **`saeforge/adapters/llama.py`** — `LlamaSelfAttention.__init__`
  constructs `q_norm = RMSNorm(cfg.head_dim, eps=eps)` and
  `k_norm = RMSNorm(cfg.head_dim, eps=eps)` when `cfg.qk_norm=True`,
  else `None`. `LlamaSelfAttention.forward` applies them per head
  between the Q/K reshape and SDPA when present; pre-this-change
  forward path runs unchanged when `cfg.qk_norm=False`.
- **`saeforge/adapters/__init__.py`** — register the new `qwen3`
  adapter alongside the existing four families.
- **`tests/conftest.py`** — new `tiny_qwen3_untied_4layer` fixture
  parallel to the Qwen2 one. Uses `pytest.importorskip("transformers",
  minversion="4.51")` so Intel installs that can't see Qwen3 skip
  gracefully.

### Validation tiering (T1-T4 ladder)

The Qwen3 adapter's CI signal is structurally weaker than past
adapters' because `[intel]`-extra installs (the dev machine the project
treats as the cross-architecture defaults-validation surface) cannot
host a Qwen3 model. The proposal accepts this and tiers as follows:

| Tier | Install / hardware | Qwen3 testable? | Owner |
|---|---|---|---|
| T0 (CI on GitHub Actions) | `[dev]` only | **No.** Skipped via `importorskip`. | Pre-merge — non-Qwen3 tests still gate |
| T1 (Intel Mac, `[intel]` extra) | torch 2.2.2 + transformers<4.50 | **No.** Skipped via `importorskip`. | Maintainer can confirm no regressions on other adapters |
| T1.5 (Intel Mac, `[torch]` extra) | torch fresh + transformers≥4.51 | **Yes — but requires upgrading off Intel-pin.** Adds friction. | Optional |
| T2 (M4 Apple Silicon) | `[torch]` extra with current transformers | **Yes.** Real tiny `Qwen3-0.6B` / `Qwen3-1.7B` end-to-end smoke. | Maintainer |
| T3 (external NVIDIA/CUDA) | `[torch]` extra | **Yes.** Larger Qwen3 dense (4B, 8B) + hybrid comparison harness. | Community / follow-up issue |

The shipping criterion is T0 + T1 green (regression gate for all
*non*-Qwen3 surfaces) plus T2 green (one-time real Qwen3 confirmation
on M4 before merge). T3 ships as a post-merge follow-up. The Intel
defaults-validation surface that worked for Qwen2 simply cannot work
for Qwen3 without an extras upgrade — that's a property of HF's release
cadence, not a fixable design issue.

### CLI surface

Unchanged. The existing `sae-forge forge` flags (`--host-model`,
`--hybrid-bridge`, etc.) work against any registered host class
including Qwen3 once `Qwen3ForCausalLM` is importable.

### Out of scope (deferred)

- **`qwen3-moe-support`.** MoE adds the router + per-expert SwiGLU
  decomposition, expert-pruning compression modes, and routing-collapse
  risk. Separate change; depends on this one. Tracked as the next
  roadmap item.
- **Bumping the Intel `[intel]` extra to support Qwen3.** Doing so
  requires moving off torch 2.2.2, which is the last x86_64 macOS
  wheel. Out of scope — would be a bigger Intel-tier rewrite (or a
  deprecation of `[intel]` in favor of CPU-only `[torch]`).
- **Cross-family comparison harness for Qwen3.** The existing
  `scripts/compare_single_vs_hybrid_gpt2.py` is GPT-2 specific. A
  generic `--host-model` extension to it is a separate ergonomic
  change tracked elsewhere.
- **Sliding-window attention.** Qwen3 has it for some context-length
  modes. The native module uses standard causal attention everywhere,
  accepting long-context drift as `ε_attn` per `docs/algorithm.md` §5.
  Same posture as Qwen2 and Gemma-2.

## Capabilities

### New Capabilities

- **`qwen3-dense-support`** — defines the `Qwen3Adapter` walk
  contract (every Qwen2-emitted key set MINUS the Q/K/V biases PLUS
  the new q_norm/k_norm passthroughs), the `qk_norm` field
  auto-detection rule, and the attention-block forward contract
  (RMSNorm applied per head between projection-and-reshape and SDPA).

### Modified Capabilities

None. `architecture-adapters` already says "new architectures are
added by registering an `ArchitectureAdapter`, not by extending the
projector" — Qwen3 fits the registration pattern. The
`hybrid-bridge-llama-family` capability already covers Qwen3 by
inheritance (Qwen3 routes through `build_llama_family_module`); no
change to that spec.

## Impact

- **No public API breakage.** Single-basis Llama/Gemma-2/Qwen2 forges
  are unchanged. Forges against any newly-supported Qwen3 host run
  end-to-end (subject to `transformers≥4.51`).
- **Hybrid bridge works for free.** Qwen3 inherits hybrid-bridge
  support via the shared `build_llama_family_module` factory wired in
  PR #20. Adding a Qwen3 integration test gate to the
  `hybrid-bridge-llama-family` capability spec's family-coverage
  requirement is part of this change.
- **CI signal shrinks for the Qwen3 surface.** All Qwen3 tests skip
  in CI (`[dev]` install) and on Intel (`[intel]` install). The user
  runs them on M4 pre-merge; community runs on CUDA post-merge.

## Sequencing

- **Depends on:** `hybrid-bridge-llama-family` (PR #20, already on
  `main`). Qwen3 builds on the Llama-family bridge wiring shipped
  there.
- **Blocks:** `qwen3-moe-support`. MoE shares the Q/K-norm
  requirement and the family-tag plumbing this change adds.
- **Single PR.** ~60 LOC source (adapter) + ~30 LOC factory edits +
  ~150 LOC tests + ~20 LOC conftest. No staged rollout — the
  byte-equivalence gate on existing families passes or it doesn't.
