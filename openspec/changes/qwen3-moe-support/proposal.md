## Why

Qwen3 ships in two structural variants:

- **Dense** (`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`,
  `Qwen3-14B`, `Qwen3-32B`) — single SwiGLU MLP per block. Shipped as
  `qwen3-dense-support` (PR #22, now on `main`).
- **MoE** (`Qwen3-30B-A3B-Base`, the "Active 3B" naming pattern Qwen3
  uses for sparse models) — the SwiGLU MLP is replaced by a router
  + N independent expert MLPs, where the router selects top-K experts
  per token and weighted-sums their outputs. Qwen3-30B-A3B has 128
  experts with 8 active per token.

The dense path supports Qwen3-MoE *projection* trivially — the
adapter dispatches, the walker emits dense-Llama keys… and then
fails, because Qwen3-MoE has no `mlp.gate_proj` etc. The host has
`mlp.gate.weight` (the router) and `mlp.experts.{i}.gate_proj.weight`
etc. (per-expert FFs). Today, pointing the dense Qwen3 path at a
Qwen3-MoE host raises `AttributeError` when the walker tries to read
the missing dense MLP.

MoE is also the first **non-dense host family** the forge would
support. The compression story changes meaningfully: 128 experts per
layer multiplies storage, and a useful new compression knob appears
(expert pruning — keep top-N most-used experts and renormalize the
router).

This change adds the `qwen3_moe` family adapter, the MoE-aware walker
(router + per-expert SwiGLU), the forged native MoE MLP, the new
`NativeModelConfig` MoE fields, and three compression modes
(`preserve` / `collapse` / `top_n`). Hybrid-bridge support comes for
free — bridges sit at the residual stream's embed/mid/lm-head
boundaries, not at the MLP boundary, so the MoE choice is orthogonal
to the bridge mechanism.

## What Changes

### Scope

Add `Qwen3MoEAdapter` (`saeforge/adapters/qwen3_moe.py`) inheriting
from `Qwen3Adapter`. Picks up the Q/K-norm and untied-bias semantics
from Qwen3 dense; overrides only the per-block MLP walk and the
native-config population. Family stamped as `"qwen3_moe"`.

Extend the Llama-family factory (`build_llama_family_module` in
`saeforge/adapters/llama.py`) so that when `cfg.num_experts > 0`, the
block's `mlp` attribute is constructed as a new `Qwen3MoEMLP` (router
+ expert ModuleList + top-K weighted dispatch) instead of the dense
`SwiGLU_MLP`. The dense MLP path is unchanged when `cfg.num_experts == 0`
(the default).

Add four new fields to `NativeModelConfig`:

| Field | Type | Default | Semantics |
|---|---|---|---|
| `num_experts` | int | `0` | Total expert count per layer. `0` = dense MLP path (existing). |
| `num_experts_per_tok` | int | `0` | Top-K router selection. Required when `num_experts > 0`. |
| `moe_intermediate_size` | int | `0` | Per-expert FF inner width. Independent of `intermediate_size` (which the MoE path ignores). |
| `norm_topk_prob` | bool | `True` | Renormalize the top-K probabilities to sum to 1 after the gate softmax. Matches the HF Qwen3MoE default. |

Add `"qwen3_moe"` to `_SUPPORTED_FAMILIES` and to the
`_build_torch_module` dispatch (still routes to
`build_llama_family_module` — same factory).

When `cfg.num_experts == 0` (the default), every existing family's
behavior is byte-identical to today. This is the load-bearing
backward-compat invariant — Llama, Gemma-2, Qwen2, Qwen3 dense, and
GPT-2 single-basis + hybrid forges all continue producing identical
safetensors.

### New artifacts

- **`saeforge/adapters/qwen3_moe.py`** (new, ~100 LOC). `Qwen3MoEAdapter`
  inherits from `Qwen3Adapter`. Overrides the per-block MLP walk to emit
  the router + per-expert keys. Overrides `build_native_config` to
  populate the four new MoE fields from host config attributes. Imports
  `Qwen3MoeForCausalLM` inside a `try` block so old transformers
  installs silently skip registration — same pattern as every other
  adapter.
- **`scripts/smoke_qwen3_moe.py`** (new, runnable). NVIDIA-scale smoke
  script targeting a real Qwen3-MoE host (`Qwen/Qwen3-30B-A3B-Base` by
  default). Loads via `device_map="auto"`, dispatches the adapter,
  walks the host, builds the native config, constructs the forged MoE
  module, runs one forward pass, optionally logs per-expert utilization
  on a short prompt. Memory budget: ≥80GB GPU recommended for the host;
  forged model is ~3–4GB and fits anywhere. Bundled in this PR per
  reviewer request so it's available the moment the adapter ships.
- **`tests/integration/test_qwen3_moe_adapter.py`** (new). Synthetic
  small Qwen3-MoE host (3 layers, 4 experts, top-2, head_dim small) for
  unit-level walker + native-module reconstruction tests. Runs on M4
  in seconds — no real-MoE host load needed.

### Modified artifacts

- **`saeforge/adapters/llama.py`** — `LlamaBlock.__init__` branches on
  `cfg.num_experts > 0`: dense path constructs `SwiGLU_MLP`, MoE path
  constructs `Qwen3MoEMLP`. Forward is unchanged at the block level —
  both MLP classes expose the same `(B, T, hidden) -> (B, T, hidden)`
  forward signature.
- **`saeforge/adapters/__init__.py`** — register the new `qwen3_moe`
  adapter alongside the existing five families.
- **`saeforge/model.py`** — `NativeModelConfig` gains the four MoE
  fields. `_SUPPORTED_FAMILIES` and `_build_torch_module` dispatch
  include `"qwen3_moe"`. `__post_init__` validates that when
  `num_experts > 0`, `num_experts_per_tok > 0` and
  `moe_intermediate_size > 0` and `num_experts_per_tok <= num_experts`.
- **`tests/conftest.py`** — new `tiny_qwen3_moe_untied` fixture (small
  synthetic MoE config for adapter unit tests; doesn't need real
  Qwen3-MoE weights).

### Compression modes (controlled by ForgePipeline)

Three modes selected via a new `moe_strategy: Literal["preserve", "collapse", "top_n"] = "preserve"`
field on `ForgePipeline`:

| Mode | Storage | Behavior fidelity |
|---|---|---|
| **`preserve`** (default) | `num_experts × forged_expert_size` per layer | Full. Forged MoE has the same expert count + top-K as host. |
| `collapse` | `1 × forged_expert_size` per layer | Degraded. Average all experts into a single dense MLP per layer; the router is removed and the block becomes effectively dense. Loses MoE-specific behavior; useful when storage is the binding constraint. |
| `top_n` | `N × forged_expert_size` per layer | Partial. Requires a calibration pass — run the host across a corpus, log per-expert activation frequency, keep the top-N most-used experts, renormalize the router. Needs `moe_keep_n` field. |

`top_n` is more complex than `collapse` because it requires a
calibration utility (`scripts/calibrate_moe_experts.py` — tracked as
a deferred follow-up, NOT shipped here). v1 ships `preserve` and
`collapse` only; `top_n` is a placeholder enum value that raises
`NotImplementedError` with a pointer to the calibration follow-up
issue.

### CLI surface

Two new flags:

```
--moe-strategy {preserve,collapse,top_n}   # default: preserve
--moe-keep-n N                              # required when --moe-strategy=top_n
```

`--moe-strategy=top_n` raises an actionable error message in v1
pointing at the calibration follow-up.

### Validation tiering

This change has a more painful CI signal story than dense Qwen3 because
even a single Qwen3-MoE host weighs 60GB+ at bf16:

| Tier | Install / hardware | What runs | Notes |
|---|---|---|---|
| T0 GitHub Actions (`[dev]`) | pytest + ruff | Lint, importorskip-gated tests skip cleanly | No transformers in `[dev]` |
| T1 Intel Mac (`[intel]`) | `transformers<4.50` | All Qwen3MoE tests skip | No Qwen3 in old transformers |
| T2 M4 Apple Silicon (`[torch]`) | `transformers>=4.51` | Synthetic small Qwen3-MoE adapter tests pass (3 layers, 4 experts, top-2 routing) | M4 cannot hold real Qwen3-30B-A3B |
| **T3 NVIDIA A100/H100 ≥80GB** | `[torch]` | `scripts/smoke_qwen3_moe.py` against real `Qwen3-30B-A3B-Base` | Load-bearing — the only place the real MoE forge actually runs end-to-end |
| T4 NVIDIA multi-GPU | `[torch]` + accelerate | Larger Qwen3-MoE if/when released (35B, 70B classes) | Community follow-up |

The shipping criterion is T0 + T1 + T2 green (regression on
non-Qwen3MoE surfaces + synthetic MoE adapter correctness) plus a
**one-time T3 NVIDIA smoke run before merge**. The `[intel]` validation
surface that worked for Qwen2 cannot validate Qwen3-MoE because:
(a) `transformers<4.50` doesn't import Qwen3MoE,
(b) even with newer transformers, the 30B-A3B host won't fit in 16GB.
M4 covers (a) via synthetic small MoE; T3 covers (b).

### Out of scope (deferred)

- **Load-balancing aux loss during MoE fine-tune.** Standard MoE
  training uses an aux loss to keep expert utilization balanced. The
  forge fine-tune currently uses pure next-token CE; whether the aux
  loss is necessary depends on whether routing collapses empirically.
  Add only if empirically required. Tracked as
  `qwen3-moe-aux-loss`.
- **`top_n` expert pruning calibration utility.** Requires running
  the host on a corpus and logging per-expert activation. Separate
  follow-up (`moe-expert-calibration`); v1 ships the enum but raises
  `NotImplementedError`.
- **Sliding-window attention.** Qwen3-MoE inherits sliding-window
  from Qwen3 for some context-length modes. The native module uses
  standard causal attention everywhere; long-context drift accepted
  as `ε_attn` per `docs/algorithm.md` §5. Same posture as Qwen3
  dense and all prior families.
- **Aux-loss-free routing with per-expert bias.** Some recent MoE
  variants (DeepSeek-V3) use a per-expert bias updated outside the
  aux loss. Qwen3-MoE doesn't use this pattern as of the
  `Qwen3-30B-A3B-Base` release; revisit if/when Qwen extends the
  pattern.
- **Optimized batched expert dispatch.** The v1 forged MoE forward
  uses a `for e in range(num_experts)` loop with `(top_i == e)`
  masking. Functionally correct but ~Nx slower than a fused
  scatter-add kernel for large N. On NVIDIA this is acceptable
  for the smoke; production training wants the optimized path.
  Tracked as `moe-fused-dispatch`.

## Capabilities

### New Capabilities

- **`qwen3-moe-support`** — defines the `Qwen3MoEAdapter` walk
  contract (router + per-expert FFs), the four new MoE
  `NativeModelConfig` fields, the conditional MoE-MLP construction
  in the Llama-family factory, the compression-mode contract
  (`preserve` is full-fidelity; `collapse` produces a dense
  approximation; `top_n` raises `NotImplementedError` in v1), and
  the NVIDIA-tier smoke-script contract.

### Modified Capabilities

None. Qwen3-MoE inherits hybrid-bridge support from
`hybrid-bridge-llama-family` (the bridges sit at residual-stream
boundaries, orthogonal to MLP structure) and Q/K-norm + Llama-family
shape contracts from `qwen3-dense-support`.

## Impact

- **No public API breakage.** Single-basis dense forges (every
  existing family) are unchanged. The new `moe_strategy` /
  `moe_keep_n` knobs default off; the four new `NativeModelConfig`
  fields default to dense behavior.
- **Hybrid bridge works for free for Qwen3-MoE.** Same factory
  routing as Qwen3 dense. Adding a Qwen3-MoE integration test to the
  `hybrid-bridge-llama-family` family-coverage requirement is part of
  this change.
- **CI signal shrinks further.** Real Qwen3-MoE only runs on T3
  NVIDIA. T2 M4 covers synthetic small-MoE adapter correctness.
  Pre-merge gate: T3 NVIDIA smoke output pasted into the PR.

## Sequencing

- **Depends on:** `qwen3-dense-support` (PR #22, already on `main`).
  Qwen3-MoE's adapter inherits from Qwen3Adapter — needs the
  qk_norm machinery already shipped.
- **Independent of:** Cross-family comparison harness extension and
  T3 Gemma-2-2B reproduction (orthogonal queued items).
- **Single PR** for source + script + tests. The compression-mode
  `top_n` path raises `NotImplementedError` and is its own follow-up.
