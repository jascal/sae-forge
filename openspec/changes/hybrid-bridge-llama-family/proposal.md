## Why

The `hybrid-bridge-forge` change (#18, on `main` at `ca1c185`) shipped the
three-basis forge mechanism end-to-end, but only wired the bridges into the
**GPT-2** native module's forward pass. For any other architecture
(Llama-3, Gemma-2, Qwen2 — all built via the shared
`saeforge/adapters/llama.py::build_llama_family_module` factory), the
half-built state is:

1. `ForgePipeline.__post_init__` accepts `hybrid_bridge=True` with
   shape-compatible bases.
2. `_build_hybrid_bundle` returns a valid `HybridBasisBundle`.
3. `SubspaceProjector.project_module(host, hybrid=bundle)` routes the
   walk correctly through the three bases (the routing layer is
   family-agnostic).
4. `NativeModelConfig` ends up with `bridges=True` and the bridge knobs
   set.
5. **`build_llama_family_module` ignores `cfg.bridges` entirely.** The
   `LlamaTransformer.__init__` does not construct any
   `BridgeModule` instances. The `LlamaTransformer.forward` loop
   (`for layer in self.layers: x = layer(x)`) has no bridge call sites.
6. The forged model trains and saves with the projected weights, **but
   bridges are silently absent from the forward path**. The config
   claims bridges are on; the runtime behaves as if they aren't.

This is the only blocker preventing the next family of work
(`qwen3-dense-support`, `qwen3-moe-support`, the planned bridge-on-Gemma-2
M4 reproduction T3 tracked in `hybrid-bridge-forge/tasks.md` §15.5) from
proceeding. The mechanism itself is correct; only the per-family `forward`
wiring is missing.

This change closes the gap with a single coordinated edit to the
Llama-family factory, mirroring the GPT-2 wiring exactly. No new
capability surface, no new knobs, no FSM changes — the same `cfg.bridges`
/ `cfg.bridge_init` / `cfg.bridge_nonlin` / `cfg.bridge_pre_layernorm`
fields drive the construction.

## What Changes

### Scope

Wire bridge construction and forward-pass insertion into the Llama-family
native module factory (`saeforge/adapters/llama.py`). The wiring SHALL
mirror the GPT-2 implementation exactly:

- Bridges constructed in `LlamaTransformer.__init__` via a
  `_build_bridges(cfg)` static helper identical in shape to
  `ForgedGPT2.Transformer._build_bridges`.
- Bridges called in `LlamaTransformer.forward` at block indices `0`
  (after the embed-region block) and `L-2` (after the last mid-region
  block, before the lm-head-region block).
- When `cfg.bridges` is `False` (the default), zero new code paths
  execute and the existing Llama-family forward behavior is byte-
  identical to today.

The same `bridges` ModuleDict naming (`emb_mid`, `mid_lm`) ships, so
the safetensors state-dict keys are stable across families:
`model._bridges.emb_mid.linear.weight` and `model._bridges.mid_lm.linear.weight`.

Wait — there's a naming subtlety. GPT-2 hangs bridges on
`transformer.bridges` (since the GPT-2 native uses `self.transformer`).
Llama-family hangs them on `model.bridges` (since the Llama native uses
`self.model`). The save-time / load-time round-trip stays family-internal;
the user-facing API doesn't expose these paths. Documented in design.md.

### New artifacts

None. This change adds no new files.

### Modified artifacts

- **`saeforge/adapters/llama.py`** — `LlamaTransformer.__init__` gains
  a `_build_bridges(cfg)` call constructing a `nn.ModuleDict` when
  `cfg.bridges`. `LlamaTransformer.forward` gains a per-index check
  applying `bridges["emb_mid"]` after block 0 and `bridges["mid_lm"]`
  after block `L-2`, gated by `len(self.layers) >= 3` to handle the
  edge case where a host model has fewer than 3 layers (matches the
  GPT-2 guard).
- **`saeforge/forge.py`** — *no changes*. The pipeline already passes
  the hybrid bundle and sets `cfg.bridges` for any family.
- **`saeforge/projector.py`** — *no changes*. Family-agnostic routing
  already works.

### Tests

- **`tests/integration/test_hybrid_bridge_llama.py` (new)** — mirrors
  `tests/integration/test_hybrid_bridge_gpt2.py` against the existing
  `tiny_llama` (untied, 4-layer-after-config-bump) fixture. Asserts:
  bridges appear in `state_dict`; forward produces finite logits;
  safetensors round-trip preserves the bridges; `hybrid_bridge=False`
  leaves the forged module byte-identical to pre-change.
- **`tests/integration/test_hybrid_bridge_qwen2.py` (new)** —
  the same shape against an untied Qwen2 fixture. Confirms the
  qkv_bias + bridges combination round-trips cleanly.

The existing `tests/integration/test_hybrid_bridge_gpt2.py` stays
unchanged — it's the GPT-2-specific surface and remains the load-bearing
T0 gate.

### Conftest updates

- **`tests/conftest.py`** — bump `tiny_llama` to 4 layers (currently 2),
  matching `tiny_gpt2_untied_4layer`. Adds a `tiny_qwen2_untied_4layer`
  fixture symmetric to the GPT-2 one. The 2-layer `tiny_llama` fixture is
  retained as `tiny_llama_2layer` for tests that explicitly need the
  small variant.

### CLI surface

Unchanged. The `--hybrid-bridge`, `--basis-embed`, `--basis-lm-head`,
`--bridge-init`, `--bridge-nonlin`, `--bridge-no-pre-ln` flags already
plumb correctly through to `cfg.bridges` for any family.

### Out of scope (deferred)

- **GPT-2 family-tag refactor.** GPT-2 hangs bridges on
  `self.transformer.bridges`; Llama-family on `self.model.bridges`. The
  naming asymmetry is honest (it reflects each host's HF naming
  convention) but a future
  `forged-module-state-dict-normalization` change could unify them
  under `bridges.*` at the top level. Out of scope here.
- **Qwen3 / Qwen3-MoE adapters.** Those need their own forged-module
  changes (`q_norm` / `k_norm` for dense; routing for MoE) and are
  tracked separately as `qwen3-dense-support` / `qwen3-moe-support`.
- **FSM-orchestrator wiring for hybrid forge.** Still imperative-path
  only. Tracked as the same parent `hybrid-bridge-forge` follow-up.
- **Re-running the T1 comparison harness on Llama.** The Intel/`gpt2`
  baseline numbers stand; adding a Llama harness is a useful follow-up
  for the next set of architecture choices, but it isn't a shipping
  gate for this change.

## Capabilities

### New Capabilities

- **`hybrid-bridge-llama-family`** — pins the bridge-insertion contract
  for the Llama-family factory. Specifies the construction conditional
  on `cfg.bridges`, the per-index forward insertion at `0` and `L-2`,
  the `len(layers) >= 3` guard, the `model.bridges` state-dict key
  prefix, and the byte-equivalence-when-disabled scenario. This is
  parallel to (not a modification of) the existing
  `hybrid-bridge-forge` capability — same contract, different family
  factory.

### Modified Capabilities

None. The parent `hybrid-bridge-forge` capability is unmodified — its
requirements are written family-generically and were always intended to
hold for any architecture; this change closes an *implementation* gap
without re-shaping the contract.

## Impact

- **No public API breakage.** Single-basis Llama/Gemma-2/Qwen2 forges
  are unchanged. The new `hybrid_bridge=True` path on these families
  goes from "silently half-broken" to "works end-to-end."
- **Test surface.** ~10 new tests across the two new integration files
  plus the conftest fixture additions. The existing GPT-2 tests stay
  unchanged.
- **Unblocks downstream.** Once this lands, Qwen3 dense (and any future
  Llama-family architecture) inherits working hybrid forging by virtue
  of building on `build_llama_family_module`.

## Sequencing

- **Depends on:** `hybrid-bridge-forge` (already on `main` at
  `ca1c185`). This change references `cfg.bridges` and friends introduced
  there.
- **Blocks:** `qwen3-dense-support`. Qwen3 dense's hybrid story
  requires Llama-family bridge insertion to actually work end-to-end;
  this change is its prerequisite.
- **Single PR.** ~80 LOC of source + 100 LOC of tests. No staged
  rollout — the byte-equivalence gate passes or it doesn't.
