## Why

sae-forge v0.1 silently produces a GPT-2-shaped, randomly-initialised model when run against any non-GPT-2 host. The code paths that produce this footgun:

- `saeforge/forge.py:185` hardcodes `transformers.GPT2LMHeadModel.from_pretrained(self.host_model_id)` — passing `"google/gemma-2-2b"` (the shipped Gemma example's host) loads the Gemma weights into a GPT-2 *config*, which HF silently warns about and emits an essentially randomly-initialised model.
- `saeforge/projector.py:154-163` runs an `isinstance` chain over `GPT2LMHeadModel` / `GPT2Model` and falls through to `NotImplementedError` for everything else — but only for objects that actually arrived as their native type. The forge.py hardcode upstream means non-GPT-2 hosts never reach the projector's failure-mode check.
- `saeforge/model.py:106-167` builds a GPT-2-shaped target (`Conv1D` matrices named `c_attn` / `c_proj` / `wte` / `wpe`, `LayerNorm`, GeLU). No SwiGLU MLP, no RMSNorm, no Q/K/V/O separation, no GQA support.
- `examples/forge_gemma2_2b.py` and the README "Hardware notes" tier table actively claim Gemma-2-2B works today. They don't.

The projection algebra in `docs/algorithm.md` is architecture-agnostic — every linear map projects the same way. The blocker is plumbing: the right HF parameter names, SwiGLU's three matrices, RMSNorm, and Llama-3 GQA's `n_kv_heads < n_q_heads`.

Polygram already handles Llama / Gemma residual capture in `polygram.behavioural.runtime._get_layer_module` (`model.model.layers`), so the upstream side is fine. This change is sae-forge-internal.

## What Changes

- **New `saeforge/adapters/` package** with a small architecture-adapter registry. `register_adapter(transformers_class, adapter)` maps an HF model class to an `ArchitectureAdapter` whose contract is `walk(host, projector, attention_width) -> dict[str, np.ndarray]` plus `build_native_config(host, n_features, attention_width) -> NativeModelConfig` and `native_module_class(config) -> nn.Module`.
- **GPT-2 adapter extracted** from the current `SubspaceProjector.project_module` and `_config_from_host` bodies — same algebra, same parameter names, just relocated. `SubspaceProjector.project_module` becomes a 5-line dispatcher; the existing GPT-2 tests stay green by construction.
- **Llama adapter** — covers `LlamaForCausalLM` from `transformers`. Walks `model.embed_tokens`, every `model.layers.{i}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}`, `model.norm`, `lm_head`. Handles GQA (`config.num_key_value_heads`); RMSNorm γ projects through `project_residual_aligned` (no β — RMSNorm is mean-free); SwiGLU's three matrices (`gate_proj`, `up_proj`, `down_proj`) all route through the existing residual-input / residual-output helpers.
- **Gemma-2 adapter** — covers `Gemma2ForCausalLM`. Shape-compatible with the Llama adapter (Gemma-2 uses the same RMSNorm + SwiGLU + Q/K/V/O + GQA primitives) so most of the walk delegates to a shared helper. Gemma-2-specific quirks (alternating local/global attention, `final_logit_softcapping`, `attn_logit_softcapping`, the pre-feedforward layernorm pair) are surfaced on the `NativeModelConfig` but accepted as `ε_nonlin` per docs/algorithm.md §5 — fine-tuning corrects the drift; we don't try to replicate softcap exactly in v0.1.
- **NativeModel becomes family-aware**. `NativeModelConfig` gains a `family: str` field (`"gpt2" | "llama" | "gemma2"`); `_build_torch_module` dispatches on family. The Llama / Gemma path uses `nn.Linear` (not `Conv1D`), `RMSNorm`, SwiGLU MLP (`gate × up → silu → down`), separate `q_proj` / `k_proj` / `v_proj` / `o_proj` linear layers, no `wpe`, optional tied `lm_head`. **BREAKING**: `NativeModelConfig.family` has no default — every existing call site picks a family explicitly.
- **`ForgePipeline.run` switches** `transformers.GPT2LMHeadModel.from_pretrained(...)` to `transformers.AutoModelForCausalLM.from_pretrained(...)` so the host loads as its actual architecture. The adapter registry then dispatches.
- **`saeforge/model.py:_config_from_host`** routes through the adapter — the GPT-2 specific knobs (`cfg.n_embd`, `cfg.n_head`, `cfg.n_inner`) move into the GPT-2 adapter.
- **Tests**: extend `tests/test_subspace_projector.py` and `tests/test_native_model.py` with tiny-synthetic Llama and Gemma-2 fixtures (`LlamaConfig` / `Gemma2Config` at `hidden_size=128`, 2 layers, 4 heads, 2 KV heads for GQA exercise). Assertion shape: every emitted name has a matching slot in the corresponding `Forged{Llama,Gemma2}` skeleton — i.e. **no weight is randomly initialised after projection**. New negative test: passing a `BertModel` (or any unregistered architecture) raises `NotImplementedError` whose message names the type and lists the registered classes.
- **Examples**: `examples/forge_gemma2_2b.py` made to actually work with `--steps 0`. New companion `examples/forge_synthetic_llama.py` exercises the Llama path on a synthetic basis without HF token requirements (mirrors the `forge_gpt2_real_sae.py` style). Smoke test wraps `forge_gemma2_2b.py` with `--steps 0` and a `pytest.importorskip` guard for the Gemma weights.
- **README**: hardware-tier table and "Integration with Polygram" section updated to reflect "v0.2 supports GPT-2 family + Llama-3 + Gemma-2; Pythia / GPT-NeoX deferred."

## Capabilities

### New Capabilities

- `architecture-adapters`: Registry-based dispatch from HF model class → `ArchitectureAdapter` (walk + native-config + native-module-class). `SubspaceProjector.project_module` and `NativeModel.from_host` route through it; unregistered architectures raise `NotImplementedError` naming the registered set. Built-in adapters cover GPT-2, Llama (Llama-3), and Gemma-2.

### Modified Capabilities

- `subspace-projector`: `project_module` switches from a hard-coded GPT-2 walker to adapter dispatch. The GPT-2 walk semantics are unchanged (same emitted parameter names, same algebra) — they move into `saeforge.adapters.gpt2.GPT2Adapter`.

## Impact

- `saeforge/adapters/__init__.py`, `base.py`, `gpt2.py`, `llama.py`, `gemma2.py` — new package (~500 lines of plumbing across the four files).
- `saeforge/projector.py:127-216` — replace the body of `project_module` with `_dispatch_adapter(host).walk(host, self, attention_width=...)`. ~80 line drop, ~10 line replacement.
- `saeforge/model.py:106-183` — `_build_torch_module` becomes a dispatcher; family-specific module factories move into the adapters or sit alongside `ForgedGPT2` as `ForgedLlama` / `ForgedGemma2`. `_config_from_host` becomes a one-line adapter call. `from_host` switches to `AutoModelForCausalLM`.
- `saeforge/forge.py:185` — `GPT2LMHeadModel.from_pretrained` → `AutoModelForCausalLM.from_pretrained`.
- `tests/conftest.py` — add `tiny_llama` and `tiny_gemma2` fixtures alongside the existing `tiny_gpt2`.
- `tests/test_subspace_projector.py` and `tests/test_native_model.py` — Llama + Gemma walker shape audits and the negative test.
- `examples/forge_gemma2_2b.py` — verified to run with `--steps 0`; small README header update naming the new prerequisites (HF token, polygram>=0.1.0).
- `examples/forge_synthetic_llama.py` — new (lighter sibling of `forge_gpt2_real_sae.py` running against a synthetic Llama config so no HF token needed).
- `README.md` — update hardware tier table and "Integration with Polygram" section.
- **Out of scope**: Pythia / GPT-NeoX. Tracked as a follow-up that also needs a small upstream polygram addition for the parallel Q/K/V parameterisation.
- **Migration**: callers of `NativeModelConfig(...)` must add `family="gpt2"` (or `"llama"` / `"gemma2"`). All in-tree callers are updated; downstream callers see a clear `TypeError` at construction.
