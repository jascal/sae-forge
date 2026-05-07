## 1. Adapter package scaffolding

- [x] 1.1 Create `saeforge/adapters/__init__.py` with `register_adapter`,
      `adapter_for`, `registered_classes`, and an internal `_REGISTRY`
      list of `(host_class, adapter)` tuples (first-match-wins).
- [x] 1.2 Create `saeforge/adapters/base.py` with the
      `ArchitectureAdapter` ABC declaring the three abstract methods
      (`walk`, `build_native_config`, `native_module_class`) plus the
      `family: str` class attribute.
- [x] 1.3 Wire `saeforge/adapters/__init__.py` to import the three
      bundled adapter modules (`gpt2`, `llama`, `gemma2`) at import
      time so `import saeforge.adapters` populates the registry.

## 2. NativeModelConfig family field

- [x] 2.1 Add `family: str` field to `NativeModelConfig` (no default)
      and update `__post_init__` to reject any value not in
      `{"gpt2", "llama", "gemma2"}`.
- [x] 2.2 Add `n_kv_heads: int | None = None` field for GQA support.
      `__post_init__` SHALL set `n_kv_heads = num_heads` when omitted.
      Reject `n_kv_heads > num_heads` and require `num_heads %
      n_kv_heads == 0`.
- [x] 2.3 Add `tied_embeddings: bool = False` field.
- [x] 2.4 Add `final_logit_softcap: float | None = None` and
      `attn_logit_softcap: float | None = None` fields (used by the
      Gemma-2 native module, ignored by GPT-2 / Llama paths).
- [x] 2.5 Add `rms_norm_eps: float | None = None` field; populated by
      Llama / Gemma adapters from `host.config.rms_norm_eps`.

## 3. Refactor the GPT-2 path through the adapter

- [x] 3.1 Create `saeforge/adapters/gpt2.py` containing
      `GPT2Adapter(ArchitectureAdapter)` whose `walk(...)` is the
      current body of `SubspaceProjector.project_module` (lines 154–214
      of `saeforge/projector.py`).
- [x] 3.2 Move `_build_torch_module`'s GPT-2 logic from `saeforge/model.py`
      into `gpt2.py` as `_build_gpt2_module(config)`. `GPT2Adapter.native_module_class`
      returns the `ForgedGPT2` class produced inside that factory.
- [x] 3.3 Move `_config_from_host`'s GPT-2 logic into
      `GPT2Adapter.build_native_config`. The result has `family="gpt2"`.
- [x] 3.4 Replace `SubspaceProjector.project_module`'s body with the
      adapter dispatch (`adapter_for(host).walk(host, self,
      attention_width=...)`); keep the lazy `transformers` import +
      friendlier `ImportError`.
- [x] 3.5 Replace `NativeModel.from_host`'s `GPT2LMHeadModel.from_pretrained`
      with `transformers.AutoModelForCausalLM.from_pretrained`; route
      the projection through the adapter dispatch.
- [x] 3.6 Replace `_build_torch_module(config)` body with a dispatch on
      `config.family`; the GPT-2 branch calls `_build_gpt2_module`.
- [x] 3.7 In `saeforge/forge.py:185`, replace `GPT2LMHeadModel.from_pretrained`
      with `AutoModelForCausalLM.from_pretrained` so non-GPT-2 hosts
      load as their actual class.
- [x] 3.8 Run the full suite — every existing GPT-2 test must pass
      unchanged.

## 4. Llama adapter

- [x] 4.1 Create `saeforge/adapters/llama.py` with `LlamaAdapter`.
- [x] 4.2 Implement `walk(host, projector, attention_width)`:
      `model.embed_tokens.weight` via `project_embed`; per-layer
      `self_attn.{q,k,v,o}_proj.weight` via the residual-input/output
      helpers (respecting GQA's `num_key_value_heads`);
      `mlp.{gate,up,down}_proj.weight` via residual-input/output;
      `input_layernorm.weight` and `post_attention_layernorm.weight`
      via `project_residual_aligned`; `model.norm.weight` likewise;
      `lm_head.weight` via `project_unembed` (omitted when
      `tie_word_embeddings=True`).
- [x] 4.3 Implement `build_native_config(host, n_features,
      attention_width)`: pull `hidden_size`, `num_attention_heads`,
      `num_key_value_heads`, `intermediate_size`, `vocab_size`,
      `rms_norm_eps`, `tie_word_embeddings`,
      `max_position_embeddings`, head_dim from `host.config`;
      construct a `NativeModelConfig(family="llama", ...)`.
- [x] 4.4 Implement `native_module_class()` returning a `ForgedLlama`
      class built by a `_build_llama_family_module(config)` helper.
- [x] 4.5 Implement `_build_llama_family_module(config)` (in
      `saeforge/adapters/llama.py` or a shared `_native.py`): RMSNorm,
      separate `nn.Linear` `q_proj` / `k_proj` / `v_proj` / `o_proj`,
      SwiGLU MLP (`gate_proj × up_proj → silu → down_proj`), no `wpe`,
      optional tied lm_head.
- [x] 4.6 Register `LlamaAdapter` for `transformers.LlamaForCausalLM`.

## 5. Gemma-2 adapter

- [x] 5.1 Create `saeforge/adapters/gemma2.py` with `Gemma2Adapter`
      that subclasses `LlamaAdapter` (or imports its helpers) and
      overrides `walk`, `build_native_config`, and
      `native_module_class`.
- [x] 5.2 Extend `walk` to emit the two extra per-layer RMSNorm keys
      (`pre_feedforward_layernorm.weight`,
      `post_feedforward_layernorm.weight`).
- [x] 5.3 Extend `build_native_config` to copy
      `final_logit_softcapping` and `attn_logit_softcapping` from
      `host.config` into the matching `NativeModelConfig` fields. Set
      `family="gemma2"`.
- [x] 5.4 Extend `_build_llama_family_module` (or branch in a Gemma-2
      module factory) to handle the four-norm-per-layer block layout
      and apply `final_logit_softcap` post-`lm_head` as
      `tanh(logits / cap) * cap` when not None.
- [x] 5.5 Register `Gemma2Adapter` for `transformers.Gemma2ForCausalLM`.

## 6. Unregistered architecture handling

- [x] 6.1 `saeforge.adapters.adapter_for` raises `NotImplementedError`
      naming the host's type and the registered class names when no
      match is found.
- [x] 6.2 `SubspaceProjector.project_module` propagates this error
      verbatim (no longer raises its own GPT-2-specific
      `NotImplementedError`).

## 7. Tests

- [x] 7.1 Add `tiny_llama` fixture to `tests/conftest.py` (`LlamaConfig`
      at `hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
      num_key_value_heads=2, intermediate_size=256, vocab_size=1024,
      tie_word_embeddings=False`); same shape for a
      `tiny_llama_tied` variant with `tie_word_embeddings=True`.
- [x] 7.2 Add `tiny_gemma2` fixture (`Gemma2Config` with
      `final_logit_softcapping=30.0, attn_logit_softcapping=50.0`).
- [x] 7.3 Add `tests/test_subspace_projector.py::test_llama_walker_keys_and_shapes`
      and the GQA-shape audit (`q_proj` rows = `n_q_heads * head_dim`;
      `k_proj` / `v_proj` rows = `n_kv_heads * head_dim`).
- [x] 7.4 Add `test_llama_tied_embeddings_omits_lm_head`.
- [x] 7.5 Add `tests/test_subspace_projector.py::test_gemma2_walker_emits_four_norms_per_block`
      and the soft-cap-config-passthrough audit.
- [x] 7.6 Add `tests/test_subspace_projector.py::test_unregistered_architecture_raises`
      using a stub class registered with `register_adapter` and a
      `BertConfig`-derived stub passed to `project_module`. Assert
      the message names the type and lists the registered classes.
- [x] 7.7 Add `tests/test_native_model.py::test_llama_native_module_state_dict_matches_walk`
      and the equivalent for Gemma-2: every key in
      `adapter.walk(host, ...)` has a slot in the native module's
      `state_dict()` with matching shape; every parameter slot in
      the native module is loaded by some walk key.
- [x] 7.8 Add `test_native_model_config_requires_family` and
      `test_native_model_config_rejects_unknown_family`.

## 8. Examples

- [x] 8.1 Verify `examples/forge_gemma2_2b.py` runs end-to-end with
      `--steps 0` against a real Gemma-2-2B host (when available).
      Update the script header to call out `polygram>=0.1.0` and the
      new `RegrowConfig` requirement.
- [x] 8.2 Add `examples/forge_synthetic_llama.py` — runs against a
      tiny synthetic `LlamaForCausalLM` (similar to the
      `tiny_llama` test fixture but slightly larger) so the example
      exercises the Llama path without an HF token.
- [x] 8.3 Add `tests/test_examples_smoke.py` invoking the synthetic
      example end-to-end; gate the Gemma-2 smoke on
      `pytest.importorskip` of an env-var or of the actual model
      weights via `transformers`.

## 9. Failure mode for ForgePipeline.run

- [x] 9.1 Switch `transformers.GPT2LMHeadModel.from_pretrained` to
      `transformers.AutoModelForCausalLM.from_pretrained` in
      `saeforge/forge.py:185` and `saeforge/model.py:217` (the latter
      already routed through the adapter, but the pretrained loader
      stays explicit).
- [x] 9.2 Add a unit test (without HF download) that constructs a
      mock host of an unregistered class and asserts
      `ForgePipeline.run(...)` raises `NotImplementedError` from
      adapter dispatch (not from a downstream shape mismatch).

## 10. Documentation

- [x] 10.1 Update `README.md` "Hardware notes" tier table:
      v0.2 supports GPT-2 family + Llama-3 + Gemma-2; Pythia /
      GPT-NeoX explicitly deferred.
- [x] 10.2 Update `README.md` "Integration with Polygram" section to
      reflect that Llama / Gemma support landed.
- [x] 10.3 Update `docs/algorithm.md` v0 implementation notes to
      mention RMSNorm / SwiGLU / GQA equivalent projection paths and
      the soft-cap / sliding-window non-goals for Gemma-2.
- [x] 10.4 CHANGELOG entry under "Added (multi-architecture-support)"
      naming the new adapters, the `family` field, and the breaking
      `NativeModelConfig` migration.

## 11. Verification

- [x] 11.1 Run `pytest -q` and confirm all tests pass (existing GPT-2
      green, new Llama / Gemma-2 green).
- [x] 11.2 Run `ruff check saeforge tests examples` clean.
- [x] 11.3 Run `openspec validate multi-architecture-support --strict`
      clean.
- [x] 11.4 Spot-check the adapter dispatch by constructing a real
      `LlamaForCausalLM` (tiny synthetic) and confirming
      `ForgePipeline.run` produces a `forged/` dir whose model.safetensors
      contains the expected per-layer RMSNorm + SwiGLU + Q/K/V/O keys
      with no random-init parameter slots remaining.
