# architecture-adapters Specification

## Purpose

The `architecture-adapters` capability defines the registry-based
dispatch from a HuggingFace transformers model class to the
sae-forge adapter that knows how to walk that architecture's weights
into a feature-basis projection and build the matching native module.
It replaces the GPT-2-only `isinstance` chain that v0.1
`SubspaceProjector.project_module` shipped with — adding a new host
architecture is now one module under `saeforge/adapters/<family>.py`
plus a `register_adapter(...)` call. Bundled adapters cover GPT-2,
Llama-3 (with GQA + tied embeddings), and Gemma-2 (with the
four-norm-per-block layout and logit soft-capping).

## Requirements

### Requirement: Adapter registry dispatches by host model class

`saeforge.adapters` SHALL expose a registry-based dispatcher with five public functions:

- `register_adapter(host_class: type, adapter: ArchitectureAdapter) -> None` — registers an adapter for a transformers model class.
- `adapter_for(host_model) -> ArchitectureAdapter` — returns the adapter whose registered class is the most-specific match for `host_model` (first-match-wins over the registration order).
- `adapter_for_family(family: str) -> ArchitectureAdapter` — returns the adapter whose `family` attribute matches the given string. Used by code paths that have only the `NativeModelConfig.family` string in hand (e.g. inside the training loop where the host class is already gone, or in `saeforge.eval.targets._default_target_for`). Raises `ValueError` naming the registered families when none match.
- `registered_classes() -> list[type]` — returns the list of currently-registered host classes for diagnostic use.
- `registered_families() -> frozenset[str]` — returns the live set of `adapter.family` values across registered adapters. Used by `_default_target_for` and `saeforge.model._supported_families()`; consumers SHOULD prefer this helper over re-deriving the family set from the registry.

When `adapter_for` cannot find a match, it SHALL raise `NotImplementedError` whose message names the host model's type and the list of registered class names. The error SHALL NOT fall back to a default adapter.

The bundled adapters (GPT-2, Llama, Gemma-2, Qwen2, Qwen3, Qwen3MoE, Whisper-encoder) SHALL register themselves at module import time. Adapters whose host class needs a newer `transformers` than the install provides (Qwen3 and Qwen3MoE need `transformers >= 4.51`) SHALL silently skip their registration via a `try/except ImportError` guard. Importing `saeforge.adapters` SHALL be sufficient to populate the registry with whatever adapters the install can support.

#### Scenario: registered adapter is returned for matching host

- **GIVEN** a host model that is an instance of `transformers.GPT2LMHeadModel`
- **WHEN** `saeforge.adapters.adapter_for(host)` is called
- **THEN** the returned adapter's `family` attribute equals `"gpt2"`

#### Scenario: unregistered architecture raises actionable error

- **WHEN** `saeforge.adapters.adapter_for(bert_model)` is called with an instance whose class has no registered adapter
- **THEN** `NotImplementedError` is raised; the message contains the type name (e.g. `"BertModel"`) and the registered class names (e.g. `"GPT2LMHeadModel"`, `"LlamaForCausalLM"`, `"Gemma2ForCausalLM"`)

#### Scenario: registered_classes lists all built-in adapters

- **WHEN** `saeforge.adapters.registered_classes()` is called after `import saeforge.adapters`
- **THEN** the returned list includes (at least) `GPT2LMHeadModel`, `LlamaForCausalLM`, and `Gemma2ForCausalLM`

### Requirement: ArchitectureAdapter contract

The `saeforge.adapters.ArchitectureAdapter` ABC SHALL declare three abstract methods, one concrete method with an override hook, and one class attribute:

- `family: str` — class attribute; one of the bundled family identifiers (`"gpt2"`, `"llama"`, `"gemma2"`, `"qwen2"`, `"qwen3"`, `"qwen3_moe"`, `"whisper_encoder"`) or a third-party-registered value. Used by `NativeModelConfig.family`.
- `walk(self, host, projector, *, attention_width: str) -> dict[str, np.ndarray]` — projects every relevant host weight via `projector` and returns a flat dict keyed by `NativeModel` parameter names. Pure-numpy; no torch operations beyond reading `host`'s parameters.
- `build_native_config(self, host, n_features: int, *, attention_width: str) -> NativeModelConfig` — pulls per-block dimensions from `host.config` into a `NativeModelConfig` whose `family` matches `self.family`.
- `native_module_class(self) -> type` — returns the `nn.Module` subclass used to instantiate forged models for this family. The returned class's `__init__` SHALL accept a `NativeModelConfig`-shaped object as its sole positional argument.
- `default_faithfulness_target(self) -> FaithfulnessTarget` — returns the family's default loop-gating scorer. Consulted by `saeforge.eval.targets._default_target_for(family)` when no explicit `ForgePipeline(faithfulness=...)` is set. The ABC's default implementation returns `KLTarget()` (lazy-imported to avoid the `saeforge.eval.targets` → `saeforge.adapters` import cycle); subclasses MAY override. `WhisperEncoderAdapter` overrides to return `CosineTarget()`; the six LM-family adapters inherit the `KLTarget()` default.

`walk` SHALL emit one entry per parameter the corresponding native module declares. The native module's `state_dict()` keys SHALL be a superset of the `walk` output, and every key in `walk` SHALL match the native module's expected shape exactly. Mismatches SHALL surface as `ValueError` from `NativeModel.from_projected_weights` with the parameter name and both shapes named.

#### Scenario: walk emits every native parameter

- **GIVEN** a registered adapter and a host model whose architecture matches
- **WHEN** `adapter.walk(host, projector, attention_width="host")` is called
- **THEN** for every key in the returned dict, the corresponding entry exists in `adapter.native_module_class()(config).state_dict()` with the same shape, and **every** weight slot in the resulting native module corresponds to a key in the walk's dict (no randomly-initialised parameter survives `NativeModel.from_projected_weights`)

#### Scenario: LM-family adapters return KLTarget by default

- **WHEN** `default_faithfulness_target()` is invoked on each of `GPT2Adapter`, `LlamaAdapter`, `Gemma2Adapter`, `Qwen2Adapter`, `Qwen3Adapter`, `Qwen3MoEAdapter`
- **THEN** the returned target is an instance of `KLTarget` whose `name == "kl"` and `better_when == "lower"`

#### Scenario: Whisper-encoder adapter overrides to CosineTarget

- **WHEN** `WhisperEncoderAdapter().default_faithfulness_target()` is invoked
- **THEN** the returned target is an instance of `CosineTarget` whose `name == "cosine"` and `better_when == "higher"`

### Requirement: Llama-3 adapter handles GQA and SwiGLU

`saeforge.adapters.llama.LlamaAdapter` SHALL handle `transformers.LlamaForCausalLM`. The walk SHALL emit:

- `model.embed_tokens.weight` — projected via `project_embed`.
- For each `model.layers.{i}`: `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`, `input_layernorm.weight`, `post_attention_layernorm.weight`.
- `model.norm.weight`.
- `lm_head.weight` (when not tied to `embed_tokens`; when tied, only `embed_tokens` is projected and the native module aliases `lm_head.weight` to `model.embed_tokens.weight` post-init).

The adapter SHALL respect the host's `config.num_key_value_heads`. Its `build_native_config` SHALL set `n_kv_heads = config.num_key_value_heads` and `n_heads = config.num_attention_heads`; when those differ (GQA), the projection of `q_proj` versus `k_proj` / `v_proj` SHALL produce shapes matching `n_q_heads * head_dim` and `n_kv_heads * head_dim` respectively.

The SwiGLU MLP SHALL be projected as three separate matrices: `gate_proj` and `up_proj` are residual-input matrices (shape `(d_model, intermediate_size)` after HF's `Linear.weight` layout transpose); `down_proj` is a residual-output matrix (shape `(intermediate_size, d_model)`).

RMSNorm γ SHALL project via `project_residual_aligned`. RMSNorm has no β; the adapter SHALL NOT emit `*.bias` keys for any RMSNorm layer.

#### Scenario: walk on tiny synthetic Llama emits expected keys

- **GIVEN** a `transformers.LlamaForCausalLM` constructed from a `LlamaConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=256, vocab_size=1024)` with random weights
- **WHEN** `LlamaAdapter().walk(host, projector, attention_width="host")` is called against a basis with 32 features
- **THEN** the returned dict contains `model.embed_tokens.weight`, `model.layers.0.self_attn.{q,k,v,o}_proj.weight`, `model.layers.0.mlp.{gate,up,down}_proj.weight`, `model.layers.0.input_layernorm.weight`, `model.layers.0.post_attention_layernorm.weight` (and the same for layer 1), `model.norm.weight`, and `lm_head.weight` (or omits `lm_head.weight` when `tie_word_embeddings=True`)

#### Scenario: GQA shapes match num_key_value_heads

- **GIVEN** a Llama host with `num_attention_heads=4` and `num_key_value_heads=2` and `head_dim=32`
- **WHEN** `LlamaAdapter().walk(host, projector_to_32_features, attention_width="host")` is called
- **THEN** the projected `q_proj.weight` has shape `(128, 32)` (i.e. `n_q_heads * head_dim = 128` rows, `n_features = 32` columns), and the projected `k_proj.weight` and `v_proj.weight` each have shape `(64, 32)` (i.e. `n_kv_heads * head_dim = 64` rows)

#### Scenario: tied embeddings produce no lm_head walk entry

- **GIVEN** a Llama host with `config.tie_word_embeddings=True`
- **WHEN** `LlamaAdapter().walk(host, projector, attention_width="host")` is called
- **THEN** the returned dict has no `lm_head.weight` key, and `LlamaAdapter().build_native_config(...).tied_embeddings` is `True`

### Requirement: Gemma-2 adapter shares Llama-family layout

`saeforge.adapters.gemma2.Gemma2Adapter` SHALL handle `transformers.Gemma2ForCausalLM`. The walk SHALL emit the same parameter set as the Llama adapter (Q/K/V/O proj, SwiGLU gate/up/down, input_layernorm, post_attention_layernorm, model.norm, lm_head when not tied) plus Gemma-2's two additional per-layer norms: `pre_feedforward_layernorm` and `post_feedforward_layernorm`.

The adapter SHALL surface Gemma-2-specific config fields on the resulting `NativeModelConfig` (at minimum: `final_logit_softcapping: float | None`, `attn_logit_softcapping: float | None`). The native module SHALL apply `final_logit_softcapping` (when not None) as `tanh(lm_head(h) / cap) * cap`. The projection itself SHALL NOT be modified by the soft-cap.

Gemma-2's alternating local/global attention pattern is OUT OF SCOPE for this change; the native module SHALL use the standard causal mask everywhere. The drift on long-context tasks is accepted as `ε_attn` per `docs/algorithm.md` §5.

#### Scenario: walk on tiny synthetic Gemma-2 emits the four-norm-per-block layout

- **GIVEN** a `transformers.Gemma2ForCausalLM` constructed from a `Gemma2Config(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=256, vocab_size=1024, final_logit_softcapping=30.0)`
- **WHEN** `Gemma2Adapter().walk(host, projector, attention_width="host")` is called
- **THEN** the returned dict contains `model.layers.0.{input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm}.weight` (four norms per layer) and `Gemma2Adapter().build_native_config(host, 32, attention_width="host").final_logit_softcap == 30.0`

### Requirement: NativeModelConfig.family field is required

`NativeModelConfig` SHALL declare a `family: str` field (no default value). Construction without `family` SHALL raise `TypeError`. Valid values are the bundled families (`"gpt2"`, `"llama"`, `"gemma2"`, `"qwen2"`, `"qwen3"`, `"qwen3_moe"`, `"whisper_encoder"`) plus any third-party family registered via `register_adapter` before the config is constructed. `__post_init__` SHALL raise `ValueError` for any other value.

`__post_init__` SHALL validate `self.family` against `saeforge.model._supported_families()`, which returns the sorted union of `saeforge.model._SUPPORTED_FAMILIES` (a static tuple of the bundled family names) and `saeforge.adapters.registered_families()`. Bundled family names SHALL be accepted unconditionally so config construction works on a base install without `transformers` (where the adapters' `try/except ImportError` registration guards short-circuit and leave `_REGISTRY` empty). Runtime dispatch sites (`_build_torch_module`, `_default_target_for`) SHALL still require an actually-registered adapter and raise a distinct dispatch-time error when one is unavailable.

`_build_torch_module(config)` SHALL dispatch on `config.family` via `adapter_for_family(config.family).native_module_class()(config)` — a registry lookup, NOT an `if/elif` family tree. The dispatched module SHALL produce parameter slots that match the corresponding adapter's `walk` output one-for-one.

#### Scenario: NativeModelConfig requires family

- **WHEN** `NativeModelConfig(hidden_size=32, qkv_inner_size=32, num_layers=2, num_heads=4, head_dim=8, intermediate_size=64, vocab_size=100)` is constructed without `family`
- **THEN** Python raises `TypeError` for the missing keyword argument `family`

#### Scenario: bundled family accepted without adapter registration

- **GIVEN** an environment where `transformers` is unavailable (base install without the `[torch]` extra) and the adapter registry is therefore empty
- **WHEN** `NativeModelConfig(family="gpt2", ...)` is constructed
- **THEN** construction succeeds; the static `_SUPPORTED_FAMILIES` tuple suffices for config-time validation even when runtime dispatch would fail

#### Scenario: unknown family rejected

- **WHEN** `NativeModelConfig(family="not-a-real-family", ...)` is constructed
- **THEN** `__post_init__` raises `ValueError` whose message names the supported values (the union of bundled families and any third-party registrations)

### Requirement: ForgePipeline.run loads the host via AutoModelForCausalLM

`saeforge.forge.ForgePipeline.run` SHALL load the host model via `transformers.AutoModelForCausalLM.from_pretrained(host_model_id)` (replacing the v0.1 `GPT2LMHeadModel.from_pretrained`). The returned model's class SHALL drive adapter dispatch via `saeforge.adapters.adapter_for(host)`. Unregistered architectures SHALL raise the dispatcher's `NotImplementedError` (no fallback to GPT-2 loading).

#### Scenario: Gemma-2 host loads as Gemma2ForCausalLM

- **GIVEN** `host_model_id="google/gemma-2-2b"` and a transformer install with Gemma-2 support
- **WHEN** `ForgePipeline.run("/output")` is called
- **THEN** the loaded host is an instance of `transformers.Gemma2ForCausalLM`, and `adapter_for(host).family == "gemma2"`. The forge does NOT silently load Gemma weights into a GPT-2 config.

#### Scenario: unregistered host architecture raises before random init

- **GIVEN** a hypothetical `host_model_id` resolving to an architecture with no registered adapter
- **WHEN** `ForgePipeline.run(...)` is called
- **THEN** `NotImplementedError` is raised by the adapter dispatcher with the offending type and the registered class list, and no model file is written to disk
