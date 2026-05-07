# subspace-projector Specification

## Purpose

`SubspaceProjector` is the v0 component that turns a `FeatureBasis`
plus an HF host model into a flat dict of projected weights ready for
`NativeModel.from_projected_weights`. The projection algebra (residual-
input matrices use `D @ W`, residual-output matrices use `W @ E`,
biases use `b @ E`, residual-aligned vectors project the same way) is
documented in the module docstring; this spec pins the shape contracts
and the coverage for each supported host architecture.

Per-architecture walking is delegated to
[`saeforge.adapters`](../architecture-adapters/spec.md). This spec
documents the shape contracts each adapter is expected to produce
when called via `SubspaceProjector.project_module(host_model)`. New
architectures are added by registering an `ArchitectureAdapter`, not
by extending the projector.

## Requirements

### Requirement: project_module covers every GPT-2 weight

For an HF `GPT2LMHeadModel` host model, `project_module` SHALL return a
dict containing exactly the following keys, each with the indicated
shape (where `f = n_features`, `d = n_embd`, `i = n_inner`,
`v = vocab_size`, `p = n_positions`):

```
transformer.wte.weight                 (v, f)
transformer.wpe.weight                 (p, f)
transformer.h.{0..n_layer-1}.ln_1.weight        (f,)
transformer.h.{...}.ln_1.bias                   (f,)
transformer.h.{...}.attn.c_attn.weight          (f, 3d)
transformer.h.{...}.attn.c_attn.bias            (3d,)
transformer.h.{...}.attn.c_proj.weight          (d, f)
transformer.h.{...}.attn.c_proj.bias            (f,)
transformer.h.{...}.ln_2.weight                 (f,)
transformer.h.{...}.ln_2.bias                   (f,)
transformer.h.{...}.mlp.c_fc.weight             (f, i)
transformer.h.{...}.mlp.c_fc.bias               (i,)
transformer.h.{...}.mlp.c_proj.weight           (i, f)
transformer.h.{...}.mlp.c_proj.bias             (f,)
transformer.ln_f.weight                (f,)
transformer.ln_f.bias                  (f,)
lm_head.weight                         (v, f)
```

For a `GPT2Model` host (the bare base model with no LM head), the
returned dict SHALL contain the same keys EXCEPT `lm_head.weight`.

Implementation lives in
`saeforge.adapters.gpt2.GPT2Adapter.walk`; `project_module` dispatches
via `saeforge.adapters.adapter_for(host)` and returns the adapter's
walk output verbatim.

#### Scenario: `tiny_gpt2` produces every key with the right shape

- **GIVEN** a `tiny_gpt2` fixture (n_embd=16, n_layer=2, n_head=4, vocab=100)
- **AND** an 8-feature `FeatureBasis` over the same `d_model=16`
- **WHEN** `project_module` is called
- **THEN** the dict has exactly the keys listed above
- **AND** every value has the shape listed above

### Requirement: Non-registered hosts raise NotImplementedError

`project_module` SHALL raise `NotImplementedError` for any host model
whose class has no registered adapter in `saeforge.adapters`. The
error message SHALL name the offending type and SHALL list the
registered class names so the user can see which architectures are
supported.

#### Scenario: an unregistered host class is rejected with an actionable message

- **GIVEN** a custom class `FakeBert` (not registered in `saeforge.adapters`)
- **WHEN** `project_module(FakeBert())` is called
- **THEN** `NotImplementedError` is raised whose message contains
  `"FakeBert"` and lists at least `"GPT2LMHeadModel"`,
  `"LlamaForCausalLM"`, and `"Gemma2ForCausalLM"`

### Requirement: project_module covers every Llama weight

`project_module` SHALL cover every Llama weight when given an HF `LlamaForCausalLM` host model. The returned dict SHALL contain the following keys with the indicated shapes (where `f = n_features`, `d_q = num_attention_heads * head_dim`, `d_kv = num_key_value_heads * head_dim`, `i = intermediate_size`, `v = vocab_size`):

```
model.embed_tokens.weight              (v, f)
model.layers.{0..n_layer-1}.input_layernorm.weight              (f,)
model.layers.{...}.self_attn.q_proj.weight                      (d_q, f)
model.layers.{...}.self_attn.k_proj.weight                      (d_kv, f)
model.layers.{...}.self_attn.v_proj.weight                      (d_kv, f)
model.layers.{...}.self_attn.o_proj.weight                      (f, d_q)
model.layers.{...}.post_attention_layernorm.weight              (f,)
model.layers.{...}.mlp.gate_proj.weight                         (i, f)
model.layers.{...}.mlp.up_proj.weight                           (i, f)
model.layers.{...}.mlp.down_proj.weight                         (f, i)
model.norm.weight                      (f,)
lm_head.weight                         (v, f)
```

When `host.config.tie_word_embeddings` is `True`, the `lm_head.weight`
key SHALL be omitted (the resulting `NativeModel` aliases
`lm_head.weight` to `model.embed_tokens.weight`).

RMSNorm has no β; the dict SHALL NOT contain any `*.bias` keys for
RMSNorm layers.

#### Scenario: tiny synthetic Llama produces the expected key set

- **GIVEN** a `tiny_llama` fixture (`hidden_size=128, num_hidden_layers=2,
  num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
  vocab_size=1024, tie_word_embeddings=False`)
- **AND** a 32-feature `FeatureBasis` over `d_model=128`
- **WHEN** `project_module(host)` is called
- **THEN** the dict's key set exactly equals
  `model.embed_tokens.weight`, the 7-key-per-layer set above for layers
  0 and 1, `model.norm.weight`, `lm_head.weight`
- **AND** every value has the shape listed above

### Requirement: project_module covers every Gemma-2 weight

`project_module` SHALL cover every Gemma-2 weight when given an HF `Gemma2ForCausalLM` host model. The returned dict's key set SHALL match the Llama key set above with these additions: each `model.layers.{i}` block contains TWO additional RMSNorm `weight` keys, `pre_feedforward_layernorm` and `post_feedforward_layernorm`, both shape `(f,)`.

Gemma-2's `final_logit_softcapping` and `attn_logit_softcapping`
config fields SHALL be surfaced on the resulting `NativeModelConfig`
(via `Gemma2Adapter.build_native_config`) but SHALL NOT modify any
projected weight. The native module applies `final_logit_softcap` (when
not None) as `tanh(lm_head(h) / cap) * cap` post-projection.

#### Scenario: tiny synthetic Gemma-2 emits the four-norm-per-block layout

- **GIVEN** a `tiny_gemma2` fixture
  (`hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
  num_key_value_heads=2, intermediate_size=256, vocab_size=1024,
  final_logit_softcapping=30.0, attn_logit_softcapping=50.0`)
- **WHEN** `project_module(host)` is called against a 32-feature basis
- **THEN** the dict contains, per layer, four RMSNorm weight keys:
  `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm`
- **AND** the resulting `NativeModelConfig` (via
  `Gemma2Adapter.build_native_config`) has
  `final_logit_softcap == 30.0` and `attn_logit_softcap == 50.0`
