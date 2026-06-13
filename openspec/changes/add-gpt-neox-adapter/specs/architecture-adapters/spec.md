# architecture-adapters Specification (delta)

## ADDED Requirements

### Requirement: GPT-NeoX-family architecture adapter

A `GPTNeoXAdapter` SHALL be registered against `transformers.GPTNeoXForCausalLM` so
`saeforge.adapters.adapter_for(host)` returns a `GPTNeoXAdapter` instance for HF GPT-NeoX / Pythia hosts. Its
`family` class attribute SHALL be `"gpt_neox"`, and `"gpt_neox"` SHALL be a member of
`saeforge.model._SUPPORTED_FAMILIES`.

Registration SHALL be lazy: if `transformers` is not importable, the adapter module SHALL NOT register and
SHALL NOT raise — `adapter_for` continues to work for all other families.

#### Scenario: dispatch to the GPT-NeoX adapter

- **GIVEN** a `GPTNeoXForCausalLM` host
- **WHEN** `adapter_for(host)` is called
- **THEN** it SHALL return a `GPTNeoXAdapter` whose `family == "gpt_neox"`

### Requirement: GPT-NeoX walk parameter inventory

`GPTNeoXAdapter.walk(host, projector, *, attention_width="host")` SHALL emit a `dict[str, np.ndarray]`
projected through the supplied `SubspaceProjector` and keyed to match the native module's `state_dict`:

- `gpt_neox.embed_in.weight` (via `project_embed`).
- For each layer `i` in `0..num_hidden_layers - 1`:
  - `input_layernorm.weight` / `.bias` and `post_attention_layernorm.weight` / `.bias` (via
    `project_residual_aligned`).
  - `attention.query_key_value.weight` (via `project_residual_output`); `attention.query_key_value.bias`
    passed through **unprojected** (head space, `3*hidden`).
  - `attention.dense.weight` (via `project_residual_input`); `attention.dense.bias` (via
    `project_residual_bias`).
  - `mlp.dense_h_to_4h.weight` (via `project_residual_output`); `mlp.dense_h_to_4h.bias` unprojected
    (MLP-hidden space).
  - `mlp.dense_4h_to_h.weight` (via `project_residual_input`); `mlp.dense_4h_to_h.bias` (via
    `project_residual_bias`).
- `gpt_neox.final_layer_norm.weight` / `.bias` (via `project_residual_aligned`).
- `embed_out.weight` (via `project_unembed`) — GPT-NeoX is untied.

Every parameter of the native module SHALL be reached by the walk (no randomly-initialised parameter remains).
`attention_width` other than `"host"` SHALL raise `NotImplementedError` in v1.

#### Scenario: walk reaches every native parameter

- **GIVEN** a GPT-NeoX host and an identity `SubspaceProjector`
- **WHEN** `walk(...)` builds the parameter dict and a `NativeModel` is constructed from it
- **THEN** every native parameter name SHALL be present in the walk dict

### Requirement: GPT-NeoX native forward (parallel residual + partial rotary)

The native module SHALL reproduce HF `GPTNeoXForCausalLM` semantics: a **parallel-residual** block
(`x + attention(input_layernorm(x)) + mlp(post_attention_layernorm(x))`), **partial rotary** on the first
`int(head_dim * partial_rotary_factor)` head dims, **LayerNorm with bias** on every norm, a **fused
`query_key_value`** projection split per-head as `[q|k|v]`, a **GELU** MLP, and an untied `embed_out`.
`build_native_config` SHALL source `partial_rotary_factor` and `rope_theta` from `cfg.rope_parameters` when
present (modern transformers), falling back to the legacy top-level `rotary_pct` / `rotary_emb_base`.

#### Scenario: identity-basis forge reproduces the host (faithfulness gate)

- **GIVEN** a GPT-NeoX host and a basis with `W_dec = I` (width == d_model), so every projection is the
  identity
- **WHEN** the host is forged and both run the same `input_ids`
- **THEN** the forged logits SHALL match the host's to float precision — **tiny-random** hosts to
  `max|Δ| < 1e-4`, and a **real Pythia** host to relative error `< 1e-4` with 100% next-token argmax agreement
