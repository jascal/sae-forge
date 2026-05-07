# native-model Specification

## Purpose

Defines `NativeModel` — the v0 forged transformer with a
feature-basis-width residual stream. The model is implemented as a
small in-tree torch nn.Module rather than a `transformers.GPT2LMHeadModel`
wrapper because attention's internal width does not generally factor
to match the basis width.

## Requirements

### Requirement: Config validates qkv_inner_size factorization

`NativeModelConfig.__post_init__` SHALL raise `ValueError` whose
message contains `"qkv_inner_size"` when
`num_heads * head_dim != qkv_inner_size`.

#### Scenario: 4 heads × 4 head_dim ≠ 15

- **WHEN** `NativeModelConfig(qkv_inner_size=15, num_heads=4,
  head_dim=4, ...)` is constructed
- **THEN** `ValueError` is raised whose message contains
  `"qkv_inner_size"`

### Requirement: Config round-trips through dict

`NativeModelConfig.to_dict` and `NativeModelConfig.from_dict` SHALL be
inverses for any well-formed config.

#### Scenario: dict round-trip preserves equality

- **GIVEN** a `NativeModelConfig` with all fields set
- **WHEN** `from_dict(to_dict())` is called
- **THEN** the returned config equals the original

### Requirement: Forward pass produces (batch, seq, vocab) logits

`NativeModel.forward(input_ids)` SHALL accept a `(batch, seq)` integer
tensor and return logits of shape `(batch, seq, vocab_size)`.

#### Scenario: 8-feature, 100-vocab forward pass

- **GIVEN** a `NativeModelConfig` with `hidden_size=8, vocab_size=100`
- **WHEN** `forward(torch.randint(0, 100, (2, 16)))` is called
- **THEN** the output has shape `(2, 16, 100)`

### Requirement: from_projected_weights covers every parameter slot

`NativeModel.from_projected_weights(config, weights)` SHALL copy every
key from `weights` into the matching `state_dict` slot. When a
projected key has no destination, `KeyError` SHALL be raised whose
message contains `"no slot"`. When a projected ndarray's shape doesn't
match the destination, `ValueError` SHALL be raised whose message
names the destination key, the projected shape, and the expected shape.

#### Scenario: unknown projected key raises KeyError

- **GIVEN** a config and a weights dict whose key `"unknown_key"` is
  not in the model's `state_dict`
- **WHEN** `from_projected_weights` is called
- **THEN** `KeyError` is raised whose message contains `"no slot"`

### Requirement: save_pretrained + load_pretrained round-trip exactly

`save_pretrained(output_dir)` followed by `load_pretrained(output_dir)`
SHALL produce a `NativeModel` that yields identical forward outputs
(within `1e-6` absolute tolerance) on the same input. The serialized
form is `config.json` (the config dict) plus `model.safetensors` (the
state dict).

#### Scenario: round-trip preserves forward output

- **GIVEN** a `NativeModel` constructed from random init and an input
  `(1, 8)` token tensor
- **WHEN** the model is saved to a tmp dir, then a fresh
  `NativeModel.load_pretrained` is built from the same dir
- **THEN** both models' forward outputs on the same input agree to
  `1e-6` absolute tolerance

### Requirement: Lazy-imports torch and transformers

`NativeModel.__init__` SHALL lazy-import torch via `require_extra`.
`NativeModel.from_host` SHALL lazy-import `transformers`. Missing
extras SHALL raise `ImportError` whose message names the `[torch]`
extra.

### Requirement: from_host derives config from the loaded HF GPT-2

`from_host` SHALL set the returned `NativeModelConfig` fields from the
host's HF GPT2Config plus the basis width:

```
hidden_size              = projector.basis.n_features
qkv_inner_size           = host.config.n_embd
num_layers               = host.config.n_layer
num_heads                = host.config.n_head
head_dim                 = host.config.n_embd // host.config.n_head
intermediate_size        = host.config.n_inner or 4 * host.config.n_embd
vocab_size               = host.config.vocab_size
max_position_embeddings  = host.config.n_positions
layer_norm_epsilon       = host.config.layer_norm_epsilon
```
