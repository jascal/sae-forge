## ADDED Requirements

### Requirement: WhisperEncoderAdapter is registered for both Whisper host classes

`saeforge.adapters.whisper.WhisperEncoderAdapter` SHALL be registered
at module import time for both
`transformers.WhisperForConditionalGeneration` and
`transformers.WhisperModel`. Both registrations SHALL resolve to the
same `WhisperEncoderAdapter` instance via `adapter_for(host)`.

The adapter's `family` class attribute SHALL be `"whisper_encoder"`.

#### Scenario: full-conditional-generation host resolves to the adapter

- **GIVEN** a `transformers.WhisperForConditionalGeneration` instance
- **WHEN** `saeforge.adapters.adapter_for(host)` is called
- **THEN** the returned adapter's `family` attribute equals
  `"whisper_encoder"`

#### Scenario: encoder-only WhisperModel host resolves to the same adapter

- **GIVEN** a `transformers.WhisperModel` instance
- **WHEN** `saeforge.adapters.adapter_for(host)` is called
- **THEN** the returned adapter is the same object as for the
  full-conditional-generation case

### Requirement: walk projects every encoder weight whose input or output touches the residual stream

`WhisperEncoderAdapter.walk` SHALL return, for a Whisper encoder host
with `n` layers, a dict containing exactly the following keys, each with
the indicated shape (where `f = n_features`, `d = d_model`, `i =
intermediate_size`, `m = n_mels`, `p = max_source_positions`).
Shapes follow HF `nn.Linear` convention: `weight` is `(out, in)` and
`bias` matches `out`. Q/K/V read the residual (in-axis projected
from `d` to `f`); `out_proj` writes the residual (out-axis
projected from `d` to `f`).

```
conv1.weight                                (d, m, 3)         # frozen-copied
conv1.bias                                  (d,)               # frozen-copied
conv2.weight                                (d, d, 3)         # frozen-copied
conv2.bias                                  (d,)               # frozen-copied
embed_positions.weight                      (p, d)            # frozen-copied
basis_encode                                (d, f)            # d→f bridge buffer
layers.{0..n-1}.self_attn_layer_norm.weight (f,)
layers.{...}.self_attn_layer_norm.bias      (f,)
layers.{...}.self_attn.q_proj.weight        (d, f)
layers.{...}.self_attn.q_proj.bias          (d,)
layers.{...}.self_attn.k_proj.weight        (d, f)
layers.{...}.self_attn.v_proj.weight        (d, f)
layers.{...}.self_attn.v_proj.bias          (d,)
layers.{...}.self_attn.out_proj.weight      (f, d)
layers.{...}.self_attn.out_proj.bias        (f,)
layers.{...}.final_layer_norm.weight        (f,)
layers.{...}.final_layer_norm.bias          (f,)
layers.{...}.fc1.weight                     (i, f)
layers.{...}.fc1.bias                       (i,)
layers.{...}.fc2.weight                     (f, i)
layers.{...}.fc2.bias                       (f,)
layer_norm.weight                           (f,)
layer_norm.bias                             (f,)
```

The conv stem stays at ``d_model`` channels (frozen-copied) and the
transformer blocks operate at ``n_features`` width, so the forged
module needs a runtime d → f projection at the conv-stem →
first-block boundary. The walker emits ``basis_encode`` carrying
``projector.basis.pseudoinverse() * projector.scale_boost`` (the
matrix-form of :meth:`SubspaceProjector.encode`); the forged module
loads it into a non-parameter buffer. It appears in
``state_dict()`` (so save/load round-trips it) but not in
``named_parameters()`` (so it doesn't participate in gradient
checkpointing, weight-decay groups, or the
no-randomly-initialised-weights invariant).

`k_proj` SHALL NOT have a bias (matches HF Whisper). The conv stem
weights and `embed_positions` SHALL be byte-identical to the host's
corresponding parameters — the adapter SHALL NOT call the projector
on them.

#### Scenario: walker emits the v0.4 key set on tiny synthetic Whisper

- **GIVEN** a `tiny_synthetic_whisper` fixture (d_model=64, encoder_layers=2,
  encoder_attention_heads=4, encoder_ffn_dim=128)
- **AND** a 32-feature `FeatureBasis` over the same `d_model=64`
- **WHEN** `WhisperEncoderAdapter().walk(host, projector)` is called
- **THEN** the dict has exactly the keys listed above
- **AND** every value's shape matches the corresponding entry

#### Scenario: frozen-copy invariant holds bit-for-bit

- **GIVEN** the same setup as above
- **WHEN** the walker runs
- **THEN** `weights["conv1.weight"]` equals `host.encoder.conv1.weight`
  cast to `float64` (bit-for-bit, post-cast)
- **AND** the same equality holds for `conv2.weight`, `conv1.bias`,
  `conv2.bias`, `embed_positions.weight`

### Requirement: build_native_config emits encoder-shaped NativeModelConfig

`WhisperEncoderAdapter.build_native_config(host, n_features)` SHALL
return a `NativeModelConfig` with:

- `family = "whisper_encoder"`
- `output_kind = "encoder_states"`
- `vocab_size = 0`
- `hidden_size = n_features`
- `qkv_inner_size = host.config.d_model`
- `num_layers = host.config.encoder_layers`
- `num_heads = host.config.encoder_attention_heads`
- `head_dim = host.config.d_model // host.config.encoder_attention_heads`
- `intermediate_size = host.config.encoder_ffn_dim`
- `n_kv_heads = num_heads` (Whisper is MHA, not GQA)
- `max_position_embeddings = host.config.max_source_positions`
- `activation = "gelu"`

#### Scenario: tiny synthetic Whisper produces a validly-shaped config

- **GIVEN** the `tiny_synthetic_whisper` fixture (d_model=64,
  encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128)
- **WHEN** `WhisperEncoderAdapter().build_native_config(host,
  n_features=32)` is called
- **THEN** the returned config has `family == "whisper_encoder"`,
  `output_kind == "encoder_states"`, `hidden_size == 32`,
  `qkv_inner_size == 64`, `num_layers == 2`, `num_heads == 4`,
  `head_dim == 16`, `intermediate_size == 128`, `n_kv_heads == 4`

### Requirement: native_module_class returns ForgedWhisperEncoder lazily

`WhisperEncoderAdapter.native_module_class()` SHALL return the
`ForgedWhisperEncoder` class. The torch import SHALL be lazy — calling
`native_module_class()` on a Whisper adapter SHALL be safe without
torch installed (it raises a clear ImportError naming the `[torch]`
extra) but SHALL succeed when torch is installed.

#### Scenario: torch installed → class returned

- **GIVEN** the `[torch]` extra is installed
- **WHEN** `WhisperEncoderAdapter().native_module_class()` is called
- **THEN** the returned class is a subclass of `torch.nn.Module`
- **AND** the class name is `ForgedWhisperEncoder`

## MODIFIED Requirements

### Requirement: Adapter registry dispatches by host model class

`saeforge.adapters.registered_classes()` SHALL include
`transformers.WhisperForConditionalGeneration` and
`transformers.WhisperModel` in its returned list once
`saeforge.adapters.whisper` has been imported (the v0.3 contract is
otherwise preserved unchanged).

The `NotImplementedError` raised on unregistered hosts SHALL list
all five registered classes (the v0.3 GPT-2/Llama/Gemma-2 plus the
two new Whisper classes) in its diagnostic message.

#### Scenario: registry exposes Whisper classes after import

- **WHEN** `saeforge.adapters` is imported
- **THEN** `saeforge.adapters.registered_classes()` contains both
  `transformers.WhisperForConditionalGeneration` and
  `transformers.WhisperModel`
- **AND** also contains the v0.3 LM classes (`GPT2LMHeadModel`,
  `GPT2Model`, `LlamaForCausalLM`, `Gemma2ForCausalLM`)

#### Scenario: unregistered host's error message lists every registered class

- **GIVEN** a custom non-registered class `FakeBert`
- **WHEN** `adapter_for(FakeBert())` is called
- **THEN** `NotImplementedError` is raised whose message contains
  `"FakeBert"` and the registered class names including
  `"WhisperForConditionalGeneration"` and `"WhisperModel"`
