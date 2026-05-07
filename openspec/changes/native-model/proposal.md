## Why

`NativeModel` is the v0 component that takes the projected weight dict
from `SubspaceProjector` and assembles a working transformer whose
residual stream has width `basis.n_features`. The bootstrap change
shipped four `NotImplementedError` stubs (`from_host`,
`from_projected_weights`, `forward`, `save_pretrained`). This change
implements all four.

The architectural constraint that drives the implementation: a forged
model's residual width is `n_features`, but the host's attention
internal width (`n_heads * head_dim`) and MLP inner width are inherited
unchanged. Those generally don't factor as `n_features = n_heads *
head_dim`, so the stock `transformers.GPT2LMHeadModel` config — which
ties attention's inner width to `n_embd` — doesn't apply. The path of
least resistance is a small in-tree transformer module that decouples
residual width from attention/MLP internals.

## What Changes

- Implement `NativeModelConfig` as a dataclass with the seven
  architecture knobs (`hidden_size`, `qkv_inner_size`, `num_layers`,
  `num_heads`, `head_dim`, `intermediate_size`, `vocab_size`) plus
  three small ones (`max_position_embeddings`, `layer_norm_epsilon`,
  `activation`). Validate `qkv_inner_size == num_heads * head_dim` in
  `__post_init__`.
- Implement an in-tree torch transformer (~80 lines):
  - `Conv1D(in, out)` — HF GPT-2 style `y = x @ W + b`.
  - `CausalSelfAttention` — splits c_attn output into Q/K/V, multi-head,
    causal mask, c_proj back to residual.
  - `MLP` — c_fc → gelu(approximate="tanh") → c_proj.
  - `Block` — pre-norm residual layout (ln_1 → attn → +; ln_2 → mlp → +).
  - `Transformer` — wte + wpe → blocks → ln_f.
  - `ForgedGPT2` — Transformer + lm_head (`nn.Linear(hidden, vocab,
    bias=False)`).
- Implement `NativeModel.from_host(host_model_id, projector, *, dtype,
  device)`: load HF GPT-2 via `from_pretrained`, run
  `projector.project_module`, derive a `NativeModelConfig` from the
  host's per-block dims, and build the native model.
- Implement `NativeModel.from_projected_weights(config, weights)`:
  copy each projected `np.ndarray` into the matching `state_dict` slot,
  validating shape and raising `KeyError` with a clear message when a
  projected key has no destination.
- Implement `forward(input_ids)`: thin pass-through to the torch module.
- Implement `save_pretrained(output_dir)` and the matching
  `load_pretrained(input_dir)` — config as JSON, state as `safetensors`.
  Round-trip exactly: same input → same output after reload.
- Lazy-import torch and transformers via `saeforge.utils.lazy.require_extra`.

## Capabilities

### New Capabilities

- `native-model-construction`: Build a small torch transformer with a
  feature-basis-width residual stream and host-inherited attention /
  MLP internal widths. Round-trip serialization (`save_pretrained` /
  `load_pretrained`) preserves outputs exactly.
- `native-model-from-host`: One-call constructor that loads an HF GPT-2
  by id, projects its weights through a `SubspaceProjector`, and
  returns a forged `NativeModel`.

### Modified Capabilities

- `bootstrap`: stubs are filled. The "NativeModelConfig constructs"
  scenario remains valid but now reflects the implemented constructor.

## Impact

- `saeforge/model.py`: ~180 lines covering the dataclass, the in-tree
  torch module factory, the public `NativeModel` class, and the
  `_config_from_host` helper.
- `tests/test_native_model.py`: 7 tests covering config validation,
  config round-trip, construction, forward pass, end-to-end host
  projection on `tiny_gpt2`, key-mismatch raise, save/load round-trip.
- No external API change beyond filling the stubs.
- The native model deliberately does NOT subclass
  `transformers.PreTrainedModel`. v0 ships a minimal API; HF
  compatibility (auto-tokenization, generation_config, the full HF
  trainer surface) is a deferred concern.
