## 1. Config dataclass

- [x] 1.1 Implement `NativeModelConfig` with the ten fields and the `qkv_inner_size == num_heads * head_dim` post-init check
- [x] 1.2 Add `to_dict` / `from_dict` for JSON round-trip

## 2. In-tree torch transformer

- [x] 2.1 Implement HF GPT-2 style `Conv1D` (weight `(in, out)`, bias `(out,)`)
- [x] 2.2 Implement `CausalSelfAttention` with c_attn → Q/K/V split → multi-head → causal mask → c_proj → residual
- [x] 2.3 Implement `MLP` with c_fc → gelu(approximate="tanh") → c_proj
- [x] 2.4 Implement `Block` with pre-norm residual layout (ln_1 → attn → +; ln_2 → mlp → +)
- [x] 2.5 Implement `Transformer` with wte + wpe → blocks → ln_f
- [x] 2.6 Implement `ForgedGPT2` with the transformer plus `nn.Linear(hidden, vocab, bias=False)` lm_head

## 3. Public API

- [x] 3.1 `NativeModel.__init__(config)` builds the torch module via `_build_torch_module`
- [x] 3.2 `NativeModel.from_host(host_model_id, projector, *, dtype, device)` loads HF GPT-2, projects weights, derives config, returns the native model on the requested device/dtype
- [x] 3.3 `NativeModel.from_projected_weights(config, weights)` copies each projected ndarray into the state_dict slot; raises `KeyError` with a clear message on key mismatch and `ValueError` on shape mismatch
- [x] 3.4 `forward(input_ids)` and `parameters()` / `num_parameters()` thin pass-throughs
- [x] 3.5 `save_pretrained(output_dir)` writes config.json + model.safetensors; `load_pretrained(input_dir)` round-trips exactly
- [x] 3.6 Lazy-import torch and transformers via `require_extra`

## 4. Tests

- [x] 4.1 Config validation: factorization check rejects mismatches; round-trip preserves the dataclass
- [x] 4.2 Construction with a synthetic config has non-zero parameter count
- [x] 4.3 Forward pass on random input has the expected `(batch, seq, vocab)` shape
- [x] 4.4 `from_projected_weights` builds a working model from `tiny_gpt2` projected weights
- [x] 4.5 Extra keys in projected weights raise `KeyError`
- [x] 4.6 `save_pretrained` + `load_pretrained` round-trip preserves forward outputs to `1e-6` tolerance

## 5. OpenSpec scaffolding

- [x] 5.1 `openspec/changes/native-model/proposal.md`
- [x] 5.2 `openspec/changes/native-model/tasks.md` (this file)
- [x] 5.3 `openspec/changes/native-model/specs/native-model/spec.md`
