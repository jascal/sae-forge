# Design — NanochatAdapter

## Weight inventory (nanochat-style GPT vs HF GPT-2)

| weight                       | nanochat              | GPT-2 (HF)            | walk strategy                              |
|------------------------------|-----------------------|-----------------------|--------------------------------------------|
| token embedding              | `transformer.wte`     | `transformer.wte`     | `project_embed`                            |
| positional embedding         | (none — rotary)       | `transformer.wpe`     | skip                                       |
| rotary cos/sin buffers       | `cos`, `sin`          | (n/a)                 | non-persistent buffers; regenerated        |
| pre-attn norm scale          | parameter-free RMS    | `ln_1.{weight,bias}`  | skip (no params to project)                |
| qkv projection               | `c_q,c_k,c_v` separate| `c_attn` fused        | three separate `project_qkv_*` calls       |
| attn output projection       | `c_proj`              | `c_proj`              | `project_attn_out`                         |
| pre-mlp norm scale           | parameter-free RMS    | `ln_2.{weight,bias}`  | skip                                       |
| mlp expand                   | `mlp.c_fc`            | `mlp.c_fc`            | `project_mlp_in`                           |
| mlp contract                 | `mlp.c_proj`          | `mlp.c_proj`          | `project_mlp_out`                          |
| activation                   | ReLU² (inline)        | GELU                  | activation lives in module class, not walk |
| value-residual embeds        | `value_embeds.{i}`    | (n/a)                 | `project_embed` (kv_dim-wide)              |
| value-residual gate          | `attn.ve_gate`        | (n/a)                 | `project_residual_aligned`                 |
| per-layer mix scalars        | `resid_lambdas`, `x0_lambdas` | (n/a)         | identity pass-through (1D scalars)         |
| final norm                   | parameter-free RMS    | `ln_f.{weight,bias}`  | skip                                       |
| lm head                      | `lm_head`             | `lm_head` (tied)      | `project_lm_head`                          |

The four nanochat-specific items the GPT-2 adapter doesn't handle are: separate qkv, value embeds + gate, per-layer scalars, no positional embedding matrix. None of them require new `SubspaceProjector` methods — the existing `project_qkv_full` / `project_embed` / `project_residual_aligned` primitives cover everything.

## Registry dispatch

`saeforge.adapters._REGISTRY` keys on `isinstance(host_model, cls)`. lm-sae's `GPT` is not an HF transformers class, so we register against the lm-sae class directly. The adapter file imports `GPT` lazily inside `walk()` to avoid a hard import-time dep on lm-sae (the same pattern `gpt2.py` uses for HF imports).

Registration:

```python
def _register():
    try:
        from lm_sae.train import GPT as _NanochatGPT
    except ImportError:
        return  # lm-sae not installed; adapter is unreachable but harmless
    register_adapter(_NanochatGPT, NanochatAdapter())

_register()
```

This keeps lm-sae as an *optional* dependency of sae-forge: not in `pyproject.toml`, just available if the user has installed it locally.

## Forge module class

The forged `nn.Module` mirrors lm-sae's `GPT.forward` exactly, with three substitutions:

1. The residual stream lives in the SAE's feature basis (dim = `n_features` instead of `n_embd`).
2. `c_q`, `c_k`, `c_v`, `c_proj`, `c_fc`, `mlp.c_proj`, `lm_head` are loaded from the projected weight dict.
3. `value_embeds` and `ve_gate` are projected analogously.

`resid_lambdas` and `x0_lambdas` are scalars — they carry through unchanged. `cos` / `sin` buffers are regenerated from `head_dim` and `sequence_len`; they're geometry, not parameters.

The cleanest implementation is to subclass lm-sae's `GPT` and override `__init__` to wire the projected weights. That requires lm-sae's `GPT` to be importable; see the registry-dispatch note above.

## Open questions for implementation

1. **Should `attention_width="feature_native"` be supported?** GPT-2 has it for the v0.2 both-sides projection. Nanochat's separate qkv makes it trivial — three separate `project_qkv_full` calls instead of one fused one. Recommend: yes, opt-in, mirror the GPT-2 design.
2. **Default `faithfulness_target`?** `TokenCosineTarget` (ESM-2's choice) is the obvious fit for a token-stream LM. The alternative `KLTarget` requires the host to expose logits — fine for nanochat (it has `lm_head`), but the cosine target is cheaper and produced cleaner forge runs in bio-sae.
3. **Sliding-window?** Upstream nanochat uses `SSSL`. The lm-sae fork dropped it for MPS portability. If a future Apple-Silicon-era lm-sae re-introduces the window, the adapter can pick it up from `host.window_sizes` and pass it through to a `windowed_sdpa` helper. Defer until the lm-sae fork itself re-enables.

## Deferred work (separate proposals)

- **GQA** — nanochat supports `n_kv_head < n_head`; the adapter walk should project `c_k` / `c_v` to `n_kv_head * head_dim` (not `n_head * head_dim`). lm-sae's port currently keeps `n_kv_head == n_head` so the simple path works; the asymmetric path can be added when needed.
- **Value-residual ResFormer ablation.** `add-resformer-ablation` could pose: does forge quality change if we project `value_embeds` to a SEPARATE basis (not the residual basis)? Speculative; not on critical path.
- **Step-budget pin in sweep_pareto_capability.** lm-sae's trajectory experiment will want each agent-edit checkpoint to receive the same number of forge calibration tokens. Today the sweep is data-scale-based; a step-based mode is a separate change.
