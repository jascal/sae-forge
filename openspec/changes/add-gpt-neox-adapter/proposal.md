# GPT-NeoX / Pythia architecture adapter — forge a causal LM with parallel residual + partial rotary

Add a `gpt_neox` host adapter so sae-forge forges `GPTNeoXForCausalLM` hosts — the **EleutherAI Pythia
ladder** (14m → 2.8b), GPT-NeoX-20B, and friends. This broadens the forge's causal-LM coverage to a new
architecture class and unblocks the **Pythia-ladder** validation of the causal trained-encoder result
(`add-causal-host-capability-sweep`) the moment Pythia SAEs are available.

## Why

The forge already covers GPT-2, Llama, Gemma-2, Qwen-2/3(+MoE), Whisper-encoder, and ESM-2 — but **not
GPT-NeoX/Pythia**, the canonical open *interpretability* ladder (one tokenizer, 7 sizes, fully released
checkpoints). The just-merged `add-causal-host-capability-sweep` showed the trained-encoder win **survives the
full forge on GPT-2**; the obvious next question — *does it scale across the Pythia ladder?* — needs the forge
to run on Pythia. Best practice is to support Pythia **in sae-forge** (one auditable adapter alongside the
others) rather than rebuild the forge elsewhere.

GPT-NeoX combines three features no single existing adapter had together:

1. **Parallel residual** (`use_parallel_residual=True`): attention reads `input_layernorm(x)` and the MLP reads
   `post_attention_layernorm(x)` — both off the **same** pre-block `x`, summed: `x + attn + mlp` (GPT-2/Llama
   are sequential, with the MLP reading the post-attention residual).
2. **Partial rotary** (`rotary_pct`, e.g. 0.25 for Pythia): RoPE on only the first
   `int(head_dim * rotary_pct)` dims of each head; the rest pass through. The existing Llama-family rope
   *raises* on `partial_rotary_factor != 1.0`.
3. **LayerNorm with bias** (not RMSNorm) on every norm + a final layer norm; **fused QKV** `query_key_value`
   reshaped per-head `[q|k|v]`; **GELU MLP** (`dense_h_to_4h`/`dense_4h_to_h`) — all with biases; **untied**
   `embed_in`/`embed_out`.

## What

### 1. Partial rotary — `saeforge._positional.rope.apply_rotary_pos_emb_partial`

`apply_rotary_pos_emb_partial(q, k, cos, sin, rot_dim)`: rotate the first `rot_dim` head dims (HF's exact
rotate-half convention, cache built for `rot_dim`), pass the rest through. Validated equal to HF's
`GPTNeoXAttention` rotary to **0.0** on identical inputs. `rot_dim == head_dim` reduces to full RoPE.

### 2. `saeforge/adapters/gpt_neox.py` — `GPTNeoXAdapter` + native module

- `walk()`: project `embed_in`, per-block (two LayerNorms w/ bias, fused `query_key_value` w/ head-space bias
  unprojected, `dense` w/ residual bias, GELU MLP w/ biases), `final_layer_norm`, untied `embed_out`. Method
  choices follow the Llama nn.Linear convention (project the `d_model` axis; reads→`project_residual_output`,
  writes→`project_residual_input`).
- `build_native_config()`: pulls dims from `host.config`. Reads rotary knobs from `cfg.rope_parameters`
  (modern transformers) with a legacy `rotary_pct`/`rotary_emb_base` fallback — **the bring-up bug**: the
  top-level `rotary_pct` is gone in transformers ≥4.5x, so a naive `getattr` silently defaulted to full
  rotary.
- `native_module_class()` → `ForgedGPTNeoX`: `GPTNeoXModel` (embed_in → parallel-residual layers →
  final_layer_norm) + untied `embed_out`. Attention does fused QKV → per-head `[q|k|v]` split → partial rotary
  → causal SDPA. v1: `attention_width="host"` only (feature_native raises).
- Registered at import in `saeforge/adapters/__init__.py`; `gpt_neox` added to
  `saeforge.model._SUPPORTED_FAMILIES`.

## Falsifiable acceptance gate (the canonical "adapter is correct" check)

**Identity-basis forge** (`scripts/prototype_gpt_neox.py`): with `W_dec = I` (basis width == d_model) every
projection is the identity, so the forged `NativeModel` is a *pure re-implementation* of the host — its logits
MUST match. Both outcomes are reported descriptively; the gate is the match.

### Gate RESULT (2026-06-13) — PASS

- **Tiny-random** GPT-NeoX (4 configs spanning head_dim 16/32/64 and rotary 0.25/0.5/1.0, up to 6 layers):
  forged logits match the host to **max|Δ| ≈ 1–9e-7** (float32), every native param reached by the walk.
- **Real Pythia-70m** (float32): **rel error 6.4e-6** (the float32 accumulation floor over 6 layers) with
  **100% next-token argmax agreement**. (Absolute |Δlogits| ≈ 7e-3 only because real Pythia logits have
  magnitude ~1e3; it is *not* a math error — confirmed identical under eager vs sdpa attention.)

The bring-up found and fixed two real issues, now regression-guarded: (i) `partial_rotary_factor` silently
defaulting to 1.0 (rope_parameters migration); (ii) the dtype trap — real Pythia checkpoints load in float16,
so faithfulness must be judged on **relative** error / argmax, not an absolute |Δ| threshold.

## What this does NOT solve

- **No Pythia SAE shipped.** The adapter makes the *forge* run on Pythia; the capability *gate* additionally
  needs a Pythia SAE (none cached locally). The Pythia-ladder trained-encoder experiment is unblocked but
  awaits an SAE.
- **`attention_width="host"` only** in v1 (feature_native QKV both-sides projection raises — a follow-up).
- **No `host_wrapped_module`** (v1 ships GPT-2 only there; inherits the base `NotImplementedError`).
- **No rope_scaling** beyond Pythia's default (NeoX hosts with linear/dynamic/yarn scaling are a follow-up).

## Related

- `add-causal-host-capability-sweep` — the merged change whose Pythia-ladder follow-up this unblocks.
- `add-llama-family-rope` — the rope module this extends with the partial variant.
- FABLE_DIRECTIONS X5 (cross-host validation) / R1 (the `τ*` law's cross-arch ladder, which spans Pythia).
