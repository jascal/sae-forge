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

Architectural diff vs the nearest existing adapters (reader aid):

| axis | GPT-2 | Llama/Gemma | **GPT-NeoX (this)** |
|---|---|---|---|
| residual | sequential | sequential | **parallel** (attn+mlp off same `x`) |
| positions | learned `wpe` | full RoPE | **partial RoPE** (`rotary_pct`) |
| norm | LayerNorm (+bias) | RMSNorm (no bias) | **LayerNorm (+bias)** |
| QKV | fused `c_attn` | separate `q/k/v` | **fused `query_key_value`** (per-head `[q\|k\|v]`) |
| MLP | GELU (+bias) | SwiGLU (no bias) | **GELU (+bias)** |
| embeddings | tied | usu. tied | **untied** (`embed_in`/`embed_out`) |

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

### Pythia-ladder RESULT (2026-06-13) — the GPT-2 forge-level win does NOT generalize

With the adapter in place, the **causal forge gate ran on Pythia** (`scripts/pythia_ladder_gate.py`, fetching
EleutherAI sparsify `sae-pythia-{70m,160m}-32k` from HF; the `gpt_neox` forged-block accessor was wired into
`sweep_capability._FORGED_BLOCK_ACCESSORS`). Same protocol as the GPT-2 forge gate (proxy-train `E` +
full-multi-layer-forge score, trained vs `pinv` retained-mAUC). Data: `scripts/pythia_ladder_gate_results.json`.

| host + SAE | n=128 Δ (trained − pinv) | verdict |
|---|---|---|
| **GPT-2** + jbloom **ReLU** SAE | **+0.039 ± 0.003** (~13σ) | SURVIVES |
| **Pythia-70m** + sparsify **TopK** SAE (3 seeds) | **−0.030 ± 0.005** (~6σ) | **ERASED** |
| **Pythia-160m** + sparsify **TopK** SAE (2 seeds) | +0.0001 ± 0.0003 | TIE |

**What the gate measures (self-contained, not deferred to prior PRs).** For each width `N`: slice the SAE
decoder to its top-`N` rows = the basis; forge the host onto it and run the full multi-layer forward; read the
forged residual at the SAE's layer back through the SAE encoder. **Labels** = the SAE's own prevalence-band
features, binarised. **retained-mAUC** = forged feature-label AUC / host feature-label AUC. The **trained**
arm proxy-trains the encoder `E` (init `pinv(W_dec)`) to preserve the host's latents; the **`pinv`** arm is the
Frobenius baseline. Δ = trained − `pinv`, held-out, multi-seed.

**The GPT-2 forge win does NOT generalize to Pythia** — it reverses to tie/negative on both rungs, falsifying
the "causal hosts win" reading from the earlier activation-level control. **But "decoder geometry, not host
causality" is a candidate explanation, NOT the established takeaway — the dominant confound is unresolved:**
GPT-2 used a **ReLU** SAE and Pythia a **TopK** SAE, so the reversal cannot be cleanly attributed to decoder
geometry vs. the encoder being mis-specified for the SAE type. Stacked caveats: (i) **SAE-type mismatch is the
leading confound — the clean isolation is a matched-SAE control (a ReLU vs a TopK SAE on the *same* host),
which this PR does not run**; (ii) the Pythia gate uses a **restricted-ReLU task encoder over the TopK SAE's
directions** (TopK gating dropped for training tractability) — not apples-to-apples; (iii) low power (2–3
seeds, two small models). The descriptive, defensible claim is just: *the GPT-2 forge win did not replicate on
Pythia under this protocol.*

**Deeper resolution (powered-R2 follow-up, run after this gate; lands in `add-trained-subspace-projector`).**
The forge gate's premise was itself a mis-transfer. fieldrun's **R2** ("a trained rank-`r` projection beats
frozen SVD") trains the **subspace** (which directions to keep); the forge's X2 only trains the **encoder into
a *fixed* SAE dictionary**, where `pinv` is already optimal — so it can win only on ill-conditioned (ReLU)
dictionaries. And R2's actual lever is **model-general**: a powered rerun (`fieldrun/lo3a/tau_star_powered.py`,
20k tokens, 3 seeds) gives trained-subspace − frozen-SVD open-class R@32 of **GPT-2 +52pp** and **Pythia-70m
+31pp** (the earlier "Pythia loses" was R2's *under-powered* 1199-token protocol). So the honest summary: the
forge result is about *which knob you train* (subspace vs encoder-into-a-fixed-dictionary) and SAE
conditioning — **not** host causality; the matched-SAE control + a `train_subspace` mode are the follow-ups
that isolate it.

## What this does NOT solve

- **Matched-SAE control still open.** The Pythia ladder points the trained-encoder win at SAE type rather than
  host class, but a same-host ReLU-vs-TopK SAE pair is the decisive isolation (offline-blocked: needs a TopK
  GPT-2 SAE or a ReLU Pythia SAE; both are fetchable / trainable as a follow-up).
- **Ladder is 2 rungs.** EleutherAI `sae-pythia-*-32k` exists for 70m + 160m only; 410m–2.8b lack that SAE
  release (the *forge* runs on all of them — only the SAE is missing).
- **`attention_width="host"` only** in v1 (feature_native QKV both-sides projection raises — a follow-up).
- **No `host_wrapped_module`** (v1 ships GPT-2 only there; inherits the base `NotImplementedError`).
- **No rope_scaling** beyond Pythia's default (NeoX hosts with linear/dynamic/yarn scaling are a follow-up).

## Related

- `add-causal-host-capability-sweep` — the merged change whose Pythia-ladder follow-up this unblocks.
- `add-llama-family-rope` — the rope module this extends with the partial variant.
- FABLE_DIRECTIONS X5 (cross-host validation) / R1 (the `τ*` law's cross-arch ladder, which spans Pythia).
