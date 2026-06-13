# Implementation tasks

## Implementation status — COMPLETE (2026-06-13)

`gpt_neox` adapter landed + validated on `feat/gpt-neox-adapter`. ruff clean; new tests
(`tests/test_gpt_neox_adapter.py`, 6) green; existing adapter suite unaffected (40 passed together).

**GATE RESULT — PASS** (`scripts/prototype_gpt_neox.py`): identity-basis forge matches the host to float32
precision — tiny-random max|Δ| ≈ 1–9e-7 (4 configs, head_dim 16/32/64, rotary 0.25/0.5/1.0), **real Pythia-70m
rel 6.4e-6 + 100% argmax agreement**. Two bring-up bugs found & regression-guarded: `partial_rotary_factor`
silently defaulting to 1.0 (rope_parameters migration), and the float16-checkpoint dtype trap. Full writeup in
`proposal.md` "Gate RESULT".

## 0. Design pre-locks (blocking)

- [x] 0.1 **Native-forward correctness spike (gating risk):** identity-basis forge of a tiny random
  GPT-NeoX reproduces host logits to float tolerance. Localise any divergence by component (parallel
  residual / partial rotary / fused QKV / LayerNorm / GELU) before broader work. *(Found: partial rotary not
  reaching attention → traced to `rope_parameters` migration; fixed.)*
- [x] 0.2 Confirm the partial-rotary helper equals HF `GPTNeoXAttention` rotary on identical q/k/cos/sin (Δ=0).
- [x] 0.3 Lock v1 scope: `attention_width="host"` only; no `host_wrapped`; no rope_scaling beyond default.

## 1. `saeforge/_positional/rope.py` — partial rotary

- [x] 1.1 `apply_rotary_pos_emb_partial(q, k, cos, sin, rot_dim)`: rotate first `rot_dim` dims, pass the rest;
  HF rotate-half convention; reduces to full RoPE at `rot_dim == head_dim`.

## 2. `saeforge/adapters/gpt_neox.py` — adapter + native module

- [x] 2.1 `GPTNeoXAdapter.walk()`: project embed_in / two LayerNorms (w+b) / fused query_key_value
  (head-space bias unprojected) / dense (residual bias) / GELU MLP (biases) / final_layer_norm / untied
  embed_out. nn.Linear projection convention (reads→`project_residual_output`, writes→`project_residual_input`).
- [x] 2.2 `build_native_config()`: read `cfg.rope_parameters` (partial_rotary_factor, rope_theta) with legacy
  `rotary_pct`/`rotary_emb_base` fallback; `qkv_bias` from `attention_bias`; LayerNorm eps.
- [x] 2.3 Native `ForgedGPTNeoX`: parallel-residual block (two norms, summed), fused-QKV attention with
  partial rotary + causal SDPA, GELU MLP, LayerNorm, untied embed_out.
- [x] 2.4 Register in `adapters/__init__.py`; add `gpt_neox` to `model._SUPPORTED_FAMILIES`.
- [x] 2.5 Tests `tests/test_gpt_neox_adapter.py`: dispatch; config reads partial_rotary from rope_parameters;
  walk reaches every param + expected key set; identity-forge reproduces host logits (3 rotary fractions);
  compressed forge runs.

## 3. Acceptance gate (blocking merge)

- [x] 3.1 `scripts/prototype_gpt_neox.py`: identity forge — tiny-random (exact) + real Pythia-70m (relative,
  argmax). PASS. Route the result into `proposal.md` "Gate RESULT". No overclaim; descriptive.
