# Implementation tasks

## Implementation status — COMPLETE (2026-06-13)

Landed on `feat/causal-host-capability-sweep`. ESM path byte-identical (32 existing sweep tests green); new
tests in `tests/test_causal_host_sweep.py` (7). **GATE RESULT — SURVIVES:** GPT-2 full-forge proxy-trained `E`
beats `pinv` at n=128 by **+0.039 ± 0.003** (3 seeds, ~13σ, overfit clean), where the matched ESM protocol
tied — the causal trained-encoder win is **forge-level**. Full writeup in `proposal.md` "Gate RESULT".
Tasks 0–4 below all done; the spike (0.1) PASSED (forward-pre-hook captures the forged basis-space resid).

## 0. Design pre-locks (blocking)

- [x] 0.1 **Forged mid-layer capture spike (gating risk):** confirm a forward-pre-hook on
  `forged_module.transformer.h[L]` captures a finite `(1, seq, N)` basis-space tensor during
  `forged_module(input_ids)`, and that `captured @ W_dec_slice` reproduces the host's `resid_pre[L]` shape
  `(seq, d_model)`. If the hook misses (functional_call / module aliasing), name it and pick the re-expression
  before other work.
- [x] 0.2 Lock the layer convention: host `hidden_states[L]` == `blocks.{L}.hook_resid_pre` == forged
  `transformer.h[L]` **input**. Verify on the real jbloom layer (L=8) that host extraction matches a
  TransformerLens-style `hook_resid_pre` to tolerance (relative SAE latents sane, recon cosine > 0.8).
- [x] 0.3 Lock v1 family scope: `gpt2` causal branch; other causal families raise `NotImplementedError`
  naming the block-path. ESM-2 (`host_layer=None`) stays byte-identical.

## 1. `_load_encoding_state` — SAELens key support

- [x] 1.1 Accept `W_dec` `(n_features, d_model)` / `W_enc` (SAELens) in addition to `decoder.weight`
  `(d_model, n_features)` / `encoder.weight`. Detect by key presence; assert shape unambiguous.
- [x] 1.2 Test: load a jbloom-shaped SAELens state dict and a bio-shaped `decoder.weight` state dict; both
  yield the same `(n_features, d_model)` `W_dec_full` + row norms.

## 2. `sweep_pareto_capability` — `host_layer` + causal extraction

- [x] 2.1 Add `host_layer: int | None = None`. Thread it into `_extract_host_activations` /
  `_extract_forged_activations`. `None` ⇒ the existing ESM path (byte-identical).
- [x] 2.2 `_extract_host_activations` causal branch: `output_hidden_states=True` → `hidden_states[host_layer]`,
  no `[1:-1]` strip; per-token (`feed="residue"`) or mean-pool (`feed="pooled"`) as today.
- [x] 2.3 `_extract_forged_activations` causal branch: forward-pre-hook on `transformer.h[host_layer]`
  capturing basis-space `resid_pre`; remove hook in `finally`. Non-`gpt2` causal families raise
  `NotImplementedError`.
- [x] 2.4 Tests: (a) ESM path unchanged (existing tests green); (b) a tiny GPT-2 host end-to-end cell
  populates host + forged activations at `host_layer` with aligned row counts; (c) non-gpt2 causal raises.

## 3. `scripts/causal_host_forge_gate.py` — the gate

- [x] 3.1 Build a `CapabilityDataset` for GPT-2 (corpus + jbloom SAE encoder + SAE-derived labels,
  `tokenizer_id="gpt2"`, `feed="residue"`). Run `sweep_pareto_capability(train_encoder=True,
  train_objective="proxy", host_layer=8)` at widths × seeds; compare trained vs `pinv` full-forge
  retained-mAUC. Reuse `causal_lm_forge_gate.py`'s corpus + label-derivation helpers (no duplication).

## 4. Acceptance gate (blocking merge) — multi-seed, both outcomes first-class

- [x] 4.1 Run at GPT-2 widths (n ∈ {64,128,256}), ≥3 seeds, compression-controlled, held-out. Report
  mean ± std of `delta_heldout` per width.
- [x] 4.2 **Descriptive verdict, pre-committed both ways:** SURVIVES (trained > pinv through the full GPT-2
  forge → the causal win is forge-level) OR ERASED (ties/loses → the forge tax erases it too). Route into
  this change's "Gate RESULT" + the parent's follow-up note. No "irreducible"/"closes the tax" language.
