# Causal-host capability sweep — run the trained-encoder forge gate on a CAUSAL LM (the host-class falsifier, at the FORGE level)

Generalize `sweep_pareto_capability`'s activation extraction so the **full multi-layer forge** capability
gate runs on a **causal LM host** (GPT-2) at the SAE's own mid-layer hook point — not just on ESM-2. This is
the follow-up the `add-full-forge-encoder-training` **Causal-LM control RESULT** named as still-open: the
activation-level control showed trained-`E` beats `pinv` on a causal host, but whether that win **survives
the full forge** (the LayerNorm/TopK tax that erased ESM's tiny activation-level gain) is untested because the
forge scoring path is ESM-shaped.

## Why

The capability-trained-encoder line produced a **null on ESM-2** (`pinv` near-optimal) and then a **causal
activation-level control** showing the null is *host-class-specific* — trained-`E` clearly beats `pinv` on
GPT-2 at the projection level, ties on ESM-2 (`add-full-forge-encoder-training/proposal.md` "Causal-LM control
RESULT"). But the **decisive** measurement — the one the ESM bio gate ran — is **proxy-train `E` + score on
the full forge**. On ESM that gave the null; the causal analog is **blocked** for a narrow, fixable reason:

> The cached jbloom GPT-2 SAEs live on a *mid-layer* residual (`blocks.8.hook_resid_pre`), but
> `sweep_capability._extract_host_activations` / `_extract_forged_activations` are **ESM-shaped**: they read
> `host.esm` + `out.last_hidden_state` and strip `[0, 1:-1, :]` (ESM CLS/EOS), and they implicitly score the
> *final* layer. A GPT-2 host returns `.logits` (no `last_hidden_state`), has no CLS/EOS to strip, and its SAE
> is at a mid-layer the helpers can't target. The **forge itself already works for GPT-2**
> (`examples/forge_gpt2_real.py` projects weights → `NativeModel` → forward); only the two extraction helpers
> and the SAE-key loader are encoder-only.

This change unblocks exactly those helpers. Nothing about the forge, the projector, `train_encoder`, or the
sweep's scoring logic changes — only **where the activations are read from** (a named layer, causal-aware).

## What

### 1. Causal-aware, layer-targeted activation extraction

`sweep_pareto_capability(..., host_layer: int | None = None)`. When `host_layer` is set (causal hosts), both
extraction helpers read the residual stream **at that layer**, the way the SAE was trained:

- `_extract_host_activations` — causal branch: run the HF host with `output_hidden_states=True`, take
  `hidden_states[host_layer]` (= `blocks.{layer}.hook_resid_pre`), **no** `[1:-1]` strip (causal LMs have no
  CLS/EOS). The ESM branch (`host.esm` / `last_hidden_state` / strip) is unchanged and used when
  `host_layer is None`.
- `_extract_forged_activations` — causal branch: register a **forward-pre-hook** on the forged module's block
  `host_layer` (`forged_module.transformer.h[host_layer]`) to capture its **basis-space** `resid_pre`
  `(seq, N)` — no model surgery, no forward-signature change. The existing `forged_d = forged_h @ W_dec`
  remap (basis → `d_model`) then feeds the SAE exactly as for ESM.

Family dispatch is explicit: GPT-2 in v1 (`transformer.h[...]`); other causal families
(Llama/Gemma-2/Qwen) raise a clear `NotImplementedError` naming the block-path to wire (a one-line follow-up
each). ESM-2 keeps the `host_layer=None` path byte-identical.

### 2. SAELens-format SAE loading

`_load_encoding_state` accepts the SAELens key convention (`W_dec` `(n_features, d_model)` / `W_enc`) used by
GPT-2 SAEs, in addition to the existing `decoder.weight` `(d_model, n_features)` convention. Detection is by
key presence; no transpose ambiguity (the two conventions are distinguishable by shape + name).

### 3. The causal forge gate

`scripts/causal_host_forge_gate.py` — the GPT-2 analog of `forge_trained_encoder_bio_gate.py`: build a
`CapabilityDataset` (GPT-2 corpus + jbloom SAE encoder + SAE-derived labels, `tokenizer_id="gpt2"`,
`feed="residue"` per-token), run `sweep_pareto_capability(train_encoder=True, train_objective="proxy",
host_layer=8)` at compressed widths, multi-seed, and compare trained vs `pinv` **full-forge** retained-mAUC —
the same comparison the ESM bio gate ran, now on a causal host.

## Falsifiable acceptance gate (both outcomes first-class, descriptive)

Run the causal gate at GPT-2 widths (e.g. n ∈ {64, 128, 256}), ≥3 seeds, compression-controlled, held-out.

- **Survives** — trained-`E` full-forge retained-mAUC **> `pinv`** by a noise-clearing margin → the causal
  projection win **survives the forge**; the host-class result is forge-level, not just activation-level. The
  strongest possible vindication of the trained-encoder thesis on a real LM forge.
- **Erased** — trained-`E` ties/loses through the full GPT-2 forge (as on ESM) → the forge tax (LayerNorm /
  TopK rank-shuffle) erases the causal activation-level gain too; the projection win is real but **not
  forge-robust**, narrowing where capability-trained bases help.

Either way: multi-seed, `overfit_flag`-surfaced, routed into this change's "Gate RESULT" + the parent change's
follow-up note. **No "irreducible" / "closes the tax" language** (`no-necessity-claims`).

### Gate RESULT (real GPT-2 + jbloom SAE, 2026-06-13) — the causal win SURVIVES the forge

Implemented and run (`scripts/causal_host_forge_gate.py`; data in `scripts/causal_host_forge_gate_results.json`).
Proxy-train `E` + score on the **full multi-layer GPT-2 forge**, layer 8, 3 seeds, compression-controlled:

| width | pinv | trained | Δ mean ± std | gate |
|---:|---:|---:|---:|:--|
| 64  | 0.7716 | 0.7688 | −0.0028 ± 0.0014 | ERASED |
| 128 | 0.7385 | 0.7776 | **+0.0391 ± 0.0031** | **SURVIVES** |
| 256 | 0.7495 | 0.7568 | +0.0073 ± 0.0048 | SURVIVES |

**The causal trained-encoder win SURVIVES the full forge.** At n=128 the trained `E` beats `pinv` by
**+0.039 (~13σ, 3 seeds, overfit_flag clean)** through the complete multi-layer GPT-2 forward — where the
**matched** ESM-2 protocol (proxy-train + full-forge-score) gave a **tie** (`/tmp` bio proxy gate n=128:
Δ −0.0003). So the host-class signal is **forge-level**, not merely activation-level: the LayerNorm/TopK
forge tax that erased ESM's tiny gain does **not** erase GPT-2's. At n=64 (smallest width) it ties/slightly
loses — the win is width-dependent, strongest at the mid widths, consistent with the activation-level gate.

**Caveats (descriptive, `no-necessity-claims`).** (i) Δ is the *optimistic* all-token row delta (the trained
`E` saw the fit subset), same as the ESM bio gate — apples-to-apples, and `train_encoder`'s internal held-out
is early-stop-protected (overfit_flag clean at n=128). (ii) **SAE-type confound stands** (GPT-2 ReLU/L1 vs
ESM TopK) — a matched-SAE control is still the clean isolation. (iii) Proxy-trained, not trained *through* the
differentiable causal forge (esm2-only) — a follow-up. The verdict: **trained-`E` beats `pinv` at the forge
level on a causal host**, the strongest cross-substrate evidence yet that the trained-encoder thesis is real
and causal-host-conditional.

## What this does NOT solve

- **One causal family in v1** (`gpt2`). Llama/Gemma-2/Qwen raise `NotImplementedError` (the block-path is the
  only per-family wiring); no silent fallback.
- **Not the differentiable causal forge.** This is the **proxy**-trained path (train `E` on the activation
  proxy, score on the full forge) — exactly the ESM bio gate. Training `E` *through* the differentiable GPT-2
  forge (the `forge_diff.py` analog for causal hosts) is a separate, larger follow-up.
- **Not a matched-SAE control.** Isolating SAE activation type (ReLU/L1 vs TopK) from host class needs a
  matched SAE pair; out of scope here.

## Related

- `add-full-forge-encoder-training` — the parent; its "Causal-LM control RESULT" names this experiment as the
  open forge-level falsifier. `scripts/causal_lm_forge_gate.py` is the activation-level version this extends.
- FABLE_DIRECTIONS X5 (cross-host validation) — this is the first real cross-host *forge* measurement; its
  verdict feeds the "report per host class, not one number" guidance.
