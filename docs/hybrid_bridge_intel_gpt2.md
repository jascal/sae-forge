# Hybrid-bridge forge: Intel/GPT-2 baseline numbers (T1)

This document captures the cross-architecture defaults-validation baseline
documented in [the hybrid-bridge-forge change proposal](../openspec/changes/hybrid-bridge-forge/design.md)
§ "Cross-architecture validation tiering" (T1 tier).

**Hardware:** 16GB Intel Mac (Core i9), no CUDA. Python 3.11.
**Date:** 2026-05-11.
**Host model:** `gpt2` (124M params, 12 layers, d_model=768), with the
untying workaround (clone `wte.weight` → `lm_head.weight`, flip
`tie_word_embeddings=False`). Numerically identical forward; the untying
purely unblocks hybrid forging.
**Bases:** PCA-derived "synthetic SAE" bases — top-`n_features` singular
vectors of the post-layer residual stream over a 32-prompt activation set.
Not trained SAEs; cheap CPU proxies for the cross-arch defaults experiment.
**Eval:** Faithfulness KL over 4 held-out prompts, pre-fine-tune (bridges
at their orthogonal-random init, not trained).
**Harness:** [`scripts/compare_single_vs_hybrid_gpt2.py`](../scripts/compare_single_vs_hybrid_gpt2.py).

## Headline: hybrid wins at n_features ∈ {64, 256}; loses at n=128

| n_features | single-mid KL | single-embed KL | single-lm KL | hybrid KL | ΔKL (hybrid − single-mid) |
|---:|---:|---:|---:|---:|---:|
|  64 | 6.07 | 5.42 | 41.28 |  **4.95** | **−1.11** |
| 128 | **6.42** | 7.83 | 17.83 |  7.08 | +0.67 |
| 256 | 8.03 | 5.24 | 20.46 |  **3.39** | **−4.63** |

Observations:

- **Hybrid wins at n=64 and n=256**, loses by a small margin at n=128. The
  non-monotonicity in `n_features` mirrors the [`project_kl_nonmonotonic`](../)
  memory and is unsurprising — at fixed feature count, the rank-vs-faithfulness
  curve is genuinely non-monotonic for synthetic PCA bases.
- **Single-lm is always terrible** (KL > 17 across all `n_features`). The
  layer-11 residual stream is already lm-head-shaped; PCA-ing it produces
  a basis that fits the unembed *output* distribution but not the
  pre-unembed transformer stack. Confirms that "one basis at the deepest
  layer" is a strictly bad single-basis strategy.
- **Single-embed is competitive but unstable** — it beats single-mid at
  n=64 and n=256 but loses at n=128. Layer-0 activations have low
  pre-block structure, so PCA over them sometimes captures a more
  general basis than the mid layer.

## Bridge-config ablation (at n_features=256)

| Bridge init | Nonlin | Pre-LN | Hybrid KL | ΔKL vs single-mid |
|---|---|---|---:|---:|
| **orthogonal** | **none** | **on** | **3.39** | **−4.63** |
| orthogonal | relu | on | 3.32 | −4.71 |
| orthogonal | none | off | 5.95 | −2.07 |
| identity | none | on | 6.30 | −1.73 |
| zero | none | on | 10.98 | +2.95 |

**The v1 default (orthogonal + linear + pre-LN) is validated.** It's
within 0.07 nats of the best config (orthogonal + ReLU + pre-LN) and
preserves the save-time-fold option discussed in the design doc. Findings:

- **Orthogonal init is load-bearing.** Identity-init bridges (which pass
  embed-basis activations into mid-region blocks unchanged) cut the win
  by ~3 nats. Zero-init bridges *invert* the result — the hybrid becomes
  worse than single-mid because the bridge clobbers the signal to zero.
  This is consistent with the design doc's framing: bridges work
  via *initialization isolation*, not added linear capacity.
- **Pre-LN is also load-bearing.** Removing it cuts the win in half
  (−4.63 → −2.07). Layer normalization stabilizes the bridge's effect
  on activation magnitudes.
- **ReLU vs linear: essentially tied (3.32 vs 3.39).** At pre-fine-tune
  the non-linearity adds almost nothing — the bridges haven't been
  trained yet, so the activation just clamps random-rotated activations.
  Post-fine-tune the ranking may flip; tracked as a follow-up
  experiment.

## What this experiment does NOT establish

1. **Trained SAE bases.** The bases here are PCA proxies. Real Polygram-
   compressed SAEs may have different geometry. The harness is intentionally
   structured so an external contributor with NVIDIA/CUDA can plug in real
   SAE bases via `--basis-embed` / `--basis-lm-head` and rerun.
2. **Post-fine-tune KL.** This is a pre-FT measurement. The proposal's
   ultimate claim is that bridges enable faster convergence. The fine-tune
   path is wired into the FSM orchestrator (`orchestrator="fsm"`); the
   imperative orchestrator used by this harness does not run fine-tune.
3. **Larger hosts.** `gpt2-medium`, `gpt2-large`, Gemma-2-2B, Llama-3-8B —
   all out of scope for the 16GB Intel box. Tracked as T2 (Intel
   post-merge), T3 (M4), and T4 (external NVIDIA/CUDA) in the design doc.

## Reproducing this table

```bash
# n_features sweep
for n in 64 128 256; do
  python scripts/compare_single_vs_hybrid_gpt2.py \
    --n-features $n --output runs/comparison_n${n}.json
done

# bridge-config ablation at n=256
python scripts/compare_single_vs_hybrid_gpt2.py --n-features 256 \
  --bridge-init orthogonal --bridge-nonlin none \
  --output runs/n256_orth_none.json

python scripts/compare_single_vs_hybrid_gpt2.py --n-features 256 \
  --bridge-init identity --bridge-nonlin none \
  --output runs/n256_id_none.json

python scripts/compare_single_vs_hybrid_gpt2.py --n-features 256 \
  --bridge-init zero --bridge-nonlin none \
  --output runs/n256_zero_none.json

python scripts/compare_single_vs_hybrid_gpt2.py --n-features 256 \
  --bridge-init orthogonal --bridge-nonlin relu \
  --output runs/n256_orth_relu.json

python scripts/compare_single_vs_hybrid_gpt2.py --n-features 256 \
  --bridge-init orthogonal --bridge-nonlin none --bridge-no-pre-ln \
  --output runs/n256_orth_none_nopreln.json
```

Each run takes 10–15 seconds on the Intel box, so the full table
reproduces in roughly two minutes.
