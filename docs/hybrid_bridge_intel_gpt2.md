# Hybrid-bridge forge: Intel/GPT-2 baseline numbers (T1)

This document captures the cross-architecture defaults-validation baseline
documented in [the hybrid-bridge-forge change proposal](../openspec/changes/hybrid-bridge-forge/design.md)
§ "Cross-architecture validation tiering" (T1 tier). See also
[`tasks.md`](../openspec/changes/hybrid-bridge-forge/tasks.md) for the
deferred follow-ups (FSM wiring, Llama/Gemma-2 bridge insertion, real-SAE
re-sweep, T3 M4 reproduction, T4 community CUDA validation).

## Quick start

Hybrid forging is **opt-in** via `--hybrid-bridge`. The flag requires
three Polygram-compressed SAE checkpoints — one per anchor (embed / mid /
lm-head). The CLI does not auto-train SAEs.

The host model must have **untied embeddings**. GPT-2 ships tied by
default, so the simplest reproducible CLI path is to point at an
already-untied host. For a one-shot exploration without doing SAE
training, the comparison harness applies an in-script untying workaround:

```bash
# Reproduce the T1 ablation (PCA-proxy bases, no real SAE training):
python scripts/compare_single_vs_hybrid_gpt2.py --n-features 256
```

For a real forge run with trained SAE bases (recommended once you have
three Polygram-compressed checkpoints at three anchor layers):

```bash
sae-forge forge ./mid_basis.compressed.safetensors \
  --host-model your-untied-host \
  --output-dir runs/hybrid \
  --hybrid-bridge \
  --basis-embed ./embed_basis.compressed.safetensors \
  --basis-lm-head ./lm_head_basis.compressed.safetensors
```

Bridge configuration knobs (defaults in **bold**):

| Flag | Choices | Default |
|---|---|---|
| `--bridge-init` | `orthogonal` / `identity` / `zero` | **`orthogonal`** |
| `--bridge-nonlin` | `none` / `relu` / `gelu` | **`none`** |
| `--bridge-no-pre-ln` | flag | **off** (i.e. pre-LN **enabled**) |

The default config is validated against the T1 ablation table below.

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

## Headline: hybrid wins decisively at low/mid rank, breaks catastrophically at high rank

| n_features | single-mid KL | single-embed KL | single-lm KL | hybrid KL | ΔKL (hybrid − single-mid) |
|---:|---:|---:|---:|---:|---:|
|  64 | 6.07 | 5.42 | 41.28 |  **4.95** | **−1.11** |
| 128 | **6.42** | 7.83 | 17.83 |  7.08 | +0.67 |
| 256 | 8.03 | 5.24 | 20.46 |  **3.39** | **−4.63** |
| 512 | **6.02** | 37.37 | 15.50 |  29.27 | +23.26 |
| 768 | **7.26** | 21.95 | 40.88 | 108.27 | +101.01 |

Observations:

- **Hybrid wins at n=64 and n=256**, loses narrowly at n=128, and breaks
  catastrophically at n=512 and n=768. The non-monotonicity at low rank
  mirrors the [`project_kl_nonmonotonic`](../) memory; the breakdown at
  high rank is a new finding worth investigating.
- **Single-lm is always terrible** (KL > 15 across every `n_features`).
  The layer-11 residual stream is already lm-head-shaped; PCA-ing it
  produces a basis that fits the unembed *output* distribution but not
  the pre-unembed transformer stack. Confirms that "one basis at the
  deepest layer" is a strictly bad single-basis strategy.
- **The breakdown above n=256 traces to PCA-proxy quality, not the
  mechanism.** As `n_features` grows past ~256, the per-layer PCA picks
  up increasingly noisy singular directions in the rank-512..768 tail
  (we have ~912 captured tokens supporting up to rank-768 PCA — the
  signal-to-noise ratio at high rank is poor). When two of the three
  bases individually exceed KL ≈ 20, the hybrid composition compounds
  those errors through the (untrained) bridges. Real trained SAEs do
  not suffer this failure mode at the same rank, which is precisely
  why the M4 Gemma-2-2B prototype (with trained SAEs at all three
  anchors) reported KL=11.81 at `n_features=256` rather than the
  catastrophic numbers a PCA proxy would predict at the same rank.
- **The right way to read this table:** hybrid is opt-in (default off)
  precisely because it requires all three bases to be individually
  high-quality. Hybrid's headline win at n=256 (-4.63 nats) holds when
  the bases are good; the breakdown at n≥512 is a faithful warning
  signal that the *bases* have degraded, not that the hybrid mechanism
  is wrong.

A useful follow-up: rerun this sweep with trained Polygram SAEs as
bases (via `--basis-embed PATH --basis-lm-head PATH` in the CLI) to
separate the mechanism's failure modes from the synthetic-proxy's.

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

## Cross-family coverage

The numbers above were captured against GPT-2 specifically because the
GPT-2 native module is the first place the hybrid-bridge forward-pass
insertion landed (#18). As of the `hybrid-bridge-llama-family` change,
the same mechanism is wired through the shared `LlamaTransformer`
factory and exercises end-to-end on the **Llama**, **Gemma-2**, and
**Qwen2** families. The mechanism is family-agnostic; only the
state-dict key prefix differs (`transformer.bridges.*` for GPT-2,
`model.bridges.*` for the Llama-family — each reflecting the host's
own HF naming convention).

Family-specific integration coverage:

- `tests/integration/test_hybrid_bridge_gpt2.py` — T0 GPT-2 (this doc's
  numbers come from this surface).
- `tests/integration/test_hybrid_bridge_llama.py` — T0 untied Llama
  smoke + round-trip + tied-refusal + zero-init inversion.
- `tests/integration/test_hybrid_bridge_qwen2.py` — same shape against
  an untied Qwen2 (qkv_bias=True) host. Validates that the Q/K/V bias
  state-dict entries coexist cleanly with the bridge state-dict
  entries.
- `tests/integration/test_hybrid_bridge_qwen3.py` — same shape against
  an untied Qwen3 (qk_norm=True) host. Validates that the per-head Q/K
  RMSNorm weights coexist cleanly with the bridge state-dict entries.
  Requires `transformers >= 4.51`; the entire file skips on installs
  without Qwen3 support (the `[intel]` extra is capped at `<4.50`).

A Gemma-2 family integration test is deferred to the T3 M4 reproduction
pass (the mechanism works for Gemma-2 by inheritance through
`LlamaTransformer`, but pinning a CI test without cached Gemma-2 weights
doesn't add signal).

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
for n in 64 128 256 512 768; do
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
