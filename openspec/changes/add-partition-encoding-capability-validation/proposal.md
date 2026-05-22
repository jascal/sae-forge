# Partition-Encoding Capability Validation — re-test Wave C under the capability framework

Re-run polygram's partitioned-basis encoding through `sweep_pareto_capability_progressive` to measure whether partition closes the data-scale-widening retained_mauc gap that the n=5000 progressive sweep surfaced on bio-sae's pooled regime. **Pure measurement experiment**, not a new architecture. Wave C shipped polygram's partition machinery in v0.14.0 but the projected forge-side payoff (10-30 % `forge_kl` improvement) didn't materialize at 0 %, so the partition was filed as "unproven." We now have evidence that `forge_kl` is the wrong metric — and a wrapper (`sweep_pareto_capability_progressive`) that asks the right question.

## Why

The series of empirical findings from the progressive sweep series converges on a specific question:

| measurement | finding |
|---|---|
| Bio-sae writeup §3.2 (n=500, single-shot, `forge_kl` metric) | Pooled regime carries a "uniform tax" of ~7 % AUC across all widths. |
| Wave C partition A/B (2026-05-21, `forge_kl` metric) | Partition shows 0 % forge_kl change vs raw_slice. Filed as "unproven." |
| Progressive sweep `[200, 500]` (PR #85, `retained_mauc` metric) | Pooled regime un-converged: plateau argmin shifts n=384 → n=256 as data scale doubles. |
| Progressive sweep `[1000, 5000]` (this work, 2026-05-22, `retained_mauc`) | Plateau argmin stable at n=256; **retained_mauc drifts 0.92 → 0.90** because host_mauc grows +3.8 % while forge_mauc grows +0.3 %. The "uniform tax" widens with data scale. |

The architectural reason: the closed-form forge is a fixed-rank linear projection. The host has access to nonlinear interactions that more data lets it exploit; the forge can't. The gap grows as more discriminating signal surfaces.

**Partition is the most-built-out architectural alternative that could fix this.** Instead of slicing the SAE's W_dec by row norm (the current `raw_slice` encoding — flat top-N rows), partition groups features into tiers (e.g. bio-sae's `categorical=24, hierarchical=40, positional=3, synthetic=8` partition from `runs/polygram_partition/uniref50_small/partition_summary.json`). The hypothesis: when downstream labels span hierarchical structure (GO terms, Pfam families), partitioning the basis along feature-cluster boundaries preserves the relations the flat slice drops.

Wave C tested this and saw 0 % `forge_kl` improvement. But `forge_kl` measures residual-stream numerical proximity — a metric that's blind to the question partition was supposed to answer. **Capability metrics ask the relevant question: does the forge preserve the downstream task's discriminative structure?** If partition is right architecturally, capability should show it where KL did not.

## What

Pure measurement experiment. **No new sae-forge surface.** Three phases:

### Phase 0 — Materialize the partition checkpoint (bio-sae side, ~1 day)

The polygram-partitioned SAE checkpoint isn't on disk yet (only the partition spec at `runs/polygram_partition/uniref50_small/partition_summary.json` describing the *intended* structure). This phase produces a usable safetensors at `bio-sae/runs/polygram_partition/uniref50_n5000/pooled_w1024_k64_partition.pt` by running polygram's compression pipeline with the partition active.

### Phase 1 — Run the falsifiable measurement (sae-forge side, ~half day)

```bash
# Raw-slice baseline (already runnable via the progressive wrapper).
sae-forge sweep-capability-progressive \
    --dataset-config bio-pooled.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --candidate-widths 16,64,128,256,512,1024 \
    --schedule 1000,5000 \
    --output-dir runs/partition_validation/raw_slice/

# Partition (new sae.pt path from Phase 0).
sae-forge sweep-capability-progressive \
    --dataset-config bio-pooled-partition.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --candidate-widths 16,64,128,256,512,1024 \
    --schedule 1000,5000 \
    --output-dir runs/partition_validation/partition/
```

Two `progressive_summary.json` files; comparison is mechanical.

### Phase 2 — Writeup (bio-sae side, ~half day)

The empirical outcome lands as a new section under `bio-sae/docs/forge-capability-bottleneck.md` §5 (or a new doc). Three possible narratives, all worth documenting:

| outcome | downstream implication |
|---|---|
| **Partition closes the data-scale drift** (retained_mauc variance across [1000, 5000] < 0.005; `converged=True`) | Wave C is re-evaluated as "right architecture, wrong metric." Capability framework validates partition. `add-progressive-finetune` becomes unnecessary on this substrate. |
| **Partition reduces drift partially** (variance halved; `converged=False` but the gap narrows) | Partition is a real but insufficient lever. Combine with `add-progressive-finetune` or other fixes. The "structural tax" decomposes into multiple causes. |
| **Partition shows the same drift** (variance ≥ 0.024; `converged=False` with no improvement) | Basis structure isn't the bottleneck. The data-scale tax is fundamentally about the forge's representational ceiling vs host's. `add-progressive-finetune` becomes the next candidate. |

## Falsifiable acceptance gate

The validation experiment is itself the acceptance gate. Predictions:

| pre-experiment expectation | metric | falsifies if |
|---|---|---|
| partition `retained_mauc` at n=256 is ≥ raw_slice's by ≥ 0.02 at the largest stage (n=5000) | `retained_mauc_vs_host` per cell | partition `retained_mauc` < raw_slice's, OR within ±0.01 (would mean no effect either way) |
| partition trajectory variance across [1000, 5000] < raw_slice's | `argmin_retained_mauc` variance across stages | partition variance equal or greater than raw_slice |
| partition `converged=True` under default strictness | `recommendation.converged` | partition is still un-converged, AND raw_slice wasn't — would mean the gap is real but partition doesn't fix it |

The three predictions partition a 2×2×2 = 8 outcome space; the writeup names which cell of the space landed and what it implies.

## Why this is NOT a sae-forge architectural change

The current `sweep_pareto_capability` takes `sae_checkpoint: str | Path` and slices W_dec by row norm. A polygram-partitioned safetensors with the same `encoder.weight` / `decoder.weight` key shape **is** an SAE checkpoint as far as the wrapper is concerned — the partition structure lives in W_dec's algebraic content, not in a new file format. Phase 1 is a drop-in path substitution.

If partition WINS the capability test, the natural follow-up is to expose multi-encoding sweeps as a first-class API (`sae-forge sweep-capability --encoding raw_slice:path --encoding partition:path`). That's a separate openspec (`add-multi-encoding-capability-sweep`), deferred until the validation result motivates it.

## Why this is worth doing first (vs jumping to `add-progressive-finetune`)

Three reasons:

1. **Partition is already shipped (polygram v0.14.0).** This is a measurement experiment, not an architecture-implementation one. ~2 day total work; clear three-outcome gate. `add-progressive-finetune` is multi-week scope.
2. **Wave C is currently in the "unproven" pile.** Resolving it positively or negatively under the capability framework retires an open question with real evidence. The current state — "we shipped partition but it didn't help on KL, never tested on capability" — is the worst possible epistemic position.
3. **The result conditions future work.** If partition wins, fine-tune is unnecessary. If partition loses, fine-tune is well-motivated. Either way the next openspec's case strengthens.

## Related

- The progressive sweep that surfaced the data-scale-widening tax: `add-progressive-capability-sweep` (PR #82-87).
- The n=5000 empirical evidence: `bio-sae/runs/forge/progressive_pooled_n5000/progressive_summary.json` (2026-05-22 background run).
- Wave C's original partition shipment: polygram v0.14.0 (2026-05-21).
- Wave C's "unproven" status: memory entry `wave-c-partition-forge-side-unproven` (2026-05-21).
- Deferred follow-up if partition loses: `add-progressive-finetune` (the warm-start counter-shape's deferred third openspec).
- Partition spec on disk: `bio-sae/runs/polygram_partition/uniref50_small/partition_summary.json` (75 features across 4 tiers).
