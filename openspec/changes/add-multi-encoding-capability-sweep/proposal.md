# Multi-Encoding Capability Sweep — compare basis choices in one run

Generalize `sweep_pareto_capability_progressive` to accept a **list** of `(encoding_label, sae_checkpoint_path)` pairs and compare them in a single sweep run, on the same protein subsamples, with apples-to-apples cell-by-cell retained_mauc deltas. **The architecturally-aware version of the mixed-χ ensemble idea** that the 2026-05-22 chat proposal asked for: instead of building an ensemble framework on top of closed-form forges (which doesn't fit; see `docs/proposals/warm-start-proposal-response.md`), expose the substantive question — *do different polygram encodings at different bond dimensions / partition strategies give different capability outcomes?* — as a direct comparison sweep.

## Why

Three converging signals make this the right next step:

1. **The data-scale tax that motivated the proposal exists.** The n=5000 progressive run on bio-sae's pooled fixture (2026-05-22) showed `retained_mauc` drifting 0.92 → 0.90 as data scale grows because the closed-form forge can't track the host's improvement. That's the gap the proposal correctly tries to address.

2. **The in-flight partition validation (#88, #89) tests ONE specific basis structure** (decoder-norm-quantile partition). If it helps, the natural follow-up is "what about other partitions? MPS bond-dim alternatives?" — exactly what this openspec enables. If it doesn't help, this openspec lets you systematically rule out OTHER encoding choices before reaching for fine-tuning.

3. **Polygram already ships multiple encodings** (`Rung5`, `MPSRung1`, `HEARung2`, plus the partition-aware basis builder this codebase just shipped at PR #89). They live in polygram's compression pipeline; their forge-side capability impact has never been directly compared because `sweep_pareto_capability` only accepts a single `sae_checkpoint`. **One sweep call should be able to compare them all at once on the same eval sample**, not multiple sequential single-encoding sweeps that recompute host activations independently and rely on the user to align frontier files by hand.

This openspec is the **substantive answer to the mixed-χ ensemble proposal** without inheriting its architecture mismatches (no model-state warm-start, no `χ-as-data-scale-axis` conflation, no ensemble framework on top of closed-form forge).

## What

### `sweep_pareto_capability(encodings=[(label, path), ...])` — multi-encoding API

```python
from saeforge import sweep_pareto_capability

rows = sweep_pareto_capability(
    encodings=[
        ("raw_slice",     "runs/uniref50_n5000/pooled_w1024_k64/sae.pt"),
        ("partition_q4",  "runs/polygram_partition/uniref50_n5000/pooled_w1024_k64_partition.pt"),
        ("mps_rung1_x4",  "runs/polygram_encoded/.../mps_rung1_x4.pt"),
        ("mps_rung1_x16", "runs/polygram_encoded/.../mps_rung1_x16.pt"),
        ("rung5",         "runs/polygram_encoded/.../rung5.pt"),
    ],
    host_model_id="facebook/esm2_t6_8M_UR50D",
    dataset=dataset,
    widths=[16, 64, 128, 256, 512, 1024],
    scale_boosts=[1.0],
    output_dir=Path("runs/multi_encoding/"),
)
```

The existing single-encoding `sae_checkpoint=` keyword is retained as a sugar for `encodings=[("raw_slice", sae_checkpoint)]`. Back-compat: every v0.8.x / v0.9.x caller continues to work unchanged.

Each cell now has shape `(encoding, width, scale_boost)` instead of `(encoding_label_informational, width, scale_boost)`. The `encoding_label` field on `ParetoFrontierRow` is now load-bearing — it identifies WHICH encoding's basis was used for that cell.

### `sweep_pareto_capability_progressive(encodings=[...])` — passthrough

The progressive wrapper accepts the same `encodings` list and runs each stage's sweep across all encodings simultaneously. Per-encoding plateau identification + per-encoding convergence detection. `ProgressiveHistory` carries per-encoding `ProgressiveRecommendation`s (one per encoding) plus a top-level "best encoding" pick.

### `ProgressiveRecommendation.per_encoding_recommendations`

Optional new field on `ProgressiveRecommendation` (back-compat: omitted when single-encoding sweep). When the sweep ran multiple encodings, this field is a `dict[str, ProgressiveRecommendation]` mapping encoding label to its per-encoding recommendation. The top-level `target_n_features_kept` + `retained_mauc_vs_host` belong to the best-converged encoding (smallest stable n at retained_mauc >= cross-encoding median).

### `sae-forge sweep-capability --encoding LABEL:PATH` (repeatable)

CLI accepts `--encoding` multiple times: `--encoding raw_slice:path1 --encoding partition_q4:path2 --encoding mps_rung1_x4:path3`. Mirrors the existing `sae-forge sweep-pareto --encoding LABEL:PATH` syntax for the non-capability sweep.

### Recommend over multi-encoding frontiers

`sae-forge recommend --target retained-mauc>=0.95` over a multi-encoding frontier picks **the encoding+width pair with the smallest n meeting the predicate**, breaking encoding ties by lowest median retained_mauc variance across the progressive stages. Output emits:

```
recommended encoding: partition_q4
  target_n_features_kept: 64
  retained_mauc_vs_host:  0.9523
  cross-encoding rank:    1/5 (n=64 was Pareto-best across all encodings tested)
```

## Falsifiable acceptance gate

Three predictions against bio-sae's pooled fixture under `[1000, 5000]` schedule, comparing five encodings (raw_slice, partition_q4, mps_rung1_x4, mps_rung1_x16, rung5):

| prediction | falsifies if |
|---|---|
| At least ONE encoding crosses the per-cell retained_mauc threshold (≥ 0.95 at n=512 at the largest stage) where raw_slice doesn't | every alternative encoding is ≤ raw_slice at every cell |
| Some encoding's plateau argmin is data-scale-stable (`per_encoding_recommendations[...].converged = True`) where raw_slice's isn't | every alternative encoding is also un-converged at default strictness |
| At least TWO encodings disagree on `target_n_features_kept` by more than one candidate-grid bucket at the same threshold | every encoding picks the same width (would mean encoding choice doesn't matter at this width grid) |

If all three predictions hold → multi-encoding sweep is load-bearing; encoding choice is a real lever. If only some hold → encoding-choice helps in specific regimes; documented. If none hold → encoding-choice doesn't move the capability frontier on this substrate; fine-tune is the next lever.

## Why this is NOT the mixed-χ ensemble proposal

The chat proposal (2026-05-22) framed χ as a *data-scale axis* (different protein counts → different optimal χ) and wanted to maintain a *model ensemble* with warm-start promotion between χ tiers. Two structural problems with that framing:

1. **χ isn't a data-scale axis**, it's an encoding-choice axis. Bond dimension is a property of the *encoding* (a polygram-side decision about how compressed the basis representation is), not of *how much data we evaluate against*. This openspec treats χ correctly — as one option among several encoding choices at any given data scale.

2. **There's no model state to ensemble** under the closed-form forge. The progressive sweep produces recommendations, not trained models. This openspec gives users a *recommendation per encoding* (or one across all encodings) without requiring an ensemble framework.

If the partition validation (#88) or this multi-encoding sweep reveals that **the closed-form forge fundamentally can't close the gap regardless of encoding**, that's the empirical motivation for `add-progressive-finetune` — and *then* the original proposal's warm-start ensemble ideas apply to a real model-state substrate.

## Scope (v1)

- **In:**
  - Multi-encoding `sweep_pareto_capability` + progressive passthrough.
  - Per-encoding `ProgressiveRecommendation` with cross-encoding "best" pick.
  - `sae-forge sweep-capability --encoding LABEL:PATH` (repeatable).
  - `sae-forge recommend` over multi-encoding frontiers.
  - Falsifiable acceptance gate on bio-sae pooled at [1000, 5000] comparing 3-5 polygram encodings.
- **Out:**
  - Polygram-side machinery to emit per-encoding shadow checkpoints at multiple bond dimensions. This openspec consumes whatever polygram ships; if polygram needs to emit new encoded checkpoints, that's a separate polygram-side change.
  - Cross-encoding ensembling at inference time (the original proposal's `MixedChiEnsemble.sample()`). Out of scope; deferred until there's a use case for "run the forged model with multiple bases active at once."
  - Optimal per-encoding scale_boost discovery. v1 uses a fixed `scale_boost=[1.0]`; advanced calibration is a separate openspec.

## Related

- The single-encoding sweep this generalizes: `sweep_pareto_capability` (PR #76) + `sweep_pareto_capability_progressive` (PR #83).
- Partition-aware basis builder this composes: `add-partition-encoding-capability-validation` (PR #88-89).
- Existing multi-encoding pattern (non-capability): `sae-forge sweep-pareto --encoding LABEL:PATH` (PR #65-ish, the original sweep).
- Polygram encoders: `polygram.encoding.rung5`, `polygram.encoding.mps_rung1`, etc. (polygram v0.14.0+).
- Counter-shape for the warm-start proposal: `docs/proposals/warm-start-proposal-response.md` (PR #86).
- Counter-shape for THIS proposal (mixed-χ ensemble) — this openspec IS that counter-shape.
