## Context

Polygram's `add-encoding-partition` Phase 2 shipped in v0.14.0 (2026-05-21). The feature groups SAE features into tiers and applies per-tier encoding under polygram's compression pipeline. Bio-sae's `runs/polygram_partition/uniref50_small/partition_summary.json` describes a 4-tier partition (categorical=24, hierarchical=40, positional=3, synthetic=8). What's documented as "shipped but unproven": the forge-side `forge_kl` A/B (2026-05-21) showed 0 % improvement vs raw_slice, falsifying the 10-30 % projected payoff.

The progressive capability sweep series (PRs #82-#87, 2026-05-22) introduces a different metric (`retained_mauc`) and a multi-scale measurement protocol that wasn't available when Wave C was tested. The `[1000, 5000]`-stage progressive run on bio-sae's pooled fixture (this session's background work) surfaced a specific gap: host_mauc grows +3.8 % with data, forge_mauc grows +0.3 %, retained_mauc drops 0.92→0.90. That gap is exactly the question partition was designed to address, and the original A/B couldn't have measured it.

This change is a **measurement experiment**, not architecture. The goal is to settle Wave C's "unproven" status under the right metric framework, with three falsifiable outcomes that each condition the next-best architectural lever (`add-progressive-finetune`, multi-encoding sweep API, or substrate-specific partition refinement).

## Goals / Non-Goals

**Goals:**
- Materialize a polygram-partitioned SAE checkpoint usable as a drop-in `sae_checkpoint` argument to `sweep_pareto_capability`.
- Run the existing progressive wrapper against both `raw_slice` and `partition` encodings at the same schedule, comparing retained_mauc drift + plateau argmin + convergence flag.
- Document the outcome under `bio-sae/docs/forge-capability-bottleneck.md` regardless of direction.

**Non-Goals:**
- Multi-encoding sweep API (`--encoding LABEL:PATH` for capability sweeps). If partition wins, that's a follow-up openspec. v1 here runs two single-encoding sweeps and compares manually.
- Re-implementing polygram's partition logic. We consume what polygram ships.
- Comparing more than two encodings. Partition vs raw_slice is the test; other encodings can layer on later.
- Cross-fixture comparison. We measure on the pooled fixture specifically because that's where the data-scale gap was surfaced. Residue (which already shows partition-irrelevant retained_mauc > 100 %) is out of scope.

## Decisions

### Decision 1 — Partition output materializes as a "shadow" SAE checkpoint

The polygram-partitioned encoding doesn't replace the SAE's encoder/decoder weights; it changes which features are kept + how their basis is structured. For the purpose of `sweep_pareto_capability`, we materialize the partition as a NEW safetensors file with:

- `encoder.weight`, `encoder.bias`: copied verbatim from the original SAE.
- `decoder.weight`, `decoder.bias`: copied verbatim from the original SAE.

The "partitioned-ness" lives in the **basis-construction step**: when the sweep slices to `target_n_features_kept`, it slices per-tier proportionally instead of by global row norm. This means a partition-aware basis at n=128 might pick top-50 from "hierarchical", top-30 from "categorical", top-15 from "synthetic", top-5 from "positional" (proportional to tier sizes 40/24/8/3 = 53/32/11/4 of 128, then round to integers).

To make this work with the current `sweep_pareto_capability` (which row-norm-slices), we extend its basis construction to ACCEPT a partition manifest at runtime. The partition manifest comes from the same safetensors file via an optional `partition_block_ids: tensor of int32, shape (n_features,)` key. If present, the sweep slices per-tier. If absent, the sweep falls back to row-norm slicing (current behaviour).

**Trade-off:** This is a minor extension to `sweep_pareto_capability`'s basis construction. Strictly v1 could avoid this by running two sweeps with two separately-prepared checkpoints, but that would require pre-materializing per-K partition slices ahead of time — much more disk + much harder to vary the candidate_widths post-hoc. The runtime-aware basis builder is the cleaner shape.

### Decision 2 — Materialization happens bio-sae-side, not in polygram

The polygram CLI doesn't currently emit a partition-tagged safetensors at per-K granularity that matches sae-forge's `sae_checkpoint` shape. Two paths:

- (a) Add a new polygram CLI emitter (`polygram compress --emit-partition-shadow`) that writes the shadow checkpoint.
- (b) Run a bio-sae-side script that reads the original SAE + the partition spec from `partition_summary.json` + computes the `partition_block_ids` vector, then writes the shadow safetensors.

**Chosen: (b).** The materialization is a derived artifact, not a new polygram primitive. Bio-sae's responsibility (it owns both the SAE and the partition spec for its fixtures). One ~50-line script under `bio-sae/scripts/materialize_partition_checkpoint.py`. If partition wins the validation, (a) becomes worth doing as cleanup; if it loses, (b)'s script is throwaway.

### Decision 3 — Two single-encoding sweeps, NOT a multi-encoding sweep

v1 runs `sweep_pareto_capability_progressive` twice: once with `raw_slice`, once with the partition shadow checkpoint. Each produces its own `progressive_summary.json`. The comparison is done by a small writeup script that reads both summaries.

Alternative: extend the wrapper to take `--encoding LABEL:PATH` (multiple times) like `sweep-pareto` already does for the non-capability sweep. This is the natural eventual API but adds real surface (refactor `sweep_pareto_capability`'s internals to multiplex per-cell across encodings). v1 punts.

### Decision 4 — Comparison criterion is structured: per-cell delta + trajectory variance + convergence flag

The validation summary report carries three comparisons:

1. **Per-cell `retained_mauc` delta.** For each (stage, width) cell where both sweeps ran, `partition_retained_mauc - raw_slice_retained_mauc`. Positive → partition helps at this cell.
2. **Trajectory variance.** `max(argmin_retained_mauc) - min(argmin_retained_mauc)` across stages, for each encoding. Smaller → more data-scale-stable.
3. **Convergence flag.** `recommendation.converged` for each encoding.

Decision tree:
- Both `converged=True` + partition delta ≥ +0.02 at the largest stage → **Partition wins** (Wave C re-evaluated).
- Partition `converged=True` AND raw_slice `converged=False` → **Partition wins** (closes the data-scale gap, even if the per-cell delta is small).
- Both `converged=False` + partition variance < 50 % of raw_slice → **Partition partial win** (reduces drift but doesn't close it).
- Both `converged=False` + partition variance ≥ raw_slice variance → **Partition no-op** (data-scale tax independent of basis structure).
- Partition `converged=False` AND raw_slice `converged=True` → **Partition makes it worse** (unexpected; document as evidence partition has subtle costs).

### Decision 5 — Acceptance gate is the experiment outcome itself

This change ships a measurement experiment, not a feature. The "acceptance gate" is the writeup that names which decision-tree cell landed and what it implies for `add-progressive-finetune`'s priority + Wave C's status. No new pytest gates beyond a sanity-check unit test for the partition-aware basis builder (verifies the per-tier slicing math).

## Risks / Trade-offs

- **Polygram-side artifact dependencies.** Materializing the partition shadow checkpoint requires bio-sae having both `partition_summary.json` (present) AND the original `sae.pt` (present at `runs/uniref50_n5000/pooled_w1024_k64/sae.pt`). If either is missing for a future fixture, the materialization script's contract documents the requirement.
- **Per-tier slicing math may not be uniquely defined.** If the user requests `target_n_features_kept=10` against a 4-tier partition with [24, 40, 3, 8], proportional rounding could yield [3, 5, 0, 1] = 9 features instead of 10. Mitigation: documented rounding rule (largest fractional remainder wins the last slot); deterministic across runs.
- **Partition shadow checkpoint is a duplicate of the original SAE weights.** Adds ~ a few MB to bio-sae's `runs/` directory. Acceptable; `runs/*` is gitignored anyway.
- **The 0.02 retained_mauc delta threshold is somewhat arbitrary.** It's calibrated against the noise floor (~0.005 at n=5000 proteins) + a 4x margin. If partition shows a delta of +0.015 at the largest stage, that's "partial win" not "win." Mitigation: the writeup names the specific numbers and lets future readers decide.
- **Negative result is still informative.** If partition loses, the result strengthens the case for `add-progressive-finetune` and weakens Wave C's potential return. Either direction is publishable in the bio-sae writeup; no risk of unfalsifiable "experiment didn't tell us anything."
