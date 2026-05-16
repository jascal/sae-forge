## Why

Today, one layer's SAE feature dictionary is compressed by exactly one polygram encoding — the `--encoding-class` choice picks `MPSRung1` (cap=8) or `Rung3` (cap=16) or `Rung4` (cap=32) or `HEA_Rung2(n_qubits=N)` (cap=2^N) for the *whole* dictionary, and every surviving feature is allocated the same parameter budget.

Real SAEs aren't uniform. A heavy-hitting feature with broad decoder support and a high `n_fires_total` plausibly needs more substrate than a feature that fires once per ten thousand tokens on a single neighbourhood. The current single-encoding regime forces the analyst to pick the *worst-case* budget for the *best-case* feature: choosing `Rung4` over `MPSRung1` because a handful of features need the capacity inflates the parameter cost for the entire dictionary, including the long tail where it isn't needed. Choosing `MPSRung1` puts the heavy features in a bucket too small to reconstruct cleanly. Neither choice is right for both subsets simultaneously.

The Axis-4 sweeps `add-auto-materialise-sweep` ships are the right venue to ask whether this matters. The `quality_tier="degenerate"` rows from `add-forge-quality-diagnostics` already flag that at low K, the basis rank doesn't span the host's residual stream. One reading of "degenerate at K=4" is "K is too small." Another is "we ran out of substrate for the surviving features." The two are confounded under uniform encoding. Partitioning the dictionary so heavy features get `Rung4` while the tail keeps `MPSRung1` lets the analyst distinguish them.

This is the prerequisite for any future per-feature substrate policy. Without it, "heterogeneous capacity" lives only in proposal slides; with it, the analyst has a concrete frontier-row knob to flip.

## What Changes

### CLI: `--encoding-partition PATH`

New `sweep-pareto` flag accepting a JSON manifest path. Mutually exclusive with the existing single `--encoding-class LABEL:CLASS` path (the two modes name different things about the same encoding decision). When supplied, the named encoding for the label is ignored; the partition manifest drives per-block encoding choice.

Manifest schema (one entry per block):

```json
{
  "label": "mps_layer8",
  "partition": [
    {"block_id": "heavy", "encoding_class": "Rung4", "feature_ids": [3, 17, 22, ...]},
    {"block_id": "tail",  "encoding_class": "MPSRung1", "feature_ids": [0, 1, 2, 4, ...]}
  ]
}
```

Feature-id sets SHALL be disjoint and SHALL cover every id in the SAE checkpoint. The cache key includes the manifest's SHA-256 so flipping partitions invalidates `_materialised/` deterministically.

### `ParetoFrontierRow` gains `partition_label: str | None`

Provenance only. Populated under `--encoding-partition` with the human-readable label (e.g. `"heavy:Rung4+tail:MPSRung1"`); default `None` for single-encoding rows. Frontier-jsonl consumers that don't ask for it see `null`; existing readers unaffected.

### `auto_materialise.py`: partition-aware materialisation chain

`_run_materialisation_chain` SHALL pass the partition manifest through to polygram's `from_sae_lens(..., encoding_partition=...)` when supplied. The partition is also threaded into `compute_cache_key` so a partition flip reports `MISS (encoding_partition)` on the next run rather than silently re-using a checkpoint built under a different partition.

### Polygram-side capability (separate proposal — `add-encoding-partition`, polygram repo)

This sae-forge change is **gated on a polygram capability** the polygram repo must ship first:

- `CompressionConfig.encoding_partition: list[BlockSpec] | None` — `None` keeps current behaviour byte-identical.
- `BlockStructuredDictionary` — container with the same `.features` contract `BehaviouralValidator` already requires. Concatenates per-block sub-dictionaries.
- Per-block Compressor merge pass; cross-block pair-scoring runs once over the union (the load-bearing claim — partitioning doesn't fragment the cancellation signal).
- `polygram.partition_by_firing_geometry(report, *, n_blocks, encoding_assignment, seed)` — deterministic firing-percentile partitioner. Heuristic helper, not load-bearing.

sae-forge plumbing is small (one flag, one row field, one cache-key entry); the algorithmic surface lives in polygram.

### Out of scope, deliberately

- **Dynamic re-partitioning during forge fine-tune.** The partition is decided pre-materialisation and frozen. A controller that adjusts blocks mid-run is its own proposal — closer in shape to `adaptive-regrow` than to this one.
- **Cross-layer composition.** That's `hybrid-bridge-forge`. This proposal stays within one layer's dictionary.
- **A new Rung.** The existing four encodings cover the comparison this proposal needs to ask.
- **Auto-discovering the partition from the SAE alone.** The firing-geometry helper is a heuristic starting point, not a learned partitioner. Learned partitioning is a research project; this proposal is the surface that lets it be evaluated against a baseline if anyone writes one.
- **Per-feature (singleton block) partitioning.** Conceptually the limit of this proposal, operationally pointless at sae-forge K-budgets; the polygram side would need to be much smarter about constant-cost overhead per block. Out of scope here.

## Falsifiable acceptance gate

A block-structured run MUST Pareto-dominate the corresponding single-encoding run on **at least one K** in an Axis-4 sweep on real GPT-2 layer 8 — strictly lower forge KL at strictly equal-or-fewer kept features. If every K is matched-or-beaten by the single-encoding baseline, the proposal is killed. No retreat to "promising on a metric we didn't pre-register" and no claim of "well, the rank-aware tier moved" — the metric is forge KL at fixed K.

Required cells (Intel, the cross-arch defaults-validation surface):
- GPT-2 layer 8 SAE, `single:Rung4` vs `block(heavy:Rung4 + tail:MPSRung1)` at K ∈ {25, 50, 100, 211}, same validation prompts, same `--rep-selection`.

Required follow-up (M4):
- Gemma-2-2B layer 12 SAE, same comparison. Skipped if Intel kills the proposal.

## Capabilities

### Modified Capabilities

- `pareto-sweep`: `ParetoFrontierRow` gains the `partition_label` provenance field. `sweep-pareto` CLI gains `--encoding-partition PATH`. Existing rows / invocations byte-identical when the new flag isn't supplied.

### Added Capabilities

- `block-structured-materialisation`: the partition-aware auto-materialise path. Cache-keyed on the manifest's SHA-256; refuses overlapping or incomplete feature-id sets at materialise time, not at forge time.

## Impact

- **Modified**: `saeforge/auto_materialise.py` (partition kwarg through `materialise` and `_run_materialisation_chain`; cache key includes partition SHA-256); `saeforge/sweep.py` (`ParetoFrontierRow.partition_label` field; propagated through `_process_row`); `saeforge/forge.py` (`ForgePipeline.sweep_pareto` pass-through); `saeforge/cli.py` (`--encoding-partition` flag, mutually exclusive with single `--encoding-class`).
- **New module**: `saeforge/partition.py` — manifest parser, schema validation (disjoint + complete coverage), SHA-256 helper. Numpy-only.
- **No breaking changes**: row schema extension is forward-compatible (existing readers see `null`); no behaviour change unless `--encoding-partition` is set.
- **Dependencies**: polygram `>=0.7.0` once the `add-encoding-partition` polygram-side capability ships. Until then, this sae-forge proposal is **blocked** at the implementation tasks step. The proposal itself is a design lock-in so both sides can be written against the same contract.
- **Risk note**: the polygram-side cross-block cancellation claim is the load-bearing assumption. If `Compressor`'s pair-scoring pass turns out to need per-block-aware similarity metrics (rather than running on the union with one threshold), the polygram-side proposal needs to surface that; this sae-forge proposal does not pre-commit to either path.
