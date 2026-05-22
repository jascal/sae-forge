## Context

`sweep_pareto_capability` (PR #76) accepts a single `sae_checkpoint` and treats `--encodings` as informational labels. The partition-aware basis builder (PR #89) added a per-encoding basis-construction path triggered by `partition_block_ids` in the SAE state dict — but still operates on one checkpoint per call.

`sweep_pareto` (the older non-capability sweep, PR ~#65) already supports `--encoding LABEL:PATH` repeatable; multiple encodings are first-class there. The capability sweep is the odd one out. This openspec brings capability-sweep up to parity with sweep-pareto's multi-encoding contract and extends the contract through the progressive wrapper.

The motivating empirical signal: the n=5000 progressive run (2026-05-22) surfaced a data-scale-widening retained_mauc gap on bio-sae's pooled fixture. The partition validation (PR #88) is testing whether ONE alternative basis structure (decoder-norm-quantile partition) closes the gap; this openspec lets us systematically compare SEVERAL — polygram's MPS encoders at various bond dimensions, partition variants, raw_slice — without spawning separate sweep invocations and aligning their frontiers by hand.

The chat proposal (2026-05-22) that motivated this openspec asked for a `MixedChiEnsemble` framework with model-state warm-start across stages. The framework doesn't fit the closed-form forge (no model state to ensemble); the *substantive question* it asks (does encoding choice matter, especially across bond dimensions?) is testable directly under the existing progressive wrapper if we widen its encoding axis. This openspec ships exactly that widening.

## Goals / Non-Goals

**Goals:**
- `sweep_pareto_capability(encodings=[(label, path), ...])` API accepting multiple encodings.
- Progressive wrapper passes them through; per-encoding plateau identification + convergence detection.
- `ProgressiveRecommendation` gains a `per_encoding_recommendations` field for multi-encoding sweeps; top-level recommendation picks the best across encodings.
- CLI repeatable `--encoding LABEL:PATH` flag matching sweep-pareto's existing surface.
- `sae-forge recommend` over multi-encoding frontiers picks the smallest-n-meeting-predicate across encodings.
- Falsifiable gate on bio-sae pooled at [1000, 5000] comparing 3-5 encodings.

**Non-Goals:**
- Polygram-side machinery to materialize encoded shadow checkpoints. We consume what polygram + the partition materialization script (#88-89) emit.
- Optimal per-encoding `scale_boost` calibration. v1 uses a fixed scale_boost list for all encodings; encoding-specific calibration is a separate openspec.
- Cross-encoding ensembling at inference. The proposal's `MixedChiEnsemble.sample()` is out of scope; we return recommendations, not ensembled models.
- Cross-fixture comparison (compare encodings on residue + pooled in one call). Each `sweep_pareto_capability` invocation is one fixture; users run multiple invocations for multi-fixture comparisons.

## Decisions

### Decision 1 — `encodings` is a list of `(label, path)` tuples, NOT a list of pre-loaded `FeatureBasis` objects

The capability sweep needs to LOAD each encoding's SAE state dict to extract W_dec + optional `partition_block_ids`. Accepting pre-loaded objects would force callers to do the loading themselves, exposing internals. The `(label, path)` shape is the same as `sweep-pareto`'s existing `--encoding LABEL:PATH` flag — consistency with the older sweep's contract is worth a lot for users.

The single-encoding `sae_checkpoint=PATH` keyword is retained as sugar — it becomes `encodings=[("raw_slice", sae_checkpoint)]` internally. Back-compat: every v0.8.x / v0.9.x call site continues working.

### Decision 2 — Host cache is SHARED across encodings; forge cache (deferred) is per-encoding

Host activations are encoding-independent — they're the residual stream of the host model on the input sequences, which doesn't change based on which encoding we're forging against. The host-extraction cache (PR #77) is therefore correctly SHARED across encodings: the cache key doesn't include encoding identity, and stage K+1's host cache from a multi-encoding sweep is read by every encoding's first cell. This is a free win — N encodings cost the same host-extraction work as 1.

The forge-activations cache (deferred to `add-forged-activations-cache` per the warm-start counter-shape, PR #86) WILL be per-encoding because it caches the output of running the forged module, which differs per encoding. This openspec doesn't ship the forge-activations cache; it leaves the door open for the deferred openspec to layer on later. **When both ship: multi-encoding cost approaches `host_cost + K × per_encoding_forge_cost`**, not `K × (host_cost + per_encoding_forge_cost)`.

### Decision 3 — Per-encoding plateau identification, NOT cross-encoding

Each encoding gets its OWN plateau computation per stage. Reason: encodings may have different overall retained_mauc levels; a global plateau across encodings would conflate encoding choice with width choice.

Per-encoding plateau:
- For encoding E at stage K, plateau = widths within `plateau_tolerance` of E's peak retained_mauc at stage K.
- Per-encoding convergence: argmin-of-E's-plateau stable across stages with retained_mauc variance within tolerance.

Cross-encoding "winner" pick: among encodings whose recommendation converged, pick the one with the smallest stable n meeting the threshold. Ties broken by lowest plateau-argmin retained_mauc variance across stages.

### Decision 4 — `ProgressiveRecommendation.per_encoding_recommendations` is optional

A single-encoding sweep produces a `ProgressiveRecommendation` with `per_encoding_recommendations = None` (or absent from the dataclass). A multi-encoding sweep populates it with the per-encoding map.

The top-level `target_n_features_kept` / `retained_mauc_vs_host` / `converged` fields on `ProgressiveRecommendation`:
- Single-encoding: as-is today.
- Multi-encoding: belong to the WINNING encoding (per Decision 3's tiebreaker). The `rationale` string names which encoding won and why.

This is a strict superset of the v0.9.x `ProgressiveRecommendation` shape — back-compat preserved.

### Decision 5 — CLI `--encoding LABEL:PATH` repeatable mirrors `sweep-pareto`'s syntax

`sae-forge sweep-capability --encoding raw_slice:p1 --encoding partition:p2` works identically to how `sae-forge sweep-pareto --encoding mps:p1 --encoding rung4:p2` already works. Users moving from one sweep to the other don't relearn the flag.

Backward-compat with the existing `sweep-capability --dataset-config YAML` (where `encoder_checkpoint` lives in the YAML): if neither `--encoding` nor `--dataset-config`'s `encoder_checkpoint` provides multiple encodings, the call is single-encoding. If both provide encodings (YAML AND `--encoding`), `--encoding` wins (explicit beats implicit); CLI emits a warning.

### Decision 6 — `χ` is treated as an encoding-choice axis, NOT a data-scale axis

The 2026-05-22 mixed-χ chat proposal framed bond dimension as a data-scale-coupled hyperparameter (different protein counts → different optimal χ). **This openspec deliberately rejects that coupling.** Bond dimension is a property of the *encoding* (how compressed the basis representation is — a polygram-side choice); the data-scale axis is about AUC-estimator variance (how much eval data we run against). These are orthogonal. An MPSRung1 encoder at χ=16 is one *encoding option* the user can compare against raw_slice, partition_q4, and Rung5 at any given data scale — the progressive wrapper's stage ladder handles the data-scale axis independently. Treating χ correctly as an encoding-choice axis is what makes the original proposal's substantive question testable without an ensemble framework.

### Decision 7 — `recommend` over multi-encoding frontiers picks across encodings

`sae-forge recommend --target retained-mauc>=0.95` against a multi-encoding `frontier.jsonl`:

1. Filter rows by predicate(s) as today.
2. Among survivors, sort by `(target_n_features_kept ASC, encoding_label_index ASC)` — smallest n wins; if multiple encodings tie on n, the one that appeared first in `--encoding` flag order wins.
3. Output names both the encoding and the width.

Rationale: the user's CLI invocation already encodes their priority order via the flag order. We trust it.

## Risks / Trade-offs

- **N-fold compute cost for K encodings.** Multi-encoding sweep with K encodings × M widths × S stages runs K × M × S cells. For K=5 encodings on bio-sae's pooled fixture at [1000, 5000], that's ~80 cells, ~4 hours on CPU. The forge-activations cache (deferred) would cut this; v1 users with K > 3 should expect long runtimes.
- **Encoding-identity mismatch in cache.** The host-extraction cache key currently doesn't include encoding identity — correct for host-cache (host activations are encoding-independent). If a future change adds an encoding-coupled cache, this needs to be revisited.
- **`ProgressiveRecommendation.per_encoding_recommendations` doubles the on-disk `progressive_summary.json` size.** For K=5 encodings × 4 stages × ~50 cells each, ~5KB → ~25KB. Negligible.
- **Outcome ambiguity.** If two encodings tie at the smallest stable n with identical retained_mauc, the tiebreaker (lowest variance, then flag order) may not be the "right" choice for the user's downstream application. Mitigation: the JSON manifest carries all per-encoding recommendations; the user can override the top-level pick.
- **`scale_boost` interaction with encoding.** The current scale_boost calibration (`scale_boost=auto`) was tuned for raw_slice. Different encodings may need different scale_boost ranges. v1 uses a uniform scale_boost list across encodings; per-encoding scale_boost is a separate openspec.
- **CLI flag-order semantics.** If the user runs `--encoding raw_slice:p1 --encoding partition:p2` and partition wins, but they expected raw_slice to win, the result is confusing. Mitigation: `recommend` output ALWAYS prints the per-encoding ranking, so the user sees WHY the winner was picked.
