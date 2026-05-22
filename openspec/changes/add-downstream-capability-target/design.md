## Context

`add-llama-family-rope` (archived 2026-05-19) shipped the structural fidelity fix that made forging Llama / Gemma-2 / Qwen produce sensible residual streams. `add-host-wrapped-forge-fallback` (in progress) addresses the under-complete-basis regime where `native_in_basis` mode fails. `add-polygram-cluster-diagnostics` (shipped) surfaced polygram's per-feature clustering structure on `ParetoFrontierRow`.

All of these landed against a faithfulness metric — KL on logits for LM hosts, cosine on encoder states for whisper / esm2 hosts. The metric measures how close the forged hidden states are to the host's hidden states.

Bio-sae's 2026-05-22 capability-bottleneck investigation (cross-referenced in proposal.md) demonstrated that this metric systematically misranks forges for downstream-task users. Two specific failure modes:

1. **Concentrated substrate** (categorical AA features clustered at host AUC=1.0). Cosine prefers wider bases; capability AUC prefers the smallest possible basis. Cosine recommends n=256 (0.095); capability recommends n=16 (retained mAUC 103 %). The right answer differs by 16×.
2. **Spread substrate** (hierarchical biology spread across AUC [0.6, 0.95]). Cosine improves monotonically with width; capability has an inverted-U with peak at n=512 (1.6× over-complete). Wider is worse past the peak.

The cosine-vs-capability divergence is not a bug in cosine — it's that cosine answers the residual-stream-fidelity question, not the downstream-task question. Different users have different questions. We need both metrics, plus a way for sweeps to use the right one per use case.

## Goals / Non-Goals

**Goals:**
- A new `FaithfulnessTarget` that scores per-feature × per-label AUC through a downstream task encoder, decoded from the basis through the encoder's host-coord input space.
- A wrapper sweep (`sweep_pareto_capability`) that surfaces retained-AUC fields on `ParetoFrontierRow` so capability rows are comparable across encodings × widths × scale_boosts.
- A `CapabilityDataset` abstraction that decouples the dataset surface (sequences, labels, encoder, tokenizer, aggregator) from the sweep machinery. Bio-sae / sm-sae / econ-sae each provide their own constructor.
- Falsifiable acceptance gate via bio-sae's two pre-existing measurements.
- Byte-identical behaviour when neither the new target nor the new sweep is invoked. New `ParetoFrontierRow` fields default `None`.

**Non-Goals:**
- A supervised forge (using labels at projection time). Out of scope; a separate proposal.
- A universal "best forge" heuristic. Each dataset's optimal is data-dependent; the sweep makes the trade-off legible, not invariant.
- URI-based dataset registry (`bio-sae://uniref50_n5000_pooled`). v1 ships explicit constructors only.
- ESMFold / structure-prediction capability targets. The proposal sketches feature-recovery AUC as the v1 metric; structure prediction (TM-score, LDDT) is a v2 extension once ESM-2 t36-scale forge validation lands.
- Closing the structural ~9 % mAUC gap on spread substrates. The gap is fundamentally about layer-norm non-commutation + TopK rank-shuffling in the encoder; remediation is its own research problem.

## Decisions

### Decision 1 — Target reads ctx for inputs, holds encoder + labels at construction

The target's `score(*, forged, host, ctx)` method receives forged + host as protocol arguments and pre-tokenised inputs via `ctx["_eval_input_ids"]`. The encoder and labels are NOT in ctx — they're set on the target instance at construction time.

**Why this split:**

- ctx is shared across all targets (the FSM populates it once per forge run). Polluting it with encoder + labels makes ctx-based dispatch brittle and creates an implicit ordering dependency.
- Encoder + labels are dataset-specific. `DownstreamCapabilityTarget(encoder=X, labels=Y)` is the natural constructor — same shape as `GroundTruthTarget(labels=Y, ...)` already ships.
- Constructor-time validation (labels.ndim==2, encoder callable, aggregator in supported set) fails fast at sweep config time, not inside the FSM mid-run.

The cost: each `ForgePipeline.run(...)` call constructs one `DownstreamCapabilityTarget` per dataset, parameterised by `(encoder, labels, aggregator, min_prevalence, decode_via_basis)`. That's one instance, not per-row; cheap.

### Decision 2 — Decode via the forged module's `basis_encode` buffer when present, else explicit `basis.W_dec`

The forged module for `esm2` and `whisper_encoder` adapters emits a `basis_encode` buffer carrying `pinv(W_dec) * scale_boost` (shape `(d_model, n_features)`). The downstream-capability path needs the *inverse* direction — basis-coord forged hidden states → host-coord activations the encoder can read. That's `forged_h @ W_dec`, shape `(d_model,)`.

The forged module DOES NOT hold `W_dec` directly today; only `basis_encode = pinv(W_dec) * scale_boost`. So the target needs to either:

(a) Pass `basis.W_dec` explicitly via ctx (additional ctx surface).
(b) Hold `basis` on the target at construction time (couples target to basis lifecycle).
(c) Recover `W_dec` from `basis_encode` via numerical pseudoinverse at score time (extra cost per row).

**Decision: (c) with a fallback to (b).** The target lazy-computes `W_dec` from `basis_encode` on first use (pinv is O(d_model × n_features), one-time per forge), caches the result. Users who already have basis in scope can pass it directly via a `basis=...` ctx field to skip the pinv; missing-basis is the common case (multi-target FSM runs).

The pinv-recovery is *only valid for orthonormal-row bases*. For non-orthonormal W_dec (the common case under real polygram compression), the recovered matrix is approximate. That's acceptable for the v1 target — the systematic error is folded into the same "fundamental forge tax" the metric is measuring. Users who need exact `W_dec` can pass the basis explicitly.

A future enhancement: `ForgeResult` carries `basis` already; piping it into `ctx["basis"]` at `_score_faithfulness_imperative` time gives every target free access. That's a separate one-line change tracked as task 0.1.

### Decision 3 — Aggregator dispatch is "pool_then_encode" by default

Two pool orders:

1. **`pool_then_encode`** (default): activations averaged across residues → encoder → score. Matches what the existing `GroundTruthTarget` does (pools the residual, then AUC). Bio-sae's `forge_capability_eval.py` default. The "smoothed" measurement.

2. **`encode_then_pool`**: encoder applied per residue → latents averaged across residues → score. Sharper measurement that exposes per-residue forge degradation; bio-sae's data shows the gap *grows* under this aggregator (host signal gets sharper, forge can't follow).

The default is `pool_then_encode` because:

- It matches the existing `GroundTruthTarget` behaviour and the README's published cov95 numbers downstream.
- It's the conservative pick for "does the forge preserve biology" — answer "no" under pool_then_encode is a stronger claim than "no" under encode_then_pool.
- Encode-then-pool is more expensive (encoder applied L times per protein instead of once) — opt-in for users who want the sharper measurement.

Users can opt into encode-then-pool for diagnostic deep-dives (bio-sae's `forge_pool_after_encode.py` already does this manually); the target exposes both behind a single string flag.

### Decision 4 — Reuse `ParetoFrontierRow`; add fields as optional

The existing `ParetoFrontierRow` schema (v0.7) carries 30+ fields covering encoding, target_n_features_kept, KL, basis sizes, polygram diagnostics, error metadata, etc. Capability is the same Pareto axis as KL — same row carries both. Adding new optional fields keeps:

- One JSONL file per sweep (no separate `capability_frontier.jsonl`).
- Existing `from_dict` round-trip compatibility (unknown fields silently ignored already).
- Tooling continuity (visualisers, `sae-forge recommend`, ad-hoc `jq` filters all just see extra columns).

Trade-off: the row gets wider. With ~15 capability fields the row becomes hard to eyeball in a terminal. The `recommend` subcommand picks the relevant subset; CLI table output gets a `--show capability` / `--show kl` flag that selects which column group to render.

### Decision 5 — `CapabilityDataset.from_bio_sae` lives in sae-forge, not bio-sae

The constructor parses bio-sae's bundle / sequences / SAE format directly. Implementing it inside sae-forge keeps sae-forge self-contained and lets the falsifiable acceptance gate run in sae-forge's own CI (no bio-sae checkout needed beyond a tiny test fixture).

Bio-sae imports `CapabilityDataset.from_bio_sae` back from sae-forge in its own `scripts/forge_capability_eval.py` (which becomes a thin wrapper post-merge per task 7.3). sm-sae / econ-sae do the analogous `from_sm_sae` / `from_econ_sae` constructors in their own repos — the contract is documented; implementations live close to the fixture format.

The alternative (every fixture repo registers a constructor in sae-forge) creates a one-way import dependency from sae-forge into every fixture repo, which is the wrong direction. Single constructor in sae-forge for the one fixture format that lives in bio-sae is the pragmatic compromise.

### Decision 6 — Falsifiable gate uses bio-sae's pre-existing measurements

The proposal pre-commits to reproducing two specific predictions:

- `runs/uniref50_small/residue` → optimal n=16, retained_mauc ≥ 1.00.
- `runs/uniref50_n5000/pooled_w1024_k64` → optimal n=512, retained_mauc ≈ 0.93 ± 0.01.

These are measured (`bio-sae/runs/forge/capability_eval_smoke/`, `bio-sae/runs/forge/capability_pooled_n500*/`) and the prediction is tight (1 mAUC point). If the new sweep produces different optimal widths, that's a falsified implementation, not a "drift" — bio-sae's manual scripts and the new sweep should be measuring the same quantity.

The 1 mAUC tolerance absorbs: (a) random variation in the protein subset (the manual scripts used a 10-protein eval for residue, 500-protein for pooled; the sweep may use different defaults), (b) floating-point drift across BLAS / numpy / torch versions, (c) any slightly-different aggregator semantics. Drift > 1 mAUC points to a real bug in the new sweep, not noise.

## Risks / Trade-offs

- **The `pinv(basis_encode)` recovery for `W_dec`** (Decision 2(c)) is approximate for non-orthonormal bases. Common case is *fine* (the error contributes to the same fundamental forge tax the metric measures), but degenerate bases (rank < n_features, columns near zero) could produce a `W_dec` that misrepresents the forge. Mitigation: log a warning when `np.linalg.matrix_rank(basis_encode) < n_features`; recommend passing `ctx["basis"]` explicitly in that case.
- **Encoder-on-CPU cost** at sweep scale. The capability sweep on bio-sae's n=5000 pooled SAE × 6 widths × 2 scale_boosts × 500 proteins took ~5 minutes on CPU per sweep. Encode-then-pool is ~5× slower. For sm-sae / econ-sae-scale fixtures (much smaller activations) this is fine; for ESM-2 t36 / Gemma-2-2B-scale fixtures, the sweep is GPU territory.
- **Singletons in unfiltered Y matrices** inflate cov95 trivially. The `min_prevalence` flag matches bio-sae's `--min-n-pos` convention. Default 0 (no filter) reproduces the README's headline numbers; users who want "robust biology" set 10. The sweep documentation should make this explicit.
- **Per-feature AUC is symmetric** (`max(auc, 1-auc)`). Features whose latents *anti-correlate* with labels score the same as positively-correlated ones. Bio-sae and `GroundTruthTarget` both use this convention; we follow.
