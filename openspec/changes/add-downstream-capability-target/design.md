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

### Decision 2 — Emit a `basis_decode` buffer on encoder-only forged modules; fall back through three paths

**Revised after PR #75 review.** The forged module for `esm2` and `whisper_encoder` adapters today emits a `basis_encode` buffer carrying `pinv(W_dec) * scale_boost` (shape `(d_model, n_features)`). The downstream-capability path needs the *inverse* direction — basis-coord forged hidden states → host-coord activations the encoder can read. That's `forged_h @ W_dec`, shape `(d_model,)`.

The reviewer flagged that recovering `W_dec` by inverting `pinv(basis_encode)` is roundabout: the adapter already computed `pinv(W_dec)` from `W_dec` at forge time, so the matrix is *already on disk* in the basis the adapter walked. We just don't emit it.

**Decision: emit a `basis_decode` buffer on encoder-only adapters, alongside `basis_encode`.** Forged modules whose adapters produce a `basis_encode` buffer (`esm2`, `whisper_encoder`, future encoder-only families) SHALL also produce a `basis_decode` buffer carrying `W_dec` directly (shape `(n_features, d_model)`). The walk emits both; the native module registers both as non-parameter buffers; `state_dict()` round-trips both.

The target then resolves `W_dec` via this precedence:

(a) `ctx["basis"]` — explicit FeatureBasis passed by the pipeline (cheapest, exact, available when ForgeResult plumbing supplies it — see follow-up below).
(b) `forged_module.basis_decode` — emitted by the adapter at walk time, exact, no pinv. **Default path** for capability sweeps over encoder-only families.
(c) `pinv(forged_module.basis_encode)` — lazy fallback for forged modules lacking `basis_decode` (third-party adapters that don't follow the bundled convention; pre-rollout forge artifacts that pre-date this change). Cached per `id(forged_module)`; one-time cost; emits a `UserWarning` recommending the buffer be added on the adapter side.

The fallback (c) is *only valid for orthonormal-row bases* — for non-orthonormal W_dec the recovered matrix is approximate. With (b) as the default path, this approximation never fires for the bundled adapters and the systematic error is removed entirely from the v1 implementation.

A future enhancement: `ForgeResult` carries `basis` already; piping it into `ctx["basis"]` at `_score_faithfulness_imperative` time gives every target free access via path (a). That's a separate one-line change tracked as task 0.2.

### Decision 3 — Aggregator dispatch is "pool_then_encode" by default; the contract is per-protein-vector

**Aggregator contract.** An aggregator is a `Callable[[Tensor], Tensor]` that takes a per-residue tensor of shape `(L, latent_or_d_model)` (single protein, CLS / EOS already stripped) and returns a per-protein tensor of shape `(latent_or_d_model,)` (collapsed across the sequence axis). The two built-in strings parameterise this contract:

- **`"pool_then_encode"`** — operates on `h_d: (L, d_model)`. Returns `encoder(h_d.mean(0, keepdim=True))[0]` — shape `(latent_width,)`.
- **`"encode_then_pool"`** — operates on `h_d: (L, d_model)`. Returns `encoder(h_d).mean(0)` — shape `(latent_width,)`.

A user-supplied callable receives `h_d` (host-coord activations after the decode step) plus the encoder via closure, and is responsible for the encode + reduce composition. The return SHALL be a 1-D tensor with `latent_width` elements; the target stacks these across proteins into a 2-D `(N_items, latent_width)` matrix and applies the chunked Mann-Whitney AUC across all label columns simultaneously.

**Multi-token semantics.** All built-in aggregators operate on the per-residue / per-token axis (axis 0 of `h_d` after CLS / EOS stripping). There is no "per-position" output — the target's labels live at the protein scope, not the residue scope. Users with residue-scope labels SHOULD set up a separate eval where the per-residue latents (not the pooled latents) are concatenated across all proteins; that's the residue-feed pattern bio-sae's `forge_capability_eval.py --feed residue` already exercises, and it'd flow through this target with a no-op aggregator (`lambda h_d: h_d.reshape(-1)` to flatten) only on a per-protein call.

**Multi-label semantics.** Labels are a `(N_items, V)` binary matrix where `V` is the number of distinct label columns. AUC is computed per latent × per label (`(V, latent_width)` matrix of AUCs), then `max_over_latents` per label (`(V,)` array of best-discriminator AUCs), then `mean_over_labels` to produce the scalar score. This matches `GroundTruthTarget`'s convention and `biosae.sae.evaluation.score_against_ground_truth`. Labels are NOT aggregated; each label column gets its own AUC.

**Why `pool_then_encode` is the default.**

Justification for the default:

- It matches the existing `GroundTruthTarget` behaviour and the README's published cov95 numbers downstream.
- It's the conservative pick for "does the forge preserve biology" — answer "no" under pool_then_encode is a stronger claim than "no" under encode_then_pool (bio-sae's data showed encode-then-pool *widens* the gap because host signal gets sharper).
- Encode-then-pool is ~L× more expensive (encoder applied L times per protein instead of once) — opt-in for users who want the sharper measurement.

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

### Decision 7 — Performance budget; recommended practices documented, not enforced

**Added after PR #75 review.** Capability sweeps are more expensive than KL / cosine sweeps because each (encoding, width, scale_boost) cell runs the downstream encoder on top of the host's forward pass. Bio-sae's empirical timing on the n=5000 pooled SAE:

| operation | cost per protein | cost @ 500 proteins × 6 widths × 2 scale_boosts |
|---|---|---|
| Host extraction (once per sweep) | ~0.05 s | ~25 s |
| Forge module construction | one-time, ~0.5 s | ~6 s |
| Forge extraction (per cell) | ~0.05 s | ~5 min |
| AUC scoring (per cell) | trivial (matmul) | ~1 s |
| **Total CPU wall** | — | **~5-6 min** |

Encode-then-pool aggregator is ~L× slower at the extract step (encoder applied per residue instead of per protein). On the same fixture that's ~25 min instead of ~5.

**Recommended practices (target's docstring + sweep CLI `--help`):**

1. **Start with a subset.** Capability eval is statistically meaningful at n_proteins ≥ 200 on the pooled feed if `min_prevalence` is set to keep ≥ ~30 informative labels. Sweep over 200 first; scale up to 1000+ for the final recommended config.
2. **Cache host extraction.** The host's per-protein activations / pooled latents are invariant across the sweep cells (one host, one tokenizer, one corpus). Cache them in `output_dir/host_activations.safetensors` on the first call and reuse across cells. Skipping this re-host-extracts on every cell — bio-sae's `forge_capability_eval.py` deliberately doesn't cache today; the sweep wrapper SHOULD.
3. **Run sweeps at `float16` / `bfloat16`.** ESM-2-shape forges have no numerical-sensitivity in the AUC scoring (rank-based). The capability metric is invariant to small precision drift; halving the activation precision halves the forward cost.
4. **Restrict the encoder eval.** Users with a wide SAE (≥ 4 K features) can pass `encoder=lambda x: full_sae(x)[1][:, :2048]` to score only the first 2 K latents — bio-sae's headline biology lives in the top-K-by-norm subset anyway.

These are documented as guidance in the target's docstring + CLI `--help`. The implementation SHALL NOT enforce any of them (callers may want full precision / full-vocab); the cache (item 2) is the one optimisation that ships *enabled by default* with an opt-out flag.

### Decision 8 — Field naming carries `retained_*` semantics, not a `downstream_` prefix

**Added after PR #75 review.** The reviewer flagged that field names like `retained_mauc` are bio-sae-shaped and might benefit from a `downstream_*` prefix for sm-sae / econ-sae adoption.

**Decision: keep `retained_*` as the public field names; the `downstream` qualifier lives on `DownstreamCapabilityTarget.name = "downstream_capability"`.** Rationale:

- The "retained" semantics ("vs host's baseline measurement") generalise to every domain — sm-sae's particle features, econ-sae's tier categories, future audio probes. The math is dataset-agnostic.
- The `downstream_*` prefix would imply a parallel set of metrics for non-downstream targets, which doesn't exist — every capability metric IS a downstream one by construction.
- The `target_name` field on each row carries `"downstream_capability"` (the target's `name` attribute) — that's where the "downstream" disambiguation lives. Filtering by target_name on the frontier is the cross-target query.
- Shorter field names render better in tabular CLI output and `jq` filters.

For sm-sae / econ-sae adoption, the contract is: each fixture repo's `CapabilityDataset.from_<repo>(...)` constructor sets up the target identically; the same `retained_mauc` / `retained_cov95` fields appear on each row, qualified by `target_name` and the dataset's `sequences_path` / `labels_path` metadata. No domain-specific aliases.

## Risks / Trade-offs

- **The `pinv(basis_encode)` recovery for `W_dec`** (Decision 2(c)) is approximate for non-orthonormal bases. Common case is *fine* (the error contributes to the same fundamental forge tax the metric measures), but degenerate bases (rank < n_features, columns near zero) could produce a `W_dec` that misrepresents the forge. Mitigation: log a warning when `np.linalg.matrix_rank(basis_encode) < n_features`; recommend passing `ctx["basis"]` explicitly in that case.
- **Encoder-on-CPU cost** at sweep scale. The capability sweep on bio-sae's n=5000 pooled SAE × 6 widths × 2 scale_boosts × 500 proteins took ~5 minutes on CPU per sweep. Encode-then-pool is ~5× slower. For sm-sae / econ-sae-scale fixtures (much smaller activations) this is fine; for ESM-2 t36 / Gemma-2-2B-scale fixtures, the sweep is GPU territory.
- **Singletons in unfiltered Y matrices** inflate cov95 trivially. The `min_prevalence` flag matches bio-sae's `--min-n-pos` convention. Default 0 (no filter) reproduces the README's headline numbers; users who want "robust biology" set 10. The sweep documentation should make this explicit.
- **Per-feature AUC is symmetric** (`max(auc, 1-auc)`). Features whose latents *anti-correlate* with labels score the same as positively-correlated ones. Bio-sae and `GroundTruthTarget` both use this convention; we follow.
