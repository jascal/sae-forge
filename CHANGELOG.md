# Changelog

All notable changes to sae-forge are tracked here. v0 entries land as
their corresponding OpenSpec change is archived.

## [Unreleased]

## [0.14.0] — 2026-06-06

**Writer-output `U_C` release (`two-basis-uc-writer-output`).** Redefines the
two-basis composition subspace from the aggregate reader-layer geometry to the
circuit WRITER heads' OV-output row spaces — the directions a downstream circuit
actually reads. On an alive single-layer GPT-2 forge this removes the
induction-predictable KL excess (−111%) where reader-geometry does not (−6%),
and the label-free attribution subspace (`∂loss/∂residual`) is ~orthogonal to it
(overlap 0.05) and fails (+14% worse): *loss-sensitivity ≠ circuit-mechanism*.
New `saeforge.circuit_heads` detects writers behaviorally. Default-off and
byte-identical when disabled (gate unchanged); the legacy reader-geometry path
remains as a documented ablation.

### Changed

- **`U_C` is now the circuit writers' OV-output (`two-basis-uc-writer-output`).**
  The composition subspace is redefined from the aggregate per-layer
  reader-geometry to the orthonormalised union of the **OV-output row spaces of
  the circuit's writer heads** (`extract_writer_subspace`). On an alive
  single-layer GPT-2 forge this removes the induction-predictable KL excess
  (−111%) where reader-geometry does not (−6%), and the label-free attribution
  subspace (`∂loss/∂residual`) is ~orthogonal to it (overlap 0.05) and fails
  (+14% worse) — *loss-sensitivity ≠ circuit-mechanism*, so writers are
  identified mechanistically. New `saeforge.circuit_heads` detects writers
  behaviorally (`prev-token` / `duplicate-token` presets, by Δ=1 /
  same-token-earlier attention). `ForgePipeline.composition_heads` now accepts a
  preset, an explicit `(layer, head)` list, or `"all"`; new
  `composition_mode={"writer-output" (default),"reader-geometry"}`; CLI gains
  `--composition-mode` and `--composition-heads` accepts presets / `L.H` lists /
  `all`. The run report records the writer heads with their detection scores and
  the mode. The legacy reader-geometry path remains as a documented ablation.
  Default-off and byte-identical when disabled (gate unchanged).

## [0.13.0] — 2026-06-06

**Two-basis forge release (`two-basis-forge`).** Preserves *two* residual
subspaces verbatim inside the projection — the assertion subspace `U_A` (sharp
atoms → recovers `cov95`) and the per-layer composition subspace `U_C` (the host
attention QK/OV read+write geometry → forged circuits stay faithful), with the
Polygram basis carrying the orthogonal remainder. Operationalizes the `lm-sae`
finding that a single residual basis cannot carry both 1-operand assertions and
2-operand composition: forged QK/OV reproduce the host on `span(U_C)` (verified
to 1e-6/1e-9), and the new `circuit_kl` (induction-predictable tokens) measures
the fidelity global KL is blind to. Default-off and byte-identical when
disabled; orthogonal to `hybrid-bridge-forge`. GPU-scale defaults decision
pending `scripts/compare_single_vs_two_basis_gpt2.py` on Intel/GPT-2.

### Added

- **Two-basis forge (`two-basis-forge`).** Opt-in forge path that preserves two
  residual subspaces verbatim inside the projection — the assertion subspace
  `U_A` (sharp atoms → recovers `cov95`) and the per-layer composition subspace
  `U_C` (host attention QK/OV read+write geometry → circuits survive forging),
  with the Polygram basis carrying the orthogonal remainder. New
  `saeforge.composition_subspace` (`extract_composition_subspace`),
  `saeforge.augmented_basis` (`AugmentedBasis`), and
  `saeforge.eval.circuit_faithfulness` (`circuit_kl`, `assertion_cov95`,
  `induction_predictable`). `SubspaceProjector.project_module` gains an
  `augmented=` arm (byte-identical when `None`); `ForgePipeline` gains
  `composition_preserve` / `assertion_preserve` / `composition_rank` /
  `composition_heads` / `assertion_k`; CLI gains the matching flags plus
  `--circuit-faithfulness`. Default-off and byte-equivalent; orthogonal to
  `hybrid-bridge-forge`. See `docs/two_basis_forge.md` and
  `scripts/compare_single_vs_two_basis_gpt2.py`.

## [0.12.0] — 2026-06-04

**Routed mixture-of-experts forge release (`add-sae-moe-forge`).** Adds
`forge_to_moe` / `ForgedMoE`: project a polygram-compressed SAE basis
into a routed MoE whose per-token decode cost scales as
`k_experts / n_experts` of the flat SAE. This promotes the runtime-MoE
play that bio-sae's fine-tune ceiling sweep re-opened — the residual
sharp-feature (cov95) tax the mAUC distillation arm could not close is a
routed-expert target ("lossless at fixed runtime cost, not fixed param
count"). v1 is inference-only with zero new parameters; the dispositive
bio-sae sharp-vs-diffuse partition experiment this unblocks runs against
this release.

### Added (add-sae-moe-forge)

- **`forge_to_moe(basis, expert_dictionary=None, *, k_experts=2, …)`** —
  single public entry. Explicit-`ExpertDictionary` path or auto-cluster
  from the basis's polygram checkpoint (`load_sae_safetensors` →
  `from_sae_lens` → `cluster_experts`).
- **`ForgedMoE`** (`torch.nn.Module`, buffers only — `.parameters()`
  empty): `forward(track_load=…)`, `route`, `expert_load`,
  `faithfulness_report`, `coherence_diagnostic`, and self-contained
  `save_pretrained` / `load_pretrained`.
- **`ForgedMoEConfig`** — frozen, JSON-round-trippable contract surface.
- **`saeforge/_moe/`** — `SubDictionaryExpertSet` (each expert a
  deterministic `W_dec` slice; vectorised masked-matmul decode;
  `effective_decode_cost`) and `PolygramHeuristicRouter` (torch-batched
  routing with bit-for-bit parity vs polygram's per-vector `route`).
- **`FeatureBasis.polygram_checkpoint_path`** (+ `to_dict` / `from_dict`).
- Lazy `__init__` exports (`ForgedMoE` / `ForgedMoEConfig` /
  `forge_to_moe`) keep a torch-free `import saeforge` cheap.
- Docs: `docs/moe-forge.md`; spec promoted to
  `openspec/specs/sae-moe-forge/`.

### Acceptance bands (all green)

- **A — fidelity collapse**: `k = n_experts` reproduces the flat SAE
  (MSE/coord ≤ 1e-5; measured 0.0).
- **B — sparsity gain**: counted decode-cost ratio within `2/E ± 0.05`.
- **C-strict — clusterable basis**: routed-vs-flat ≤ 0.5× flat-vs-host
  (0.111 on the d=768 synthetic fixture, matching the prototype's 0.12);
  **C-advisory** on isotropic bases via the reported
  `coherence_diagnostic`.
- **D — round-trip**: config + reloaded-module forward byte-identical.

### Deferred (named follow-ups)

- Native `W_enc` encoder (`add-moe-encoder-side`); v1 uses the
  `pinv(W_dec)` `SubspaceProjector` encoder.
- `tiny_mlp` / `residual_block` experts; `linear` / `mlp` trained routers
  — all raise a clean `NotImplementedError` naming the proposal.
- Gather/sparse decode kernel (wall-clock saving vs the counted gain),
  matryoshka/steering surfaces, residual-stream-layer insertion.
- A real clustered-SAE smoke fixture (tasks §2.4; Band C-strict is
  validated against the synthetic fixture only).

## [0.10.0] — 2026-05-22

**Multi-encoding capability sweep + partition-aware basis builder
release.** Combines six PRs (#88, #89, #92, #93, #94, #95) into one
minor version covering partition validation + multi-encoding API +
CLI surface + falsifiable acceptance gate (validated on bio-sae's
pooled fixture).

### Headline empirical validation

Multi-encoding acceptance gate ran K=3 encodings (raw_slice +
partition_q4 + partition_q8) on bio-sae's pooled fixture at
[1000, 5000] proteins:

| metric | raw_slice | partition_q4 (winner) | partition_q8 |
|---|---|---|---|
| recommendation target_n_features_kept | n=256 | **n=128** | n=64 |
| retained_mauc_vs_host at rec | 0.8975 | **0.9096** | 0.9004 |
| converged at default strictness | False | False | False |

- **Rec_n factor of 4×** between encodings — multi-encoding sweep
  correctly distinguishes per-encoding recommendations.
- **Pareto-shift**: same retained_mauc at fewer parameters
  (partition_q4 at n=128 matches raw_slice at n=256).
- **None converged at default strictness** — data-scale tax on
  spread regime persists across encoding choices.

Full writeup: `bio-sae/docs/forge-capability-bottleneck.md` §5.6.

### Added (add-multi-encoding-capability-sweep)

- **`sweep_pareto_capability(encodings=[(label, path), ...])`** —
  multi-encoding API (PR #92). Per-encoding basis state; per-cell
  rows carry encoding_label.
- **`sweep_pareto_capability_progressive(encodings=[...])`** — same
  on progressive (PR #93). Per-encoding plateau + convergence +
  cross-encoding winner-pick tiebreaker.
- **`ProgressiveStageResult.per_encoding_plateau_widths`** —
  per-encoding plateau dict.
- **`ProgressiveRecommendation.per_encoding_recommendations`** —
  optional dict, populated for multi-encoding sweeps.
- **`ProgressiveRecommendation.winning_encoding`** — winning
  encoding string identifier.
- **`sae-forge sweep-capability --encoding LABEL:PATH`** + same on
  `sweep-capability-progressive` (PR #94). Repeatable.
- **`sae-forge sweep-capability --dry-run`** + `--dollars-per-gpu-hr`.
  Counts cells × projects wall-time without running.
- **`sae-forge recommend` multi-encoding ranking table**. Picks
  smallest target_n_features_kept among survivors; tiebreaks by
  CLI flag order.
- **Falsifiable acceptance gate** (PR #95) on bio-sae's pooled
  fixture. Three predictions tested; all three pass under the
  revised Pareto-shift framing.

### Added (add-partition-encoding-capability-validation)

- **Partition-aware basis builder** (PR #89). `sweep_pareto_capability` and
  `sweep_pareto_capability_progressive` now honour an optional
  `partition_block_ids` tensor in the SAE state dict; when present,
  the per-cell basis builder slices per-tier proportionally (with
  largest-fractional-remainder rounding) instead of flat-by-row-norm.
  Absent: current row-norm slicing preserved byte-equivalently.
- **`_slice_partition_aware` helper** in `saeforge.sweep_capability`
  (private; not in `__all__`). Pure function; testable in isolation.
- **7 new tests** under `tests/test_sweep_pareto_capability.py`
  "Suite 5: partition-aware basis slicing". Covers proportional
  allocation, largest-residual tiebreak, within-tier top-K,
  exceeds-total ValueError, fallback-when-absent, used-when-present
  end-to-end, shape-mismatch raises.

### Validated (add-partition-encoding-capability-validation)

The validation experiment ran on bio-sae's pooled fixture
(`uniref50_n5000/pooled_w1024_k64`) at the [1000, 5000] progressive
schedule. Result: **`PARTITION_PARTIAL_WIN`** per the openspec's
decision-tree:

- Partition trajectory variance: **0.0018** (vs raw_slice 0.0235 —
  13× lower).
- Partition recommendation: `target_n_features_kept = n=128` (vs
  raw_slice n=256 — half the parameters at comparable retained_mauc
  0.9096 vs 0.8975).
- Both regimes `converged=False` at default strictness, but for
  different reasons (raw_slice's retained_mauc drifts; partition's
  argmin shifts).
- Per-cell deltas mixed: 6 partition wins, 5 raw_slice wins, 2 ties
  out of 14 cells. Partition's biggest wins at small-n cells;
  raw_slice's biggest wins at mid-n cells.

**Wave C resolved.** Polygram's partition encoding (shipped at
polygram v0.14.0 in 2026-05-21) was filed as "unproven" because its
`forge_kl` A/B showed 0 % improvement vs raw_slice. Under the
capability framework + progressive wrapper, the same partition
delivers a 13× variance reduction + half-the-parameters Pareto
improvement. **`forge_kl` was the wrong metric**; the partition's
forge-side payoff materialised under capability scoring as
predicted by the original Wave C proposal — it just wasn't visible
from KL.

Full writeup at `bio-sae/docs/forge-capability-bottleneck.md` §5.5
(per-cell delta table, reproduction recipe, per-cell-pattern
interpretation).

### Added (add-progressive-capability-sweep)

- **`sweep_pareto_capability_progressive(...)`** — new top-level
  entry point. Drives a multi-stage capability-aware Pareto sweep
  with cumulative protein subsamples + plateau-based pruning +
  neighbour expansion + convergence detection. Returns a stable
  recommendation (smallest n robust to data scale) instead of an
  argmax-on-one-sample. The recommendation contract is Occam's
  razor applied to forge basis selection: among widths that retain
  host capability equally well across data scales, pick the
  smallest.
- **`ProgressiveStageResult` / `ProgressiveRecommendation` /
  `ProgressiveHistory` / `ConvergenceTrajectoryEntry`** — four new
  frozen dataclasses. `ProgressiveHistory.to_json_dict()` emits
  `progressive_summary.json` carrying the per-stage trajectory +
  the final recommendation + the convergence narrative. External
  benchmarking can count un-converged ratios from the on-disk
  artefacts without in-library telemetry.
- **`ParetoFrontierRow.stage`** — optional `int | None` field
  default `None`. Populated on progressive-sweep cells; omitted
  from JSON when None (v0.8.x back-compat preserved).
- **`sae-forge sweep-capability-progressive`** CLI subcommand.
  Mirrors `sweep-capability`'s YAML config schema; adds
  `--candidate-widths`, `--schedule`, `--retained-mauc-tolerance`,
  `--plateau-tolerance`, `--min-plateau-widths`,
  `--convergence-n-stages`. Exit codes: 0 converged / 1 schedule-
  exhausted-but-recommendation-emitted / 2 config error.
- **`sae-forge recommend --accept-unconverged`** — opt-in to accept
  an un-converged progressive frontier. Default behaviour: refuse
  with a rich diagnostic naming the recommended n + retained_mauc,
  the list of shifted stages, the on-disk rationale string, and
  four informed opt-outs (`--accept-unconverged`, longer schedule,
  looser `plateau_tolerance`, `convergence_n_stages=1`). Single-
  shot frontiers (no `stage` field) bypass the check entirely —
  v0.8.x recommend semantics preserved.
- **`CapabilityDataset.from_bio_sae` residues-per-protein
  metadata** — under `feed='residue'`, the constructor now
  populates `metadata['residues_per_protein']` (per-protein
  residue counts derived from the bundle's `residue_index[:, 0]`).
  Required by the progressive wrapper's subsampling under residue
  feed. Computed via `np.bincount` for clean O(N_residues)
  performance.
- **Bio-sae acceptance gate** — three slow integration tests
  (`tests/test_progressive_acceptance_gate.py`) pinning the
  empirical predictions against real bio-sae fixtures: residue
  regime converges in ≤ 3 stages with rec ∈ [12, 64] and
  retained_mauc ≥ 0.98; pooled regime under default strictness
  flags the plateau argmin shift (n=384→n=256 between 200 and 500
  proteins) and refuses the recommendation cleanly; pooled regime
  under documented opt-out (`convergence_n_stages=1`) converges.
- **README "Progressive capability sweep" section** + **algorithm.md
  §5 cross-reference** + this CHANGELOG entry. Covers the
  recommendation contract, the schedule shape, the un-converged
  refusal UX, the two opt-outs, and the empirical reference points
  from bio-sae's runs.

### Notes on the empirical finding

The openspec's initial prediction was "the pooled regime is single-
shot stable per writeup §3.2; converges in 1 data-scale transition."
The empirical run surfaced that §3.2 was measuring the *peak*
position (argmax retained_mauc) as stable across data scales, not
the *smallest-plateau-member* position. The wrapper's argmin-of-
plateau contract correctly picks the plateau's left edge, which
shifts because the plateau membership contracts as the AUC estimate
tightens with more data. The acceptance-gate test pair documents
this honestly: the strict-default test validates the wrapper's
correct refusal behaviour; the opt-out test validates the documented
looser-strictness path works on the substrate that defeats the
strict path. See
`openspec/changes/add-progressive-capability-sweep/specs/pareto-sweep/spec.md`
for the corrected prediction in the spec deltas.

## [0.8.1] — 2026-05-22

Patch release on top of v0.8.0 shipping the residue-feed extension
to `CapabilityDataset` + `sweep_pareto_capability`. Closes the
follow-up filed by bio-sae's acceptance gate
(`bio-sae/tests/test_forge_capability_acceptance.py`): the n=16
residue-SAE prediction becomes directly testable upstream.

### Added (add-downstream-capability-target — residue-feed support)

- **`feed="residue"` on `CapabilityDataset` + `sweep_pareto_capability`.**
  Previously the sweep wrapper unconditionally mean-pooled per
  protein, which made it incompatible with substrate-level
  evaluation (residue-scope GT labels). New `feed` field is a
  load-bearing dataclass field (default `"pooled"` for back-compat)
  with construction-time validation that `labels.shape[0]` matches
  the feed: `== len(sequences)` for pooled, `>= len(sequences)` for
  residue.
- **`_extract_host_activations` / `_extract_forged_activations`
  honor feed.** Under residue feed, the extractors skip the mean-
  pool and concatenate per-residue states across proteins
  (protein-major ordering matching bio-sae's `residue_index`).
- **Runtime alignment check.** Under residue feed, the sweep wrapper
  verifies `host_X.shape[0] == dataset.labels.shape[0]` after
  extraction and raises a clear `RuntimeError` on mismatch (usually
  caused by `max_seq_len` drift between bundle build time and read
  time). Pre-change residue-feed users would have silently produced
  nonsense AUCs.
- **`HostCacheKey.feed` field.** Cache key now includes the feed
  string; pooled and residue runs over the same sequences produce
  distinct cache files instead of colliding.
- Closes the follow-up filed by bio-sae's acceptance gate
  (`bio-sae/tests/test_forge_capability_acceptance.py` module
  docstring). The n=16 prediction on bio-sae's residue SAE +
  residue feed becomes directly testable upstream.
- Test surface: +6 new tests (3 dataset validation, 3 sweep-side
  residue-feed end-to-end including pooled-vs-residue cache
  isolation + label-alignment failure mode). Total focused suite
  38 passing; full suite 744 passing.

## [0.8.0] — 2026-05-22

The 0.8.0 release ships two coordinated capabilities: the **ESM-2
adapter** (first encoder-only protein-LM host outside the audio
family) and **capability-aware forge tuning** (`DownstreamCapabilityTarget`
+ `sweep_pareto_capability` + `sae-forge sweep-capability` /
`recommend` CLI). The unifying observation is that residual-cosine /
KL faithfulness systematically misranks forges for capability-bound
users — bio-sae's empirical work on the ESM-2 substrate showed cosine
recommends n=256 features while the same forge at n=16 retains 103 %
of host capability (16× discrepancy on optimal width). The new
target answers "does the forge retain the downstream task?" instead
of "are the forged hidden states numerically close to host?".

### Minor-version bump (vs patch)

- New public surface: `DownstreamCapabilityTarget`,
  `CapabilityDataset`, `sweep_pareto_capability` exported from
  `saeforge`; `Esm2Adapter` registered alongside the bundled
  adapters; 14 new optional fields on `ParetoFrontierRow`; two new
  `sae-forge` CLI subcommands.
- New runtime contract: `ctx["basis"]` is now populated by
  `ForgePipeline`'s imperative + synthetic score paths.
  Cost-free for targets that don't read this key (every bundled
  target ignores it except `DownstreamCapabilityTarget`).
- New adapter-side requirement: encoder-only adapters
  (`esm2`, `whisper_encoder`) emit a `basis_decode` buffer
  alongside `basis_encode`. Pre-change forge artefacts trigger a
  one-shot UserWarning under capability scoring (path c pinv
  fallback); production runs against the bundled adapters never
  hit this path.

### Polygram pin

- `polygram>=0.15.0` (was `>=0.12.0`). v0.15.0 ships the masked-LM
  host dispatcher needed for the polygram side of the chain to
  reach ESM-2 hosts.

### Added (add-downstream-capability-target)

- **`DownstreamCapabilityTarget`** — new
  `saeforge.eval.faithfulness.FaithfulnessTarget` that scores per-
  feature × per-label AUC through a caller-supplied downstream task
  encoder. Answers "does the forge retain the downstream task?"
  instead of "are the forged hidden states numerically close to
  host?". Bio-sae's empirical work showed those two questions yield
  different Pareto-optimal widths — cosine recommends n=256,
  capability recommends n=16, on the same ESM-2 / bio-sae substrate
  (16× discrepancy). Three-path `W_dec` recovery precedence
  (`ctx["basis"]` → `forged_module.basis_decode` →
  `pinv(basis_encode)`). Aggregator dispatch:
  `pool_then_encode` / `encode_then_pool` / callable. Silenceable
  pinv warning via `warn_on_pinv=False`. Never family-defaulted —
  opt-in via `ForgePipeline(faithfulness=...)`.
- **`CapabilityDataset`** — frozen dataclass at
  `saeforge.datasets.CapabilityDataset`. Bundles sequences + labels
  + encoder + aggregator + tokenizer_id + metadata.
  `from_bio_sae(...)` constructor parses a bio-sae bundle (sae.pt +
  bundle safetensors + sequences parquet) without importing biosae.
  Sm-sae / econ-sae will register their own `from_<repo>`
  constructors in their respective repos.
- **`basis_decode` buffer** — emitted alongside `basis_encode` by
  encoder-only adapters (`Esm2Adapter`, `WhisperEncoderAdapter`).
  Shape `(n_features, d_model)`; carries `W_dec` directly. Removes
  the `pinv(basis_encode)` roundtrip for the bundled families and
  makes the capability target's path (b) the default decode route
  when `ctx["basis"]` isn't piped through.
- **`sweep_pareto_capability(...)`** — new wrapper over
  `sweep_pareto`'s machinery that uses `DownstreamCapabilityTarget`
  as the metric. Drives the (encoding × width × scale_boost) cube;
  emits `frontier.jsonl` with optional capability fields populated
  on each row. Host-extraction cache enabled by default
  (`<output-dir>/host_cache/`, opt-out via `cache_host=False`);
  content-addressed key over `(host_model_id, sequences_hash,
  aggregator, max_seq_len)`.
- **`ParetoFrontierRow` capability fields** — 14 new optional
  fields: `host_baseline_mauc`, `forge_mauc`,
  `retained_mauc_vs_host`, `retained_cov95_vs_host`, `gap_median`,
  `gap_p25` / `gap_p75` / `gap_p95`, `n_features_gap_above_0_1`,
  `n_features_negative_gap`, `capability_aggregator`,
  `capability_min_prevalence`, etc. All `Optional[…]` defaulting to
  `None`. `to_json_dict()` omits the capability block when no field
  is populated — v0.7 frontier files stay byte-equivalent for
  non-capability rows. `from_json_dict()` loads pre-change rows
  unchanged.
- **`ForgeResult.basis` piping into `ctx["basis"]`** — both
  imperative and synthetic score paths now populate
  `ctx["basis"] = self.basis`. Resolves the capability target's
  three-path decode precedence to path (a) "exact W_dec from
  explicit basis" by default when the pipeline drives the target.
  Zero-cost for targets that don't read this key.
- **CLI subcommands** —
  `sae-forge sweep-capability --dataset-config CONFIG.yaml --host
  HOST_ID --widths W1,W2,...` invokes the capability sweep.
  `sae-forge recommend --frontier frontier.jsonl --target
  retained-mauc>=0.95` filters survivors by AND-combined predicates
  and picks the smallest `target_n_features_kept`. Predicate parser
  accepts kebab-case / snake_case + shorthand aliases for the
  load-bearing capability fields. Tabular default + `--json` mode.
- **README** — new "Capability-aware forge tuning" section under
  `## Components` with the end-to-end CLI example.
- **`docs/algorithm.md` §5** — cross-reference from "Error sources"
  to the capability target. The amplification of `ε_attn` +
  `ε_nonlin` is invisible to cosine when the rank deficit is in
  non-information-bearing directions; visible to a capability
  metric that asks whether the downstream task is preserved.
- Test surface: 16 new capability-eval tests (target + dataset),
  11 new sweep tests (row schema + cache + end-to-end), 5 new
  acceptance-gate tests (synthetic substrate × structural plumbing).
  Total +32 tests; existing surface unchanged.

### Added (add-esm2-adapter)

- **`Esm2Adapter` — first encoder-only host outside the audio
  family.** Walks an HF `EsmModel` or `EsmForMaskedLM` (model_type
  `esm`, position_embedding_type `rotary`) into the projected weight
  dict + native config that drive forging. Surfaces the protein-LM
  side of polygram's downstream substrate matrix
  (sm-sae / econ-sae / bio-sae). Validated by an identity-basis
  byte-equivalence test: with `W_dec = I`, the forged `ForgedEsm2`
  reproduces HF's `EsmModel.last_hidden_state` exactly (max abs diff
  == 0.0 on `esm2_t6_8M`-shaped configs).
  - `saeforge/adapters/esm2.py` — adapter + `ForgedEsm2` nn.Module.
    Bidirectional attention (no causal mask), RoPE shared with
    `_positional/rope.py`, ESM-specific GELU + query-scaling-
    before-RoPE order, pre-LN inside both attention and FFN
    sublayers, final `emb_layer_norm_after`. Emits a `basis_encode`
    buffer alongside the projected weights so under-complete-basis
    cosine eval can project host states into basis space (same
    contract as `WhisperEncoderAdapter`).
  - `saeforge/eval/targets/token_cosine.py` — new
    `TokenCosineTarget`. Per-residue cosine on encoder hidden states,
    stripping CLS / EOS positions to match bio-sae's `EsmExtractor`.
    Default faithfulness target for `esm2` (parallel to
    `CosineTarget` for `whisper_encoder`).
  - `saeforge/model.py` — `_SUPPORTED_FAMILIES` adds `esm2`;
    `_ENCODER_STATES_FAMILIES = {whisper_encoder, esm2}`. The
    `encoder_states` validator splits family-specific vocab_size
    rules: `whisper_encoder` requires `vocab_size == 0` (no
    embeddings); `esm2` requires `vocab_size > 0` (sizes the
    amino-acid embedding table).
  - `saeforge/utils/host_loader.py` — new
    `load_host_for_forge(host_model_id)` dispatcher. Tries
    `AutoModelForCausalLM` first (the historical default for GPT-2 /
    Llama / Gemma-2 / Qwen) and falls back to `AutoModelForMaskedLM`
    when the config is unrecognised (ESM-2's path). Back-compat with
    every existing test that mocks `AutoModelForCausalLM.from_pretrained`.
  - `saeforge/forge.py` — `_run_real_imperative` and `_run_real_fsm`
    use the new dispatcher; `_resolve_positional_encoding` returns
    `"rotary"` for `esm2`.
  - **`tests/test_esm2_adapter.py`** — 8 new tests: adapter
    registration (EsmModel + EsmForMaskedLM), non-rotary rejection,
    native_config invariants (MHA, theta=10000, encoder_states,
    vocab_size > 0), the byte-identity load-bearing test, walk-key
    coverage, default target dispatch, and TokenCosineTarget under
    identity basis.

### Added (add-concept-anchored-finetune)

- **Opt-in supervised concept-anchoring loss term in `run_finetune`.**
  Transposes econ-sae's Phase 6.2 dual-head + focal-loss recipe (which
  lifted regime-tier mAUC from 0.595 to 0.991) to the forge fine-tune.
  Six new `TrainingConfig` fields (`concept_alpha`,
  `concept_pool_weight`, `concept_channel_weight`, `concept_focal_gamma`,
  `concept_label_source`, `concept_label_source_kwargs`); six matching
  `ForgePipeline` kwargs (`finetune_concept_*`). Default
  `concept_alpha=0.0` skips the entire branch (no label-source
  instantiation, no head construction, no extra forward) — byte-identical
  to the v0.3 LM-CE path and the distillation extension.
- **`saeforge/training/heads.py`** — `PooledConceptHead` (mean-pool
  + linear), `PerChannelConceptHead` (per-concept affine readout of the
  last N residual dims), and `focal_bce_loss(logits, labels, gamma=...)`.
- **`saeforge/training/concept_anchor.py`** — `LabelSource` protocol,
  `LABEL_SOURCE_REGISTRY`, `register_label_source(...)` decorator, and
  the v1 backend `PolygramClusterLabelSource` (self-supervised,
  pseudo-labels from per-cluster firings of the pre-fine-tune student).
- **Four new public exports:** `LabelSource`, `LABEL_SOURCE_REGISTRY`,
  `register_label_source` (the seam for follow-up backends —
  `corpus-tags`, `host-probe` — to register without touching the loss
  code), and `focal_bce_loss` (lifted from the heads module so
  analysts can use the same focal-BCE term standalone in custom
  losses outside the concept-anchoring path).
- **Residual-stream capture via forward pre-hook on `module.lm_head`.**
  Avoids broadening every architecture adapter's `forward` signature
  with an `output_hidden_states` flag.

### Added (add-polygram-cluster-diagnostics)

- **`ParetoFrontierRow` polygram concept-structure fields.** Four
  new optional fields piped from the polygram
  `compression_report.json` colocated with each per-K SAE:
  `polygram_n_clusters` (distinct concept clusters), `polygram_n_zeroed`
  (slots zeroed as redundant), `polygram_redundancy_ratio`
  (`n_zeroed / (n_clusters + n_zeroed)`), and
  `polygram_encoding_capacity` (Rung3=16, Rung4=32, Rung5=128,
  HEA_Rung2(n)=2ⁿ). All default `None` for backwards compatibility
  with older `frontier.jsonl` files and polygram outputs lacking the
  fields. Populated even on row failure (computed pre-forge), so an
  analyst can distinguish "doomed input" from "bug in the forge."
- **`saeforge.polygram_diagnostics` module.** Public helpers
  `load_polygram_report`, `compute_redundancy_ratio`, and
  `resolve_encoding_capacity`. The capacity resolver accepts the
  same encoding label strings the sweep CLI surfaces.
- **Pre-flight saturation advisory.** When the largest-K SAE in any
  encoding's manifest reports `polygram_n_clusters ==
  polygram_encoding_capacity`, `advise_sweep_quality` appends a
  one-line note suggesting the next encoding rung. Informational
  only — `--quality-floor` continues to react to `quality_ratio`
  alone; the saturation check never refuses.
- **README**: new "Polygram concept-structure diagnostics" section
  under `#### Pareto sweep (Axis 4)` documenting the four fields,
  the `jq` filter idiom, and the cross-encoding sweep recipe for
  measuring cluster-count saturation.

### Proposed (not yet implemented)

- **`add-sae-moe-forge`** — turn a polygram-clustered SAE into a
  routed mixture-of-experts via `forge_to_moe(basis, …)`. Proposal +
  prototype + smoke-gate landed in
  `openspec/changes/add-sae-moe-forge/`; production code follows in
  a separate PR (will introduce `saeforge/moe.py` + `saeforge/_moe/`).
  Mechanical bands (k=E collapse, k=2 sparsity gain, config
  round-trip) hold universally on the prototype; faithfulness is
  basis-dependent (0.12× flat-vs-host on clusterable bases, ~4.6× on
  near-isotropic — the spec's Band C splits strict / advisory along
  this axis).

## [0.7.0] — 2026-05-19

The 0.7.0 release closes the Llama-family RoPE gap that this cycle
opened, investigated, and fixed end-to-end. Pre-cycle, Llama /
Gemma-2 / Qwen2-3 forged attention silently dropped RoPE entirely —
`LlamaSelfAttention.forward` went from `q_proj`/`k_proj` straight to
the scaled dot-product with no rotation, despite the adapter
docstrings claiming the forge matched host positional handling. #62
proposed the fix, #63 implemented it (`apply_rotary_pos_emb` on Q/K
before optional Q/K-norm and SDPA, gated by `cfg.rope_mode in
{"standard", "none"}`), #64 archived the change-dir spec into
`openspec/specs/architecture-adapters/spec.md`, and #66 fixed two
follow-on issues the at-scale validation surfaced:

- **bf16 RoPE precision bug.** `compute_rope_cache` honored
  `dtype=q.dtype`, so on M4 bf16/MPS it built `inv_freq`, the
  position grid, and cos/sin entirely in bf16. At Gemma-2's
  `head_dim=256`, `arange(seq_len, dtype=bf16)` aliases integer
  positions above 256 (bf16's 7-bit mantissa) and the smallest
  `inv_freq` (~1e-4) loses ~half its precision. Measured cos drift
  saturated at the full ±1 range by seq_len=512; pre-softmax
  attention-logit relative drift hit 91% at seq_len=8192 — a
  randomised attention pattern, not small numerical error.
  Post-fix: forge-vs-HF Llama at identity basis matches host to
  float noise at fp32 (~1e-7) and to bf16 activation noise at bf16
  (~5e-3, constant in seq_len). The fix matches HF convention:
  build the cache in fp32 unconditionally, cast to the caller dtype
  at the end.

- **`partial_rotary_factor` silent no-op.** Was read into
  `NativeModelConfig` but never consulted by
  `LlamaSelfAttention.forward`. Llama / Gemma-2 / Qwen2-3 all use
  1.0 so this was silently fine for them, but Gemma-3 / GPT-J /
  NeoX-style hosts with `<1.0` would have rotated all `head_dim`
  instead of the partial slice. Now raises `NotImplementedError` at
  first forward, parallel to the existing `rope_scaling.type`
  guard.

Acceptance gates (post-#66): 627 tests + 10 skipped on Intel 16GB
CPU; forge-vs-HF Llama on identity basis at `head_dim ∈ {8, 32}`
and `seq_len ∈ {64, 512, 2048}` shows ~1e-7 relative error at fp32
and ~5e-3 (constant in seq_len) at bf16; full
`ForgePipeline.run_synthetic` on a tiny Llama hits KL ~1e-8 at fp32
and 1.7e-4 → 3.7e-4 going T=64 → 512 at bf16 (bounded, not
unbounded — pre-fix this would have grown without limit).

### Added

- **`saeforge/_positional/rope.py`** — `compute_rope_cache` and
  `apply_rotary_pos_emb` helpers. Pure torch, HF-compatible math,
  fp32-build-then-cast convention. Lazy-imported; no impact on the
  no-`[torch]` install path.
- **`ForgeResult.positional_encoding`** field ∈ `{"absolute_projected",
  "rotary", "none_skipped"}`, populated by
  `_resolve_positional_encoding` and propagated through
  `forge_result.json`. Makes silent positional skips loud.
- **`NativeModelConfig.rope_mode` / `.rope_theta` / `.rope_scaling`
  / `.partial_rotary_factor`** fields, populated by the Llama-family
  adapter's `build_native_config`. Legacy configs (pre-#63
  payloads) round-trip through `from_dict` with default fills.
- **CLI**: `--rope-mode {standard,none}` flag on the `forge`
  subcommand (regression-diff arm; default `standard`).

### Changed

- **Llama-family forged attention applies RoPE** at default
  `rope_mode="standard"`. Behaviour change for every existing
  Llama / Gemma-2 / Qwen2 / Qwen3 / Qwen3-MoE forge — outputs will
  differ from `0.6.0` because the prior outputs were missing the
  rotation entirely. The `rope_mode="none"` arm reproduces the
  pre-fix buggy behaviour byte-identically for regression diffing.
- **polygram dependency floor** bumped from `>=0.9.0` to `>=0.10.0`
  (both the `polygram` extra and the `all` extra). Promotes
  `EpochReport.redundancy_ratio` and `n_features_input` to
  first-class fields (schema v2 with v1 forward-compat shim) and
  emits a `UserWarning` on `BlockFormation` degenerate partitions
  — both directly relevant to the queued
  `add-polygram-cluster-diagnostics` proposal.

### Fixed

- **`compute_rope_cache` bf16 precision** — see the cycle overview
  above. Pinned by `tests/test_rope.py::test_compute_rope_cache_bf16_matches_fp32_cast`,
  which demands bit-equality (no `atol`) against the
  fp32-built-then-cast path at Gemma-2 `head_dim=256, seq_len=512`.
- **`partial_rotary_factor != 1.0` silent mis-rotation** — now
  raises `NotImplementedError` at first forward. Pinned by
  `tests/test_positional_encoding_assertion.py::test_partial_rotary_factor_non_unity_raises_at_forward`.

### Why minor (not patch)

Forged-output behaviour on every Llama-family host changes
observably between 0.6.0 and 0.7.0 — pre-cycle, RoPE was absent;
post-cycle it's applied. Users with a 0.6.0 Llama forge sitting on
disk should expect different logits after upgrading; the change is
correct (it now matches the host's positional handling), but it is
observable. New runtime guard surface
(`partial_rotary_factor != 1.0` now raises) also lands in this
cycle.

### Acknowledged but not in this release

- **M4 Gemma-2-2B at-scale re-measurement.** Pre-cycle baseline hit
  KL=13.19 on a forge that was missing RoPE *and* corrupting the
  bf16 cos/sin cache. Both gaps are closed in 0.7.0 but the at-scale
  number has not been re-collected; the release ships on the
  end-to-end correctness gates above (627 tests + tiny-Llama
  forge-vs-HF + full `run_synthetic`), with the M4 number tracked as
  a follow-up benchmark rather than a release blocker.
- **Other `rope_scaling.type` variants** (`linear` / `dynamic` /
  `yarn` / `longrope`) and **`partial_rotary_factor < 1.0`** —
  raise `NotImplementedError`; queued for the
  `add-rope-scaling-types` follow-up.

## [0.6.0] — 2026-05-19

The 0.6.0 release ships `forge-forward-mode` — the structural fix
for the rank-dependent KL amplification documented in
`fix-scale-boost-calibration`. `NativeModel` gains a new forward
implementation (`host_wrapped`) that runs every transformer block on
the host's exact, unprojected weights with `decode → host_op →
encode` wrapping each block; the existing `native_in_basis` path is
unchanged. Dispatch is by basis quality tier (`good`/`saturated` →
native; `undersized`/`degenerate` → host-wrapped) under the new
`forward_mode="auto"` default.

The 2026-05-19 acceptance gate on the GPT-2 layer-8 jbloom K=211
fixture: forge KL drops 89.9 → 15.4 nats (5.8× reduction); no
adjacent K-pair ΔKL exceeds 10 nats; KL ≈ 0 on a synthetic
orthonormal n=d basis (host-wrapped converges to host exactly when
decode/encode is identity). Per-layer instrumentation localised the
amplifier to block 0 specifically; root cause is `LayerNorm`
parameters projected via `pinv()` as a per-coord gain — a category
error since per-coord gains don't have an isomorphism in a
non-orthonormal basis. The full diagnosis and audit trail (including
a falsified alternative — decode-LN_host-encode that only fixes one
op) lives at
`openspec/changes/archive/2026-05-19-add-host-wrapped-forge-fallback/smoke-results.md`.

The minor-version bump (vs. patch) reflects the new public surface
(`forward_mode` field on `NativeModelConfig` and `ForgePipeline`,
`--forward-mode` CLI flag on the `forge` subcommand) and the
observable behaviour change for under-complete-basis users at the
default `forward_mode="auto"` — under-complete forges now route
through the host-wrapped fallback instead of producing the documented
blow-up.

### Added (forge-forward-mode)

- **`forward_mode` dispatch** — `NativeModelConfig.forward_mode` and
  `ForgePipeline.forward_mode` accept `"auto"` (default),
  `"native_in_basis"`, or `"host_wrapped"`. Auto dispatches by basis
  quality tier: good/saturated → existing native_in_basis path;
  undersized/degenerate → new host_wrapped path that wraps host's
  exact transformer with decode/encode at every block boundary.
  Canonical spec at `openspec/specs/forge-forward-mode/spec.md`;
  full audit trail archived at
  `openspec/changes/archive/2026-05-19-add-host-wrapped-forge-fallback/`.
  v1 host_wrapped is GPT-2 only and inference-only.
- **`saeforge.forward_mode.resolve_forward_mode(basis, requested)`**
  — pure function exposing the dispatch resolution. Used internally
  by `ForgePipeline` and `NativeModel.from_host`; also callable by
  examples and external tooling that need to surface the resolved
  mode before constructing the model.
- **`ArchitectureAdapter.host_wrapped_module(host, basis,
  scale_boost)`** — new ABC method. `GPT2Adapter` ships the v1
  implementation; six other bundled adapters inherit the base-class
  `NotImplementedError` default pointing at the queued per-family
  rollout proposals (`add-host-wrapped-{llama,gemma2,qwen,whisper}`
  in the openspec follow-up queue).
- **`sae-forge forge --forward-mode {auto,native_in_basis,host_wrapped}`** —
  CLI flag threading `forward_mode` through `ForgePipeline`. The
  `sweep-pareto` extension lands in a follow-up PR.
- **`sae-forge forge --llm-scale`** — preset bumping
  `cosine_threshold` to 0.85 and `regrow.n_init` to 8 per the
  sm-sae LLM-scale provisional recommendations. Explicit flag
  values still win. `save_intermediate_reports=True` (the third
  sm-sae recommendation) isn't plumbed through `ForgePipeline` yet —
  noted in `--help`.
- **`sae-forge forge --regrow-n-init`** — direct CLI control over
  `RegrowConfig.n_init` (polygram default 4; sm-sae recommends 8+
  at LLM scale).
- **`examples/forge_gemma2_2b.py`** surfaces the resolved
  `forward_mode` and (when present) polygram cluster diagnostics
  (`n_clusters`, `n_zeroed`, `redundancy_ratio`) in the run summary.
- **`docs/flagship-gemma2-2b-demo.md`** — runbook for the at-scale
  Gemma-2-2B demo, with command, acceptance bands, and red-flag
  troubleshooting.

### Changed

- **`polygram>=0.9.0`** floor (was `>=0.8.1`). 0.9.0 promotes
  `cluster_experts` / `ExpertDictionary` to the public surface
  (PR #87) — the foundation for the queued `add-sae-moe-forge`
  capability (proposal + prototype + smoke gate landed in
  `openspec/changes/add-sae-moe-forge/`). No breaking API shift for
  existing pipelines; 609 tests green against 0.9.0.

### Tests

- `tests/test_forward_mode_dispatch.py` (16 tests) — covers
  `resolve_forward_mode` per quality tier + explicit + invalid;
  `NativeModelConfig.forward_mode` validation + legacy round-trip;
  GPT-2 host-wrapped module construction + forward-shape sanity;
  KL ≈ 0 on orthonormal n=d synthetic basis; non-GPT-2 adapter
  `NotImplementedError` shape; `ForgePipeline` rejection of
  host_wrapped + finetune / hybrid_bridge; CLI parser validation
  for both new flags.
- Total: 605 → 609 (host-wrapped impl + dispatch tests; the
  earlier increment to 607 reflects the intermediate state).

## [0.5.1] — 2026-05-18

The 0.5.1 release ships `world-model-protocol` — the architecture
seam every bundled host adapter satisfies structurally. Family
dispatch in `saeforge.model._build_torch_module` and
`saeforge.eval.targets._default_target_for` moves off two hardcoded
tables (`_LM_FAMILIES`, the `if family == "gpt2"` if/elif tree)
onto a registry lookup against the new `WorldModel` Protocol.

Behaviour on the seven bundled families
(`gpt2`/`llama`/`gemma2`/`qwen2`/`qwen3`/`qwen3_moe`/`whisper_encoder`)
is byte-identical to v0.5.0, pinned by a new per-family digest
guard in `tests/test_world_model_byte_identity.py`. One intentional
widening: `qwen3_moe` was a latent gap in the old `_LM_FAMILIES`
frozenset and now inherits the `KLTarget()` default like its
sibling LM families.

The patch-version bump (vs. a minor) reflects that the public
surface change is additive and the behaviour change on bundled
families is byte-identical.

### Added (world-model-protocol)

- **`saeforge.WorldModel`** — `@runtime_checkable` `typing.Protocol`
  defining the four-member contract every host-architecture
  adapter satisfies. Re-exported from `saeforge.adapters` and
  `saeforge` top-level. Third-party adapters MAY implement
  `WorldModel` structurally without inheriting from the bundled
  `ArchitectureAdapter` ABC.
- **`ArchitectureAdapter.default_faithfulness_target() -> FaithfulnessTarget`**
  — new ABC method; default returns `KLTarget()` (lazy-imported to
  break the `saeforge.eval.targets` → `saeforge.adapters` import
  cycle). `WhisperEncoderAdapter` overrides to `CosineTarget()`;
  the six LM-family adapters inherit the default.
- **`saeforge.adapters.registered_families() -> frozenset[str]`**
  — public helper returning the live set of `adapter.family`
  values across registered adapters. Single source of truth for
  "which families does this build support."

### Changed (world-model-protocol)

- **`_default_target_for(family)`** — body is now a 4-line registry
  lookup (`adapter_for_family(family).default_faithfulness_target()`)
  with a same-shape `ValueError` on unknown families. The
  `_LM_FAMILIES` frozenset is removed.
- **`_build_torch_module(config)`** — body is now a 2-line registry
  lookup (`adapter_for_family(config.family).native_module_class()(config)`).
  The `if family == "gpt2" / elif family in ("llama", …)` family
  tree is removed.
- **`NativeModelConfig.__post_init__`** — validates `self.family`
  against the union of bundled `_SUPPORTED_FAMILIES` and runtime
  `registered_families()`. Bundled names are accepted
  unconditionally so config construction works on a base install
  without `transformers`; runtime dispatch sites still require an
  actually-registered adapter and raise a distinct dispatch-time
  error.
- **`saeforge.model._SUPPORTED_FAMILIES`** retains its module-level
  position for back-compat with any direct reader. A new
  `_supported_families()` helper returns the sorted union with
  `registered_families()`; the post_init check uses the helper.

### Fixed

- `qwen3_moe` no longer raises `ValueError` from
  `_default_target_for("qwen3_moe")`; it was missing from the
  v0.5.0 `_LM_FAMILIES` set and now inherits `KLTarget()` like
  its sibling LM families. Pinned by the parametrised default-
  target test.

### Tests

- `tests/test_world_model_protocol.py` (15 tests) — protocol
  conformance, isinstance behaviour, error-message-shape pinning,
  default-target parity per family.
- `tests/test_world_model_byte_identity.py` (4 parametrised
  tests, qwen3 / qwen3_moe skipping when their adapters are
  unregistered) — pinned SHA-256 digest of
  `(n_params, round(faithfulness, 8), faithfulness_target_name,
  basis.W_dec.tobytes())` per family. First-run capture path
  documented in the file's docstring.

## [0.5.0] — 2026-05-18

The 0.5.0 release ships `add-gt-alignment-target` — the third
built-in `FaithfulnessTarget`, motivated by `jascal/sm-sae`'s
production `GroundTruthAlignment` scorer. Family defaults (KL for
LM hosts, cosine for whisper) are byte-identical to v0.4.0;
GT-alignment is opt-in only via
`ForgePipeline(faithfulness=GroundTruthTarget(labels=L))`.

The minor-version bump (vs. a patch) reflects the new `scipy>=1.10`
runtime dependency: technically a breaking change for callers with
strict pins, even though the default surface is additive.

### Added (add-gt-alignment-target)

Third built-in `FaithfulnessTarget`:
`saeforge.eval.targets.GroundTruthTarget` (also re-exported as
`saeforge.eval.GroundTruthTarget`). It scores forged residual-stream
activations against an `(N, M)` binary label matrix via per-feature
× per-label AUC — the right gate when your eval fixture carries
known per-sample categories (synthetic mixtures, BERT-probe-derived
datasets, concept-bottleneck suites). Supported pool strategies:
`"mean"` / `"max"` / `"last"`. Default `hidden_extractor` covers the
six bundled LM-shape families (gpt2 / llama / gemma2 / qwen2 /
qwen3 / qwen3_moe) via duck typing; Whisper / exotic forges supply
their own.

Demo: `examples/forge_with_gt_alignment.py` (mixture-of-gaussians,
~20s on CPU).

The pluggable-faithfulness protocol is unchanged; KL / cosine
family defaults are byte-identical. `GroundTruthTarget` is never a
family default — pass it explicitly via
`ForgePipeline(faithfulness=GroundTruthTarget(labels=L))`.

New runtime dependency: `scipy>=1.10` (powers
`scipy.stats.rankdata`-based average-rank ties handling in the AUC
helper, matching `sklearn.metrics.roc_auc_score` bit-for-bit
without taking on sklearn itself).

## [0.4.0] — 2026-05-17

The 0.4.0 release bundles every change archived between 0.3.0
(2026-05-09) and now. The headline item is `pluggable-faithfulness`
— `ForgePipeline.faithfulness` accepts a user-supplied scorer via
the new `FaithfulnessTarget` protocol, and `ForgeResult.faithfulness_kl`
is deprecated in favour of the generic `faithfulness` /
`faithfulness_target_name` pair (one minor-version removal window).

Two follow-up specs land alongside without code:
`world-model-protocol` (the seam for non-transformer host adapters)
is proposed; concrete non-transformer adapters are explicit
follow-ups against it.

Everything below was previously accumulated under `[Unreleased]`
and is now bundled into this release. The default surface stays
byte-identical with v0.3.0 for every non-deprecated call site.

### Added (pluggable-faithfulness)

`ForgePipeline` now accepts an optional `faithfulness` argument
implementing the new `saeforge.eval.faithfulness.FaithfulnessTarget`
protocol. The protocol generalises the loop-gating signal beyond
hard-coded KL: built-in `KLTarget` and `CosineTarget` preserve v0.4
behaviour as family-dispatched defaults, and any user-supplied target
(GT-alignment, probe accuracy, monosemanticity, …) overrides them.

`ForgePipeline(faithfulness=None, ...)` (the default) is byte-identical
to the previous behaviour — the family-based default policy picks
`KLTarget` for LM hosts (`gpt2` / `llama` / `gemma2` / `qwen2` / `qwen3`)
and `CosineTarget` for `whisper_encoder`.

`ForgeResult.faithfulness_kl` is deprecated in favour of two new
fields: `ForgeResult.faithfulness` (the active target's score) and
`ForgeResult.faithfulness_target_name` (the active target's `name`).
The property keeps working for one minor version and emits a
`DeprecationWarning` on read; the constructor still accepts
`faithfulness_kl=` as a kwarg shim that forwards to `faithfulness=` /
`faithfulness_target_name="kl"` (also with `DeprecationWarning`).
Removal is scheduled for the next minor version after this lands.

Migration:

```text
Before (still works, emits DeprecationWarning):
    result = pipeline.run(...)
    print(result.faithfulness_kl)

After (KL default — no code change required):
    result = pipeline.run(...)
    print(result.faithfulness)                       # same value

After (custom target):
    from saeforge.eval.faithfulness import FaithfulnessTarget
    result = ForgePipeline(faithfulness=MyTarget(), ...).run(...)
    print(result.faithfulness, result.faithfulness_target_name)
```

`forge_result.json` gains `faithfulness` and `faithfulness_target_name`
keys alongside the existing `faithfulness_kl` (which is `null` when
the active target is not `"kl"`; removed alongside the property in the
same release).

New artifacts: `saeforge/eval/faithfulness.py::FaithfulnessTarget`,
`saeforge/eval/targets/{kl,cosine,__init__}.py`,
`examples/forge_with_gt_alignment.py`,
`tests/test_faithfulness_target_protocol.py`,
`tests/test_pipeline_with_custom_target.py`,
`tests/test_forge_result_deprecation.py`. Docs:
`docs/finetune-recipe.md` gains a "Swapping the faithfulness target"
section; `docs/advanced-fsm-options.md` documents the `faithfulness`
knob in the basis-loop knobs table.

### Added (fix-scale-boost-calibration — diagnostics-only)

This change started as a `scale_boost="calibrate"` auto-picker. The
2026-05-16 smoke gate falsified the premise — three successive
proxies for the forge's faithfulness KL all picked the wrong
`scale_boost`. The change as merged is **diagnostics-only**: it adds
the surface that explains WHY a sweep produced bad forge KL, without
attempting to fix it automatically. See
`openspec/changes/fix-scale-boost-calibration/design.md` Decision 1
for the full empirical record.

- **Two new `ParetoFrontierRow` diagnostic fields** populated when
  the sweep runs with `--magnitude-diagnostics`:
  - `logit_std_ratio`: forged-logit std ÷ host-logit std on the
    calibration corpus (layer-L shortcut). Diagnoses
    magnitude-matching independently of the forge's `faithfulness_kl`.
  - `top1_anomalous`: mode top-1 prediction in the curated
    SolidGoldMagikarp-family set. Catches the documented "broken
    forge predicts glitch tokens" signature.
  Both default to `None`; forward-compatible with existing readers.
- **`--magnitude-diagnostics VALUE` CLI flag** on `sweep-pareto`.
  Accepts `tokens:N` (built-in token-capped English corpus) or
  `prompts:PATH` (JSONL). Requires `--layer`. Post-sweep advisory
  prints per-row ratios and any anomalous-canary fires.
- **`--rank-monotonicity-check` CLI flag** on `sweep-pareto`.
  Post-sweep advisory (no refusal) that flags adjacent K pairs within
  an encoding whose `faithfulness_kl` rises by more than 0.1 nats —
  the documented blow-up pattern at default `scale_boost=1.0`.
- **`saeforge.calibration` module** exposes the load helpers
  (`load_calibration_corpus`, `load_host_unembed`), the pure-numpy
  diagnostic helpers (`compute_host_logit_std`,
  `compute_forged_logit_std`, `top1_is_anomalous`), and the
  `ANOMALOUS_TOKEN_IDS` per-tokenizer map.
- **README guidance** on `scale_boost` modes (literal / auto only;
  calibrate dropped).

`SubspaceProjector` behaviour is unchanged — only `"auto"` and
literal-float remain. The structural KL blow-up the original proposal
targeted lives in the projected NativeModel's stacked-layer
compounding (not in `scale_boost` magnitude); fixing it is a separate
proposal.

### Added (qwen3-moe-support)

- **Qwen3-MoE architecture adapter** — `Qwen3MoEAdapter` inherits from
  `Qwen3Adapter`, stamping `family="qwen3_moe"` and populating four new
  `NativeModelConfig` MoE fields (`num_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`, `norm_topk_prob`). The shared
  `LlamaAdapter.walk` gains a host-attribute-gated MoE branch
  (`hasattr(block.mlp, "experts")`) that emits the router + per-expert
  SwiGLU keys. The Llama-family factory's `LlamaBlock` constructs
  `Qwen3MoEMLP` (router + expert ModuleList + softmax-then-topk dispatch
  with `index_add_`) when `cfg.num_experts > 0`, else the dense
  `SwiGLU_MLP` (existing behavior). All other families default to
  `num_experts=0`; byte-identical behavior preserved.

- **Two compression strategies via `ForgePipeline.moe_strategy`:**
  - `preserve` (default) — per-expert projection, full fidelity
  - `collapse` — average all experts into a single dense MLP per layer;
    downgrade family to `qwen3`; storage-aggressive, behavior-degraded
  - `top_n` — v1 placeholder; raises `NotImplementedError` pointing at
    the `moe-expert-calibration` follow-up

- **NVIDIA smoke** — `scripts/smoke_qwen3_moe.py` targets a real
  `Qwen/Qwen3-30B-A3B-Base` host on an NVIDIA ≥80GB GPU.

- Requires `transformers >= 4.51`. The `[intel]` extras silently skip
  registration. Synthetic small-MoE adapter tests
  (3 layers × 4 experts × top-2) cover the M4 surface.

### Added (add-auto-materialise-sweep)

- **One-tool Axis-4 workflow.** `sae-forge sweep-pareto --auto-materialise`
  bundles polygram's `BehaviouralValidator → Compressor.plan_pareto →
  apply` into the same invocation, with the
  validation-vs-eval-prompts leakage firewall as a first-class API
  constraint (refused same-path resolution by default;
  `--allow-validation-eval-overlap` surfaces the choice in every
  frontier row's `validation_eval_overlap` field).
- **New CLI flags** on `sweep-pareto`: `--auto-materialise`,
  `--validation-prompts`, `--pareto`, `--layer`,
  `--validation-threshold`, `--validation-jaccard-threshold`,
  `--score-field`, `--rep-selection` (passes polygram 0.5.0's
  `kl_attribution` through), `--encoding-class LABEL:CLASS`
  (repeatable), `--encoding-qubits LABEL:N` (repeatable),
  `--allow-validation-eval-overlap`, `--force-rematerialise`,
  `--plan-only`.
- **`ParetoFrontierRow` gains three methodological provenance
  fields**: `validation_threshold`, `encoding_class`,
  `validation_eval_overlap`. Populated only under
  `--auto-materialise`; default `None`. Backwards-compatible (old
  consumers see null).
- **Cache under `<output-dir>/_materialised/<label>/`**, content-
  addressed via SHA-256 of the SAE checkpoint and validation prompts
  plus the threshold/encoding/layer/targets fields. Reruns with
  unchanged inputs skip the validator + Compressor entirely.
  `--force-rematerialise` is the escape hatch.
- **`--plan-only`**: prints per-encoding cache status
  (`HIT` / `MISS` with diffing-fields), SHA-256 fingerprints,
  target K list, validator-forward-count estimate, then exits 0
  without invoking validator / Compressor / forge. Mutually
  exclusive with `--frontier-only` (different lifecycle stages).
- **`saeforge.auto_materialise` module**: `AutoMaterialiseSpec`
  dataclass, `compute_cache_key`, `is_cache_hit`,
  `materialise()`, `format_plan_only_block`. Numpy-only on the cold
  paths; lazy polygram + transformers imports.
- **CLI refusal behaviour** spelled out in the spec: validator-tuning
  flags require `--auto-materialise`; mixed mode (auto + directory
  encoding paths) refused; same-path validation/eval prompts refused
  unless overridden; unknown encoding class names refused at parse
  time with the supported set listed; `HEA_Rung2` without
  `--encoding-qubits` defaults `n_qubits=3` (polygram default).
- **ClusteredDictionary explicitly excluded.** The supported encoding
  class set is `MPSRung1` / `Rung3` / `Rung4` / `HEA_Rung2` —
  `BehaviouralValidator.__post_init__` requires `.features` access
  that `ClusteredDictionary` doesn't satisfy. For N>8 SAEs, use
  `HEA_Rung2(n_qubits=N)`.

### Added (add-forge-quality-diagnostics)

- **Forge-quality diagnostics on every sweep row.** `ParetoFrontierRow`
  gains four new optional fields populated when the sweep can resolve
  the host's residual stream width:
  - `host_d_model` — `AutoConfig.from_pretrained(host_model_id).hidden_size`
    (config-only fetch; cached once per sweep).
  - `basis_rank` — `numpy.linalg.matrix_rank(W_dec_kept)` for the
    surviving (non-zero) rows of the polygram-compressed SAE.
  - `quality_ratio` — `basis_rank / host_d_model`.
  - `quality_tier` — heuristic four-tier categorical (`saturated` ≥
    1.0, `good` ≥ 0.5, `undersized` ≥ 0.0625, else `degenerate`).
    Tweakable via `--quality-tier-thresholds`.
- **Pre-flight stderr advisory** when any encoding's smallest-K basis
  is in the `undersized` or `degenerate` tier. Names the encoding,
  K, basis_rank, host_d_model, computed ratio, suggested K floor,
  and a fixed clarification sentence: "'degenerate' describes the
  rank ratio, not the validity of the run; exploratory low-rank
  smokes remain valid for impl validation."
- **Opt-in `--quality-floor RATIO`** refuses the sweep before any
  forge call when any encoding's smallest-K ratio falls below the
  floor. Default behaviour is advisory-only.
- **`--quality-tier-thresholds STR`** overrides the heuristic
  boundaries (e.g.,
  `--quality-tier-thresholds saturated:2.0,good:1.0,undersized:0.25`).
  Parser enforces format, name set, and ordering constraint.
- **Diagnostics populated regardless of forge outcome.** Failure
  rows (`error_message` populated) and `--frontier-only` rows both
  carry the four diagnostic fields, so analysts can distinguish
  "forge bug" from "structurally doomed setup" without reading row
  metrics.
- **`QualityTier` and `QualityThresholds` exported from `saeforge`**
  for downstream tooling that wants to consume the schema.
- **Public surface bumped** to include `QualityTier` and
  `QualityThresholds`; backwards-compatible (existing readers see
  `null` for the four new fields).
- **No new dependencies.** Uses the existing `transformers` extra
  for `AutoConfig` (already pulled in by `[torch]`/`[intel]`).
  Failure to resolve `host_d_model` (offline, gated model, non-LM
  host) silently disables diagnostics — the sweep proceeds with
  all four fields as `None` and no advisory printed.

### Added (add-pareto-sweep-driver)

- **Bundled fix: `torch_dtype=` for transformers compat.** Two
  `AutoModelForCausalLM.from_pretrained(..., dtype=...)` call sites
  (`forge.py` `_run_real_imperative` and `_run_real_fsm`) used the
  transformers≥4.50 `dtype=` alias, which doesn't exist on the
  `[intel]` extra's pinned `transformers>=4.46,<4.50`. Switched both
  to `torch_dtype=` — canonical name, works on both pin lines. Caught
  during the live Axis-4 MBP smoke for this PR (latent regression from
  PR #9, surfaced because the sweep is the first user-facing
  multi-row path that triggers `from_pretrained` repeatedly on Intel).
- **Pareto sweep driver.** New `saeforge sweep-pareto` CLI subcommand
  and `ForgePipeline.sweep_pareto()` method that forge across per-K
  materialised SAE checkpoints produced by
  `polygram compress --pareto --pareto-materialize`. Optionally spans
  multiple labelled encodings (e.g. MPS vs Rung4) — pass
  `--encoding LABEL:PATH` repeatedly. Emits one JSONL row per
  `(encoding, target_n_features_kept)` capturing kept-feature count,
  downstream KL, perplexity, fine-tune loss, and elapsed seconds.
  The load-bearing primitive for Axis 4 of polygram's rung-viability
  methodology — end-to-end downstream confirmation that the Axis 1
  compression-coverage lift cashes out in forged-model KL space.
- **Three lifecycle states per row.** *Success* (forge ran),
  *frontier-only* (`--frontier-only` flag, no forge), and
  *row failure* (forge raised). Downstream consumers filter on
  `error_message is None` before reading metric fields. Failure rows
  are recorded with `error_message` populated; the sweep continues to
  the next row.
- **Resumable.** `frontier.jsonl` is append-only; rerunning the sweep
  skips already-completed `(label, K)` pairs. Truncated last lines
  (mid-write crashes) are detected, dropped, and rewritten on the
  next invocation. No lockfiles or sentinel files.
- **`--frontier-only` mode** emits manifest-derived columns only
  (`target_n_features_kept`, `n_features_kept_actual`,
  `pareto_reached_target`) without invoking the forge — cheap
  exploratory triage. Pipe through `jq` to find candidate K values
  before committing forge compute. Falls back to non-zero-row counting
  on the SAE checkpoint when `pareto.json` is absent.
- **`ParetoFrontierRow` dataclass** exported from `saeforge`, with
  `to_json_dict` / `from_json_dict` round-trip. Schema documented in
  the `pareto-sweep` capability spec.
- **Polygram pin bumped to `>=0.4.0`.** The new
  `CompressionConfig.target_n_features_kept` and `score_field` fields
  flow through the existing `_ConfigMixin.to_dict/from_dict` ctx
  round-trip in `polygram-tuning-passthrough` with no sae-forge-side
  code change — `Compressor` dispatches to `plan_with_target` when
  the field is set.
- **No FSM change.** The sweep is a flat Python loop; each row's
  forge call uses the existing `StreamMachine → RefineMachine →
  BasisMachine` hierarchy. The driver hot-swaps `pipeline.basis` and
  `pipeline.projector` per row via a context manager that restores
  the originals afterwards.
- Tests: `tests/test_sweep.py` — 27 tests covering row validation +
  JSON round-trip, manifest parsing, checkpoint enumeration (both
  `pareto/` subdir and flat layouts), multi-K sweep, resumability,
  multi-encoding, per-row failure isolation, retry-on-next-sweep,
  frontier-only with and without manifest, CLI argument parsing,
  and a `--frontier-only` end-to-end CLI smoke.

### Added (add-host-distillation-finetune-loss)

- **Host distillation in fine-tune.** `TrainingConfig` gains
  `distill_alpha` (default 1.0 = pure LM-CE, byte-identical to
  v0.3) and `distill_temperature` (default 2.0). When
  `distill_alpha < 1.0`, the loss becomes
  `α·CE(corpus) + (1-α)·τ²·KL(host ‖ forged)` — Hinton-style
  soft-label distillation with the same KL direction as
  `faithfulness_kl` (so the training objective matches the eval
  metric). The host forward runs under `torch.no_grad()` in the
  same autocast context as the student.
- **`ForgePipeline` exposes the same knobs** as
  `finetune_distill_alpha` / `finetune_distill_temperature`,
  threading them into the per-step `TrainingConfig` via the
  existing ctx-build path.
- **`α=1.0` is zero-cost.** When `distill_alpha >= 1.0` the host
  forward is skipped entirely; pre-change pipeline tests
  pass unchanged.
- **`run_finetune` rejects `host=None` + `α<1.0` at the top of
  the function** before any batches are consumed, so the
  misconfiguration can't waste work.
- Docs: new "Host distillation" section in
  `docs/finetune-recipe.md`. Tests:
  `tests/test_distillation.py` (14 tests covering field
  validation, byte-identity at `α=1.0`, gradient-flow at
  `α=0.5`, host-unchanged invariant, `α=0.0` pure-KD path,
  pipeline kwargs plumbing).

### Added (forge-whisper-encoder)

- **Whisper-encoder forging — first non-causal-LM architecture in the
  registry.** New `WhisperEncoderAdapter` walks the encoder of either
  `WhisperForConditionalGeneration` or `WhisperModel` into the
  projected weight dict the matching native module consumes. The
  decoder is out of scope for v0.4 (tracked as `forge-whisper-decoder`).
- **`ForgedWhisperEncoder` native module.** Pre-LN block layout
  matching HF Whisper, GELU MLP, MHA (no GQA). The conv stem
  (`conv1`/`conv2`) and `embed_positions` are frozen-copied from the
  host bit-for-bit — ε_conv accounting per `docs/algorithm.md` §10.5.
  A `basis_encode` buffer carries the d → f bridge
  (`projector.basis.pseudoinverse() * scale_boost`) at the conv-stem
  → first-block boundary; state-dict-resident but not a parameter, so
  the no-randomly-initialised-weights invariant applies cleanly.
- **`NativeModelConfig.output_kind`** — new field, defaults to
  `"logits"`. Accepts `"encoder_states"` for the Whisper-encoder
  family. `vocab_size` now defaults to `0` and is gated by
  `output_kind`. Cross-constraints enforced at construction. Existing
  LM callers see byte-identical behaviour.
- **`saeforge.audio_eval.cosine_faithfulness`** — per-frame cosine
  similarity between forged encoder states and host states projected
  through the forge's own `basis_encode` buffer. Optional
  `precomputed_host_states` kwarg skips the host forward when the FSM
  has pre-captured states.
- **Family-aware `evaluate_faithfulness` dispatch.** LM families go
  through `_kl_from_input_ids` verbatim (FSM byte-equivalence net
  green); `whisper_encoder` goes through `cosine_faithfulness`. The
  `faithfulness` ctx field carries the family-appropriate scalar;
  `perplexity` carries `1 - cosine` for encoder so the existing
  `perplexity < best_perplexity` progress check keeps the right
  direction. `min_faithfulness` is reinterpreted per family (KL
  negation for LM; positive cosine threshold for encoder).
- **`ForgePipeline.eval_audio_features` and `eval_encoder_states`.**
  Pipeline-level fields plumbed through `_build_fsm_ctx`. Mutually
  exclusive with `eval_prompts` at construction. The
  `eval_encoder_states` field is the audio-side analog of pre-
  tokenised `_eval_input_ids` — when set, the host forward is
  skipped inside the FSM.
- **`saeforge.audio_data.synthetic_mel_features`** — pure-numpy
  sine-sweep + Gaussian noise synthesiser producing
  `(batch, 80, n_frames)` tensors shaped like Whisper input. Used
  by the synthetic example + tests; no `[audio]` extra required.
- **`sae-forge forge --audio-features-path FILE.pt`** — CLI flag,
  argparse-level mutually exclusive with `--eval-prompts`. Loads a
  `torch.save`'d tensor and passes it through to
  `ForgePipeline.eval_audio_features`.
- **`[audio]` pyproject extra** pinning `librosa>=0.10`. Optional —
  only the real-audio `.wav`/`.flac` mel-extraction path needs it.
  Added to `[all]`.
- **New examples and docs.** `examples/forge_whisper_synthetic.py`
  runs the full pipeline on a tiny synthetic Whisper without HF
  download or audio files. `docs/audio-forge.md` is the user-facing
  reference; `docs/algorithm.md` §10.5 documents the algorithmic
  surface (output_kind, vocab_size=0, the d→f bridge, ε_conv).
- **Spec correction in the same change.** The architecture-adapters
  spec delta for Whisper originally listed q/k/v_proj.weight as
  `(f, d)` and out_proj.weight as `(d, f)`; under HF
  `nn.Linear (out, in)` convention these need to be `(d, f)` and
  `(f, d)` respectively. The `(d,)` `q_proj.bias` alongside the
  original `(f, d)` `q_proj.weight` was self-inconsistent (Linear
  bias must match the first weight axis). Spec now matches the
  implementation and HF convention.

### Added (qwen3-dense-support)

- **Qwen3 dense architecture adapter.** `Qwen3Adapter` inherits from
  `Qwen2Adapter` and stamps `family="qwen3"` + auto-detects the
  per-head Q/K RMSNorm (`qk_norm=True`). The shared `LlamaAdapter.walk`
  now emits `q_norm`/`k_norm` weights as head-dim-aligned pass-through
  whenever the host has those submodules (host-attribute-gated, no-ops
  for Llama / Gemma-2 / Qwen2). The Llama-family `LlamaSelfAttention`
  conditionally constructs `RMSNorm(head_dim)` on Q and K when
  `cfg.qk_norm=True` and applies them between projection-reshape and
  SDPA. Qwen3 inherits hybrid-bridge support automatically via the
  shared `build_llama_family_module` factory. Requires
  `transformers >= 4.51`; the `[intel]` extra is capped at `<4.50` and
  silently skips Qwen3 registration.

### Added (hybrid-bridge-llama-family)

- **Hybrid-bridge insertion into the Llama-family native module
  forward path.** `LlamaTransformer` now constructs `BridgeModule`
  instances when `cfg.bridges=True` and applies them at block indices
  `0` and `L-2` in its per-block loop, mirroring the GPT-2 wiring.
  Closes the half-built state shipped in #18 where `hybrid_bridge=True`
  on a Llama / Gemma-2 / Qwen2 host accepted the flag, projected the
  weights through three bases, and then silently dropped the bridges
  on the forward pass. Llama, Gemma-2, and Qwen2 hybrid forges now
  work end-to-end. Default-off behavior is byte-identical to today.
### Added (adaptive-regrow)

- **Adaptive regrow controller** in `BasisMachine`. Opt-in via
  `--adaptive-regrow` (or `ForgePipeline(adaptive_regrow=True)`).
  Consumes the polygram-side `n_features_kept` signal and grows the
  basis toward `--n-features-target`, bounded by
  `[regrow_count, regrow_max]` and damped by `--regrow-damping`.
  Defaults preserve byte-equivalence with the v0.2 fixed-regrow path
  (the master toggle is off by default; the byte-equivalence gate
  continues to pass unmodified).
- `saeforge.basis.RegrowController.next_count(...)` — deterministic
  pure-function controller; testable in isolation.
- `saeforge.actions.adapt_and_regrow` — composed action that wraps
  `perform_regrowth` with the controller. Short-circuits to
  `perform_regrowth` under disabled / cold-start, so v0.2 behavior is
  bit-for-bit identical.
- Four new CLI flags on `sae-forge forge`: `--adaptive-regrow`,
  `--regrow-max`, `--n-features-target`, `--regrow-damping`.
- Four new `ForgePipeline` fields: `adaptive_regrow`, `regrow_max`,
  `n_features_target`, `regrow_damping`. Validated in
  `__post_init__` when the master toggle is on (require
  `regrow_max > regrow_count` AND `n_features_target > 0`); silently
  inert otherwise.

### Changed (adaptive-regrow)

- `BasisMachine`'s `compressed → regrown` transition action renames
  from `perform_regrowth` to `adapt_and_regrow`. State set,
  transition graph, and guard expressions are unchanged — the
  topology test (`tests/fsm/test_topology.py`) continues to pass.
  The committed Mermaid diagram in `docs/advanced-fsm-options.md`
  regenerates with one label change.
- `transitions_log` schema is additive — under
  `adaptive_regrow=True`, each regrow cycle gains one extra entry
  (`adapt_regrow_count`) before the existing `perform_regrowth`
  entry. Under `adaptive_regrow=False` the log shape is byte-identical
  to v0.2.

### Changed (hierarchical-fsm)

- **FSM refactored into a three-machine hierarchy** —
  `saeforge/machines/sae_forge.orca.md` (the v0.2 flat ten-state
  machine) is replaced with three composed sub-machines under the
  same directory: `stream.orca.md` (outermost, shard handling),
  `refine.orca.md` (middle, per-shard convergence), and
  `basis.orca.md` (innermost, compress/regrow loop). Composition
  uses `orca_runtime_python`'s native `- invoke:` directive +
  `parse_orca_md_multi`. Internal-only refactor: no public API,
  CLI, on-disk artifact, or runtime-behavior change. The
  byte-equivalence acceptance gate
  (`test_imperative_and_fsm_byte_equivalent`) is green.
- `transitions_log` entries gain a `machine_path` field
  (`"stream"` / `"stream/refine"` / `"stream/refine/basis"`) for
  debugging — additive; existing readers that ignore unknown keys
  are unaffected.
- Failure propagation records a new `error_origin_machine` ctx
  field (deepest origin wins) alongside the unchanged
  `error_message` — additive; the byte-equivalence test filters it.

### Added (hierarchical-fsm)

- `saeforge.machines.visualize.to_mermaid` — auto-generates a
  `stateDiagram-v2` block from the parsed hierarchy. Embedded in
  `docs/advanced-fsm-options.md`; `tests/fsm/test_diagram_drift.py`
  asserts the doc matches the live emit so drift can't land.
- `sae-forge inspect --fsm-diagram` — CLI flag that emits the
  Mermaid diagram to stdout. Mutually exclusive with the
  `checkpoint` positional argument.
- `tests/fsm/` test package with sub-machine topology checks,
  multi-shard hierarchy integration, the runtime compound-state
  probe, and the diagram-drift gate.

### Fixed (hierarchical-fsm)

- `saeforge.actions.scan_activations` referenced a non-existent
  `basis.directions` attribute on the `protect_top_k > 0` path
  (the attribute is `basis.W_dec`). Surfaced by the new
  `tests/fsm/test_load_and_scan_ordering.py` — the only test that
  exercises this path with a real basis. One-line correction.

## [0.3.0] — 2026-05-09

### Added (forge-continual-learning-loop)

- **Three-loop FSM topology** ([PR #11](../../pull/11)) layered on top
  of the v0.1 single-shard pipeline:
  - **Stream loop** — `evaluated → loaded` re-entry to consume the
    next shard. Triggered by `task_trigger` (one of `labeled` /
    `token_budget` / `loss_delta`).
  - **Refine loop** — preserved v0.1 `evaluated → compressed`
    re-entry for same-shard convergence.
  - **Basis loop** — new `compressed ↔ regrown` self-loop for
    `inner_refine_passes` rounds before exiting to `projected`.
- **New `activations_scanned` state** between `loaded` and
  `compressed`, hosting the `scan_activations` action that scores
  features and selects a protected set when `protect_top_k > 0`. True
  no-op (no basis load, no torch import) under the v0.2.0-default
  `protect_top_k = 0`.
- **Protected features** — structural EWC analogue at the basis
  level. `compress_with_polygram` post-filters the
  `ValidationReport` so protected indices cannot be merged or
  removed by Polygram's Compressor. The do-not-remove kwarg is the
  preferred long-term path; the workaround is documented in
  `tasks.md` §10.4 and tracked for upstreaming.
- **Replay buffer + MixedIterator** — new `saeforge.training.replay`
  module exposing `ReplayBuffer` (three policies: `reservoir` /
  `recent_window` / `per_task`) and `MixedIterator` with
  deterministic 100-cycle replay scheduling. Pure Python, no torch
  dependency at module import.
- **TaskStream abstraction** — new `saeforge.training.task_stream`
  module with `LabeledTaskStream`, `TokenBudgetTaskStream`,
  `LossDriftTaskStream`, plus a process-local registry mapping
  ``task_iterator_id`` strings to live stream instances.
- **12 new `ForgePipeline` fields**: `n_tasks`, `task_trigger`,
  `token_budget_per_task`, `loss_delta_threshold`,
  `inner_refine_passes`, `protect_top_k`, `protect_score`,
  `activation_buffer_size`, `replay_ratio`, `replay_buffer_size`,
  `replay_policy`, `task_stream`. All default to v0.1-equivalent
  values.
- **Construction-time validation** for the new continual-learning
  knobs — invalid combinations (e.g. `replay_ratio > 0` with
  `replay_buffer_size = 0`, or `replay_policy="per_task"` with
  `task_trigger != "labeled"`) raise `ValueError` at
  `ForgePipeline(...)` time, not at run.
- **`docs/advanced-fsm-options.md`** — user-facing reference covering
  the three-loop topology, every new context field, every new CLI
  flag, the three `task_trigger` modes, the three `protect_score`
  strategies, the three `replay_policy` strategies, plus a worked
  recipe per pattern (per-task / protected-features / drift-triggered).
- **24 new tests** — `tests/test_replay_buffer.py` (11),
  `tests/test_task_stream.py` (7), and
  `tests/test_continual_learning_loop.py` (6 stub-driven FSM-level
  tests covering basis-loop / stream-loop / refine-loop preservation
  / stream-dominance contract).

### Changed (forge-continual-learning-loop)

- **FSM uses orca-runtime-python rich guard grammar directly**.
  `refine_same_shard` is now the orca expression
  `ctx.advance_stream == false and ctx.should_continue == true`
  evaluated by the runtime; previously the v0.1 design called for
  precomputing flat-bool flags in Python actions. Three ctx fields
  (`next_basis_step`, `refine_same_shard`, `terminate_run`) and the
  hardcoded `_NEXT_EVENT_FOR_STATE` map are gone — the runtime and
  the parsed `MachineDef.transitions` are now the source of truth
  for control flow.
- **Machine state count: 9 → 10** (added `activations_scanned`).
  Updated `test_machine_loads_and_has_nine_states` →
  `test_machine_loads_and_has_ten_states` per the spec's MODIFIED
  requirement.
- **`README.md`** — Status section now lists the recent landed
  openspec changes; new "Continual learning" Quickstart subsection
  shows the knobs + `LabeledTaskStream` wiring; ambiguous v0.x
  version labels dropped from the How-it-works callouts.
- **`AGENTS.md`** — orca-lang dependency contract section updated
  to document the rich-guard pattern and link to the
  continual-learning advanced-options doc.

### Backwards compatibility

- **No breaking changes.** Defaults preserve v0.1 byte-identical
  behavior. The `test_imperative_and_fsm_byte_equivalent` safety net
  passes unchanged. All 20 existing FSM tests pass.

### Out of scope (deferred follow-ups)

- True activation-driven `protect_score` (current 0.3.0 ships a
  direction-L2 stub; activation-driven scoring needs host-model
  residual capture).
- Polygram `do_not_remove` kwarg upstream — the
  `ValidationReport` post-filter is the workaround until then.
- Per-loop-level scan tuning, feature-axis sampling, raw trigger
  signal exposure in ctx, basis-size growth across tasks, per-task
  evaluation matrix, token-level replay buffer, and CLI flags for
  the new continual knobs are tracked in
  `openspec/changes/forge-continual-learning-loop/tasks.md` §12.

## [0.2.4] — 2026-05-07

### Added

- **`SubspaceProjector(scale_boost="auto")`** ([PR #8](../../pull/8))
  resolves to `min(1.0, d_model / n_features)` — a defensible
  starting heuristic for over-complete bases (`n_features > d_model`).
  For under/equal-complete bases the heuristic returns `1.0`
  (identity-preserving). Existing positive-float values are
  unchanged; the default remains `1.0`.
- **`--scale-boost` CLI flag** on `examples/forge_gemma2_2b.py` and
  `examples/forge_synthetic_llama.py` (both default to `"auto"`).
  `examples/forge_gpt2_real_sae.py` adds a `scale_boost` function
  parameter (notebook-driven, no argparse).

### Fixed

- **Silent footgun on over-complete bases** ([PR #8](../../pull/8)).
  Empirical anchor surfaced during a Gemma-2-2B forge attempt:
  GPT-2 (`d_model=768`) with a 1024-feature basis required
  `scale_boost ≈ 0.25` for stable training; the default `1.0` was
  too large and silently produced NaNs / saturated softmax / KL
  explosion. Construction now emits a `UserWarning` when
  `n_features > d_model` and `scale_boost == 1.0`, naming the
  empirical anchor and pointing at `"auto"` or a hand-picked
  value as the next step. Suppressed when an explicit numeric
  or `"auto"` is supplied — no scolding when the user acted
  intentionally.

## [0.2.3] — 2026-05-07

### Fixed

- **Grad checkpointing crashed on Llama / Gemma-2 hosts**
  ([PR #7](../../pull/7)). `saeforge/training/loop.py:_enable_grad_checkpointing`
  hardcoded GPT-2 submodule names (`module.transformer.h`,
  `module.transformer.wte.weight`); ForgedLlama (used by both
  `family="llama"` and `family="gemma2"`) exposes
  `module.model.layers` and `module.model.embed_tokens.weight`. Any
  `--grad-checkpoint` run on a non-GPT-2 host raised
  `'ForgedLlama' object has no attribute 'transformer'` inside the
  FSM. Fix: adapter-driven layout via a new
  `ArchitectureAdapter.grad_checkpoint_targets(module)` method with
  per-family overrides; `_enable_grad_checkpointing` dispatches via
  a new `adapter_for_family(family_str)` registry helper.

- **FSM failures surfaced as silent KL=0.0 returns**
  ([PR #7](../../pull/7)). When an FSM action raised, the failure was
  swallowed into `final_state: "failed"` and `ForgePipeline.run()`
  returned a `ForgeResult` with `n_params=0`, `faithfulness_kl=0.0`,
  exit code 0 — no diagnostic signal. Fix: new
  `saeforge.ForgeFailed` exception (subclass of `RuntimeError`) with
  `error_message`, `transitions_log`, and `extras` attached; both
  FSM dispatch paths (`_run_real_fsm`, `_run_synthetic_fsm`) raise
  it after `run_machine` when the trailing transition is `log_error`.

### Added

- **`saeforge.ForgeFailed`** exception ([PR #7](../../pull/7)) —
  re-exported from the top-level package; subclass of `RuntimeError`
  so existing exception handlers don't change shape.
- **`saeforge.adapters.adapter_for_family(family_str)`** helper —
  for code paths that have only the `NativeModelConfig.family`
  string in hand (e.g. inside the training loop, where the host
  class is already gone).
- **`ArchitectureAdapter.grad_checkpoint_targets(module)`** —
  abstract-with-default-NotImplementedError on the ABC; per-family
  overrides return `(blocks, embedding_param)` for activation
  checkpointing.

## [0.2.2] — 2026-05-07

### Fixed

- **Fine-tune recipe now runs on real-host `ForgePipeline.run()`**
  ([PR #6](../../pull/6)). Since v0.3 forge-finetune-recipe landed,
  `run()` against a real HF host (`host_model_id` set) silently
  dropped every `finetune_*` field on the floor — the recipe was
  wired into the FSM action only, but `run()` always took the
  imperative path. The headline `examples/forge_gemma2_2b.py`
  documented a 1k-step fine-tune flow that had never executed.
  `run()` now branches on `self.orchestrator`:
  `"fsm"` routes through a new `_run_real_fsm` mirroring the
  synthetic FSM path; `"imperative"` (the default) emits a
  `UserWarning` when `finetune_corpus` is set so the silent skip
  cannot recur. `examples/forge_gemma2_2b.py` sets
  `orchestrator="fsm"` when `--steps > 0`.

### Added

- **`ForgePipeline.run(finetune_iterator=...)`** ([PR #6](../../pull/6))
  — pre-built iterator bypasses the `AutoTokenizer + datasets`
  round-trip the recipe action would do via `finetune_corpus`.
  Mirrors the existing `run_synthetic` kwarg.

## [0.2.1] — 2026-05-07

### Fixed

- **`NativeModel.save_pretrained` / `load_pretrained` round-trip on
  tied-embedding hosts** ([PR #5](../../pull/5)). The `ForgedLlama`
  constructor aliases `lm_head.weight` to `model.embed_tokens.weight`
  when `config.tied_embeddings` is True (Gemma-2 default + tied
  Llama configs), but `safetensors.torch.save_file` rejects
  shared-storage tensors. The fix drops `lm_head.weight` from the
  saved state_dict when tied; `load_pretrained` reconstructs the
  alias via the constructor and relaxes `load_state_dict(strict=False)`
  for the missing slot. Without this fix the Gemma-2-2B forge crashed
  at stage 4 save, after polygram + projection had already succeeded.

- **`examples/forge_gemma2_2b.py` SAE filename templating**
  ([PR #5](../../pull/5)). The previous hard-coded `average_l0_71`
  doesn't exist for layer 12 of `google/gemma-scope-2b-pt-res`
  (layer 12 publishes `{22, 41, 82, 176, 445}`). New `--l0` flag
  (default 82) templates into the `SAE_FILE_TEMPLATE` path.

## [0.2.0] — 2026-05-07

### Added (multi-architecture-support)

- **`saeforge/adapters/` package** — registry-based dispatch from HF
  model class to a `ArchitectureAdapter` whose contract is `walk` +
  `build_native_config` + `native_module_class`. Bundled adapters cover
  `GPT2LMHeadModel`, `GPT2Model`, `LlamaForCausalLM`, and
  `Gemma2ForCausalLM`. Unregistered architectures raise
  `NotImplementedError` naming the offending type and the registered
  set.
- **Llama-3 / Llama-2 support** — Q/K/V/O proj, SwiGLU MLP
  (gate/up/down), GQA via `num_key_value_heads`, RMSNorm γ, optional
  tied embeddings.
- **Gemma-2 support** — Llama-shaped + the two extra per-layer
  RMSNorms (`pre_feedforward_layernorm`, `post_feedforward_layernorm`)
  and post-`lm_head` `tanh(x / cap) * cap` soft-capping. Sliding-window
  alternating attention is NOT replicated in v0.2 (accepted as
  `ε_attn` per `docs/algorithm.md` §5).
- **`examples/forge_synthetic_llama.py`** — runs the full Llama
  forge pipeline against a tiny synthetic host with no HF token
  requirement; useful for CI and laptops.
- **Tests** — 22 new tests in `tests/test_architecture_adapters.py`
  covering registry dispatch, walker shape audits (incl. GQA), tied
  embeddings, four-norm Gemma-2 layout, soft-cap config passthrough,
  the no-randomly-initialised-weight invariant, and family-field
  validation. Plus `test_examples_smoke.py` (synthetic-Llama
  end-to-end smoke + Gemma-2 skip-if-unreachable) and
  `test_forge_pipeline_unregistered_arch.py`.

### Changed (Breaking — multi-architecture-support)

- **`NativeModelConfig.family: str` is now required** with no default.
  Valid values are `"gpt2"`, `"llama"`, `"gemma2"`. The pre-change
  config silently produced a GPT-2-shaped module for any inputs;
  forcing an explicit family removes the silent footgun. Callers
  migrate by adding `family="gpt2"` to existing `NativeModelConfig(...)`
  calls.
- **`NativeModelConfig` gains `n_kv_heads`, `tied_embeddings`,
  `rms_norm_eps`, `final_logit_softcap`, `attn_logit_softcap`** for
  the Llama / Gemma-2 paths. Defaults preserve the GPT-2 behaviour
  (`n_kv_heads=None` collapses to `num_heads` at `__post_init__`;
  the soft-caps default to `None` and are no-ops).
- **`SubspaceProjector.project_module`** now dispatches via the
  adapter registry instead of a hard-coded GPT-2 walker. The GPT-2
  walk semantics are unchanged. Unregistered architectures raise a
  registry-aware `NotImplementedError`; the v0.1 `"GPT-2"`-prefixed
  message is gone.
- **`ForgePipeline.run`** loads the host via
  `transformers.AutoModelForCausalLM.from_pretrained` (was
  `GPT2LMHeadModel.from_pretrained`). Non-GPT-2 hosts now load as
  their actual class — the pre-change path silently produced a
  randomly-initialised GPT-2 for any non-GPT-2 host and is the bug
  this change fixes.

### Out of scope

- **Pythia / GPT-NeoX** — deferred; needs a parallel Q/K/V upstream
  addition in polygram.
- **Gemma-2 sliding-window alternating attention** — replicating the
  exact attention pattern is future work; the native module uses the
  standard causal mask everywhere.

## [0.1.0] — 2026-05-07

### Added (forge-polygram-tuning-passthrough)

- Three typed polygram-tuning fields on `ForgePipeline`:
  `compression: CompressionConfig | None`,
  `epoch_compression: EpochCompressionConfig | None`,
  `regrow: RegrowConfig | None`. Each round-trips through the FSM
  context as a JSON-friendly dict (`cfg.to_dict()` →
  `<Config>.from_dict(ctx[key])`).
- `ForgePipeline.from_dict(data)` classmethod for YAML/JSON config
  loading; emits `UserWarning` for unknown top-level keys.
- New CLI flags: `--coverage-target`, `--cosine-threshold`,
  `--max-compress-iterations`, `--regrow-count`, `--regrow-layer`,
  `--regrow-strategy`. Long-tail tuning lives behind
  `ForgePipeline.from_dict`.
- `docs/forge_config_example.yaml` showing the
  `ForgePipeline.from_dict(yaml.safe_load(...))` shape end-to-end.
- `tests/test_polygram_tuning_passthrough.py` (15 tests) and
  `tests/test_cli.py` (5 tests).

### Changed (Breaking — forge-polygram-tuning-passthrough)

- **Removed flat `compression_strategy` and `rep_selection` fields
  on `ForgePipeline`.** Passing either now raises `TypeError` at
  construction. Migrate to
  `compression=CompressionConfig(strategy=..., rep_selection=...)`.
- **`regrow_count > 0` requires explicit `regrow=RegrowConfig(...)`.**
  `__post_init__` raises `ValueError` otherwise.
- **`perform_regrowth` action requires `ctx["regrow"]`** when
  `regrow_count > 0`. The previous `ctx.get("regrow_layer", 10)` and
  `ctx.get("host_model_id") or "gpt2"` fallbacks were removed in
  lock-step with polygram 0.1.0 dropping the matching defaults from
  `Regrower.from_compression_report`.
- Pinned `polygram>=0.1.0` (was `>=0.0.1`).

### Migration

- Replace `ForgePipeline(compression_strategy="merge",
  rep_selection="scale_aware", ...)` with
  `ForgePipeline(compression=CompressionConfig(strategy="merge",
  rep_selection="scale_aware"), ...)`.
- Callers with `regrow_count > 0` now must pass
  `regrow=RegrowConfig(model_name=<host>, layer=<int>)`. Layer is
  host-specific and no longer has a GPT-2 default.

### Internal

- Two pre-existing CI tests (`test_forge_pipeline_run_requires_host_model_id`,
  `test_project_module_unsupported_arch_raises`) gated with
  `pytest.importorskip("torch")` so the no-extras CI install stays
  green.

### Added

- Repository scaffolding: `pyproject.toml`, `README.md`, `AGENTS.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, CI workflow,
  `saeforge/` package skeleton with stub `FeatureBasis`,
  `SubspaceProjector`, `NativeModel`, `ForgePipeline`, and `cli.main`.
- OpenSpec change `bootstrap-package` defining the v0 milestone.
