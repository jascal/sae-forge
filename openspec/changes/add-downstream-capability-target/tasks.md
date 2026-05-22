# Implementation tasks

## 0. Design pre-locks (blocking)

- [ ] 0.1 Confirm `FaithfulnessTarget` protocol's `score(*, forged, host, ctx)` signature is sufficient for the new target. The downstream-encoder + labels are constructor args on `DownstreamCapabilityTarget`, not ctx fields — keeps the protocol surface unchanged. Re-affirm by reading `saeforge/eval/faithfulness.py:55-60`.
- [ ] 0.2 Pipe `ForgeResult.basis` into `ctx["basis"]` at `_score_faithfulness_imperative` time (the path (a) free `W_dec` source — Decision 2). Existing target callers don't read `ctx["basis"]`, so this is a one-line additive change.
- [ ] 0.3 Lock the `(score, perplexity_analog)` return convention: score is mean(max_over_features(AUC)) like `GroundTruthTarget`; perplexity is `max(0, 1 - score)` for `better_when="higher"` per the protocol docstring.
- [ ] 0.4 Decide encoder calling convention: encoder is `Callable[[torch.Tensor], torch.Tensor]` returning latents only (no reconstruction tuple). Bio-sae's `_ReferenceSAE.forward` returns `(recon, z)` — adapter helper `lambda x: encoder(x)[1]` documented in the target's docstring for users who pass an nn.Module SAE.

## 1. `saeforge/eval/targets/downstream_capability.py` — new built-in target

- [ ] 1.1 New module `saeforge/eval/targets/downstream_capability.py`. Lazy-imports torch via `require_extra`. No transformers import at module scope.
- [ ] 1.2 `DownstreamCapabilityTarget(FaithfulnessTarget)` class:
  - `__init__(*, encoder, labels, aggregator="pool_then_encode", min_prevalence=0, decode_via_basis=True)`.
  - Construction-time validation: labels is 2D non-empty, aggregator in `{"pool_then_encode", "encode_then_pool"}` or callable, encoder is callable.
  - `name = "downstream_capability"`; `better_when = "higher"`.
- [ ] 1.3 `score(*, forged, host, ctx)` method:
  - Pull `_eval_input_ids` from ctx (required; raises `KeyError` with the same message style as `CosineTarget` / `TokenCosineTarget`).
  - Tokenizer-driven re-extraction is NOT done here — eval prompts arrive already tokenised, matching every other built-in target.
  - For each input row: forged forward → strip bookkeeping tokens → decode via `forged_module.basis_encode` buffer (esm2 / whisper_encoder families that emit it) OR via passed-in `basis.W_dec` when not buffered.
  - Apply aggregator: `pool_then_encode` (mean over residues, then encoder) or `encode_then_pool` (encoder per residue, then mean).
  - Score: chunked Mann-Whitney AUC matmul (same as `GroundTruthTarget`); prevalence filter via `min_prevalence`.
  - Return `(mean_best_auc, max(0, 1 - mean_best_auc))`.
- [ ] 1.4 Unit tests at `tests/test_downstream_capability_target.py`:
  - Identity-encoder + identity-basis: target returns 1.0 (perfect retention) on a fixture with deterministic labels.
  - Encoder that returns zeros: target returns 0.5 (chance — symmetric AUC bottoms at 0.5).
  - Aggregator dispatch: pool_then_encode and encode_then_pool produce different scores on a fixture where they disagree (bio-sae's data shows them differing by ~3 mAUC points on the pooled SAE).
  - Construction-time validation: raises on bad labels shape, bad aggregator string, etc.
- [ ] 1.5 Export `DownstreamCapabilityTarget` from `saeforge.eval.targets.__init__` and `saeforge.__init__`.

## 1.6 Adapter-side: emit `basis_decode` buffer on encoder-only families

- [ ] 1.6.1 `saeforge/adapters/esm2.py:Esm2Adapter.walk()`: emit a `basis_decode` key alongside `basis_encode`, value = `basis.W_dec` (shape `(n_features, d_model)`). One additional line under the existing `basis_encode` emission.
- [ ] 1.6.2 `saeforge/adapters/esm2.py:ForgedEsm2.__init__`: register a `basis_decode` non-parameter buffer (default-init to zeros, populated by `from_projected_weights` from the walk's emission). Same pattern as `basis_encode`.
- [ ] 1.6.3 `saeforge/adapters/whisper.py:WhisperEncoderAdapter.walk()` + `ForgedWhisperEncoder.__init__`: same two-line additions. Whisper-encoder is the second encoder-only family covered by the v1 spec.
- [ ] 1.6.4 Test: round-trip a forge through `save_pretrained` / `load_pretrained`; assert `basis_decode` matches the input basis's `W_dec` to fp precision.
- [ ] 1.6.5 Test: `DownstreamCapabilityTarget` on an esm2 / whisper forge follows path (b) (`forged_module.basis_decode`) and does NOT call `pinv` (assert via monkeypatch on `numpy.linalg.pinv`).

## 2. `saeforge/datasets/capability.py` — `CapabilityDataset` + bio-sae loader

- [ ] 2.1 New package `saeforge/datasets/` (the first dedicated dataset surface). `__init__.py` exports `CapabilityDataset`.
- [ ] 2.2 `CapabilityDataset` dataclass (frozen). Fields per proposal §3: `sequences`, `labels`, `encoder`, `tokenizer_id`, `aggregator`, `min_prevalence`, `decode_via_basis`.
- [ ] 2.3 `CapabilityDataset.from_bio_sae(run_dir, bundle_path, sequences_path, *, feed="pooled", n_proteins=None, max_seq_len=512)` constructor:
  - Loads bio-sae's `sae.pt` → `_ReferenceSAE`-shaped state dict → wraps the encoder.weight + encoder.bias + topk dispatch as a callable.
  - Loads the bundle's `labels_protein_Y` (pooled feed) or `labels_residue_Y` (residue feed) + per-protein sequences from the parquet.
  - Returns a `CapabilityDataset`. No dependency on bio-sae the package; this is a pure-data constructor.
  - This constructor lives in sae-forge so sae-forge stays self-contained; bio-sae imports it back, not vice versa.
- [ ] 2.4 Tests:
  - Round-trip: build dataset from a tiny synthetic bundle fixture; assert `sequences` / `labels.shape` / `encoder.callable` invariants.
  - Feed dispatch: `feed="pooled"` and `feed="residue"` produce datasets with the right label-matrix shapes.

## 3. `saeforge/sweep.py` — extend `ParetoFrontierRow` + add `sweep_pareto_capability`

- [ ] 3.1 Add optional fields to `ParetoFrontierRow` (all default `None`; serialisation back-compat with the existing v0.7 schema — pre-change rows lacking these fields load unchanged):
  - `host_baseline_mauc: float | None`
  - `host_baseline_cov95: float | None`
  - `forge_mauc: float | None`
  - `forge_cov95: float | None`
  - `retained_mauc_vs_host: float | None`
  - `retained_cov95_vs_host: float | None`
  - `gap_median: float | None`, `gap_p25`, `gap_p75`, `gap_p95: float | None`
  - `n_features_gap_above_0_1: int | None`
  - `n_features_negative_gap: int | None`
- [ ] 3.2 `sweep_pareto_capability(sae_checkpoint, host_model_id, dataset, *, widths, encodings, scale_boosts, output_dir, **sweep_kwargs)`:
  - Constructs `DownstreamCapabilityTarget` from `dataset` (encoder + labels).
  - For each `(encoding, target_n_features_kept, scale_boost)`: runs `ForgePipeline` with the target, captures the score + perplexity, populates the new ParetoFrontierRow fields.
  - Emits `frontier.jsonl` with the augmented rows; existing `sweep_pareto` consumers ignore the new optional fields.
- [ ] 3.3 Tests:
  - Smoke at tiny scale: 1 encoding, 2 widths, 1 scale_boost, 1 dataset. Asserts the rows carry populated capability fields.
  - Schema back-compat: load a pre-change `frontier.jsonl` via the new `ParetoFrontierRow.from_dict`; assert it loads with new fields as `None`.
  - Forward-compat: load a new-schema row via a hypothetically-frozen old parser; assert unknown fields don't crash (existing `from_dict` ignores extras).

## 4. CLI surface

- [ ] 4.1 `sae-forge sweep capability` subcommand in `saeforge/cli.py`:
  - Flags: `--sae`, `--host`, `--dataset-config <yaml>` (a YAML spec that constructs `CapabilityDataset` — keys: `sequences_path`, `labels_path`, `encoder_checkpoint`, `tokenizer_id`, `aggregator`, `min_prevalence`).
  - Calls `sweep_pareto_capability(...)`, writes `frontier.jsonl`.
- [ ] 4.2 `sae-forge recommend --frontier frontier.jsonl --target retained-mauc>=0.95`:
  - Parses the frontier, filters by target predicate, returns the row with minimum `n_params_forged` (or `target_n_features_kept` when n_params not populated).
  - Tabular output by default; `--json` flag emits the picked row as JSON.
- [ ] 4.3 CLI tests:
  - Smoke: invoke the subcommand with a tiny synthetic dataset config; assert frontier file exists and has the right field set.
  - `recommend`: feed a hand-crafted frontier with one row above + one below the target; assert the right row is picked.

## 5. Falsifiable acceptance gate (proposal §"Falsifiable acceptance gate")

- [ ] 5.1 Reproduce bio-sae's two-fixture predictions:
  - Fixture A: `runs/uniref50_small/residue` — predicted optimal n=16, retained_mauc ≥ 1.00.
  - Fixture B: `runs/uniref50_n5000/pooled_w1024_k64` — predicted optimal n=512, retained_mauc ≈ 0.93 ± 0.01.
- [ ] 5.2 The sweep must recommend exactly these widths via `sae-forge recommend --target retained-mauc>=<X>` with X tuned per fixture.
- [ ] 5.3 Re-runs over a fresh resolved venv must hit retained-mauc within 1.0 mAUC point of bio-sae's manual measurements; widening the tolerance further is the signal that something has drifted upstream (polygram / transformers / numpy).

## 6. Documentation

- [ ] 6.1 README: new "Capability-aware forge tuning" section pointing at the new target + sweep.
- [ ] 6.2 `docs/algorithm.md`: cross-reference from §5 (rank-dependent amplification) to the capability target — "the amplification is invisible to cosine; use `DownstreamCapabilityTarget` when downstream task fidelity matters".
- [ ] 6.3 CHANGELOG entry under the next `[Unreleased]` block.
- [ ] 6.4 Target docstring: document the recommended-practices block per Decision 7 (subset-first, host-cache, fp16, encoder-restrict) plus the encoder calling convention (single-tensor return, `lambda x: nn_sae(x)[1]` wrapper for callers passing `(recon, z)`-returning SAEs).
- [ ] 6.5 Sweep CLI `--help`: surface the wall-time estimate per cell (~5 s/protein/cell at default precision; ~L×slower under `encode_then_pool`) so users can size sweeps appropriately.
- [ ] 6.6 End-to-end usage example in `examples/forge_capability_bio_sae.py` mirroring the YAML + CLI example in proposal.md §5, against a bundled tiny fixture (no real ESM-2 t6_8M download required for the example to import / lint).

## 7. Sibling-repo validation (post-merge, separate PRs in those repos)

- [ ] 7.1 sm-sae: build `CapabilityDataset.from_sm_sae(...)` constructor in `smsae` for its factorial vocab; run the sweep; assert categorical-substrate-like cliff cov95 profile.
- [ ] 7.2 econ-sae: same for the tier-stratified econ vocab; expect the sweep to find different optimal widths per tier (concentrated tiers: small basis; spread tiers: mid basis).
- [ ] 7.3 bio-sae: replace `scripts/forge_capability_eval.py` with a thin wrapper over `sae-forge sweep capability`; assert reproduction of the two-fixture predictions from §5.
