# Implementation tasks

## 0. Pre-locks (blocking)

- [ ] 0.1 Confirm the in-flight partition validation (PR #89's measurement) lands either positive or partial-win. If FULL no-op, re-examine whether multi-encoding sweep is worth shipping (the architectural win it enables is contingent on at least ONE non-raw_slice encoding being measurably different).
- [ ] 0.2 Lock the `ParetoFrontierRow.encoding_label` field's load-bearing semantics: it now identifies WHICH encoding was used for the cell (not just an informational label). Document this transition in the CHANGELOG entry.

## 1. `saeforge/sweep_capability.py` — multi-encoding refactor

- [ ] 1.1 `sweep_pareto_capability(...)` signature: add `encodings: list[tuple[str, str | Path]] | None = None` parameter. When provided, supersedes `sae_checkpoint`. When None AND `sae_checkpoint` provided, internally constructs `encodings = [("raw_slice", sae_checkpoint)]`. When both None: `ValueError`. When both provided: `ValueError` with hint to pass one or the other.
- [ ] 1.2 Refactor the existing per-cell loop. Currently: `cells = [(encoding_label, width, sb) for ... ]` iterates over informational encoding labels with a single global W_dec. Now: each encoding has its own W_dec + optional partition_block_ids, loaded once per sweep call. The cell loop becomes `for encoding_label, encoding_state in encodings_loaded: for width, sb in ...`.
- [ ] 1.3 Per-encoding state loading: a new `_load_encoding_state(path)` helper returns `{"W_dec": ..., "row_norms": ..., "order": ..., "partition_block_ids": Optional[...]}`. Called once per encoding at sweep start.
- [ ] 1.4 `_run_capability_cell` already takes `partition_block_ids: ndarray | None`; the only change is that the W_dec / row_norms / order / partition_block_ids it receives now come from the per-encoding state instead of a single global. Cell signature unchanged.
- [ ] 1.5 Validation: encoding labels SHALL be unique. Duplicate labels → ValueError naming the duplicate.

## 2. `saeforge/sweep_capability_progressive.py` — multi-encoding passthrough

- [ ] 2.1 `sweep_pareto_capability_progressive(encodings=[...])` accepts the same shape; passes through to per-stage `sweep_pareto_capability` calls.
- [ ] 2.2 Per-encoding plateau identification: rewrite `_identify_plateau` to take a frontier row LIST plus an `encoding_filter: str` argument, return that encoding's plateau. Top-level progressive loop now identifies per-encoding plateaus + computes per-encoding convergence.
- [ ] 2.3 `ProgressiveStageResult` adds `per_encoding_plateaus: dict[str, tuple[int, ...]]` (encoding label → plateau widths). Single-encoding rows have one-entry dict.
- [ ] 2.4 `ProgressiveRecommendation` gains `per_encoding_recommendations: dict[str, ProgressiveRecommendation] | None = None`. When multi-encoding, populated with one entry per encoding. The top-level `target_n_features_kept` / `retained_mauc_vs_host` / `converged` / `rationale` belong to the winning encoding per design.md Decision 3.
- [ ] 2.5 Cross-encoding tiebreaker: smallest stable n at retained_mauc ≥ cross-encoding median; ties broken by lowest argmin retained_mauc variance across stages; further ties broken by CLI-flag-order (or `encodings` list order) for determinism.
- [ ] 2.6 `ProgressiveHistory.to_json_dict()` emits per-encoding state for multi-encoding sweeps; back-compat: single-encoding emits the same shape as before.

## 3. CLI surface

- [ ] 3.1 `sae-forge sweep-capability --encoding LABEL:PATH` (repeatable). Parses the colon-separated form into `[(label, path), ...]`. Reused from sweep-pareto's existing parser if possible.
- [ ] 3.2 `sae-forge sweep-capability-progressive --encoding LABEL:PATH` (repeatable). Same parser.
- [ ] 3.3 YAML dataset config's `encoder_checkpoint` continues to work as the single-encoding sugar. When `--encoding` is also provided, `--encoding` wins; emit a warning.
- [ ] 3.4 `sae-forge recommend` over a multi-encoding frontier (detected via multiple distinct `encoding_label` values among rows): output emits the per-encoding ranking PLUS the picked encoding+width pair. Tiebreaker per design.md Decision 6.
- [ ] 3.5 `--help` text on `--encoding` documents the format + repeatable + winner-pick tiebreaker.

## 4. Tests

- [ ] 4.1 `tests/test_sweep_pareto_capability.py`:
  - `test_multi_encoding_sweep_smoke`: pass two encodings (the synthetic ESM fixture's sae.pt + a partition-shadow variant of the same); assert both encodings produce cells; rows carry distinct encoding_label values.
  - `test_multi_encoding_back_compat_sae_checkpoint_keyword`: `sweep_pareto_capability(sae_checkpoint=PATH, ...)` continues to work; produces single-encoding rows with `encoding_label = "raw_slice"`.
  - `test_multi_encoding_rejects_both_sae_checkpoint_and_encodings`: passing both is a ValueError.
  - `test_multi_encoding_duplicate_label_raises`: `encodings=[("a", p1), ("a", p2)]` → ValueError.
- [ ] 4.2 `tests/test_sweep_progressive.py`:
  - `test_progressive_multi_encoding_per_encoding_recommendation`: 2-encoding 2-stage progressive sweep; assert `per_encoding_recommendations` populated with two entries; top-level recommendation belongs to one of them.
  - `test_progressive_multi_encoding_winner_tiebreaker_flag_order`: deliberate tie in plateau argmin + retained_mauc variance; assert winner is the encoding listed first in `encodings`.
- [ ] 4.3 `tests/test_progressive_cli.py`:
  - `test_cli_multi_encoding_parses`: `--encoding raw_slice:p1 --encoding partition:p2` produces a 2-entry `args.encoding` list.
  - `test_cli_multi_encoding_recommend_picks_winner`: end-to-end via main(); recommend output names BOTH encoding and width.

## 5. Falsifiable acceptance gate

- [ ] 5.1 `tests/test_multi_encoding_acceptance_gate.py` (slow, gated on bio-sae fixtures + polygram-encoded shadows being on disk): progressive sweep at `[1000, 5000]` against the pooled fixture with `[raw_slice, partition_q4, mps_rung1_x4, mps_rung1_x16, rung5]`. Asserts:
  - At least ONE encoding crosses retained_mauc ≥ 0.95 at n=512 at the largest stage, OR all encodings ≤ raw_slice (the latter falsifies the openspec's premise that encoding choice helps).
  - At least ONE encoding's recommendation is converged at default strictness.
  - At least TWO encodings disagree on `target_n_features_kept` by > 1 candidate-grid bucket.

  If all three hold → multi-encoding sweep validated. If 2 of 3 → documented partial win. If 0 of 3 → multi-encoding doesn't help; close this openspec as "shipped but unproven" (mirrors Wave C's history honestly).

## 6. Documentation

- [ ] 6.1 README: extend the "Capability-aware forge tuning" + "Progressive capability sweep" sections with the multi-encoding example (3-5 encodings in one call).
- [ ] 6.2 `docs/algorithm.md`: cross-reference at §5 to the multi-encoding sweep for users who want to choose a basis encoding empirically.
- [ ] 6.3 CHANGELOG entry under `[Unreleased]` for `add-multi-encoding-capability-sweep`.

## 7. Bio-sae-side adoption (post-merge)

- [ ] 7.1 Generate polygram-encoded shadow checkpoints for the pooled fixture at multiple bond dimensions. Either via:
  - (a) Polygram CLI `polygram compress` with `--encoding-class MPSRung1 --bond-dim X` (if that surface exists);
  - (b) A bio-sae-side script `materialize_encoding_checkpoint.py` extending the partition materialization to include MPS-encoded variants.
- [ ] 7.2 Run the multi-encoding progressive sweep on the pooled fixture; capture the side-by-side comparison.
- [ ] 7.3 Writeup under `bio-sae/docs/forge-capability-bottleneck.md` (extend the partition validation §5 OR new §6): which encoding won; how the data-scale tax decomposed across encoding choices; implications for `add-progressive-finetune`'s priority.

## 8. Release

- [ ] 8.1 Bump `__version__` to `0.10.0` (new public surface: `encodings` kwarg on `sweep_pareto_capability` + `per_encoding_recommendations` field).
- [ ] 8.2 Tag `v0.10.0` on the merge commit. Bio-sae bumps pin.
