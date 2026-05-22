# Implementation tasks

## 0. Pre-locks (blocking)

- [ ] 0.1 Confirm the bio-sae `runs/uniref50_n5000/pooled_w1024_k64/sae.pt` checkpoint matches the SAE referenced by the `partition_summary.json` at `runs/polygram_partition/uniref50_small/`. The partition summary is from the `uniref50_small` line; pooled_w1024_k64 may have its own partition spec or none. If the spec doesn't exist for the pooled fixture, this task generates one via bio-sae's polygram passthrough (or scopes the experiment to the residue fixture's partition where the spec already exists).
- [ ] 0.2 Lock the per-tier slicing rule: proportional-with-largest-fractional-remainder-wins (per design.md Decision 1's edge case). Document the rule in the basis-builder docstring.

## 1. Bio-sae-side: partition shadow checkpoint materialization

- [ ] 1.1 `bio-sae/scripts/materialize_partition_checkpoint.py`: reads the source SAE (`sae.pt`) + partition spec (`partition_summary.json`), computes per-feature `partition_block_ids` vector by mapping each kept feature id to its tier, writes a new safetensors at `runs/polygram_partition/uniref50_n5000/pooled_w1024_k64_partition.pt` with the same `encoder.*` / `decoder.*` keys + an additional `partition_block_ids` tensor.
- [ ] 1.2 Smoke test: `tests/test_materialize_partition_checkpoint.py` against a synthetic partition spec → assert output safetensors has the expected keys + block-id shape.
- [ ] 1.3 Run the materialization against the bio-sae pooled fixture; commit the script + a README note pointing to where the artifact lives. The artifact itself is gitignored (under `runs/*`).

## 2. sae-forge-side: partition-aware basis builder

- [ ] 2.1 Extend `saeforge/sweep_capability.py::_run_capability_cell` to check the SAE state dict for a `partition_block_ids` tensor. If absent: fall back to current row-norm slicing (no behavioural change for any existing caller). If present: invoke a new `_slice_partition_aware(W_dec_full, row_norms, partition_block_ids, target_n_features_kept)` helper.
- [ ] 2.2 `_slice_partition_aware`: proportional per-tier allocation with largest-fractional-remainder rounding for the last slot. Within each tier, top-K by row norm. Returns the `kept` index vector for `FeatureBasis` construction.
- [ ] 2.3 Unit tests in `tests/test_sweep_pareto_capability.py`:
  - `test_partition_aware_basis_slicing_proportional`: known tier sizes + target → expected per-tier counts.
  - `test_partition_aware_basis_slicing_rounding`: edge case where proportional rounding yields off-by-one → assert largest-fractional-remainder wins.
  - `test_partition_aware_basis_slicing_within_tier_topk`: within each tier, the kept feature ids are the top-K by row norm.
  - `test_partition_aware_basis_falls_back_when_no_block_ids`: SAE state without `partition_block_ids` → current row-norm slicing (back-compat).

## 3. Falsifiable measurement run

- [ ] 3.1 Run progressive sweep against raw_slice on the pooled fixture at `[1000, 5000]`, output under `runs/forge/partition_validation/raw_slice/`. **Already runnable today** — the artifact from this session's background n=5000 run satisfies this slot; just copy it under the validation output root.
- [ ] 3.2 Run progressive sweep against the partition shadow checkpoint at the same schedule, output under `runs/forge/partition_validation/partition/`. Expected wall time: ~45 min on CPU.
- [ ] 3.3 Comparison script: `bio-sae/scripts/compare_partition_vs_raw_slice.py` reads both `progressive_summary.json` files + both `frontier.jsonl` files; emits a side-by-side table (per-cell delta + trajectory variance + convergence flag) to stdout + `partition_validation_summary.json`.

## 4. Writeup

- [ ] 4.1 Section under `bio-sae/docs/forge-capability-bottleneck.md` (new §5 or extend §4) capturing the experiment outcome. The section names which decision-tree cell landed (per design.md Decision 4) and what it implies for:
  - Wave C's "unproven" status (now resolved positively or negatively under the right metric).
  - `add-progressive-finetune`'s priority.
  - `add-multi-encoding-capability-sweep`'s motivation (only if partition wins).
- [ ] 4.2 Update bio-sae's auto-memory `wave-c-partition-forge-side-unproven` entry with the new resolution.
- [ ] 4.3 Cross-reference the sae-forge CHANGELOG entry for the partition-aware basis-builder additive change (~1 paragraph under `[Unreleased]`).

## 5. Follow-up scoping (after writeup)

- [ ] 5.1 If partition wins → file `add-multi-encoding-capability-sweep` openspec exposing `sae-forge sweep-capability --encoding LABEL:PATH` as the natural multi-encoding API.
- [ ] 5.2 If partition loses → file `add-progressive-finetune` openspec promoting it from "deferred" to "next-up." Use the n=5000 retained_mauc-drift evidence + the partition-doesn't-fix-it evidence as motivation.
- [ ] 5.3 If partition partial-wins → file BOTH (multi-encoding AND fine-tune) with explicit "complement, don't substitute" framing.
