# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `sweep_pareto_capability` honours optional `partition_block_ids` in the SAE state dict

`sweep_pareto_capability._run_capability_cell` SHALL check the SAE state dict (loaded via `torch.load`) for an optional `partition_block_ids` tensor. The tensor SHALL have:

- dtype: `torch.int64` (cast to int as needed).
- shape: `(n_features,)` where `n_features == W_dec_full.shape[0]` (rows of the decoder).
- values: integer tier ids assigning each feature to a partition block.

If the tensor is present, the cell's basis construction SHALL use **partition-aware slicing** (per Requirement 2 below) instead of the default row-norm slicing. If absent, the cell SHALL use row-norm slicing (current v0.8.x behaviour preserved — back-compat is byte-equivalent for any state dict that doesn't carry the new key).

### Requirement: Partition-aware proportional slicing

When `partition_block_ids` is present, the basis-construction step SHALL allocate `target_n_features_kept` across tiers proportionally:

1. Let `tier_sizes[t] = count of features in tier t`.
2. Let `proportional[t] = target_n_features_kept * tier_sizes[t] / sum(tier_sizes)`.
3. Floor: `allocated[t] = floor(proportional[t])`. Remaining slots: `target_n_features_kept - sum(allocated)`.
4. Distribute the remaining slots to the tiers with the largest fractional remainder (`proportional[t] - allocated[t]`), ties broken by lowest tier id for determinism.

Within each tier, the kept features SHALL be the top-K by row norm of W_dec (where K = `allocated[t]`).

### Requirement: Behaviour when `target_n_features_kept` cannot be exactly allocated

For very small `target_n_features_kept` values where proportional allocation would yield zero features in some tier, the rule still applies (some tiers may get zero). The basis is allowed to span fewer than all tiers in this case; no warning is emitted. Users requesting `target_n_features_kept` smaller than the number of tiers accept the implication.

### Requirement: Documentation of the partition-aware path

The `sweep_pareto_capability` docstring SHALL mention the optional `partition_block_ids` key + its semantics under "Parameter guidance" or an "Optional state-dict keys" subsection. The README's "Capability-aware forge tuning" section SHALL link to the partition validation experiment outcome at `bio-sae/docs/forge-capability-bottleneck.md` once that lands.

### Requirement: Experiment outcome is the acceptance gate

This change ships a measurement experiment, not a feature. The acceptance gate is the writeup at `bio-sae/docs/forge-capability-bottleneck.md` (added/extended by this change) naming:

1. The decision-tree cell that landed (per design.md Decision 4: partition wins / partial / no-op / makes-worse).
2. The implication for Wave C's "unproven" status.
3. The implication for `add-progressive-finetune`'s priority.
4. The implication for a potential `add-multi-encoding-capability-sweep` follow-up.

The writeup SHALL include the side-by-side comparison table (per-cell retained_mauc delta, trajectory variance, convergence flag) for both encodings at every stage of the validation schedule.

### Requirement: No new public sae-forge surface

This change SHALL NOT add new public API symbols to `saeforge.__all__` or new CLI subcommands. The partition-aware basis builder is an internal extension to `_run_capability_cell` triggered by an optional state-dict key. Users invoke it by pointing `--dataset-config`'s `encoder_checkpoint` at a partition-shadow safetensors instead of the original SAE.

If the experiment outcome motivates richer multi-encoding sweeps, that's a separate openspec (`add-multi-encoding-capability-sweep`) whose surface decision happens after the data lands.
