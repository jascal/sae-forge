# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `sweep_pareto_capability(encodings=[(label, path), ...])` multi-encoding API

`saeforge.sweep_pareto_capability(...)` SHALL accept a new
`encodings: list[tuple[str, str | Path]] | None = None` keyword
argument with the following semantics:

- When `encodings` is provided, it is the canonical list of
  encoding labels + SAE checkpoint paths to sweep over.
- When `encodings is None` AND `sae_checkpoint` is provided, the
  wrapper SHALL internally construct
  `encodings = [("raw_slice", sae_checkpoint)]`. This preserves
  v0.8.x / v0.9.x back-compat byte-equivalently.
- When `encodings is None` AND `sae_checkpoint is None`:
  `ValueError` naming the required arguments.
- When both `encodings` AND `sae_checkpoint` are provided:
  `ValueError` with hint to pass one or the other.

The wrapper SHALL load each encoding's SAE state dict exactly once
per sweep call via a `_load_encoding_state(path)` helper. The
host-extraction cache SHALL be shared across encodings (host
activations are encoding-independent).

Cell loop shape becomes `for (label, state) in loaded_encodings:
for (width, scale_boost) in ...`. The per-cell runner
(`_run_capability_cell`) receives W_dec, row_norms, order, and
partition_block_ids from the per-encoding state, not from a single
global. The cell signature is unchanged from v0.9.x.

### Requirement: Unique encoding labels

Encoding labels in the `encodings` list SHALL be unique. Duplicate
labels SHALL raise `ValueError` naming the duplicate label, before
any forge cost is paid.

### Requirement: `ParetoFrontierRow.encoding_label` is load-bearing

Effective with this change, `ParetoFrontierRow.encoding_label`
identifies WHICH encoding's basis was used for the cell — not an
informational label. Downstream consumers (recommend, scaling
summary emitter when it lands) SHALL partition rows by encoding
when comparing.

This is a semantic strengthening, not a schema change: the field
existed in v0.8.x but carried only informational content.
Pre-change frontier files load identically; their
`encoding_label="raw_slice"` (the historical default) is correctly
interpreted as "this row used the raw row-norm slicing path".

### Requirement: `ProgressiveRecommendation.per_encoding_recommendations`

`saeforge.ProgressiveRecommendation` SHALL gain an optional new
field:

```
per_encoding_recommendations:
    dict[str, ProgressiveRecommendation] | None = None
```

- `None` for single-encoding sweeps (back-compat preserved).
- `dict[encoding_label, ProgressiveRecommendation]` for multi-
  encoding sweeps, with one entry per encoding that ran.

The top-level fields (`target_n_features_kept`,
`retained_mauc_vs_host`, `converged`, `rationale`, etc.) belong to
the **winning encoding** chosen by:

1. Filter to encodings whose `per_encoding_recommendations[E].converged`
   is True.
2. Among those, pick the one with the smallest stable n at
   retained_mauc ≥ cross-encoding median of converged-encoding
   retained_mauc values.
3. Tiebreak by lowest argmin-retained_mauc variance across stages.
4. Final tiebreak by encoding-list order (deterministic).

If NO encoding converged, the top-level recommendation falls back
to the encoding with the lowest argmin-retained_mauc variance (most
data-scale-stable, even if non-converged); top-level `converged`
is `False`; `rationale` names which encoding was picked and why.

### Requirement: Multi-encoding progressive plateau identification

For each encoding E and each progressive stage K, plateau
identification (per `add-progressive-capability-sweep`'s spec)
runs INDEPENDENTLY:

- E's plateau at stage K = widths within `plateau_tolerance` of
  E's peak retained_mauc at stage K.
- Per-encoding `min_plateau_widths` floor applies independently.
- Per-encoding convergence detection runs over E's per-stage
  plateau argmins.

`ProgressiveStageResult.plateau_widths` becomes
`ProgressiveStageResult.per_encoding_plateaus: dict[str, tuple[int, ...]]`
mapping encoding label to plateau widths.

Single-encoding sweeps emit a one-entry dict; the dict shape is
canonical.

### Requirement: CLI `--encoding LABEL:PATH` (repeatable)

`sae-forge sweep-capability` and `sae-forge sweep-capability-progressive`
SHALL accept a `--encoding LABEL:PATH` flag that can be repeated.
Multiple `--encoding` flags accumulate into the `encodings` list
in CLI flag order.

When neither `--encoding` flags nor the YAML config's
`encoder_checkpoint` provides multiple entries, the call is
single-encoding (back-compat). When the YAML provides
`encoder_checkpoint` AND `--encoding` flags appear,
`--encoding` wins (explicit > implicit); a stderr warning names
the conflict.

Encoding label SHALL match the regex `[A-Za-z0-9_]+` (no colons,
spaces, or special characters). Bad labels → ValueError with the
specific label flagged.

### Requirement: `sae-forge recommend` over multi-encoding frontiers

When `sae-forge recommend` reads a frontier carrying multiple
distinct `encoding_label` values among its rows:

1. Apply predicates as today (filter rows that pass `--target`).
2. Sort survivors by `(target_n_features_kept ASC,
   encoding_list_order_index ASC)`.
3. Output names BOTH the encoding and the width:

```
recommended encoding: partition_q4
  target_n_features_kept: 64
  retained_mauc_vs_host:  0.9523
  cross-encoding rank:    1/5
```

Single-encoding frontiers (only one distinct `encoding_label`)
behave identically to v0.9.x.

### Requirement: Falsifiable acceptance gate

The change SHALL include a slow integration test
(`tests/test_multi_encoding_acceptance_gate.py`, gated on bio-sae
fixtures AND polygram-encoded shadow checkpoints being on disk)
running a multi-encoding progressive sweep on the pooled fixture
at `[1000, 5000]` schedule with at least three encodings.

Three predictions to test (writeup names which landed):

1. At least ONE encoding crosses retained_mauc ≥ 0.95 at n=512 at
   the largest stage, OR no encoding does and the gate documents
   that as the data-scale-tax-is-independent-of-encoding outcome.
2. At least ONE encoding's recommendation is `converged=True` at
   default strictness.
3. At least TWO encodings disagree on `target_n_features_kept` by
   more than one candidate-grid bucket at the same predicate.

If all three hold → multi-encoding sweep validated; encoding
choice is a real lever.
If 2 of 3 → partial validation; documented in the bio-sae writeup.
If 0 of 3 → multi-encoding doesn't help on this substrate; the
gate documents the negative result honestly (mirrors Wave C's
"shipped but unproven" history pattern).
