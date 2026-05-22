# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `sweep_pareto_capability` entry point

`saeforge.sweep_pareto_capability(sae_checkpoint, host_model_id,
dataset, *, widths, encodings, scale_boosts, output_dir,
**sweep_kwargs)` SHALL be a thin wrapper over `sweep_pareto` that:

1. Constructs a `DownstreamCapabilityTarget` from the supplied
   `CapabilityDataset` (encoder + labels + aggregator + min_prevalence).
2. For each (encoding, target_n_features_kept, scale_boost) triple:
   a. Runs the existing `ForgePipeline` with the target.
   b. Captures the target's `(score, perplexity_analog)` and
      reads the post-scoring per-feature AUC arrays off the target
      instance (the target SHALL expose its last-call `host_pf_auc`
      / `forge_pf_auc` numpy arrays for downstream gap computation;
      this is a public side-channel for sweep observability).
   c. Computes gap statistics: median, p25, p75, p95,
      `n_features_gap_above_0_1`, `n_features_negative_gap`.
   d. Populates a `ParetoFrontierRow` with the new fields per the
      `ParetoFrontierRow` requirement below.
3. Writes one `frontier.jsonl` file with the augmented rows.

The wrapper SHALL re-use `sweep_pareto`'s existing per-encoding cache
key plumbing (SAE-content SHA + threshold + encoding class + kwargs
+ layer + targets) and SHALL augment the cache key with `aggregator`
+ `min_prevalence` + `decode_via_basis` so capability sweeps over
the same SAE with different aggregators don't collide.

Argument validation:

- `dataset` SHALL be a `CapabilityDataset`; SHALL raise `TypeError`
  otherwise.
- `widths` SHALL be a non-empty list of positive integers.
- `encodings` SHALL be a non-empty list following the same encoding-
  label syntax `sweep_pareto` accepts.
- `scale_boosts` SHALL be a list of floats and/or the literal string
  `"auto"`. Defaults to `[1.0, "auto"]`.

### Requirement: Host-extraction caching across sweep cells

`sweep_pareto_capability` SHALL cache the host's per-protein
activations (or per-residue / pooled latents under the appropriate
aggregator) on the first sweep cell that uses a given
`(host_model_id, sequences_hash, aggregator, max_seq_len)` tuple and
reuse the cached tensor across subsequent cells. The cache lives at
`output_dir / "host_activations.safetensors"` and is keyed on the
SHA-256 of the sequences list (deterministic across runs over the
same dataset).

The cache SHALL be opt-out via `--no-host-cache` on the CLI / a
`cache_host=False` kwarg on the function. Disable when:

- The host model is non-deterministic (dropout enabled, stochastic
  attention masking — currently not a v1 concern, all bundled
  adapters are eval-mode).
- Disk space is scarce (host_activations.safetensors can be ~200 MB
  for n=5000 proteins × d_model=320 × fp32).

The cache SHALL be invalidated when any cache-key component changes.
Stale-cache errors SHALL be loud: a key mismatch raises with the
divergent component named.

### Requirement: `ParetoFrontierRow` capability fields

`saeforge.ParetoFrontierRow` SHALL accept the following additional
fields, all `Optional[…]` with default `None`:

- `host_baseline_mauc: Optional[float]`
- `host_baseline_cov95: Optional[float]`
- `forge_mauc: Optional[float]`
- `forge_cov95: Optional[float]`
- `retained_mauc_vs_host: Optional[float]`
- `retained_cov95_vs_host: Optional[float]`
- `gap_median: Optional[float]`
- `gap_p25: Optional[float]`
- `gap_p75: Optional[float]`
- `gap_p95: Optional[float]`
- `n_features_gap_above_0_1: Optional[int]`
- `n_features_negative_gap: Optional[int]`
- `capability_aggregator: Optional[str]`
- `capability_min_prevalence: Optional[int]`

The `retained_*` field names are dataset-agnostic per
`add-downstream-capability-target/design.md` Decision 8: the
"retained vs host baseline" semantics generalise to every domain
(sm-sae particles, econ-sae tiers, audio probes). The
`target_name` field on each row carries `"downstream_capability"`
when this target produced the row — that's where cross-target
disambiguation lives. No `downstream_*` field-name aliases ship in
v1; if a future domain needs them, a `field_aliases` map on
`ParetoFrontierRow.to_dict` can emit both forms.

`to_dict` SHALL include these fields in the serialised dictionary
when they are not `None`, and SHALL omit them when they are. This
preserves byte-equivalence with v0.7 frontier files for rows produced
by the non-capability sweep path.

`from_dict` SHALL accept rows lacking these fields (loading them as
`None`); rows lacking the existing v0.7 fields SHALL continue to
load unchanged. Schema-version bump is NOT required — the fields are
purely additive.

### Requirement: `sae-forge sweep capability` CLI subcommand

The CLI SHALL ship a new `sae-forge sweep capability` subcommand that
constructs a `CapabilityDataset` from a YAML config and calls
`sweep_pareto_capability`. The YAML config schema:

```yaml
encoder_checkpoint: path/to/sae.pt   # bio-sae-style sae.pt OR safetensors
sequences_path: data/sequences.parquet
labels_path: data/labels.parquet OR data/bio_bundle.safetensors  # the labels_protein_Y or labels_residue_Y key
labels_key: labels_protein_Y         # which key inside safetensors (if applicable)
tokenizer_id: facebook/esm2_t6_8M_UR50D
aggregator: pool_then_encode
min_prevalence: 10
sae_variant: topk                    # for bio-sae-style _ReferenceSAE construction
sae_k: 64                            # for variant=topk
```

The subcommand SHALL refuse with a non-zero exit code on:

- Missing required keys in the YAML.
- A mismatch between `len(sequences)` and `labels.shape[0]`.
- An `encoder_checkpoint` whose state dict shape doesn't match the
  configured `sae_variant`.

### Requirement: `sae-forge recommend` CLI subcommand

The CLI SHALL ship a new `sae-forge recommend --frontier
PATH/frontier.jsonl --target EXPR` subcommand. `EXPR` SHALL be a
simple predicate over the `ParetoFrontierRow` field set:

```
retained-mauc>=0.95
retained-cov95>=0.50
gap-p95<=0.05
forge-n-params<=10_000_000
```

Comparison operators SHALL be `>=`, `<=`, `==`, `<`, `>`. Field
names SHALL accept either kebab-case or snake_case forms. Multiple
predicates SHALL be combinable with `--target` repeated; semantics
are AND.

The subcommand SHALL:

1. Parse the frontier.
2. Filter rows by the predicates.
3. Among surviving rows, return the row minimising
   `n_params_forged` (or, when that field is `None`, minimising
   `target_n_features_kept`).
4. Print a tabular summary by default; emit the picked row as JSON
   with `--json`.

When no rows survive the predicates, the subcommand SHALL exit with
a non-zero code and print the closest row + the predicate it
violates.

### Requirement: Falsifiable bio-sae acceptance gate

The `add-downstream-capability-target` change SHALL include
integration tests that:

1. Build a `CapabilityDataset.from_bio_sae(...)` from a small bio-sae
   fixture bundled into the sae-forge test data directory.
2. Run `sweep_pareto_capability` over `widths=[16, 64, 128, 256, 512, 1024]`,
   `encodings=["Rung5:n_amp_qubits=2"]`,
   `scale_boosts=[1.0, "auto"]`.
3. Assert the row recommended by `sae-forge recommend
   --frontier ... --target retained-mauc>=0.95` matches the
   bio-sae-predicted optimal width (n=16 for the concentrated
   residue fixture; n=512 for the spread pooled fixture).
4. Assert retained-mAUC for that row is within 0.01 of bio-sae's
   manually-measured value.

Failure of either prediction SHALL block the change from being
archived.
