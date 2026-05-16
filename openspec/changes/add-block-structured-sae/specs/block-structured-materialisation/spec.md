# block-structured-materialisation Specification (delta)

## ADDED Requirements

### Requirement: Encoding-partition manifest schema

A `--encoding-partition LABEL:PATH` manifest file SHALL be a JSON object conforming to this schema:

```json
{
  "label": "<string, must match the LABEL passed on the CLI>",
  "partition": [
    {
      "block_id": "<string, unique within partition>",
      "encoding_class": "<one of: MPSRung1 | Rung3 | Rung4 | Rung5 | HEA_Rung2>",
      "encoding_kwargs": {"<class-specific kwargs>": "<value>"},
      "learn_axis_assignment": <bool, optional, defaults to false>,
      "feature_ids": [<int>, <int>, ...]
    },
    ...
  ]
}
```

Constraints:

- `label` SHALL be a non-empty string. The CLI SHALL verify it matches the LABEL portion of `--encoding-partition LABEL:PATH` at parse time and refuse with non-zero exit on mismatch.
- `partition` SHALL be a non-empty list of block objects.
- Each `block_id` SHALL be unique within the manifest. Duplicates SHALL be refused at parse time.
- `encoding_class` SHALL be one of the supported five strings. Anything else SHALL be refused at parse time with an error message listing the supported set.
- `encoding_kwargs` SHALL be an object. Class-specific requirements:
  - `MPSRung1`, `Rung3`, `Rung4`: `encoding_kwargs` SHOULD be empty `{}`. The parser SHALL refuse unrecognised kwargs (no silent drop).
  - `Rung5`: `encoding_kwargs` SHALL contain `"n_amp_qubits": <int>`. The parser SHALL refuse a `Rung5` block without this key.
  - `HEA_Rung2`: `encoding_kwargs` SHALL contain `"n_qubits": <int>`. The parser SHALL refuse an `HEA_Rung2` block without this key.
- `learn_axis_assignment` SHALL be a boolean. Absent → defaults to `false`. Other types (numeric, string) SHALL be refused at parse time.
- `feature_ids` SHALL be a non-empty list of integers. Order within a block is irrelevant to materialisation but is preserved in the on-disk JSON (the file's bytes are SHA'd as-is).

Cross-block constraints (verified at materialise time, not parse time, because the expected feature count comes from the SAE):

- The `feature_ids` lists SHALL be pairwise disjoint. The materialise driver SHALL refuse a manifest whose blocks share any feature id, with an error message listing up to the first 10 duplicated ids.
- The union of all `feature_ids` SHALL equal `set(range(N))` where `N` is the SAE's actual feature count. The materialise driver SHALL refuse a manifest with missing or extra ids, naming up to the first 10 of each.

#### Scenario: well-formed two-block manifest parses

- **GIVEN** a file with `label="mps"`, two blocks (`heavy` → `Rung5` with `n_amp_qubits=4`, `learn_axis_assignment=true`; `tail` → `MPSRung1` with empty kwargs), and disjoint+complete `feature_ids`
- **WHEN** `parse_partition_manifest(path)` is called
- **THEN** it returns an `EncodingPartition` with `manifest_sha256` populated, two `BlockSpec`s, and `format_partition_label` returns `"heavy:Rung5(k=4,learn)+tail:MPSRung1"`

#### Scenario: Rung5 block without n_amp_qubits is refused at parse time

- **GIVEN** a manifest where one block has `encoding_class="Rung5"` and `encoding_kwargs={}`
- **WHEN** `parse_partition_manifest(path)` is called
- **THEN** it raises `PartitionError` whose message names the offending `block_id` and the missing `n_amp_qubits` key

#### Scenario: unknown encoding_class is refused at parse time

- **GIVEN** a manifest where one block has `encoding_class="Bogus"`
- **WHEN** `parse_partition_manifest(path)` is called
- **THEN** it raises `PartitionError` listing the supported set (`MPSRung1`, `Rung3`, `Rung4`, `Rung5`, `HEA_Rung2`)

#### Scenario: overlapping feature ids are refused at materialise time

- **GIVEN** a manifest whose `heavy` and `tail` blocks both contain feature id `42`
- **WHEN** `validate_coverage(partition, expected_feature_count=N)` is called during materialisation
- **THEN** it raises `PartitionError` whose message names `42` as duplicated and identifies both block_ids

#### Scenario: incomplete coverage is refused at materialise time

- **GIVEN** a manifest with two blocks whose union covers `set(range(N)) - {17}` (one id missing)
- **WHEN** `validate_coverage(partition, expected_feature_count=N)` is called
- **THEN** it raises `PartitionError` whose message names `17` as a missing id

### Requirement: Partition label format

The `partition_label` string written to `ParetoFrontierRow` SHALL be a deterministic function of the manifest content. Two manifests with identical content (modulo trailing whitespace — see Decision 2 in design.md) SHALL produce identical labels.

Format per block: `block_id + ":" + encoding_class + suffix` where:

| Class | learn=False | learn=True |
|-------|-------------|------------|
| `MPSRung1` | `""` | `"(learn)"` |
| `Rung3` | `""` | `"(learn)"` |
| `Rung4` | `""` | `"(learn)"` |
| `Rung5` (with `n_amp_qubits=k`) | `"(k=K)"` | `"(k=K,learn)"` |
| `HEA_Rung2` (with `n_qubits=N`) | `"(n=N)"` | `"(n=N,learn)"` |

Blocks SHALL be joined by `+` in manifest order (not reordered by `block_id`).

#### Scenario: format_partition_label produces the documented shape

- **GIVEN** an `EncodingPartition` with one `Rung5(k=4, learn=True)` block named `heavy` and one `MPSRung1(learn=False)` block named `tail` (in that order)
- **WHEN** `format_partition_label(partition)` is called
- **THEN** it returns the literal string `"heavy:Rung5(k=4,learn)+tail:MPSRung1"`

#### Scenario: format_partition_label preserves block order

- **GIVEN** the same partition but with `tail` block listed before `heavy` in the manifest
- **WHEN** `format_partition_label(partition)` is called
- **THEN** it returns `"tail:MPSRung1+heavy:Rung5(k=4,learn)"` (different label, despite same set of blocks — the manifest's bytes also differ, so the cache key differs too)

### Requirement: Materialisation contract under `--encoding-partition`

When `AutoMaterialiseSpec.encoding_partition is not None`, `materialise(...)` SHALL:

1. Compute the cache key including `encoding_partition_sha256` (the manifest's SHA-256). The fields `encoding_class`, `encoding_kwargs`, `learn_axis_assignment` SHALL be recorded as `null` in the meta.
2. On cache hit, return the cached materialised dir as today (no polygram work).
3. On cache miss, load the SAE checkpoint to determine `N` (the feature count), then call `validate_coverage(partition, expected_feature_count=N)`. Raise `PartitionError` (re-raised with the encoding label prepended) on failure.
4. Pass the parsed `EncodingPartition` through to polygram's `from_sae_lens(..., encoding_partition=...)`. The single-encoding kwargs (`encoding=`, `learn_axis_assignment=`) SHALL NOT also be passed (the partition owns those decisions per-block).
5. Run the existing chain: `BehaviouralValidator.run()` → `Compressor.plan_pareto()` → `Compressor.apply()` per K.
6. Write `auto_materialise_meta.json` with the new field set.

The materialised dir layout is unchanged from `add-auto-materialise-sweep` (one `pareto/k_<K>.safetensors` per requested K, one `validation_report.json`, one `pareto.json`, one `auto_materialise_meta.json`). The block-structured nature lives inside the safetensors files (via polygram's `BlockStructuredDictionary` serialisation) and the cache-key meta; sae-forge does not split the on-disk artifacts per block.

#### Scenario: partition flows through to polygram's from_sae_lens

- **GIVEN** an `AutoMaterialiseSpec` with a parsed two-block `EncodingPartition` and the cache cold
- **WHEN** `materialise(spec, ...)` runs
- **THEN** `polygram.from_sae_lens` is called once with `encoding_partition=<the BlockSpec list>` and no `encoding=` kwarg; `BehaviouralValidator` and `Compressor` consume the resulting `BlockStructuredDictionary` (instance check) and the run completes

#### Scenario: cache key on first partitioned run records encoding_partition_sha256

- **GIVEN** a cold-cache invocation `--auto-materialise --encoding-partition mps:p.json`
- **WHEN** `materialise(...)` writes `auto_materialise_meta.json`
- **THEN** the file contains `"encoding_partition_sha256": "<hex>"`, `"encoding_class": null`, `"encoding_kwargs": null`, `"learn_axis_assignment": null`

### Requirement: `--plan-only` extension for partitioned encodings

When `--encoding-partition LABEL:PATH` is set under `--plan-only`, the per-encoding stderr block for that label SHALL print:

- `label`: the encoding label
- `cache_status`: `HIT` or `MISS`. On `MISS`, the list of cache-key fields that differ (or `cold` if no cache exists). `encoding_partition_sha256` appears in this list when the manifest's bytes changed.
- `sae_sha256`: as today
- `validation_prompts_sha256`: as today
- `targets`: as today
- `partition_manifest_path`: the absolute path to the manifest file
- `partition_manifest_sha256`: hex-encoded SHA-256 of the manifest bytes
- `partition_label`: the deterministic label string
- `block_count`: the number of blocks
- `validator_forward_count_estimate`: as today

The driver SHALL NOT print `encoding_class` / `encoding_kwargs` for partitioned encodings (those are block-level, not encoding-level). For mixed sweeps, non-partitioned labels print the existing block (with `encoding_class`/`encoding_kwargs`) and partitioned labels print the extended block.

#### Scenario: `--plan-only` on cold cache prints MISS with partition fields

- **GIVEN** a sweep with one `--encoding-partition mps:p.json` and no `_materialised/mps/` directory present
- **WHEN** the invocation is run with `--plan-only --auto-materialise`
- **THEN** stderr contains a `mps` block with `cache_status: MISS (cold)`, `partition_manifest_sha256: <hex>`, `partition_label: <formatted>`, `block_count: N`, no `encoding_class` field, no `encoding_kwargs` field; exit code 0; `_materialised/` is NOT created

#### Scenario: `--plan-only` on warm cache after manifest edit reports the partition-SHA diff

- **GIVEN** a previous successful partitioned run cached `_materialised/mps/`
- **WHEN** the manifest is edited (e.g. flipping one block's `learn_axis_assignment`) and `--plan-only` is rerun
- **THEN** the `mps` block reports `cache_status: MISS (encoding_partition_sha256)`; exit code 0; no materialisation happens
