# pareto-sweep Specification (delta)

## MODIFIED Requirements

### Requirement: ParetoFrontierRow dataclass

The `saeforge.sweep.ParetoFrontierRow` SHALL retain all existing fields and gain one new optional field `partition_label: str | None = None` capturing the block-structured provenance when the row was produced under `--encoding-partition`. The class SHALL continue to expose `.to_json_dict()` and `.from_json_dict(cls, data)`; `from_json_dict` SHALL accept dicts missing the new key and default it to `None` (backwards compat with frontier.jsonl emitted by sae-forge prior to this change).

Field semantics:

| Field | Type | Single-encoding row | Block-structured row | Description |
|-------|------|---------------------|----------------------|-------------|
| `partition_label` | `str \| None` | `None` | populated, deterministic from manifest content | Human-readable shape of the block partition that produced this row, e.g. `"heavy:Rung5(k=4,learn)+tail:MPSRung1"` |
| `encoding_class` (existing) | `str \| None` | populated (`"MPSRung1"`, `"Rung5"`, ...) | populated as literal `"BlockStructured"` | So consumers grouping by `encoding_class` see block-structured rows as a distinct bucket |

The literal `"BlockStructured"` value for `encoding_class` on partition rows is normative: downstream consumers can rely on it for grouping without inspecting `partition_label`. The actual per-block class set lives in `partition_label`.

#### Scenario: partition row round-trips with partition_label populated

- **WHEN** a row with `encoding_class="BlockStructured"`, `partition_label="heavy:Rung5(k=4,learn)+tail:MPSRung1"` is serialised via `.to_json_dict()` and reconstructed via `.from_json_dict(...)`
- **THEN** the reconstructed instance equals the original on both fields

#### Scenario: ParetoFrontierRow.from_json_dict tolerates missing partition_label

- **WHEN** a dict missing `partition_label` is passed to `from_json_dict`
- **THEN** the resulting instance has `partition_label=None` without raising (backwards compat with pre-change frontier.jsonl files)

#### Scenario: single-encoding row carries partition_label=None

- **WHEN** a row produced under `--encoding-class mps:Rung5 --encoding-amp-qubits mps:4` (no `--encoding-partition`) is serialised
- **THEN** the JSON contains `"partition_label": null` and `"encoding_class": "Rung5"`

### Requirement: CLI subcommand `saeforge sweep-pareto`

The existing `sweep-pareto` subcommand SHALL retain all its current flags. A new repeatable flag SHALL be added under `--auto-materialise` mode:

- `--encoding-partition LABEL:PATH` (repeatable, auto-materialise-only) — selects a JSON manifest of per-block `(encoding_class, encoding_kwargs, learn_axis_assignment, feature_ids)` triples for the named encoding label. The PATH is a filesystem path to a JSON file conforming to the `block-structured-materialisation` capability spec. When supplied for a label, the per-label single-encoding flags (`--encoding-class LABEL:...`, `--encoding-amp-qubits LABEL:...`, `--encoding-qubits LABEL:...`) for that same LABEL are refused.

The CLI SHALL refuse the following invocations with non-zero exit and a clear error message (additive to the existing refusal set):

1. `--encoding-partition` set without `--auto-materialise`. The flag is auto-materialise-only.
2. `--encoding-partition LABEL:...` AND any of `--encoding-class LABEL:...`, `--encoding-amp-qubits LABEL:...`, `--encoding-qubits LABEL:...` set for the same `LABEL`. Error message SHALL list the conflicting flags.
3. `--encoding-partition LABEL:...` where `LABEL` does not appear in any `--encoding LABEL:PATH` spec. Error message SHALL name the unknown label and list the known encoding labels.
4. Every encoding label has `--encoding-partition` set AND `--learn-axis-assignment` is also set. The global flag is refused because every label's axis-assignment policy is owned by its manifest. In mixed sweeps where at least one encoding label has no partition, the global flag is permitted and applies to those non-partitioned labels only.
5. The manifest at PATH does not parse, is missing the top-level `label` or `partition` keys, lists overlapping feature ids across blocks, omits any feature id required by the SAE's actual feature count, uses an unknown `encoding_class`, omits `n_amp_qubits` for a `Rung5` block, or omits `n_qubits` for an `HEA_Rung2` block.

#### Scenario: `--encoding-partition` without `--auto-materialise` is refused

- **WHEN** `saeforge sweep-pareto --encoding-partition mps:p.json --encoding mps:/dir --host-model gpt2 --output-dir out/` is invoked (no `--auto-materialise`)
- **THEN** the CLI exits non-zero; stderr names `--auto-materialise` as the gating flag

#### Scenario: per-label exclusivity refusal lists conflicting flags

- **WHEN** `--encoding-partition mps:p.json --encoding-class mps:Rung5 --encoding-amp-qubits mps:4` is passed together (all scoped to the same `mps` label)
- **THEN** the CLI exits non-zero; stderr names `--encoding-class mps:...` and `--encoding-amp-qubits mps:...` as conflicting with `--encoding-partition mps:...`

#### Scenario: mixed sweep with one partitioned and one single-encoding label

- **WHEN** `--encoding mps:sae1.safetensors --encoding hea:sae2.safetensors --encoding-partition mps:p.json --encoding-class hea:HEA_Rung2 --encoding-qubits hea:5` is passed under `--auto-materialise`
- **THEN** the CLI accepts the invocation; the `mps` label materialises via the partition manifest; the `hea` label materialises via single-encoding `HEA_Rung2(n_qubits=5)`; the resulting frontier.jsonl has `mps` rows with `encoding_class="BlockStructured"`, `partition_label="..."` and `hea` rows with `encoding_class="HEA_Rung2"`, `partition_label=null`

#### Scenario: unknown manifest label is refused

- **WHEN** `--encoding mps:sae.safetensors --encoding-partition unknown:p.json` is passed
- **THEN** the CLI exits non-zero; stderr names `unknown` as unknown and lists `mps` as the only declared label

#### Scenario: global `--learn-axis-assignment` is refused when every label is partitioned

- **WHEN** `--encoding mps:sae.safetensors --encoding-partition mps:p.json --learn-axis-assignment` is passed (only one encoding, partitioned)
- **THEN** the CLI exits non-zero; stderr explains that the global flag is meaningless because every label owns its own per-block axis-assignment policy

#### Scenario: global `--learn-axis-assignment` is permitted on a mixed sweep

- **WHEN** `--encoding mps:sae1.safetensors --encoding hea:sae2.safetensors --encoding-partition mps:p.json --encoding-class hea:HEA_Rung2 --encoding-qubits hea:5 --learn-axis-assignment` is passed
- **THEN** the CLI accepts the invocation; the global flag applies only to the `hea` label (non-partitioned); the `mps` label's per-block flags from its manifest are untouched

## MODIFIED Requirements

### Requirement: Auto-materialise cache under `<output-dir>/_materialised/`

The cache-key inputs recorded in `auto_materialise_meta.json` SHALL gain one optional field:

- `encoding_partition_sha256: str | None` — hex-encoded SHA-256 of the bytes of the partition manifest file when `--encoding-partition` was set for this label, else `null`.

The field is always present in the meta (as `null` when not partitioned) so the diff-against-on-disk check in `is_cache_hit` reliably detects partition transitions. The other recorded fields (`encoding_class`, `encoding_kwargs`, `learn_axis_assignment`) SHALL be recorded as `null` when `encoding_partition_sha256` is non-null (the partition manifest owns those fields per-block).

#### Scenario: cache miss when partition manifest content changes

- **GIVEN** a previous `--auto-materialise --encoding-partition mps:p.json` run cached `_materialised/mps/` with `encoding_partition_sha256=<old-sha>`
- **WHEN** the manifest at `p.json` is edited (e.g. flipping `learn_axis_assignment` on one block) and the same invocation is re-issued
- **THEN** the cache is invalidated; `auto_materialise_meta.json` records the new SHA after the rerun; the diff field reported under `--plan-only` is `encoding_partition_sha256`

#### Scenario: cache miss when transitioning from single-encoding to partitioned

- **GIVEN** a previous `--auto-materialise --encoding-class mps:Rung5 --encoding-amp-qubits mps:4` run cached `_materialised/mps/` with `encoding_partition_sha256=null`, `encoding_class="Rung5"`, `encoding_kwargs={"n_amp_qubits": 4}`
- **WHEN** the same invocation is re-issued with `--encoding-partition mps:p.json` and the per-label single-encoding flags removed
- **THEN** the cache is invalidated; the diff fields reported are `encoding_partition_sha256`, `encoding_class`, `encoding_kwargs`

#### Scenario: byte-identity preserved when `--encoding-partition` is absent

- **WHEN** a sweep is run with all existing pre-materialised + auto-materialise flow flags and no `--encoding-partition`
- **THEN** the resulting `auto_materialise_meta.json` content is byte-identical to the pre-change reference except for a single new field `"encoding_partition_sha256": null`; the frontier.jsonl is byte-identical except for a single new field `"partition_label": null` per row
