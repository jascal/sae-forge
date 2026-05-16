## 0. Polygram-side prerequisite (blocking)

- [ ] 0.1 Open the `add-encoding-partition` proposal in the polygram repo. This sae-forge proposal cannot leave Phase 2 until that polygram capability ships and is pinned via `polygram>=0.9.0` (or whichever version lands the contract).
- [ ] 0.2 Lock the polygram-side API surface at: `CompressionConfig.encoding_partition: list[BlockSpec] | None`, `BlockSpec(encoding_class, encoding_kwargs, learn_axis_assignment, feature_ids)`, `from_sae_lens(..., encoding_partition=...)`. Any drift in this shape blocks Phase 2 here.

## 1. `saeforge/partition.py` — manifest parser + label formatter

- [ ] 1.1 New module `saeforge/partition.py`. Numpy-only, no torch / polygram imports at module scope (importable in slim envs).
- [ ] 1.2 `BlockSpec` dataclass: `block_id: str`, `encoding_class: str`, `encoding_kwargs: dict[str, Any]`, `learn_axis_assignment: bool`, `feature_ids: list[int]`.
- [ ] 1.3 `EncodingPartition` dataclass: `label: str`, `blocks: list[BlockSpec]`, `manifest_sha256: str`, `manifest_path: Path`.
- [ ] 1.4 `parse_partition_manifest(path: Path) -> EncodingPartition`:
  - Read file bytes (preserves the SHA-256 over the raw bytes — see design Decision 2).
  - JSON-parse and validate top-level shape: `{"label": str, "partition": [block, ...]}`.
  - Per block: validate `block_id: str`, `encoding_class` in the supported set (`MPSRung1`, `Rung3`, `Rung4`, `Rung5`, `HEA_Rung2`), `feature_ids: list[int]` non-empty.
  - Per-block kwargs validation: `Rung5` SHALL have `encoding_kwargs["n_amp_qubits"]: int`; `HEA_Rung2` SHALL have `encoding_kwargs["n_qubits"]: int`. Same shape as the existing `--encoding-amp-qubits` / `--encoding-qubits` validation in `cli.py`.
  - `learn_axis_assignment` defaults to `False` when absent.
- [ ] 1.5 `validate_coverage(partition: EncodingPartition, *, expected_feature_count: int) -> None`:
  - Disjointness: no feature id appears in more than one block. On violation, raise `PartitionError` naming the duplicated ids (capped at first 10).
  - Completeness: the union of all blocks' `feature_ids` equals `set(range(expected_feature_count))`. On violation, raise `PartitionError` naming missing/extra ids (capped at first 10).
  - This check requires `expected_feature_count`, available only at materialise time after the SAE is loaded. The parser does NOT call this; the materialise driver does.
- [ ] 1.6 `format_partition_label(partition: EncodingPartition) -> str`:
  - Per block, build `block_id + ":" + encoding_class + suffix` where `suffix` is `""` for `MPSRung1`/`Rung3`/`Rung4` with `learn_axis_assignment=False`, `"(learn)"` for those with `learn_axis_assignment=True`, `"(k=K)"` for `Rung5` without learn, `"(k=K,learn)"` for `Rung5` with learn, `"(n=N)"` / `"(n=N,learn)"` for `HEA_Rung2`.
  - Join blocks with `+` in manifest order.
  - Deterministic: same manifest content → same label.
- [ ] 1.7 Round-trip test: parse a known-good manifest, format-label, assert string matches the design.md examples.
- [ ] 1.8 Refusal tests: overlapping ids, missing ids, unknown encoding class, `Rung5` block missing `n_amp_qubits`, `HEA_Rung2` block missing `n_qubits`, empty `feature_ids` list, duplicate `block_id`.

## 2. `auto_materialise.py` — partition-aware materialisation

- [ ] 2.1 Extend `AutoMaterialiseSpec` with optional `encoding_partition: EncodingPartition | None = None`. When set, `encoding_class` / `encoding_kwargs` on the spec SHALL be `None` (mutually exclusive at the dataclass level).
- [ ] 2.2 Extend `compute_cache_key` to record `encoding_partition_sha256: str | None` (the manifest's SHA-256) when `spec.encoding_partition is not None`. When `None`, the field is recorded as `None` (always present, for consistency with `assign_phase_knobs` / `learn_axis_assignment`).
- [ ] 2.3 In `_run_materialisation_chain`, when `spec.encoding_partition is not None`:
  - Call `validate_coverage(partition, expected_feature_count=len(records))` immediately after loading the SAE. On `PartitionError`, re-raise with the encoding label prepended.
  - Build the polygram `encoding_partition` argument from the manifest's `BlockSpec` list. Pass it to `from_sae_lens(..., encoding_partition=...)` instead of the single-encoding `encoding=` kwarg.
  - The `learn_axis_assignment` kwarg to `from_sae_lens` SHALL be `False` (the per-block flag lives inside `encoding_partition`).
- [ ] 2.4 `is_cache_hit` already diffs all cache-key fields; the new `encoding_partition_sha256` is picked up automatically. Confirm via a parametrised test: flip the manifest SHA and assert `MISS (encoding_partition_sha256)`.
- [ ] 2.5 `materialise(...)` signature gains no new positional kwargs (everything rides on `spec.encoding_partition`).
- [ ] 2.6 Plan-only output: when `spec.encoding_partition is not None`, the per-encoding `--plan-only` block SHALL print:
  - `partition_manifest_sha256: <hex>`
  - `partition_label: <formatted label>`
  - `block_count: N`
  - and SHALL omit `encoding_class` / `encoding_kwargs` (block-level, not encoding-level).

## 3. `sweep.py` — `ParetoFrontierRow.partition_label`

- [ ] 3.1 Add `partition_label: str | None = None` to `ParetoFrontierRow`.
- [ ] 3.2 `to_json_dict` / `from_json_dict` round-trip the new field. `from_json_dict` SHALL tolerate dicts missing the key (defaults to `None`).
- [ ] 3.3 Update the schema table in `openspec/changes/add-block-structured-sae/specs/pareto-sweep/spec.md` (this proposal's spec delta).
- [ ] 3.4 `_process_row` propagates the partition label from the auto-materialise spec when present; `None` otherwise.
- [ ] 3.5 Backwards-compat test: load a frontier.jsonl emitted before this change; assert `partition_label is None` on every row.

## 4. `forge.py` — `ForgePipeline.sweep_pareto` pass-through

- [ ] 4.1 `sweep_pareto(...)` gains an internal `auto_materialise_specs` interpretation extension: a spec with `encoding_partition` set bypasses the single-encoding propagation. No new public kwargs on `sweep_pareto`.
- [ ] 4.2 Ensure `validation_threshold`, `encoding_class`, `validation_eval_overlap`, and now `partition_label` are propagated to every row produced by the partitioned label.
- [ ] 4.3 When `spec.encoding_partition is not None`, the row's `encoding_class` field SHALL be set to the literal string `"BlockStructured"` (so single-encoding consumers grouping by `encoding_class` see this as a distinct bucket rather than `None`). The `partition_label` carries the actual content.

## 5. CLI surface

- [ ] 5.1 Add `--encoding-partition LABEL:PATH` to the `sweep-pareto` subparser. Repeatable. Documented as auto-materialise-only.
- [ ] 5.2 Parse each `--encoding-partition` entry into `(label, Path)` pairs at argparse time. Build the `EncodingPartition` via `parse_partition_manifest(path)`.
- [ ] 5.3 Refusal in `_cmd_sweep_pareto`:
  - `--encoding-partition` set without `--auto-materialise` → exit non-zero; stderr names `--auto-materialise` as the gating flag.
  - `--encoding-partition LABEL:...` AND any of `--encoding-class LABEL:...`, `--encoding-amp-qubits LABEL:...`, `--encoding-qubits LABEL:...` for the same `LABEL` → exit non-zero; stderr lists the conflicting flags.
  - `--encoding-partition LABEL:...` for a label not in `--encoding` → exit non-zero; stderr names the unknown label.
  - All labels have `--encoding-partition` AND `--learn-axis-assignment` is set → exit non-zero (the global flag is meaningless when every label owns its own per-block setting). When a mixed sweep has some partitioned and some single-encoding labels, the global flag applies to the non-partitioned ones and is permitted; this is documented in the refusal message text.
- [ ] 5.4 When `--encoding-partition LABEL:PATH` is set, the corresponding `AutoMaterialiseSpec` for that label SHALL be built with `encoding_class=None`, `encoding_kwargs=None`, `encoding_partition=<parsed EncodingPartition>`.
- [ ] 5.5 `--plan-only` output extends per Decision in Phase 2.6.

## 6. Tests (saeforge/tests)

- [ ] 6.1 `tests/test_partition.py` — covers Phase 1.7 / 1.8 (parser, label formatter, coverage validator).
- [ ] 6.2 `tests/test_auto_materialise_partition.py` — uses a tiny in-memory SAE fixture and a mocked polygram seam to assert:
  - Partition flows through to `from_sae_lens(encoding_partition=...)`.
  - Cache key includes the manifest SHA; flipping SHA invalidates the cache.
  - Coverage violation at materialise time raises with the label prefix.
- [ ] 6.3 `tests/test_cli_partition.py` — argparse-level tests for every refusal in Phase 5.3, plus a successful parse-time path with two encodings (one partitioned, one single).
- [ ] 6.4 `tests/test_sweep_partition.py` — `ParetoFrontierRow.partition_label` round-trip + propagation from a mocked `auto_materialise_specs`.
- [ ] 6.5 Smoke: GPT-2 layer 8 `--plan-only` with a real two-block manifest on the Axis-4 cell. Cache cold → MISS, then cache warm → HIT. No host-model forward passes.

## 7. Falsifiable acceptance gate

- [ ] 7.1 Author the two paired manifests for the Intel cell:
  - `manifests/gpt2_l8_heavy_rung5_tail_mps.json` — heavy block: `Rung5(k=4)`, `learn_axis_assignment=True`. Tail block: `MPSRung1`, `learn_axis_assignment=False`. Heavy/tail split via `polygram.partition_by_firing_geometry(n_blocks=2, ...)`.
  - `manifests/gpt2_l8_heavy_learn_tail_no_learn.json` — both blocks `Rung5(k=4)`. Heavy has `learn=True`, tail has `learn=False`. Same heavy/tail split.
- [ ] 7.2 Run the four-cell sweep on Intel:
  - `single:Rung5(k=4)` baseline.
  - `single:Rung5(k=4,learn)` baseline.
  - `block(heavy:Rung5(k=4,learn) + tail:MPSRung1)`.
  - `block(heavy:Rung5(k=4,learn) + tail:Rung5(k=4))`.
- [ ] 7.3 Frontier consumer: at K ∈ {25, 50, 100, 211}, assert each block-structured row's `(n_features_kept_actual, faithfulness_kl)` strictly Pareto-dominates the matched single-encoding baseline on at least one K. Capacity heterogeneity (cell 3 vs cell 1) is the primary gate; learn-axis heterogeneity (cell 4 vs cell 2) is informational.
- [ ] 7.4 If the primary gate fails on every K, archive the proposal to `openspec/changes/archive/<date>-add-block-structured-sae/` and update `openspec/changes/archive/README.md` (if one exists) noting the killed conclusion.
- [ ] 7.5 If the primary gate passes, repeat the comparison on M4 with Gemma-2-2B layer 12 SAE before declaring the capability shippable.

## 8. Docs

- [ ] 8.1 README example showing a two-block manifest, the `--encoding-partition` invocation, and the resulting `partition_label` in a frontier row.
- [ ] 8.2 CLI `--help` text for `--encoding-partition` references `add-encoding-partition` (polygram-side) as the gating dependency.
- [ ] 8.3 Update `docs/research/` runbook with the Axis-4 partition cell.
