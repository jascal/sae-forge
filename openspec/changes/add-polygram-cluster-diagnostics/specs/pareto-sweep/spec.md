# pareto-sweep Specification (delta)

## MODIFIED Requirements

### Requirement: ParetoFrontierRow dataclass

The `saeforge.sweep.ParetoFrontierRow` SHALL retain all existing fields (including the four forge-quality diagnostic fields from `add-forge-quality-diagnostics`) and gain four new optional polygram-side concept-structure diagnostic fields, populated when the per-row basis comes from a polygram-compressed checkpoint and the compression report is loadable. The class SHALL continue to expose `.to_json_dict()` and `.from_json_dict(cls, data)`; `from_json_dict` SHALL accept dicts missing the new keys and default them to `None` (backwards compat with `frontier.jsonl` files emitted by sae-forge prior to this change).

Row schema additions (declared at the end of the existing schema, immediately after the four `add-forge-quality-diagnostics` fields):

| Field | Type | Populated when polygram report is loadable | Populated on row failure | Populated under --frontier-only |
|-------|------|---------------------------------------------|--------------------------|---------------------------------|
| **`polygram_n_clusters`** | **`int \| None`** | **populated** (from `compression_report.json["n_clusters"]`) | **populated** | **populated** |
| **`polygram_n_zeroed`** | **`int \| None`** | **populated when the report field exists** (older polygram outputs may omit it) | **populated** | **populated** |
| **`polygram_redundancy_ratio`** | **`float \| None`** | **populated** when both `n_clusters` and `n_zeroed` are non-None and sum > 0 | **populated** | **populated** |
| **`polygram_encoding_capacity`** | **`int \| None`** | **populated** for known encodings (Rung3/Rung4/Rung5/HEA_Rung2); `None` for unknown encodings | **populated** | **populated** |

The polygram fields describe the *concept-structure of the dictionary* the forge inherited — not the forge's output. They SHALL be populated even when the forge raised (the row's `error_message` is non-None), for the same reason the rank-side diagnostic fields are: the analyst needs to distinguish bug from doomed-input.

When the polygram report is missing, malformed, or pre-dates a field, the corresponding row fields are `None` and the sweep proceeds normally. The polygram diagnostic fields are independent of the rank-side diagnostic fields — either set may be populated while the other is `None`.

`__post_init__` validation:
- `polygram_n_clusters` when non-None SHALL be `>= 0`.
- `polygram_n_zeroed` when non-None SHALL be `>= 0`.
- `polygram_redundancy_ratio` when non-None SHALL be in `[0.0, 1.0]`.
- `polygram_encoding_capacity` when non-None SHALL be `>= 1`.

#### Scenario: ParetoFrontierRow round-trips with polygram fields

- **WHEN** a row with `polygram_n_clusters=6`, `polygram_n_zeroed=88`, `polygram_redundancy_ratio=0.936`, `polygram_encoding_capacity=128` is serialised via `.to_json_dict()` and reconstructed via `.from_json_dict(...)`
- **THEN** the reconstructed instance equals the original

#### Scenario: ParetoFrontierRow.from_json_dict tolerates missing polygram keys

- **WHEN** a dict missing `polygram_n_clusters`, `polygram_n_zeroed`, `polygram_redundancy_ratio`, `polygram_encoding_capacity` is passed to `from_json_dict`
- **THEN** the resulting instance has those fields set to `None` without raising (backwards compat with pre-change `frontier.jsonl` files)

#### Scenario: ParetoFrontierRow rejects negative polygram counts

- **WHEN** `ParetoFrontierRow(..., polygram_n_clusters=-1)` or `ParetoFrontierRow(..., polygram_n_zeroed=-3)` is constructed
- **THEN** `__post_init__` raises `ValueError`; message names the offending field

#### Scenario: ParetoFrontierRow rejects out-of-range polygram_redundancy_ratio

- **WHEN** `ParetoFrontierRow(..., polygram_redundancy_ratio=1.5)` or `polygram_redundancy_ratio=-0.1` is constructed
- **THEN** `__post_init__` raises `ValueError`; message states the valid range `[0.0, 1.0]`

#### Scenario: polygram fields populated on row failure

- **GIVEN** a sweep where `pipeline.run` raises for a specific K AND the basis was loaded from a polygram-compressed checkpoint with a loadable report
- **WHEN** the resulting row is inspected
- **THEN** `error_message` is non-None AND all four polygram diagnostic fields are populated (the diagnostic was computed before the forge call)

#### Scenario: polygram fields null when report is absent

- **GIVEN** a sweep against an SAE checkpoint with NO colocated compression report
- **WHEN** rows are emitted
- **THEN** every row has `polygram_n_clusters is None`, `polygram_n_zeroed is None`, `polygram_redundancy_ratio is None`; the sweep proceeds to forge runs as it did before this change. `polygram_encoding_capacity` MAY still be non-None (it's resolved from the encoding spec, not the report).

#### Scenario: polygram_redundancy_ratio is None when n_zeroed is missing from older reports

- **GIVEN** a polygram report that contains `n_clusters` but not `n_zeroed` (older polygram output)
- **WHEN** the row is emitted
- **THEN** `polygram_n_clusters` is populated, `polygram_n_zeroed` is `None`, `polygram_redundancy_ratio` is `None`. The sweep proceeds; no advisory regression.

### Requirement: Pre-flight quality advisory

The `advise_sweep_quality` function from `add-forge-quality-diagnostics` SHALL retain its existing rank-tier advisory behaviour AND SHALL append a polygram-side **saturation note** when the largest-K SAE in any encoding's manifest reports `polygram_n_clusters == polygram_encoding_capacity` and both values are non-None.

The saturation-note text SHALL match this template (single line, terminated by newline):

`Note: polygram_n_clusters ({n_clusters}) equals encoding capacity ({capacity}) — the encoding may be saturated. Consider re-running polygram compress with a larger encoding ({suggested_next_encoding}) to see whether additional concepts are present.`

Suggested-next-encoding rules:
- `rung3` (cap 16) → `Rung4`
- `rung4` (cap 32) → `Rung5`
- `rung5` (cap 128) → `HEA_Rung2(n_qubits=8)` (cap 256, the next power-of-two)
- `hea_rung2(n_qubits=N)` → `HEA_Rung2(n_qubits=N+1)`

The saturation note SHALL be appended to the existing rank-tier advisory when both kinds of advisory apply. When the rank-tier check is silent (good/saturated) but cluster saturation fires, `advise_sweep_quality` SHALL return a single-line advisory containing only the saturation note (instead of returning `None`).

The saturation check SHALL NEVER trigger `--quality-floor` refusal. `--quality-floor` continues to react only to `quality_ratio`, never to polygram fields. Cluster saturation is *descriptive*, not a gate.

When `polygram_encoding_capacity` is `None` (unknown encoding spec) OR the report is unloadable, the saturation check SHALL NOT fire (no false-positive saturation notes).

#### Scenario: saturation note appended when n_clusters equals capacity

- **GIVEN** a sweep with `--encoding rung5` whose largest-K SAE has a polygram report containing `n_clusters=128`
- **WHEN** the advisory is built
- **THEN** the returned advisory string contains the literal phrase `polygram_n_clusters (128) equals encoding capacity (128) — the encoding may be saturated` AND names `HEA_Rung2(n_qubits=8)` as the suggested next encoding

#### Scenario: saturation note alone when no rank-tier advisory was warranted

- **GIVEN** a sweep whose smallest-K basis is in the `good` tier (no rank-tier advisory) AND whose largest-K SAE reports `n_clusters == capacity`
- **WHEN** `advise_sweep_quality` is called
- **THEN** the function returns a non-None single-line string containing only the saturation note

#### Scenario: no saturation note when clusters below capacity

- **GIVEN** a `rung5` sweep whose largest-K SAE reports `n_clusters=6`
- **WHEN** the advisory is built
- **THEN** the advisory body does NOT contain the saturation-note wording; the rank-tier message (if any) is unchanged

#### Scenario: no saturation note when capacity is unknown

- **GIVEN** a sweep invoked with an encoding spec that does NOT parse to a known capacity (e.g., a future encoding family)
- **WHEN** the advisory is built
- **THEN** the saturation check is skipped; rank-tier behaviour is unchanged

#### Scenario: --quality-floor ignores polygram saturation

- **GIVEN** a sweep where `quality_ratio` is in the `good` tier AND `polygram_n_clusters == polygram_encoding_capacity`
- **WHEN** invoked with `--quality-floor 0.5`
- **THEN** the sweep proceeds to forge runs (`--quality-floor` reacts only to `quality_ratio`); stderr contains the saturation note; exit code is 0
