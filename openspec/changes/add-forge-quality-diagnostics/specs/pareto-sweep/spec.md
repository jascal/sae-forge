# pareto-sweep Specification (delta)

## MODIFIED Requirements

### Requirement: ParetoFrontierRow dataclass

The `saeforge.sweep.ParetoFrontierRow` SHALL retain all existing fields and gain four new optional fields that capture forge-feasibility diagnostics, populated regardless of forge outcome when the sweep can resolve `host_d_model`. The class SHALL continue to expose `.to_json_dict()` and `.from_json_dict(cls, data)`; `from_json_dict` SHALL accept dicts missing the new keys and default them to `None` (backwards compat with frontier.jsonl emitted by sae-forge prior to this change).

Row schema (in declaration order; the four diagnostic fields are appended at the end of the existing schema):

| Field | Type | Populated when host d_model resolved | Populated on row failure | Populated under --frontier-only |
|-------|------|--------------------------------------|--------------------------|---------------------------------|
| ... existing fields (encoding_label through error_message) ... | ... | unchanged | unchanged | unchanged |
| **`host_d_model`** | **`int \| None`** | **populated** | **populated** | **populated** |
| **`basis_rank`** | **`int \| None`** | **populated** | **populated** | **populated** |
| **`quality_ratio`** | **`float \| None`** | **populated** | **populated** | **populated** |
| **`quality_tier`** | **`str \| None`** | **populated** (one of `saturated` / `good` / `undersized` / `degenerate`) | **populated** | **populated** |

The four diagnostic fields describe the *structural setup* the forge inherited — not the forge's output. They SHALL be populated even when the forge raised (the row's `error_message` is non-None) because the diagnostic helps the analyst distinguish "forge errored due to bug" from "forge inherited a structurally doomed basis."

When `host_d_model` cannot be resolved (network error, unsupported host, no transformers import), all four diagnostic fields SHALL be `None` and the sweep proceeds normally.

`__post_init__` validation:
- `quality_tier` when non-None SHALL be one of the four `QualityTier` string values.
- `quality_ratio` when non-None SHALL be `>= 0`.
- `host_d_model` when non-None SHALL be `>= 1`.
- `basis_rank` when non-None SHALL be `>= 0`.

#### Scenario: ParetoFrontierRow round-trips with diagnostic fields

- **WHEN** a row with `host_d_model=768`, `basis_rank=4`, `quality_ratio=0.0052`, `quality_tier="degenerate"` is serialised via `.to_json_dict()` and reconstructed via `.from_json_dict(...)`
- **THEN** the reconstructed instance equals the original

#### Scenario: ParetoFrontierRow.from_json_dict tolerates missing diagnostic keys

- **WHEN** a dict missing `host_d_model`, `basis_rank`, `quality_ratio`, `quality_tier` is passed to `from_json_dict`
- **THEN** the resulting instance has those fields set to `None` without raising (backwards compat with pre-change frontier.jsonl files)

#### Scenario: ParetoFrontierRow rejects invalid quality_tier

- **WHEN** `ParetoFrontierRow(..., quality_tier="bogus")` is constructed
- **THEN** `__post_init__` raises `ValueError`; message lists the four supported values

#### Scenario: diagnostic fields populated on row failure

- **GIVEN** a sweep where `pipeline.run` raises for a specific K
- **WHEN** the resulting row is inspected
- **THEN** `error_message` is non-None AND `host_d_model`, `basis_rank`, `quality_ratio`, `quality_tier` are all populated (the diagnostic was computed before the forge call)

#### Scenario: diagnostic fields populated under --frontier-only

- **GIVEN** a sweep invoked with `--frontier-only` and a resolvable host
- **WHEN** rows are emitted
- **THEN** all four diagnostic fields are populated; `faithfulness_kl` etc. remain `None` per the existing frontier-only contract

#### Scenario: diagnostic fields null when d_model cannot be resolved

- **GIVEN** a sweep against a host whose `AutoConfig.from_pretrained` fails (e.g., gated model, offline)
- **WHEN** the sweep runs
- **THEN** every row has `host_d_model is None` AND `basis_rank is None` AND `quality_ratio is None` AND `quality_tier is None`; no advisory is printed; the sweep proceeds to forge runs as it did before this change

### Requirement: CLI subcommand `saeforge sweep-pareto`

The existing `sweep-pareto` subcommand SHALL retain all its current flags and gain two new optional flags for forge-quality control:

- `--quality-floor RATIO` — when set, the sweep SHALL refuse (exit non-zero, raise before any forge call) if any encoding's smallest-K `quality_ratio` falls below RATIO. Refusal happens AFTER the advisory prints, so the user sees both the warning and the refusal reason. `RATIO` is a float in `[0, 1]`; invalid values exit non-zero at parse time.
- `--quality-tier-thresholds STR` — comma-separated `name:value` pairs overriding the default `QualityThresholds`. Form: `saturated:1.0,good:0.5,undersized:0.0625`. All three names must be present; values must satisfy `saturated > good > undersized >= 0`.

#### Scenario: --quality-floor refuses degenerate setup before any forge

- **GIVEN** a sweep whose smallest-K basis would have `quality_ratio < 0.1`
- **WHEN** invoked with `--quality-floor 0.5`
- **THEN** the CLI exits non-zero BEFORE any forge call (mock or instrument `pipeline.run` to assert zero invocations); stderr contains the computed ratio and the failing K

#### Scenario: --quality-floor accepts good setup

- **WHEN** invoked with `--quality-floor 0.5` against a setup where every K's `quality_ratio >= 0.5`
- **THEN** the sweep proceeds normally and exits 0

#### Scenario: --quality-floor out of range refused at parse time

- **WHEN** invoked with `--quality-floor 1.5` or `--quality-floor -0.1`
- **THEN** the CLI exits non-zero at parse time; no advisory, no sweep work

#### Scenario: --quality-tier-thresholds parses and applies

- **GIVEN** the CLI is invoked with `--quality-tier-thresholds saturated:2.0,good:1.0,undersized:0.25`
- **WHEN** rows are emitted from a sweep where `quality_ratio = 1.5`
- **THEN** `quality_tier == "good"` (1.5 falls in the new `[1.0, 2.0)` good band)

#### Scenario: --quality-tier-thresholds rejects malformed input

- **WHEN** invoked with `--quality-tier-thresholds bogus` or `--quality-tier-thresholds saturated:0.5,good:1.0` (ordering violation)
- **THEN** the CLI exits non-zero at parse time with a clear error message naming the malformation

## ADDED Requirements

### Requirement: Pre-flight quality advisory

When `sweep-pareto` runs against a resolvable host (`resolve_host_d_model(host_model_id)` returns a non-None value), the driver SHALL examine each encoding's smallest-K materialised SAE and compute its `basis_rank`. If the resulting `quality_tier` for any encoding's smallest K is `undersized` or `degenerate`, the driver SHALL print a stderr advisory before the first forge call.

The advisory message SHALL include:
- The affected encoding label
- The smallest K and its computed `basis_rank`
- The computed `quality_ratio` (`basis_rank / host_d_model`)
- The resolved `host_d_model`
- A suggested K floor — the smallest K from the encoding's manifest whose `basis_rank` meets the `good` threshold (defaults to `host_d_model / 2`)
- A one-line note that `degenerate`/`undersized` describes the rank ratio, not the validity of the run (exploratory smokes legitimately operate in low-rank regimes)

The advisory is informational by default. Refusal requires the explicit `--quality-floor` flag.

When `host_d_model` cannot be resolved, the advisory SHALL NOT be printed (the diagnostic data isn't available).

#### Scenario: advisory fires on degenerate smallest-K basis

- **GIVEN** a sweep against `gpt2` (host_d_model=768) where the smallest-K SAE has 1 surviving feature
- **WHEN** the sweep is invoked
- **THEN** stderr contains a multi-line advisory naming the encoding, K=N (the smallest target), basis_rank=1, ratio≈0.0013, host_d_model=768, and a suggested K floor (the smallest K from the manifest whose surviving-feature count is ≥ 384); the sweep then proceeds to forge runs

#### Scenario: advisory is silent on good setup

- **GIVEN** every encoding's smallest-K `quality_ratio >= 0.5`
- **WHEN** the sweep is invoked
- **THEN** no stderr advisory is printed; the sweep proceeds silently to forge runs

#### Scenario: advisory absent when d_model is unresolvable

- **GIVEN** `resolve_host_d_model` returns `None` (network failure, unsupported host)
- **WHEN** the sweep is invoked
- **THEN** no advisory is printed; the sweep proceeds; rows have `host_d_model is None` and the other three diagnostic fields are also `None`

#### Scenario: advisory suggests a sensible K floor from the manifest

- **GIVEN** a 4-K manifest where the per-K kept-feature counts are `[1, 1, 1, 4]` for K=`[4, 8, 16, 32]` (matching the live N=32 Rung4 smoke pattern) on a host_d_model=768
- **WHEN** the advisory fires
- **THEN** the suggested K floor names the smallest K whose kept-feature count would meet the `good` threshold — typically a value larger than the largest K present in the manifest, with the advisory noting that the user's `--pareto` list is entirely in the `degenerate` regime and that they should re-run polygram's compression with larger K targets
