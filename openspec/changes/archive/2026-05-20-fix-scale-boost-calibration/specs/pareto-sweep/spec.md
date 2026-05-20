# pareto-sweep Specification (delta)

## MODIFIED Requirements

### Requirement: ParetoFrontierRow dataclass

The `saeforge.sweep.ParetoFrontierRow` SHALL retain all existing fields
and gain two new optional fields that capture forge-magnitude
diagnostics, populated when the sweep ran with
`--magnitude-diagnostics`. The class SHALL continue to expose
`.to_json_dict()` and `.from_json_dict(cls, data)`; `from_json_dict`
SHALL accept dicts missing the new keys and default them to `None`
(backwards compat with frontier.jsonl emitted prior to this change).

Row schema additions (in declaration order; appended after the
auto-materialise provenance fields):

| Field | Type | Populated when --magnitude-diagnostics set | Populated on row failure | Populated under --frontier-only |
|-------|------|--------------------------------------------|--------------------------|---------------------------------|
| **`logit_std_ratio`** | **`float \| None`** | **populated** (forged std / host std, layer-L shortcut) | `None` (no projector built) | `None` |
| **`top1_anomalous`** | **`bool \| None`** | **populated** (mode top-1 prediction in the curated SolidGoldMagikarp set) | `None` | `None` |

The two fields describe the *projector's behaviour on the calibration
corpus* — a complement to `faithfulness_kl` (which measures the
fully-projected NativeModel on eval prompts). They are post-mortem
diagnostics: they explain why `faithfulness_kl` might be poor without
themselves changing any forge behaviour.

When `--magnitude-diagnostics` is NOT supplied, both fields SHALL
remain `None` and the sweep proceeds byte-identical to its pre-change
behaviour.

`__post_init__` validation:

- `logit_std_ratio` when non-None SHALL be `>= 0`.

#### Scenario: row round-trips with magnitude diagnostic fields

- **GIVEN** a `ParetoFrontierRow` constructed with both diagnostic
  fields populated (e.g. `logit_std_ratio=1.03`,
  `top1_anomalous=False`)
- **WHEN** `to_json_dict` and `from_json_dict` are applied in sequence
- **THEN** the resulting row equals the original

#### Scenario: legacy row without diagnostic fields parses cleanly

- **GIVEN** a JSON dict matching the row schema as emitted prior to
  this change (no diagnostic keys)
- **WHEN** `from_json_dict` parses it
- **THEN** the constructed row has both diagnostic fields as `None`

#### Scenario: negative logit_std_ratio rejected

- **GIVEN** a `ParetoFrontierRow` constructor call with
  `logit_std_ratio=-1.0`
- **WHEN** `__post_init__` runs
- **THEN** it raises `ValueError` naming `logit_std_ratio`

### Requirement: sweep-pareto CLI subcommand

The `sweep-pareto` CLI SHALL accept the existing flags plus two new
optional flags introduced by this change:

- `--magnitude-diagnostics VALUE`: opt-in forge-magnitude diagnostics.
  VALUE SHALL match one of `tokens:N` (use the built-in token-capped
  English corpus capped at `N` tokens) or `prompts:PATH` (load JSONL
  with `{"text": "..."}` per line). Any other format exits with code
  2 and a clear stderr message naming both legal forms. When set,
  every row's `logit_std_ratio` and `top1_anomalous` fields are
  populated and a post-sweep magnitude-diagnostics advisory is
  printed. SHALL require `--layer` to be set and a resolvable
  `--host-model`; otherwise exits with code 2.
- `--rank-monotonicity-check`: boolean flag. When set, after the
  sweep's row loop completes, the driver groups completed rows by
  `encoding_label`, sorts by `n_features_kept_actual` ascending, and
  for each adjacent pair where
  `faithfulness_kl[high] - faithfulness_kl[low] > 0.1` prints a
  stderr advisory listing the offending tuple. SHALL be advisory only
  — no refusal — so analysts can deliberately sweep at default
  `scale_boost=1.0` to characterise the documented non-monotonicity
  pattern. SHALL default to off.

Neither flag is required; absent both, sweep behaviour is byte-
identical to pre-change.

#### Scenario: --magnitude-diagnostics parses tokens:N

- **GIVEN** `--magnitude-diagnostics tokens:1024`
- **WHEN** the CLI parses argv
- **THEN** the resulting plan threads an int `n_tokens=1024` through to
  `pipeline.sweep_pareto(magnitude_diagnostics=1024, ...)`

#### Scenario: --magnitude-diagnostics parses prompts:PATH

- **GIVEN** `--magnitude-diagnostics prompts:/tmp/calib.jsonl` where
  the file exists
- **WHEN** the CLI parses argv
- **THEN** the resulting plan threads a `Path` through to
  `pipeline.sweep_pareto(magnitude_diagnostics=Path("/tmp/calib.jsonl"), ...)`

#### Scenario: --magnitude-diagnostics rejects bad format

- **GIVEN** `--magnitude-diagnostics bogus:value` (or `1024` without
  `:`)
- **WHEN** the CLI parses argv
- **THEN** the CLI exits non-zero (code 2) with a stderr message
  naming both legal forms (`tokens:N`, `prompts:PATH`)

#### Scenario: --magnitude-diagnostics requires --layer

- **GIVEN** `--magnitude-diagnostics tokens:512` but no `--layer`
- **WHEN** the CLI parses argv
- **THEN** the CLI exits non-zero (code 2) with a stderr message
  noting `--layer` is required and explaining why (must match the
  SAE's training layer)

#### Scenario: --rank-monotonicity-check on monotone rows is silent

- **GIVEN** a sweep run with `--rank-monotonicity-check` where every
  encoding's `faithfulness_kl` is non-increasing in
  `n_features_kept_actual` (within 0.1-nat tolerance)
- **WHEN** the sweep completes
- **THEN** no stderr advisory is printed

#### Scenario: --rank-monotonicity-check flags violations without refusing

- **GIVEN** a sweep run with `--rank-monotonicity-check` where one
  encoding's KL goes from 6.96 at K=25 to 55.6 at K=211 (the
  documented blow-up pattern)
- **WHEN** the sweep completes
- **THEN** stderr contains an advisory naming the encoding, the K
  pair, both KL values, and the delta; the sweep's return code is
  unchanged (no refusal); `frontier.jsonl` contains all rows

## ADDED Requirements

### Requirement: Post-sweep magnitude-diagnostics advisory

The sweep driver SHALL print a multi-line stderr advisory at the end
of each sweep whose rows have non-None `logit_std_ratio`. The advisory
fires when at least one row has `logit_std_ratio is not None` — i.e.
the sweep ran with `--magnitude-diagnostics`. It lists for each such
row:

- `encoding={label} K={n_features_kept_actual} logit_std_ratio={ratio:.4f}`

And for each row where `top1_anomalous is True`, an additional line:

- `[!] anomalous-token canary fired on encoding={label}, K={n_features_kept_actual}`

The advisory is post-sweep (not pre-flight) and complements the
existing pre-flight forge-quality advisory rather than replacing it.
When no rows have diagnostics, the advisory SHALL return `None` and
produce no output.

#### Scenario: advisory lists logit_std_ratio per diagnostic row

- **GIVEN** two rows under the same encoding with
  `logit_std_ratio=0.97` and `logit_std_ratio=23.0` respectively
- **WHEN** the post-sweep advisory runs
- **THEN** stderr contains both rows' ratios

#### Scenario: anomalous-token canary line per offending row

- **GIVEN** one row with `top1_anomalous=True` at encoding `hea`, K=16
- **WHEN** the post-sweep advisory runs
- **THEN** stderr contains the literal substring `[!] anomalous-token
  canary fired on encoding=hea, K=16`

#### Scenario: advisory silent on a no-diagnostics sweep

- **GIVEN** a sweep where no row's `logit_std_ratio` is non-None
  (legacy invocation without `--magnitude-diagnostics`)
- **WHEN** the post-sweep advisory runs
- **THEN** the advisory string is `None` and no stderr output is
  produced
