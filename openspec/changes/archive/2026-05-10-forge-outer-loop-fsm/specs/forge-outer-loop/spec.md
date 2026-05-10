# forge-outer-loop Specification

## Purpose

The `forge-outer-loop` capability defines an orca-lang FSM that drives
the full forge pipeline — load → compress → optional regrow → project →
fine-tune → evaluate → optional iterate — with statically verifiable
control flow. It replaces the v0 `ForgePipeline.run` imperative
orchestrator with a machine-checked equivalent while keeping every
numerical action behind a Python function that is unit-testable in
isolation.

## Requirements

### Requirement: Canonical machine ships with the package

sae-forge SHALL ship the FSM definition at
`saeforge/machines/sae_forge.orca.md` as a package data file. The
orchestrator SHALL load it via `importlib.resources.files`, never via a
filesystem path, so the machine works under wheel installs, zipapp
installs, and editable installs equally.

#### Scenario: machine loads from an installed wheel

- **GIVEN** sae-forge installed via `pip install sae-forge[orca]` into a
  fresh venv
- **WHEN** `python -c "from saeforge.orchestrator import _load_machine;
  _load_machine()"` is run
- **THEN** the import succeeds and `_load_machine()` returns a
  non-empty machine object whose `name` is `"SaeForge"`

### Requirement: Machine has nine states with the v0.1 topology

The machine SHALL declare exactly nine states: `init` (initial),
`loaded`, `compressed`, `regrown`, `projected`, `finetuned`,
`evaluated`, `done` (final), `failed` (final).

The transition table SHALL include exactly these non-error edges:

| Source       | Event             | Guard                      | Target       |
|--------------|-------------------|----------------------------|--------------|
| `init`       | `start`           | —                          | `loaded`     |
| `loaded`     | `load_done`       | —                          | `compressed` |
| `compressed` | `compress_done`   | `regrow_count > 0`         | `regrown`    |
| `compressed` | `compress_done`   | `regrow_count == 0`        | `projected`  |
| `regrown`    | `regrowth_done`   | —                          | `projected`  |
| `projected`  | `projection_done` | —                          | `finetuned`  |
| `finetuned`  | `finetune_done`   | —                          | `evaluated`  |
| `evaluated`  | `eval_done`       | `should_continue_loop()`   | `compressed` |
| `evaluated`  | `eval_done`       | `!should_continue_loop()`  | `done`       |

Every state in `{loaded, compressed, regrown, projected, finetuned,
evaluated}` SHALL declare an additional `error → failed` transition.
`init`, `done`, and `failed` SHALL NOT declare error transitions.

#### Scenario: orca verifier accepts the machine

- **WHEN** `orca verify saeforge/machines/sae_forge.orca.md` is run
- **THEN** the verifier exits zero with no warnings about dead states,
  unreachable transitions, or guard-coverage gaps

#### Scenario: regrow-count branching is disjoint and total

- **WHEN** the orca verifier inspects the two `compressed →
  compress_done` transitions
- **THEN** it reports the guards `regrow_count > 0` and
  `regrow_count == 0` as a complete partition of the integer domain
  (no event firing leaves both branches unselected, and none fires
  both)

### Requirement: Single-pass default reaches `done` without re-entering `compressed`

When `iterations=1` and `regrow_count=0`, a successful forge SHALL
traverse the state sequence `init → loaded → compressed → projected →
finetuned → evaluated → done` exactly once. The `compressed` state is
entered once and never re-entered.

#### Scenario: single-pass transition log

- **GIVEN** a `ForgeContext` with `iterations=1`, `regrow_count=0`,
  and actions stubbed to succeed
- **WHEN** `saeforge.orchestrator.run_machine(ctx)` is invoked
- **THEN** the `transitions.jsonl` log contains exactly seven entries
  whose `to_state` values, in order, are `loaded`, `compressed`,
  `projected`, `finetuned`, `evaluated`, `done`, *and no further
  entries*
- **AND** no entry has `to_state == "compressed"` after index 1

### Requirement: Multi-pass loop respects `should_continue_loop`

When `iterations > 1`, the FSM SHALL loop `evaluated → compressed`
exactly when `should_continue_loop(ctx)` is true:

- `ctx.current_iter + 1 < ctx.iterations`, AND
- `ctx.faithfulness >= ctx.min_faithfulness`, AND
- `ctx.perplexity < ctx.best_perplexity`.

`increment_iter` SHALL run on the loop edge and update both
`current_iter` and `best_perplexity`.

#### Scenario: three-iter forge with monotone perplexity improvement

- **GIVEN** `iterations=3`, `min_faithfulness=0.0`, and stubbed actions
  where each `evaluate_faithfulness` writes a strictly-decreasing
  `perplexity` and `faithfulness=0.95`
- **WHEN** the orchestrator runs
- **THEN** the transition log shows three traversals of the loop edge
  `evaluated → compressed` followed by one terminal `evaluated → done`
- **AND** the final `ctx.current_iter == 2` and `ctx.best_perplexity`
  equals the smallest of the three observed perplexities

#### Scenario: faithfulness regression terminates early

- **GIVEN** `iterations=5`, `min_faithfulness=0.90`, and a stubbed
  `evaluate_faithfulness` that returns `faithfulness=0.85` on the
  second iteration
- **WHEN** the orchestrator runs
- **THEN** the FSM terminates at `done` after iteration 2 without
  entering a third `compressed` state

#### Scenario: perplexity stagnation terminates early

- **GIVEN** `iterations=5` and a stubbed `evaluate_faithfulness` that
  returns the same `perplexity` on iterations 1 and 2
- **WHEN** the orchestrator runs
- **THEN** `should_continue_loop` returns false at the end of iteration
  2 and the FSM terminates at `done`

### Requirement: Per-state errors reach `failed` cleanly

Any action raising an exception SHALL drive the FSM into `failed` via
the source state's `error → failed` transition. `log_error` SHALL run
on entry to `failed` and populate `ctx.error_message` with the
exception class name, message, and source state.

#### Scenario: action raise in `compressed` reaches `failed`

- **GIVEN** a stubbed `compress_with_polygram` that raises `RuntimeError("boom")`
- **WHEN** the orchestrator runs
- **THEN** the final state is `failed`
- **AND** `ctx.error_message` contains `"compressed"`, `"RuntimeError"`,
  and `"boom"`
- **AND** the transition log's last entry has `event="error"` and
  `from_state="compressed"`

### Requirement: Quantum-aware path leaves topology unchanged

The `--quantum-aware` flag SHALL set `ctx.quantum_aware = True`. The
FSM topology, transition table, and guard set SHALL be byte-identical
between `quantum_aware=True` and `quantum_aware=False` runs. The flag
is read **only** by `compress_with_polygram`, which uses it to select
Polygram's `confirmer` strategy.

#### Scenario: quantum-aware does not add states or transitions

- **WHEN** the machine is loaded under either `quantum_aware` setting
- **THEN** the state set, transition set, and guard table returned by
  the orca-lang runtime are equal across both settings (set equality)

#### Scenario: q-orca-lang is not imported on the default path

- **GIVEN** `quantum_aware=False` (the default)
- **WHEN** a single-pass forge runs to `done`
- **THEN** `q_orca_lang` does not appear in `sys.modules`

### Requirement: Imperative and FSM orchestrators produce identical output

`ForgePipeline(orchestrator="imperative")` and
`ForgePipeline(orchestrator="fsm")` SHALL produce byte-identical
forged-weight tensors when given the same `ForgeContext` inputs and
the same numpy / torch RNG seeds. This is the migration safety net for
v0.1 → v0.2.

#### Scenario: byte-identical toy forge

- **GIVEN** the toy GPT-2 example `examples/forge_gpt2_toy.py` with
  RNG seed `0`
- **WHEN** the example is run twice — once with
  `orchestrator="imperative"` and once with `orchestrator="fsm"` — and
  both produce a `forged.safetensors` file
- **THEN** the two files have identical SHA-256 hashes

### Requirement: orca-lang is opt-in via the `[orca]` extra

The default `pip install sae-forge` SHALL NOT pull in `orca-lang`.
`saeforge.orchestrator` SHALL lazy-import `orca_lang.runtime` and raise
`ImportError` (via `saeforge.utils.lazy.require_extra`) with a message
naming the `[orca]` extra when the dependency is missing.

#### Scenario: missing extra produces a clear error

- **GIVEN** sae-forge installed without the `[orca]` extra
- **WHEN** `saeforge.orchestrator.run_machine(ctx)` is called
- **THEN** `ImportError` is raised whose message contains both
  `orca-lang` and `[orca]`

### Requirement: Default CLI path stays imperative in v0.1

`sae-forge forge` without the `--fsm` flag SHALL run the imperative
`ForgePipeline.run` from v0 unchanged. `--fsm` SHALL be the only
opt-in for the FSM path in v0.1. v0.2 promotes `--fsm` to default and
deprecates the imperative path; v1.0 removes it.

#### Scenario: --fsm flag exists and toggles the orchestrator

- **WHEN** `sae-forge forge --help` is run
- **THEN** the help text lists `--fsm` with a description naming the
  orca-lang FSM orchestrator
- **AND** invoking `sae-forge forge ... --fsm` constructs the pipeline
  with `orchestrator="fsm"`
