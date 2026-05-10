# hierarchical-fsm Specification

## Purpose

The `hierarchical-fsm` capability defines the composed three-machine
forge FSM that supersedes the v0.2 flat ten-state machine. It
specifies which states live in which sub-machine, how compound
states invoke their child machines, how the orchestrator loads and
composes the three orca files, how the runtime tracks the current
machine path on context, and how the auto-generated Mermaid
visualization reflects the parsed hierarchy.

This capability does NOT redefine the action signatures, the ctx
field semantics, or the guard truth tables — those are inherited
from `forge-outer-loop` and modified there. This capability is
purely about the *structural* shape of the FSM.

## ADDED Requirements

### Requirement: Three sub-machines compose into one runtime hierarchy

The package SHALL ship exactly three orca machine files under
`saeforge/machines/`:

- `stream.orca.md` — declares `# machine StreamMachine`
- `refine.orca.md` — declares `# machine RefineMachine`
- `basis.orca.md` — declares `# machine BasisMachine`

The orchestrator's `load_machine_hierarchy()` SHALL read all three
files, concatenate them with a `\n---\n` separator, and pass the
concatenated text to `orca_runtime_python.parse_orca_md` (or the
runtime's documented multi-file equivalent). The parsed result
SHALL be a single composed machine definition where:

- `StreamMachine.streaming` is a compound state that declares
  `> invoke: RefineMachine` in its body.
- `RefineMachine.refining` is a compound state that declares
  `> invoke: BasisMachine` in its body.
- `BasisMachine` declares no `invoke:` directives (it is the leaf).

The package SHALL NOT ship a flat `sae_forge.orca.md` file. The
v0.2 file of that name SHALL be deleted in the same commit that
adds the three replacements.

#### Scenario: package ships exactly the three machine files

- **WHEN** the package is installed
- **THEN** `importlib.resources.files("saeforge.machines")` exposes
  `stream.orca.md`, `refine.orca.md`, `basis.orca.md`
- **AND** does NOT expose `sae_forge.orca.md`

#### Scenario: composed parse resolves all invoke directives

- **WHEN** `load_machine_hierarchy()` is called
- **THEN** the returned definition's compound-state map links
  `StreamMachine.streaming` to `RefineMachine` and
  `RefineMachine.refining` to `BasisMachine`
- **AND** no unresolved `invoke:` reference is present

#### Scenario: orca topology checker passes on the composed hierarchy

- **GIVEN** the three sub-machine files
- **WHEN** the orca verifier runs against each file in isolation
- **THEN** each verifier call returns no errors
- **AND** the same verifier run against the concatenated three-file
  text also returns no errors

### Requirement: State identifiers are stable across the v0.2 → hierarchy migration

The hierarchy SHALL NOT change the state identifiers observable to
consumers of `transitions_log`. The v0.2 state names that participate
in the basis loop and projection / fine-tune / evaluate phases SHALL
be preserved:

- `compressed`, `regrown`, `projected`, `finetuned` SHALL exist
  in `BasisMachine` (or in `RefineMachine.evaluating` for
  evaluation-adjacent states).
- `evaluated` is renamed to `RefineMachine.evaluating` but the
  `transitions_log` action entries SHALL continue to record
  `state: "evaluated"` for backward compatibility (the action
  helper `_log` reads from a stable per-state name table, not
  directly from the runtime state name).

`loaded` and `activations_scanned` SHALL be collapsed into a
single `RefineMachine.entering` state whose action is the
composed `load_and_scan` helper. This is the single observable
state-name change. The `transitions_log` for the entering action
SHALL record two entries (`load_sae_and_corpus` then
`scan_activations`) — matching the v0.2 log shape — even though
the FSM is in a single state for the duration. This preserves
byte-equivalence with the imperative reference path.

#### Scenario: transitions_log preserves the v0.2 action sequence

- **GIVEN** a forge run with default knobs (single shard, no
  regrow, no protect)
- **WHEN** the run completes via the hierarchy
- **THEN** the sequence of `action` field values in
  `transitions_log` is exactly:
  `["load_sae_and_corpus", "scan_activations",
  "compress_with_polygram", "project_to_subspace",
  "fine_tune_model", "evaluate_faithfulness", "save_final_model"]`
- **AND** this sequence equals the sequence produced by the
  imperative reference path on the same seed

### Requirement: Runtime tracks current machine path on ctx

The orca runtime, while traversing compound states, SHALL update
the ctx key `_machine_path` to a forward-slash-separated string of
machine names from outermost to innermost:

- `_machine_path == "stream"` while the active state lives in
  `StreamMachine` directly.
- `_machine_path == "stream/refine"` while the active state lives
  in `RefineMachine` (under `StreamMachine.streaming`).
- `_machine_path == "stream/refine/basis"` while the active state
  lives in `BasisMachine` (under `RefineMachine.refining`).

Actions SHALL treat `_machine_path` as read-only — they SHALL NOT
mutate it. The action helper `_log` SHALL read
`ctx.get("_machine_path", "stream")` and append it to every
`transitions_log` entry as the `machine_path` key.

The initial ctx built by `ForgePipeline._build_fsm_ctx` SHALL set
`_machine_path = "stream"`.

#### Scenario: machine_path is recorded on every log entry

- **GIVEN** a forge run with default knobs
- **WHEN** the run completes
- **THEN** every entry in `transitions_log` has a `machine_path`
  string field
- **AND** `compress_with_polygram` entries record
  `machine_path == "stream/refine/basis"`
- **AND** `evaluate_faithfulness` entries record
  `machine_path == "stream/refine"`
- **AND** `save_final_model` entries record
  `machine_path == "stream"`

#### Scenario: actions do not mutate _machine_path

- **GIVEN** any registered action's return delta
- **WHEN** the orchestrator merges the delta into ctx
- **THEN** the delta does NOT contain `_machine_path` as a key
  (asserted via a static check on the action implementations and a
  runtime guard in `_log`)

### Requirement: Mermaid visualization reflects the parsed hierarchy

The package SHALL ship `saeforge/machines/visualize.py` exposing
`to_mermaid(hierarchy_def) -> str`. The function SHALL:

1. Accept the parsed composed definition returned by
   `load_machine_hierarchy()`.
2. Emit a single `stateDiagram-v2` Mermaid block.
3. Render each sub-machine as a nested `state X { ... }` subgraph
   with the parent's compound state as the outer container.
4. Include every state, every transition, and every guard label
   from all three machines in the block.

The CLI flag `sae-forge inspect --fsm-diagram` SHALL call
`to_mermaid(load_machine_hierarchy())` and write the result to
stdout. The flag SHALL be mutually exclusive with `--print-spec`.

#### Scenario: emitted Mermaid contains every state in the hierarchy

- **WHEN** `to_mermaid(load_machine_hierarchy())` runs
- **THEN** the returned string contains the substring
  `stateDiagram-v2`
- **AND** contains state-name tokens for all sixteen states across
  the three machines (5 stream + 5 refine + 6 basis)
- **AND** contains the substring `state "RefineMachine"` nested
  inside the `StreamMachine` block
- **AND** contains the substring `state "BasisMachine"` nested
  inside the `RefineMachine` block

#### Scenario: CLI flag emits the diagram to stdout

- **WHEN** the user runs `sae-forge inspect --fsm-diagram`
- **THEN** the process exits with code 0
- **AND** stdout contains a non-empty `stateDiagram-v2` block
- **AND** stderr is empty

#### Scenario: --fsm-diagram is mutually exclusive with --print-spec

- **WHEN** the user runs `sae-forge inspect --fsm-diagram --print-spec`
- **THEN** argparse rejects the invocation with a non-zero exit
  code
- **AND** the error message names both flags

### Requirement: Sub-machines are independently runnable for testing

Each of the three sub-machines SHALL be parseable and executable
in isolation (without its parent or child) for unit-testing
purposes. The runtime SHALL accept a single sub-machine file and
run it to completion under a stubbed ctx that satisfies its
declared field contract.

When a sub-machine is run in isolation:

- `BasisMachine` SHALL run from `compressed` to `done` given the
  ctx fields it owns plus the actions it invokes.
- `RefineMachine` SHALL run from `entering` to `exiting` given a
  stub for the `> invoke: BasisMachine` directive (a no-op compound
  state that immediately fires the parent's `basis_done` event).
- `StreamMachine` SHALL run from `init` to `done` given a stub for
  the `> invoke: RefineMachine` directive.

This is a *test* requirement, not a production requirement —
production always runs the composed hierarchy. The isolation tests
exist so that bugs surface inside the smallest possible machine.

#### Scenario: BasisMachine isolation test reaches done

- **GIVEN** `BasisMachine` parsed from `basis.orca.md` alone
- **AND** a stubbed ctx with `inner_refine_passes = 1`,
  `regrow_count = 0`, and stub implementations of
  `compress_with_polygram`, `project_to_subspace`, `fine_tune_model`
- **WHEN** the runtime drives the machine from `compressed`
- **THEN** the machine reaches the `done` final state
- **AND** the `transitions_log` records the four basis actions in
  order: `compress_with_polygram`, `project_to_subspace`,
  `fine_tune_model`

#### Scenario: RefineMachine isolation test reaches exiting

- **GIVEN** `RefineMachine` parsed from `refine.orca.md` alone
- **AND** a stub that replaces the `> invoke: BasisMachine` body
  with a single-state pass-through
- **AND** a stubbed ctx with `iterations = 1`,
  `should_continue = False`
- **WHEN** the runtime drives the machine from `entering`
- **THEN** the machine reaches the `exiting` final state
- **AND** `transitions_log` includes `load_sae_and_corpus`,
  `scan_activations`, `evaluate_faithfulness`

### Requirement: Failure-state propagation matches v0.2 single-failed semantics

Each sub-machine SHALL declare a local `failed` final state. An
error inside `BasisMachine` SHALL propagate as follows:

1. The action raises; the orchestrator's `_step` helper catches
   and writes `error_message` to ctx.
2. `BasisMachine` enters its local `failed` state.
3. `RefineMachine` observes the child reaching a final state and,
   because `error_message` is non-empty, fires its own `error`
   event and enters `RefineMachine.failed`.
4. `StreamMachine` does the same and enters `StreamMachine.failed`.

The final ctx after this propagation SHALL contain the same
`error_message` that the v0.2 flat machine produced for the same
underlying action error.

#### Scenario: error in compress_with_polygram reaches the outermost failed state

- **GIVEN** a forge run where `compress_with_polygram` raises
  `RuntimeError("synthetic error")`
- **WHEN** the run completes
- **THEN** the final state is `StreamMachine.failed`
- **AND** `ctx["error_message"] == "RuntimeError: synthetic error"`
- **AND** the value of `error_message` matches what the v0.2 flat
  machine produced for the same forced error
