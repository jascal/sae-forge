## MODIFIED Requirements

### Requirement: Machine has ten states with the v0.2 topology

The forge FSM SHALL declare the same ten observable state
identifiers as v0.2, but distributed across three composed orca
machines per the `hierarchical-fsm` capability:

- `StreamMachine`: `init`, `streaming` (compound,
  `> invoke: RefineMachine`), `next_shard`, `done`, `failed`
- `RefineMachine`: `entering`, `refining` (compound,
  `> invoke: BasisMachine`), `evaluating`, `exiting`, `failed`
- `BasisMachine`: `compressed`, `regrown`, `projected`,
  `finetuned`, `done`, `failed`

The v0.2 states `loaded` and `activations_scanned` are collapsed
into the single `RefineMachine.entering` state whose action is
the composed `load_and_scan` helper. `transitions_log` SHALL
continue to record both `load_sae_and_corpus` and
`scan_activations` as separate entries (preserving the v0.2
log shape) — see the byte-equivalence requirement below.

The v0.2 state `evaluated` is renamed to `RefineMachine.evaluating`
internally; `transitions_log` action entries continue to record
the action names that were associated with `evaluated` in v0.2
(`evaluate_faithfulness`, `advance_to_next_task`,
`rotate_for_next_iter`, `save_final_model`). State-name churn is
not observable to consumers that read action names from the log.

#### Scenario: composed hierarchy declares all v0.2 state names

- **WHEN** the orchestrator parses the composed three-file hierarchy
- **THEN** the union of state names across the three machines
  contains the v0.2 state set with `loaded` / `activations_scanned`
  collapsed and `evaluated` renamed to `evaluating`
- **AND** the canonical-event derivation handles every non-final
  source state in the union (no `RuntimeError` from the
  orchestrator's "no canonical event derivable" guard)

### Requirement: Stream loop advances to the next task on `advance_stream`

The composed FSM SHALL preserve the v0.2 three-edge guard truth
table for the transitions that fire when `RefineMachine` reaches
its `exiting` final state, distributed across the hierarchy as
follows:

`StreamMachine.streaming → next_shard` on event `refine_done`
guarded by `stream_advance` (reads `ctx.advance_stream == true`)
runs action `advance_to_next_task`.

`StreamMachine.streaming → done` on event `refine_done` guarded by
`terminate_run` (reads
`ctx.advance_stream == false ∧ ctx.should_continue == false`)
runs action `save_final_model`.

The third v0.2 case — refine-same-shard — is internalized in
`RefineMachine`: `RefineMachine.evaluating → refining` on event
`eval_done` guarded by `refine_continue` (reads
`ctx.advance_stream == false ∧ ctx.should_continue == true`)
runs action `rotate_for_next_iter`. `RefineMachine` only fires its
final `exiting` event when `refine_continue` is false, at which
point `StreamMachine` arbitrates between `next_shard` and `done`.

The three guards SHALL remain jointly exhaustive and pairwise
disjoint over `(advance_stream, should_continue) ∈ {true, false}²`.

#### Scenario: stream loop dominates refine loop

- **GIVEN** a `RefineMachine.evaluating` state with
  `ctx.advance_stream = true` AND `ctx.should_continue = true`
- **WHEN** the FSM fires `eval_done`
- **THEN** `refine_continue` evaluates to false (because
  `advance_stream` is true)
- **AND** `RefineMachine` transitions to `exiting`
- **AND** `StreamMachine` then transitions from `streaming` to
  `next_shard` (not `done`)
- **AND** `advance_to_next_task` runs

#### Scenario: stream loop respects n_tasks budget

- **GIVEN** `ctx.n_tasks = 3` and `task_trigger = "labeled"`
- **WHEN** the FSM completes the third refine cycle
- **THEN** `ctx.advance_stream == false` (set by
  `evaluate_task_advance`) and the FSM terminates at
  `StreamMachine.done`
- **AND** `ctx.task_idx == 2` at termination

### Requirement: Basis loop refines the basis before projection

`BasisMachine` SHALL localize the basis-loop transitions and preserve the v0.2 truth table verbatim:

| Source       | Event           | Guard                       | Target       | Action                  |
|--------------|-----------------|-----------------------------|--------------|-------------------------|
| `compressed` | `compress_done` | `should_regrow`             | `regrown`    | `perform_regrowth`      |
| `compressed` | `compress_done` | `no_regrow_more_passes`     | `compressed` | `compress_with_polygram`|
| `compressed` | `compress_done` | `no_regrow_done`            | `projected`  | `project_to_subspace`   |
| `regrown`    | `regrowth_done` | `basis_loop_continue`       | `compressed` | `compress_with_polygram`|
| `regrown`    | `regrowth_done` | `basis_loop_done`           | `projected`  | `project_to_subspace`   |
| `projected`  | `projection_done` |                           | `finetuned`  | `fine_tune_model`       |
| `finetuned`  | `finetune_done` |                             | `done`       |                         |

`BasisMachine` reaches its `done` final state after `fine_tune_model`
completes. `RefineMachine` then fires the `basis_done` event and
runs `evaluate_faithfulness` from its `evaluating` state, exactly
as v0.2 ran `evaluate_faithfulness` from the `evaluated` state.

The five basis-loop guard expressions (`should_regrow`,
`no_regrow_more_passes`, `no_regrow_done`, `basis_loop_continue`,
`basis_loop_done`) SHALL be copied byte-for-byte from the v0.2
flat machine. No expression is rewritten.

#### Scenario: inner_refine_passes drives the basis self-loop

- **GIVEN** `ctx.inner_refine_passes = 3`, `ctx.regrow_count = 0`,
  `ctx.inner_refine_idx = 0`
- **WHEN** the FSM enters `BasisMachine.compressed`
- **THEN** the machine self-loops on `compressed` three times
  (incrementing `inner_refine_idx` in `compress_with_polygram`)
- **AND** transitions to `projected` on the fourth `compress_done`
  event (when `no_regrow_done` fires)

### Requirement: Imperative byte-equivalence test continues to pass

The test `test_imperative_and_fsm_byte_equivalent` SHALL continue
to compare the imperative orchestrator path against the FSM path
on a fixed seed, with one filtering update:

- The `machine_path` field on `transitions_log` entries (added by
  `hierarchical-fsm`) SHALL be filtered out before comparison.
- All other comparison axes — action sequence, final ctx scalar
  fields, `final_model_path` artifact bytes — SHALL match exactly.

The filtering update SHALL be the *only* change to the test. If
the test fails for any other reason after the hierarchy migration,
the migration is incorrect and SHALL be fixed; the test SHALL NOT
be rebaselined.

#### Scenario: byte-equivalence holds across the hierarchy migration

- **GIVEN** a fixed seed and a default-knobs ForgePipeline
- **WHEN** both the imperative path and the hierarchy run to
  completion
- **THEN** `transitions_log[*].action` sequences match exactly
- **AND** `final_model_path` artifact bytes are identical
- **AND** the final ctx scalar fields (`faithfulness`,
  `perplexity`, `task_idx`, `inner_refine_idx`) match exactly
- **AND** the only difference between the two `transitions_log`
  shapes is the presence of the `machine_path` key on the FSM-side
  entries

### Requirement: Loop budget protects against guard-write bugs

The orchestrator SHALL retain its existing transition-count
budget (a hard cap on total transitions across the composed
hierarchy) as a guard against guard-expression bugs that produce
infinite loops. The budget applies to the total transition count
across all three machines, not per-machine — a runaway basis loop
inside a runaway stream loop is bounded by the same limit as
either alone.

The budget value, the budget-exceeded error path, and the resulting
`failed` state contract are unchanged from v0.2.

#### Scenario: runaway basis loop hits the global budget

- **GIVEN** a buggy `compress_with_polygram` that never sets
  `inner_refine_idx >= inner_refine_passes`
- **WHEN** the orchestrator drives the hierarchy
- **THEN** the run terminates at `StreamMachine.failed` with an
  `error_message` that names the budget cap
- **AND** the `transitions_log` length equals the budget cap (no
  silent truncation)
