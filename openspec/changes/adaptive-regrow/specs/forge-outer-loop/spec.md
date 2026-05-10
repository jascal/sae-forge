## MODIFIED Requirements

### Requirement: Basis loop refines the basis before projection

`BasisMachine` SHALL localize the basis-loop transitions and
preserve the v0.2 truth table as documented under
`hierarchical-fsm`, with **exactly one transition action rename**
to support the `adaptive-regrow` capability:

| Source       | Event           | Guard                       | Target       | Action                  |
|--------------|-----------------|-----------------------------|--------------|-------------------------|
| `compressed` | `compress_done` | `should_regrow`             | `regrown`    | **`adapt_and_regrow`**  |
| `compressed` | `compress_done` | `no_regrow_more_passes`     | `compressed` | `compress_with_polygram`|
| `compressed` | `compress_done` | `no_regrow_done`            | `projected`  | `project_to_subspace`   |
| `regrown`    | `regrowth_done` | `basis_loop_continue`       | `compressed` | `compress_with_polygram`|
| `regrown`    | `regrowth_done` | `basis_loop_done`           | `projected`  | `project_to_subspace`   |
| `projected`  | `projection_done` |                           | `finetuned`  | `fine_tune_model`       |
| `finetuned`  | `finetune_done` |                             | `done`       |                         |

The bolded `adapt_and_regrow` is the new composed action: it
short-circuits to `perform_regrowth` when `adaptive_regrow=False`
(byte-identical to v0.2) and calls the `RegrowController` then
`perform_regrowth` when `adaptive_regrow=True`.

The state set, event names, guard expressions, and transition
targets SHALL be unchanged from the post-`hierarchical-fsm`
topology. The state count of `BasisMachine` SHALL remain at 7
(`starting`, `compressed`, `regrown`, `projected`, `finetuned`,
`done`, `failed`) — the topology test in
`tests/fsm/test_topology.py` SHALL continue to pass without
modification.

The auto-generated Mermaid diagram in
`docs/advanced-fsm-options.md` SHALL be regenerated to reflect
the one renamed action label. The drift CI
(`tests/fsm/test_diagram_drift.py`) SHALL fail until the doc is
updated, and pass after.

#### Scenario: state set unchanged

- **WHEN** `load_machine_hierarchy()` is called against the
  post-`adaptive-regrow` tree
- **THEN** `{s.name for s in BasisMachine.states}` equals
  `{"starting", "compressed", "regrown", "projected",
   "finetuned", "done", "failed"}`

#### Scenario: regrow transition action is adapt_and_regrow

- **WHEN** the post-`adaptive-regrow` `BasisMachine.transitions`
  table is inspected
- **THEN** the row with
  `(source="compressed", event="compress_done",
   guard="should_regrow")` has `action == "adapt_and_regrow"`
- **AND** every other row's action is unchanged from the
  `hierarchical-fsm` baseline

#### Scenario: byte-equivalence gate continues to pass under default knobs

- **GIVEN** the existing `test_imperative_and_fsm_byte_equivalent`
  setup (no `adaptive_regrow` knobs set; defaults to False)
- **WHEN** the test runs against the post-this-change tree
- **THEN** the imperative-path and FSM-path forged weights are
  byte-identical
- **AND** the `transitions_log` action sequence is unchanged
  from v0.2 (the composed `adapt_and_regrow` action's
  short-circuit means only `perform_regrowth` runs)
