## ADDED Requirements

### Requirement: Machine has ten states with the v0.2 topology

The machine SHALL declare exactly ten states: `init` (initial),
`loaded`, `activations_scanned`, `compressed`, `regrown`,
`projected`, `finetuned`, `evaluated`, `done` (final),
`failed` (final).

The state `activations_scanned` SHALL sit between `loaded` and
`compressed`. Its action `scan_activations` SHALL be a no-op
pass-through when `protect_top_k == 0`.

#### Scenario: machine declares ten states under v0.2

- **WHEN** the orchestrator parses `saeforge/machines/sae_forge.orca.md`
- **THEN** `machine.states` contains exactly the ten state names listed
  above
- **AND** `loaded` has a single non-error outgoing edge whose target is
  `activations_scanned`

### Requirement: Stream loop advances to the next task on `advance_stream`

The transition table SHALL include exactly these `evaluated → eval_done`
edges (replacing the v0.1 two-edge set):

| Source       | Event       | Guard                               | Target       | Action                  |
|--------------|-------------|-------------------------------------|--------------|-------------------------|
| `evaluated`  | `eval_done` | `advance_stream`                    | `loaded`     | `advance_to_next_task`  |
| `evaluated`  | `eval_done` | `refine_same_shard`                 | `compressed` | `rotate_for_next_iter`  |
| `evaluated`  | `eval_done` | `terminate_run`                     | `done`       | `save_final_model`      |

`advance_stream` SHALL read `ctx.advance_stream == true`.
`refine_same_shard` SHALL read
`ctx.advance_stream == false ∧ ctx.should_continue == true`.
`terminate_run` SHALL read
`ctx.advance_stream == false ∧ ctx.should_continue == false`.

The three guards SHALL be jointly exhaustive and pairwise disjoint
over `(advance_stream, should_continue) ∈ {true, false}²`.

#### Scenario: stream loop dominates refine loop

- **GIVEN** an `evaluated` state with `ctx.advance_stream = true` AND
  `ctx.should_continue = true`
- **WHEN** the FSM fires `eval_done`
- **THEN** the FSM transitions to `loaded` (not `compressed`) and
  `advance_to_next_task` runs

#### Scenario: stream loop respects n_tasks budget

- **GIVEN** `ctx.n_tasks = 3` and `task_trigger = "labeled"`
- **WHEN** the FSM completes the third `evaluated` state
- **THEN** `ctx.advance_stream = false` (set by
  `evaluate_task_advance`) and the FSM terminates at `done`
- **AND** `ctx.task_idx == 2` at termination

### Requirement: Basis loop refines the basis before projection

The transition table SHALL include exactly these `compressed →
compress_done` and `regrown → regrowth_done` edges:

| Source       | Event             | Guard                          | Target       | Action                   |
|--------------|-------------------|--------------------------------|--------------|--------------------------|
| `compressed` | `compress_done`   | `next_step_is_regrow`          | `regrown`    | `perform_regrowth`       |
| `compressed` | `compress_done`   | `next_step_is_compress`        | `compressed` | `compress_with_polygram` |
| `compressed` | `compress_done`   | `next_step_is_project`         | `projected`  | `project_to_subspace`    |
| `regrown`    | `regrowth_done`   | `next_step_is_compress`        | `compressed` | `compress_with_polygram` |
| `regrown`    | `regrowth_done`   | `next_step_is_project`         | `projected`  | `project_to_subspace`    |

Each guard SHALL read the single field `ctx.next_basis_step` for its
respective enum value (`"regrow" | "compress" | "project"`). The
guards SHALL be pairwise disjoint and jointly exhaustive.

`compress_with_polygram` SHALL write `ctx.next_basis_step` per the
rules in `tasks.md` §4.4. `perform_regrowth` SHALL write
`ctx.next_basis_step` per the rules in `tasks.md` §4.5.

#### Scenario: basis loop runs N inner passes

- **GIVEN** `inner_refine_passes = 3`, `regrow_count = 1`
- **WHEN** a single shard runs through the FSM
- **THEN** the transition log contains exactly three traversals of
  `compressed → regrown` and three of `regrown → compressed`
- **AND** the final exit edge from `regrown` targets `projected`

#### Scenario: basis loop short-circuits when regrow is disabled

- **GIVEN** `inner_refine_passes = 5`, `regrow_count = 0`
- **WHEN** a single shard runs through the FSM
- **THEN** the transition log contains zero `compressed → regrown`
  traversals
- **AND** `compressed` exits to `projected` on its first compress

### Requirement: Protected features survive compression

When `ctx.protect_top_k > 0`, `scan_activations` SHALL write
`ctx.protected_features` as a list of `protect_top_k` integer
indices in `[0, n_features)` selected per `protect_score`.

`compress_with_polygram` SHALL ensure that every index in
`ctx.protected_features` is present in the compressed basis (i.e.
the indices are not pruned by `Compressor`). This MAY be enforced
by passing a do-not-remove set to Polygram, or by post-filtering the
`ValidationReport` to mark those features as confirmed in every
validation pair.

#### Scenario: protected features survive an aggressive compress

- **GIVEN** an SAE with `n_features = 64`, a `ValidationReport` that
  would reduce it to 16 features, and `ctx.protect_top_k = 4`
- **WHEN** `scan_activations` then `compress_with_polygram` run
- **THEN** the compressed basis has at most 16 features
- **AND** every index in `ctx.protected_features` appears in the
  compressed basis's index map

### Requirement: `task_trigger` writes a single boolean

`evaluate_task_advance` SHALL be called from inside
`evaluate_faithfulness` and SHALL set `ctx.advance_stream` to
exactly one boolean per the configured `task_trigger`:

- `"labeled"`: `advance_stream = (task_idx + 1 < n_tasks)`
- `"token_budget"`: `advance_stream = (tokens_seen_in_task >=
  token_budget_per_task) AND (task_idx + 1 < n_tasks)`
- `"loss_delta"`: `advance_stream = window_full AND
  (mean(recent_eval_losses[-3:-1]) - recent_eval_losses[-1] >
  loss_delta_threshold) AND (task_idx + 1 < n_tasks)` where
  `window_full` means `len(recent_eval_losses) >= 3`

The FSM SHALL read `ctx.advance_stream` only via the
`advance_stream` guard. No FSM guard or transition SHALL inspect
`task_trigger` or any of its trigger-specific fields directly.

#### Scenario: token-budget trigger fires on threshold crossing

- **GIVEN** `task_trigger = "token_budget"`, `token_budget_per_task = 100`,
  and a `fine_tune_model` run that processes 120 tokens
- **WHEN** `evaluate_task_advance` runs
- **THEN** `ctx.advance_stream == true`

#### Scenario: loss-delta trigger waits for window fill

- **GIVEN** `task_trigger = "loss_delta"`, `loss_delta_threshold = 0.05`
- **WHEN** the first two evaluations complete (window not yet full)
- **THEN** `ctx.advance_stream == false` after both
- **WHEN** the third evaluation reports a loss higher than the mean of
  the prior two by more than 0.05
- **THEN** `ctx.advance_stream == true`

### Requirement: Default knobs preserve v0.1 transition log

When all new context fields are at their defaults
(`n_tasks=1`, `inner_refine_passes=1`, `protect_top_k=0`,
`replay_ratio=0`, `replay_buffer_size=0`, `task_trigger="labeled"`),
the transition log's `to_state` sequence SHALL equal the v0.1
single-pass sequence with exactly one extra `activations_scanned`
entry inserted between `loaded` and `compressed`.

Specifically the sequence SHALL be:

```
loaded, activations_scanned, compressed, projected, finetuned,
evaluated, done
```

#### Scenario: default-knob run has the v0.2-default sequence

- **GIVEN** a `ForgeContext` with every new field at its default
- **WHEN** `run_machine(ctx)` runs to completion
- **THEN** `ctx["transitions_log"]`'s `to_state` sequence equals the
  seven-entry sequence above
- **AND** no entry has `to_state == "regrown"`

### Requirement: Imperative byte-equivalence test continues to pass

The v0.1 byte-equivalence test
`test_imperative_and_fsm_byte_equivalent` SHALL continue to pass
without modification. Adding `activations_scanned` as a no-op
pass-through (under `protect_top_k = 0`) SHALL NOT perturb RNG state
or otherwise change the forged weights.

#### Scenario: byte-identical toy forge under v0.2 defaults

- **GIVEN** the toy GPT-2 example with RNG seed `0` and v0.2 default
  knobs
- **WHEN** the example is run twice — once with
  `orchestrator="imperative"` and once with `orchestrator="fsm"`
- **THEN** the two `forged.safetensors` files have identical SHA-256
  hashes

### Requirement: Loop budget protects against guard-write bugs

The orchestrator SHALL track the number of FSM transitions in
`ctx["_transition_count"]` and SHALL raise `RuntimeError` with a
diagnostic message if the count exceeds `1000`. This is a defensive
limit against bugs where an action fails to advance
`next_basis_step` and the FSM enters an infinite self-loop.

#### Scenario: runaway basis loop is caught

- **GIVEN** a stubbed `compress_with_polygram` that erroneously always
  writes `ctx.next_basis_step = "compress"`
- **WHEN** `run_machine(ctx)` is invoked
- **THEN** the orchestrator raises `RuntimeError` whose message names
  the transition budget and the last-fired event

## MODIFIED Requirements

### Requirement: Machine has nine states with the v0.1 topology

This requirement is REPLACED by **Machine has ten states with the v0.2
topology** (above). The v0.1 nine-state topology is no longer
canonical; the v0.2 topology adds `activations_scanned` and modifies
the `compressed`, `regrown`, and `evaluated` outgoing edges as
described above.

### Requirement: Single-pass default reaches `done` without re-entering `compressed`

This requirement is REPLACED by **Default knobs preserve v0.1
transition log** (above). The new requirement guarantees the same
property — `compressed` is entered once and never re-entered under
defaults — and additionally pins the exact seven-entry sequence
including the new `activations_scanned` hop.

### Requirement: Multi-pass loop respects `should_continue_loop`

The v0.1 refine loop semantics are PRESERVED. The
`refine_same_shard` guard reads
`ctx.advance_stream == false ∧ ctx.should_continue == true`, so the
loop edge fires only when the new stream loop is *not* advancing.
`rotate_for_next_iter` is unchanged.

The three v0.1 scenarios under this requirement (three-iter forge,
faithfulness regression, perplexity stagnation) SHALL all continue
to pass with `n_tasks = 1` (the default).

#### Scenario: refine loop still works under v0.2

- **GIVEN** the v0.1 three-iter scenario with `n_tasks = 1`,
  `iterations = 3`, `inner_refine_passes = 1`, `protect_top_k = 0`
- **WHEN** the orchestrator runs
- **THEN** the transition log shows exactly three traversals of
  `evaluated → compressed` and one terminal `evaluated → done`
- **AND** `ctx.task_idx == 0` at termination

### Requirement: Per-state errors reach `failed` cleanly

The v0.1 error-routing requirement is PRESERVED and EXTENDED to cover
the new `activations_scanned` state. Any action raising an exception
in `activations_scanned` SHALL drive the FSM into `failed` via an
`error → failed` transition. `log_error` SHALL run on entry and
populate `ctx.error_message`.

#### Scenario: scan_activations raise reaches `failed`

- **GIVEN** a stubbed `scan_activations` that raises
  `RuntimeError("scan boom")`
- **WHEN** the orchestrator runs
- **THEN** the final state is `failed`
- **AND** `ctx.error_message` contains `"activations_scanned"`,
  `"RuntimeError"`, and `"scan boom"`
