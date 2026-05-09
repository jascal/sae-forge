## Why

The v0.1 `forge-outer-loop` FSM (`forge-outer-loop-fsm`) drives a
*single-shard* forge: load → compress → optional regrow → project →
fine-tune → evaluate → optional iterate-on-the-same-data → done. The
existing `should_continue_loop` edge re-enters `compressed` against the
**same** loaded SAE/corpus to re-converge the basis, not to advance to
a new data shard.

That is not enough for continual-learning workflows. Three orthogonal
extensions keep coming up in design discussions and have no place to
live in the current FSM:

1. **Multi-shard streaming.** A continual-learning run consumes a
   *stream* of data shards (task-labeled, token-budgeted, or
   drift-detected) and re-runs compress/regrow/project/fine-tune on
   each. There is no "next shard" edge in the FSM today; the outer
   driver would have to call `run_machine` repeatedly from Python and
   re-marshal context across runs, which loses the FSM's verifiable
   control-flow contract.
2. **Inner basis refinement.** Repeating compress↔regrow several
   times *before* projecting (so dead regrown features get pruned by
   the next compress pass) is cheap — no host-model touch — but the
   current FSM only has one compress and one regrow edge. Doing it
   from Python means hidden iteration that the FSM cannot verify.
3. **Protected features (structural EWC).** During continual learning
   the basis itself carries old-task semantics. Removing features that
   carried prior tasks is the SAE-stack equivalent of catastrophic
   forgetting. The natural fix is to forbid `Compressor` from removing
   a configurable set of "protected" features, scored from the previous
   task's activations. That requires a place in the FSM to *compute
   the protected set* before `compress_with_polygram` runs.

This change generalizes the FSM to a three-level loop —
**stream → refine → basis** — with all new behavior gated behind
defaults that preserve byte-identical v0.1 semantics. The drift case
(no task labels) is covered by a configurable `task_trigger` whose
three implementations (`labeled`, `token_budget`, `loss_delta`) all
write a flat boolean into context for the FSM guard to read, matching
the existing `should_continue` pattern.

## What Changes

### Scope

`forge-outer-loop` expands from "drive a single forge" to "drive a
configurable continual-learning forge stream":

- **Stream loop (new, outermost):** `evaluated → loaded` re-entry to
  consume the next shard. Triggered by guard `advance_stream`.
- **Refine loop (existing):** `evaluated → compressed` re-entry on
  the same shard. Unchanged.
- **Basis loop (new, innermost):** `compressed ↔ regrown` self-loop
  for `inner_refine_passes` rounds before exiting to `projected`.

One new state, `activations_scanned`, sits between `loaded` and
`compressed` and hosts the new `scan_activations` action that
populates the activation buffer and (when enabled) the protected
feature set.

### New context fields

All default to values that recover today's behavior byte-identically.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `n_tasks` | int | 1 | Outer stream length |
| `task_idx` | int | 0 | Current task index |
| `task_trigger` | enum | `"labeled"` | `labeled` \| `token_budget` \| `loss_delta` |
| `token_budget_per_task` | int | 0 | Trigger threshold for `token_budget` |
| `tokens_seen_in_task` | int | 0 | Counter incremented during fine-tune |
| `loss_delta_threshold` | float | 0.0 | Trigger threshold for `loss_delta` |
| `recent_eval_losses` | list[float] | `[]` | Sliding window for `loss_delta` |
| `advance_stream` | bool | false | Computed flag the FSM guard reads |
| `inner_refine_passes` | int | 1 | Basis-loop iteration count |
| `inner_refine_idx` | int | 0 | Basis-loop counter |
| `next_basis_step` | enum | `"project"` | `"regrow"` \| `"compress"` \| `"project"` |
| `protect_top_k` | int | 0 | Protected feature set size (0 = disabled) |
| `protect_score` | enum | `"mean_act"` | `mean_act` \| `usage` \| `grad_importance` |
| `protected_features` | list[int] | `[]` | Indices excluded from compression |
| `activation_buffer_size` | int | 4096 | Tokens scored against the SAE encoder |
| `replay_ratio` | float | 0.0 | Fraction of replay tokens in fine-tune |
| `replay_policy` | enum | `"reservoir"` | `reservoir` \| `recent_window` \| `per_task` |
| `replay_buffer_size` | int | 0 | Buffer capacity (0 = disabled) |
| `task_iterator_id` | string | `""` | Process-local handle to a `TaskStream` |

### New actions

- `scan_activations` — runs the SAE encoder over `activation_buffer_size`
  tokens of the current shard, writes feature usage scores, and (when
  `protect_top_k > 0`) the protected-feature index set.
- `advance_to_next_task` — increments `task_idx`, advances the task
  iterator, resets `inner_refine_idx`, `tokens_seen_in_task`, and
  `current_iter`. Carries forward the host model and the SAE basis.
- `evaluate_task_advance` — runs after `evaluate_faithfulness`,
  computes `advance_stream` per `task_trigger`, appends to
  `recent_eval_losses`.
- `update_replay_buffer` — populates the replay buffer per
  `replay_policy` from the current task's tokens (called from
  `fine_tune_model`, no separate FSM state).

### Modified actions

- `compress_with_polygram` reads `ctx.protected_features` and forwards
  it to Polygram's `Compressor` as a do-not-remove set. Unchanged
  semantics when the set is empty.
- `fine_tune_model` mixes replay tokens into the iterator at
  `replay_ratio`, increments `tokens_seen_in_task`, and calls
  `update_replay_buffer` at end of run.
- `evaluate_faithfulness` is unchanged; the new advance logic moves
  into `evaluate_task_advance` (a separate action), preserving the
  existing FSM's eval edge for the existing `should_continue` knob.

### FSM topology delta

Net new states: 1 (`activations_scanned`).
Net new transitions: 5 (split `compressed → compress_done`, split
`regrown → regrowth_done`, new `evaluated → eval_done` advance edge,
new `activations_scanned` chain).

The full transition table delta is in
`specs/forge-outer-loop/spec.md`. Key invariants preserved:

- Default knobs (`n_tasks=1`, `inner_refine_passes=1`,
  `protect_top_k=0`, `replay_ratio=0`) traverse exactly the v0.1
  state sequence — the only difference is the extra
  `activations_scanned` hop, which is a no-op pass-through when
  `protect_top_k == 0`.
- All new guards are flat ctx reads (boolean fields written by Python
  actions), matching the existing `should_continue` pattern. No new
  guard expression complexity in the `.orca.md`.
- Stream loop and refine loop are mutually exclusive: when both
  `advance_stream == true` and `should_continue == true`, the stream
  loop wins (next-task semantics dominate same-task refinement).

### CLI surface

- `--n-tasks N` (default 1)
- `--task-trigger {labeled,token_budget,loss_delta}` (default `labeled`)
- `--token-budget-per-task N` (default 0)
- `--inner-refine-passes N` (default 1)
- `--protect-top-k N` (default 0)
- `--replay-ratio F` (default 0.0)
- `--replay-buffer-size N` (default 0)

All flags are no-ops at their defaults.

## Capabilities

### Modified Capabilities

- `forge-outer-loop`: Adds the stream and basis loops, the
  `activations_scanned` state, and the protected-features /
  replay-buffer hooks. The v0.1 single-pass guarantee
  (`n_tasks=1, inner_refine_passes=1` traverses the v0.1 state
  sequence) is preserved as an explicit requirement.

No new capabilities; no removed capabilities.

## Impact

- Files touched:
  - `saeforge/machines/sae_forge.orca.md` — new state, new transitions,
    new context fields, new guards.
  - `saeforge/actions/__init__.py` — three new actions, two modified.
  - `saeforge/orchestrator.py` — extend `_NEXT_EVENT_FOR_STATE` for
    `activations_scanned`.
  - `saeforge/training/replay.py` — **new**, replay-buffer
    implementation (reservoir / recent_window / per_task).
  - `saeforge/training/task_stream.py` — **new**, `TaskStream`
    abstraction unifying labeled-shard / token-budget / drift sources.
  - `saeforge/forge.py` — wire new context fields through
    `ForgePipeline`; serialize task-stream config into ctx.
  - `pyproject.toml` — no new deps. Replay/task-stream are pure-Python.
  - `tests/` — three new test files (basis loop, stream loop,
    protected-features); no existing tests change.
  - `docs/advanced-fsm-options.md` — **new**, dedicated user-facing
    reference for the three-loop semantics, every new context field,
    every new CLI flag, the three `task_trigger` options, the
    `protect_score` strategies, and the `replay_policy` strategies.
    Includes a worked configuration recipe per pattern (1)/(2)/(3)
    and a state diagram of the extended FSM.
- **No breaking changes.** Default knobs preserve byte-identical v0.1
  behavior. The byte-equivalence test
  (`test_imperative_and_fsm_byte_equivalent`) continues to pass.
- **No new optional extras.** Continual-learning machinery is part of
  the existing `[orca]` extra.
- `AGENTS.md` gains a brief subsection on the three-loop semantics and
  the `task_trigger` contract, and links to `docs/advanced-fsm-options.md`
  as the authoritative reference.
- `README.md` "How it works" section gains a one-paragraph link out to
  `docs/advanced-fsm-options.md`; the README itself stays terse.
- `examples/` is **not** extended in this change; an example is
  deferred to a follow-up `examples-continual-learning` change so the
  scope of this PR stays in spec + FSM + actions + docs.
