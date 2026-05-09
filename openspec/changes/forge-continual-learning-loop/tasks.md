## 1. Context schema

- [ ] 1.1 Extend `## context` table in `saeforge/machines/sae_forge.orca.md` with the 17 new fields listed in `proposal.md` (stream + basis loop + protected features + replay)
- [ ] 1.2 Default values for every new field MUST recover v0.1 semantics: `n_tasks=1`, `inner_refine_passes=1`, `protect_top_k=0`, `replay_ratio=0.0`, `replay_buffer_size=0`, `task_trigger="labeled"`
- [ ] 1.3 Add `next_basis_step` enum with values `"regrow" | "compress" | "project"` (default `"project"`); document the per-state-source meaning in a short comment block above the context table

## 2. Machine definition (FSM topology)

- [ ] 2.1 Add state `activations_scanned` between `loaded` and `compressed`; declare per-state `error → failed` transition for it
- [ ] 2.2 Replace the single `loaded → compressed` transition with `loaded → activations_scanned (action: scan_activations)` followed by `activations_scanned → compressed (action: compress_with_polygram)`
- [ ] 2.3 Replace the two `compressed → compress_done` transitions with three guarded edges driven by `next_basis_step`:
  - `next_basis_step == "regrow"` → `regrown` action `perform_regrowth`
  - `next_basis_step == "compress"` → `compressed` (self-loop) action `compress_with_polygram`
  - `next_basis_step == "project"` → `projected` action `project_to_subspace`
- [ ] 2.4 Replace the single `regrown → regrowth_done` transition with two guarded edges:
  - `next_basis_step == "compress"` → `compressed` action `compress_with_polygram`
  - `next_basis_step == "project"` → `projected` action `project_to_subspace`
- [ ] 2.5 Replace the two `evaluated → eval_done` transitions with three guarded edges:
  - `advance_stream == true` → `loaded` action `advance_to_next_task`
  - `advance_stream == false ∧ should_continue == true` → `compressed` action `rotate_for_next_iter`
  - `advance_stream == false ∧ should_continue == false` → `done` action `save_final_model`
- [ ] 2.6 Add new guards: `next_step_is_regrow`, `next_step_is_compress`, `next_step_is_project`, `advance_stream`, `refine_same_shard`, `terminate_run` — all flat ctx comparisons
- [ ] 2.7 `test_machine_loads_and_has_ten_states` passes (replaces the v0.1 nine-state test); old test renamed and updated

## 3. Orchestrator

- [ ] 3.1 Extend `_NEXT_EVENT_FOR_STATE` in `saeforge/orchestrator.py` with `"activations_scanned": "scan_done"`
- [ ] 3.2 Verify the existing `while machine.state.value not in _FINAL_STATES` driver handles the new self-loops on `compressed` correctly (it should — but add a budget check to fail fast on runaway loops, e.g. >1000 transitions)
- [ ] 3.3 Add transition counter `ctx["_transition_count"]` and raise `RuntimeError` if it exceeds `1000` (defensive — protects against guard-write bugs)

## 4. Actions

- [ ] 4.1 New action `scan_activations` in `saeforge/actions/__init__.py`:
  - Reads `activation_buffer_size` tokens from the current corpus
  - Runs `SAE.encode` on the buffer (loaded from `current_sae_path`)
  - Writes `ctx["feature_usage"]: list[float]` of length `n_features`
  - When `protect_top_k > 0`, writes `ctx["protected_features"]` as the top-k indices per `protect_score`
  - Pass-through (no encoder call) when both `protect_top_k == 0` AND no other consumer of `feature_usage` is configured
- [ ] 4.2 New action `evaluate_task_advance` (called inside `evaluate_faithfulness` to keep the FSM action count tight):
  - Per `task_trigger`, computes and writes `ctx["advance_stream"]`
  - For `loss_delta`, appends to `ctx["recent_eval_losses"]` (capped at length 3)
- [ ] 4.3 New action `advance_to_next_task`:
  - Increments `task_idx`, resets `inner_refine_idx`, `tokens_seen_in_task`, `current_iter`, `recent_eval_losses`
  - Calls `TaskStream.next()` via `task_iterator_id` to install the next shard's iterator
  - Carries forward `final_model_path`, `current_sae_path`, `protected_features` (the protected set persists across tasks unless explicitly cleared)
- [ ] 4.4 Modify `compress_with_polygram`:
  - Read `ctx["protected_features"]`; pass to Polygram's `Compressor`
  - **If Polygram does not expose a `do_not_remove` arg today**, post-process the `ValidationReport`: mark protected features as confirmed in every pair so `Compressor` cannot drop them. Document the workaround in code; file a Polygram upstream issue
  - At end of action, write `ctx["next_basis_step"]`:
    - `"regrow"` if `regrow_count > 0` and this is not the final basis pass
    - `"compress"` if `inner_refine_idx < inner_refine_passes - 1` and `regrow_count == 0` (rare path; means re-compress without regrowing — usually a no-op)
    - `"project"` otherwise; also increments `inner_refine_idx`
- [ ] 4.5 Modify `perform_regrowth`:
  - At end of action, increment `inner_refine_idx`
  - Write `ctx["next_basis_step"]`:
    - `"compress"` if `inner_refine_idx < inner_refine_passes`
    - `"project"` otherwise
- [ ] 4.6 Modify `fine_tune_model`:
  - When `replay_ratio > 0` and `replay_buffer_size > 0`, wrap the iterator in `MixedIterator` (see §5)
  - Increment `ctx["tokens_seen_in_task"]` per step (sum of input_ids batch shape)
  - At end of run, call `update_replay_buffer` with the just-seen sequences
- [ ] 4.7 Modify `evaluate_faithfulness`:
  - After existing logic, call `evaluate_task_advance(ctx)` so `advance_stream` is set before the FSM reads the eval_done guard
- [ ] 4.8 Update `ACTION_TABLE` exports to include the three new actions

## 5. Replay + task stream modules

- [ ] 5.1 New module `saeforge/training/replay.py` exposing:
  - `ReplayBuffer(size: int, policy: str)` with `.add(sequences)` and `.sample(n)`
  - Reservoir / recent_window / per_task strategies; `per_task` requires a `task_id` arg on `add`
  - `MixedIterator(primary, replay, replay_ratio)` round-robin yields per the configured ratio
  - 100% pure-Python; no torch dependency at module level (lazy import inside)
- [ ] 5.2 New module `saeforge/training/task_stream.py` exposing:
  - `TaskStream` ABC with `.next() -> CorpusIterator | None`
  - `LabeledTaskStream(corpora: list[Path | str])` — finite labeled list
  - `TokenBudgetTaskStream(source, tokens_per_task)` — wraps a single stream and chunks
  - `LossDriftTaskStream(source)` — single stream; `next()` returns `None` once `advance_stream` fires (driven by ctx, not by the iterator itself)
  - Process-local registry mapping `task_iterator_id: str` → `TaskStream` instance, populated by `ForgePipeline` and read by `advance_to_next_task`

## 6. ForgePipeline integration

- [ ] 6.1 Add new `ForgePipeline` fields: `n_tasks`, `task_trigger`, `token_budget_per_task`, `inner_refine_passes`, `protect_top_k`, `protect_score`, `replay_ratio`, `replay_buffer_size`, `replay_policy`
- [ ] 6.2 `ForgePipeline.run` registers the configured `TaskStream` in the process-local registry, stores the handle in `ctx["task_iterator_id"]`
- [ ] 6.3 At pipeline teardown, deregister the handle (avoid leaks across pipelines in the same Python process)

## 7. CLI

- [ ] 7.1 Wire all new flags listed in `proposal.md` §CLI to `sae-forge forge`
- [ ] 7.2 Validation: `--task-trigger token_budget` requires `--token-budget-per-task > 0`; `--task-trigger loss_delta` requires `--loss-delta-threshold > 0`; `--replay-ratio > 0` requires `--replay-buffer-size > 0`. Argparse-level validation with actionable error messages
- [ ] 7.3 `--n-tasks` default 1 means single-shard; users wanting continual learning must explicitly set `> 1`

## 8. Tests

### Default-knob equivalence (the hard contract)

- [ ] 8.1 `test_continual_default_knobs_match_v01_topology`: with all new fields at default, the transition log's `to_state` sequence equals the v0.1 sequence with one extra `activations_scanned` entry
- [ ] 8.2 `test_imperative_and_fsm_byte_equivalent` from `forge-outer-loop-fsm` continues to pass unchanged

### Basis loop

- [ ] 8.3 `test_basis_loop_runs_inner_refine_passes`: with `inner_refine_passes=3, regrow_count=1`, the transition log contains exactly three `compressed → regrown` traversals before `regrown → projected`
- [ ] 8.4 `test_basis_loop_zero_regrow_no_self_loop`: with `inner_refine_passes=3, regrow_count=0`, the FSM goes straight `compressed → projected` (compressing without regrow has no effect, no point self-looping)
- [ ] 8.5 `test_inner_refine_idx_resets_on_task_advance`: after `advance_to_next_task`, `ctx["inner_refine_idx"] == 0`

### Stream loop

- [ ] 8.6 `test_stream_loop_labeled_three_tasks`: with `n_tasks=3, task_trigger="labeled"`, the FSM enters `loaded` exactly three times; `task_idx` reaches 2 before terminating
- [ ] 8.7 `test_stream_loop_token_budget_advances_on_threshold`: with `task_trigger="token_budget", token_budget_per_task=100`, advance fires after the fine-tune step that crosses 100 tokens. End-to-end through the FSM, not action-level
- [ ] 8.7b `test_stream_loop_token_budget_smoke_two_shards`: integration smoke — a real two-shard token-budget run reaches `done` with `task_idx == 1` and the artifact tree on disk; tolerant assertions on per-task losses just to confirm fine-tune happened on each shard
- [ ] 8.8 `test_stream_loop_loss_delta_advances_on_regression`: with `task_trigger="loss_delta", loss_delta_threshold=0.05`, advance fires after a stubbed eval that returns rising loss
- [ ] 8.8b `test_stream_loop_loss_delta_smoke_window_fill`: integration smoke — three stubbed evals with synthetic loss trajectory `[1.0, 1.0, 1.2]` and threshold `0.15` fires advance exactly once at index 2; FSM reaches `done`

### Protected features

- [ ] 8.9 `test_scan_activations_writes_protected_set`: with `protect_top_k=4`, `ctx["protected_features"]` has length 4 and contains the indices of the four highest `feature_usage` entries
- [ ] 8.10 `test_compress_respects_protected_features`: **unit test on the `compress_with_polygram` wrapper** — feed it a basis where the top-4-usage features are unambiguous and `ctx["protected_features"] = [those four indices]`. Assert all four survive the compression and the survival is *because of* the protected set, not coincidence (verify by re-running with `protected_features=[]` and checking those features get dropped)
- [ ] 8.10b `test_protected_features_polygram_workaround_path`: when the Polygram do-not-remove API is unavailable and the workaround path runs, the post-filtered `ValidationReport` marks every protected index as confirmed in every pair; the resulting compressed basis has the same protected-feature survival guarantee as the upstream-API path
- [ ] 8.11 `test_protected_features_persist_across_task_advance`: after one full stream loop iteration, `ctx["protected_features"]` is non-empty and continues to gate compression on the next task

### Replay

- [ ] 8.12 `test_replay_buffer_reservoir_uniform`: 10k adds into size-100 reservoir; sampled distribution covers all eras (chi-squared check, lenient threshold)
- [ ] 8.13 `test_replay_buffer_recent_window_fifo`: size-32 buffer, add 64 items; assert the held set is items 32..63
- [ ] 8.14 `test_mixed_iterator_replay_ratio`: with `replay_ratio=0.5`, exactly half of the iterator's first 100 outputs come from the replay source

### Stream-loop dominance

- [ ] 8.15 `test_advance_stream_dominates_should_continue`: with both `advance_stream=true` and `should_continue=true`, the FSM takes the stream edge (loaded), not the refine edge (compressed)

### RNG determinism

- [ ] 8.16 `test_activations_scanned_no_op_preserves_rng_state`: with `protect_top_k=0`, capture `torch.get_rng_state()` and the numpy RNG state immediately before and after `scan_activations` runs; assert byte-equality. This is the pre-condition for the v0.1 byte-equivalence test continuing to pass
- [ ] 8.17 `test_scan_activations_seeded_deterministic`: with `protect_top_k=4` and a fixed seed, two independent runs of `scan_activations` on the same buffer produce identical `feature_usage` vectors and identical `protected_features` lists
- [ ] 8.18 `test_full_pipeline_seeded_two_runs_byte_identical`: end-to-end FSM run with `n_tasks=2, protect_top_k=4, replay_ratio=0.25`, fixed seeds — two runs produce SHA-256-identical `forged.safetensors`. Catches RNG leaks in any of the new actions (scan, replay sampling, task advance)

## 9. Documentation

- [x] 9.0 Draft `openspec/changes/forge-continual-learning-loop/docs/advanced-fsm-options.md` (the spec-aligned draft) so the proposal can be reviewed against the user-facing surface
- [ ] 9.0b Extend the docs draft with: (a) a single-table summary of every new knob with default + v0.1-equivalent effect (e.g., `protect_top_k=0 → no protection, scan is no-op`), and (b) a "Choosing `task_trigger`" subsection with concrete usage scenarios (continual pretraining on a sliding web crawl → `token_budget`; sequential fine-tune across labeled domain corpora → `labeled`; uncertain-cadence drift on a single source → `loss_delta`)
- [ ] 9.1 Move the draft to `docs/advanced-fsm-options.md` during implementation; finalize once the actions land. The doc covers:
  - The three-loop diagram (ASCII state graph)
  - Every new context field with default + valid range + when to change it
  - All seven new CLI flags with worked examples
  - The three `task_trigger` options with a short paragraph each on when to pick which
  - The three `protect_score` options with the same treatment
  - The three `replay_policy` options
  - **A worked recipe per pattern (1)/(2)/(3)** from the design discussion, showing exact CLI invocations and expected behavior
  - A "Why three loops?" subsection cribbed from `design.md`
  - A debugging subsection: how to read `transitions_log` for each loop
- [ ] 9.2 Update `AGENTS.md` to add a "continual-learning loop" subsection with one paragraph on the three-loop semantics and a link to `docs/advanced-fsm-options.md` as the canonical reference
- [ ] 9.3 Update `README.md` "How it works" section with one paragraph linking to `docs/advanced-fsm-options.md`; do not duplicate content into README
- [ ] 9.4 `docs/advanced-fsm-options.md` includes a footer noting the open Polygram do_not_remove API question (tasks 4.4) and a link to the upstream issue once filed

## 10. Polygram coordination

- [ ] 10.1 Audit Polygram's `Compressor` for an existing do-not-remove / pinned-feature API; if absent, file an upstream issue with a proposed signature: `Compressor(report, do_not_remove: set[int] | None = None)`
- [ ] 10.2 Wire either the upstream addition or the post-filter workaround into `compress_with_polygram` per task 4.4
- [ ] 10.3 If using the workaround, add a TODO with a link to the Polygram issue and a note in `docs/advanced-fsm-options.md` that the protected-features path uses a Polygram-side workaround pending upstream support

## 11. OpenSpec scaffolding

- [x] 11.1 `openspec/changes/forge-continual-learning-loop/proposal.md`
- [x] 11.2 `openspec/changes/forge-continual-learning-loop/design.md`
- [x] 11.3 `openspec/changes/forge-continual-learning-loop/tasks.md` (this file)
- [x] 11.4 `openspec/changes/forge-continual-learning-loop/specs/forge-outer-loop/spec.md` (delta)

## 12. Deferred follow-ups (intentionally out of scope here)

These are recorded so future changes can reference them, not implemented in this PR.

- [ ] 12.1 **Per-loop-level scan tuning.** Today `activation_buffer_size` is a single scalar. A future change exposes per-stream-iteration / per-basis-iteration overrides so users can pay full encoder cost only on the first basis pass and downsample on inner refinements
- [ ] 12.2 **Feature-axis sampling in `scan_activations`.** For `n_features >> 100k` add a random-subset-with-importance-reweighting variant. Currently always full-axis
- [ ] 12.3 **Expose raw trigger signals in ctx.** `recent_eval_losses`, `tokens_seen_in_task`, `feature_usage` are written today but only the boolean `advance_stream` is contractual. A future change promotes the raw signals to part of the public ctx contract so power users can write custom Python triggers without forking `evaluate_task_advance`
- [ ] 12.4 **Token-level replay buffer.** Current `ReplayBuffer` is sequence-level. Token-level reservoir is a future option for memory-bound configurations
- [ ] 12.5 **Per-task evaluation matrix.** Reporting forgetting as a `task_idx × task_idx` KL matrix needs a separate evaluator state and a per-task held-out probe corpus. Out of scope here; valuable as a follow-up change `forge-continual-eval-matrix`
- [ ] 12.6 **Basis-size growth across tasks.** `basis_size_policy` is named in the proposal as a future knob; this change ships `"fixed"` only. Reshaping the projected model mid-stream is non-trivial — separate change
- [ ] 12.7 **Polygram `do_not_remove` upstream.** Resolution preference: upstream the argument to Polygram's `Compressor` rather than ship the post-filter workaround long-term. File and link the upstream issue from `docs/advanced-fsm-options.md` once the wrapper lands
