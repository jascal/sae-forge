# Design: forge-continual-learning-loop

## Three-loop semantics

The extended FSM has three nested loops with disjoint triggers:

```
┌─────────────── stream loop (per task / shard) ───────────────┐
│  loaded → activations_scanned → ┌── basis loop ──┐ → projected│
│                                 │ compressed ↔   │            │
│                                 │   regrown      │            │
│                                 └────────────────┘            │
│   → finetuned → evaluated ─┐                                  │
└────────────────────────────┘ advance_stream                   │
                                 │                              │
                                 ▼                              │
                              loaded (next shard)               │
                                                                │
                       │ should_continue (same shard)           │
                       ▼                                        │
                    compressed (refine loop, existing)          │
└──────────────────────────────────────────────────────────────-┘
```

| Loop | Edge | Triggered by | Counter | Semantics |
|------|------|--------------|---------|-----------|
| Stream | `evaluated → loaded` | `advance_stream == true` | `task_idx` | Move to next data shard, reset basis loop counters, carry forward host model + SAE |
| Refine | `evaluated → compressed` | `advance_stream == false ∧ should_continue == true` | `current_iter` | Re-converge same shard's basis (existing v0.1 behavior) |
| Basis | `compressed ↔ regrown` self-loop | `inner_refine_idx < inner_refine_passes - 1` | `inner_refine_idx` | Refine the basis before projecting |

Precedence is enforced by **disjoint** guard expressions, not by
edge ordering. Each guard reads a single boolean ctx field that
upstream Python actions are responsible for keeping consistent.

## Why three loops, not one

A single counter could in principle drive everything, but conflating
them loses information the user needs:

- The stream loop is **data-driven**: end condition is "stream
  exhausted" or "external `n_tasks` budget hit." The counter is
  position-in-stream.
- The refine loop is **convergence-driven**: end condition is
  "perplexity stopped improving." The counter is iteration-on-shard.
- The basis loop is **structural**: end condition is "compress→regrow
  has settled, no more dead features being created." The counter is
  pre-projection refinement passes.

Mixing them would mean a single "should we keep going?" predicate
hides what kind of progress (or stagnation) is happening. The three
counters are also useful eval signals on their own.

## Guard discipline

orca-runtime-python's guard expressions are flat ctx comparisons.
Disjunctions like `regrow_count > 0 OR inner_refine_idx < ...` would
need to be encoded across two transition rows; that becomes
quadratic-ugly fast.

The v0.1 FSM already established the cleaner pattern: complex
predicates run in Python actions and write a flat boolean field.
`should_continue` is the precedent. We extend it:

| Boolean field | Written by | Read by guard |
|---------------|------------|---------------|
| `should_continue` | `evaluate_faithfulness` | `should_continue_loop` |
| `advance_stream` | `evaluate_task_advance` (new) | `advance_stream` |
| `next_basis_step == "regrow"` | `compress_with_polygram` | `compressed → regrown` edge |
| `next_basis_step == "compress"` | `perform_regrowth` | `regrown → compressed` edge |
| `next_basis_step == "project"` | both | exit edges to `projected` |

`next_basis_step` is an enum because the *meaning* of "continue" differs
between `compressed` and `regrown` source states. A single bool
would have to be re-derived per state, which is exactly the
state-machine smell we are avoiding.

## `task_trigger`: three implementations, one ctx field

All three triggers share the contract: write `ctx.advance_stream` (and
nothing else FSM-visible) at the end of `evaluate_task_advance`.

### `labeled` (default)

The `TaskStream` source enumerates discrete shards. After each shard's
fine-tune+eval, advance unconditionally until `task_idx + 1 == n_tasks`.

- Predicate: `ctx.task_idx + 1 < ctx.n_tasks`
- This is the "task-labeled shards" case from the design discussion.
- `advance_to_next_task` calls `TaskStream.next()` to load the new
  corpus / replace `_finetune_iterator`.

### `token_budget`

Drift case A: there is no label, but we want to chunk by tokens
processed.

- `fine_tune_model` increments `ctx.tokens_seen_in_task` per step.
- Predicate: `ctx.tokens_seen_in_task >= ctx.token_budget_per_task`
  AND `ctx.task_idx + 1 < ctx.n_tasks`.
- `advance_to_next_task` resets `tokens_seen_in_task = 0`.

### `loss_delta`

Drift case B: advance when held-out probe loss starts climbing.

- `evaluate_faithfulness` already computes a KL on a held-out probe.
  We append it to a sliding window `recent_eval_losses` (size 3 by
  default — small to keep config minimal).
- Predicate: window full AND
  `mean(recent_eval_losses[-3:-1]) - recent_eval_losses[-1]
   > loss_delta_threshold` (sign chosen so larger threshold means
  more sensitivity to *deterioration*).
- `advance_to_next_task` clears `recent_eval_losses`.

The drift triggers are intentionally simple. More sophisticated
drift detection (e.g. CUSUM, ADWIN) is out of scope; the contract
that *any* drift detector writes a single bool into ctx makes
swapping in something better a non-FSM change.

## Activation buffer + protected features

`scan_activations` runs *before* `compress_with_polygram`. It samples
`activation_buffer_size` tokens from the current shard, runs them
through the SAE encoder (not the projected model — the SAE is the
compression target, and we want the basis-side feature usage), and
writes `ctx.feature_usage: list[float]` of length `n_features`.

When `protect_top_k > 0`, it also writes
`ctx.protected_features = argsort(feature_usage)[-protect_top_k:]`.

`compress_with_polygram` forwards the protected list to Polygram's
`Compressor` (via the `do_not_remove` arg if Polygram exposes one,
otherwise via post-filtering the `ValidationReport` to mark those
features as "must-keep"). Polygram support gap is tracked in
tasks.md §4.4 — if Polygram's API doesn't yet support a do-not-remove
set, we wrap the compression report manually.

### Why score against the SAE encoder, not the projected model

The projected model is *downstream* of the basis. Scoring features
against it conflates "feature is informative for this task" with
"feature aligned well with the host model's residual stream after
projection." The first is what we want; the second is a noisy
proxy. Encoder-side scoring also avoids a chicken-and-egg on task 0
(no projected model exists yet — no need to special-case it).

### `protect_score` strategies

- `mean_act` (default): `feature_usage[i] = mean over buffer of |z_i|`
  where `z = SAE.encode(tokens)`. Simple, cheap.
- `usage`: fraction of tokens where `z_i > 0` (sparsity). Picks
  features that fire often, not necessarily strongly.
- `grad_importance`: `mean over buffer of |z_i * dL/dz_i|` against
  reconstruction loss. More expensive (one backward per buffer);
  closer to EWC's Fisher information in spirit.

`mean_act` is the default because it's cheap and lines up with the
KL-vs-rank memory: the features that contribute most to faithfulness
are the high-magnitude ones, not the high-frequency ones.

## Replay buffer

Replay is a fine-tune concern only — the FSM doesn't need to know
about it. `fine_tune_model` wraps its iterator:

```
mixed = MixedIterator(
    primary=task_iterator,
    replay=ReplayBufferIterator(buffer, policy=replay_policy),
    replay_ratio=ctx.replay_ratio,
)
```

After fine-tune, `update_replay_buffer` samples from the just-seen
tokens per `replay_policy`:

- `reservoir`: classic reservoir sampling, uniform across all history.
- `recent_window`: keep the last `replay_buffer_size` tokens, FIFO.
  Cheap, biased toward recency.
- `per_task`: stratify capacity across `n_tasks`, allocate
  `replay_buffer_size / n_tasks` slots per task. Best for catastrophic
  forgetting; requires `task_trigger == "labeled"` to know task
  boundaries (enforced by an init-time check).

When `replay_buffer_size == 0` the whole replay path is a no-op.

## What we are *not* doing in this change

- **No basis-size growth across tasks.** `basis_size_policy` is
  named in the proposal as a future knob; this change ships
  `"fixed"` only. Growing `n_features` mid-stream requires re-shaping
  the projected model, which deserves its own change.
- **No EWC on parameters.** Protected-features is the structural
  analogue at the basis level. Parameter-level EWC would need
  Fisher accumulation across the whole projected model, which
  contradicts the design intent of locating continual-learning
  semantics in the basis.
- **No per-task evaluation matrix.** Reporting forgetting as a
  task-by-task KL matrix is a follow-up evaluator. This change
  ships only the per-shard eval that already exists.
- **No CLI for `task_corpus_glob` or similar.** Stream sources are
  constructed in Python and registered with a process-local handle
  (`task_iterator_id`) the FSM context references. CLI-driven
  multi-shard runs are deferred to the follow-up examples change.

## Migration discipline

The default-knob byte-equivalence guarantee is the hard contract.
`tests/test_imperative_and_fsm_byte_equivalent` (from
`forge-outer-loop-fsm` task 6.5) MUST keep passing without
modification. If the introduction of `activations_scanned` as a
no-op pass-through perturbs RNG state (it should not — `scan_activations`
is read-only when `protect_top_k == 0`), that test catches it.

A new test, `test_continual_default_knobs_match_v01_topology`, asserts
that the *transition log* under defaults contains the same `to_state`
sequence as v0.1 modulo one extra `activations_scanned` entry —
documenting the contract that defaults add at most one no-op hop.

## Open questions (with resolutions)

Reviewed in PR #10. Resolutions captured in `tasks.md` §12.

- **Polygram API for protected features.** Resolution: **upstream the
  `do_not_remove` argument to Polygram's `Compressor` as the long-term
  path.** Ship the `ValidationReport` post-filter as a v0.2 workaround
  if the upstream change has not landed in time, but file the upstream
  issue *first* so the workaround is short-lived. Tracked in tasks.md
  §4.4 and §12.7.
- **`scan_activations` cost on large bases.** For `n_features >> 100k`
  we may want feature-axis sampling. Out of scope here; tracked as
  follow-up in tasks.md §12.2. Also tracked: per-loop-level scan
  buffer config (§12.1) so users can run full scans on the first basis
  pass and cheap scans on inner refinements.
- **Replay sampling unit.** Token-level vs. sequence-level. We ship
  sequence-level; token-level is tracked in tasks.md §12.4.
- **Basis-size growth across tasks.** Out of scope; tracked in
  tasks.md §12.6 as a separate future change. Reshaping the projected
  model mid-stream is non-trivial enough to deserve its own
  capability spec.
- **Per-task evaluation matrix.** Out of scope; tracked in tasks.md
  §12.5 as the follow-up `forge-continual-eval-matrix` change.
- **Custom triggers.** The `task_trigger` contract (single bool
  written by Python, read by flat FSM guard) is closed. Promoting
  the raw underlying signals (`recent_eval_losses`, `tokens_seen_in_task`,
  `feature_usage`) to the public ctx contract for user-written
  triggers without forking actions is tracked in tasks.md §12.3.
