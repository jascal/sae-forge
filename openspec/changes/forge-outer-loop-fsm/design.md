# Design: forge outer loop as an orca-lang FSM

## Why an FSM here, and not elsewhere

orca-lang's value is highest where (a) control flow has multiple legal
paths, (b) wrong paths are silently dangerous, and (c) the per-step
work is large enough that orchestration overhead is irrelevant. The
forge outer loop hits all three:

- Multiple legal paths: regrowth on/off, single-pass vs multi-cycle,
  early-stop on faithfulness, early-stop on perplexity regression.
- Silent danger: re-entering `compressed` from `done` (lost output),
  skipping `evaluated` (silent faithfulness regression), running
  `regrowth` without a prior `compressed` (no zeroed slots to fill).
- Large steps: each action is seconds-to-hours; FSM dispatch is
  microseconds.

Below the outer loop — inside `SubspaceProjector.project_module`,
inside `compress_with_polygram` — control flow is short, linear, and
hot. An FSM would be ceremony there. The boundary is intentional.

## State graph

```
init ──start──▶ loaded ──load_done──▶ compressed
                                          │
                  ┌───────── regrow_count == 0 ─────────┐
                  ▼                                      ▼
              projected ◀── regrowth_done ── regrown    │
                  │                                      │
                  ▼                                      │
              finetuned ──finetune_done──▶ evaluated     │
                                              │          │
                  ┌── should_continue_loop() ─┘          │
                  ▼                                      │
              compressed (next iter)                     │
                                                          │
              evaluated ──!should_continue_loop()──▶ done│
                                                          │
              any state ──error──▶ failed ◀──────────────┘
```

Static-verification checks the orca-lang verifier enforces:

- Every non-final state has at least one outgoing non-`error`
  transition (no dead-end-not-final states).
- Every final state has zero outgoing transitions.
- The loop edge `evaluated → compressed` is reachable only when
  `should_continue_loop()` is true; the `evaluated → done` edge is
  reachable only when it is false; together they cover the guard space.
- `error` is the only event that targets `failed`, and every state
  that performs side-effecting work (`loaded`, `compressed`, `regrown`,
  `projected`, `finetuned`, `evaluated`) declares an `error → failed`
  transition.

## Context dataclass

```python
@dataclass
class ForgeContext:
    # Inputs (set at start, not mutated except as noted)
    sae_checkpoint: str
    corpus_path: str
    host_model_id: str
    output_dir: str

    # Knobs
    iterations: int = 1
    regrow_count: int = 0
    target_feature_ratio: float = 0.25
    compression_strategy: str = "scale_aware_zero"
    min_faithfulness: float = 0.90
    quantum_aware: bool = False

    # Mutable state
    current_iter: int = 0
    current_sae_path: str = ""        # rotated each iteration
    compressed_sae_path: str = ""     # output of compress_with_polygram
    regrown_sae_path: str = ""        # output of perform_regrowth (when run)
    current_feature_count: int = 0    # n_kept after the latest compress
    projected_weights_path: str = ""  # output of project_to_subspace
    finetuned_model_path: str = ""    # output of fine_tune_model
    faithfulness: float = 0.0         # written by evaluate_faithfulness
    perplexity: float = float("inf")
    best_perplexity: float = float("inf")
    final_model_path: str = ""        # written by save_final_model
    error_message: str = ""           # written by log_error
```

`current_sae_path` is the field that rotates each iteration.
`load_sae_and_corpus` initializes it to `sae_checkpoint`;
`rotate_for_next_iter` rebinds it to the most recent
`regrown_sae_path or compressed_sae_path` so the next compression
operates on the previous iteration's output.

## Guards

`should_continue_loop()` is the only non-trivial guard:

```
ctx.current_iter + 1 < ctx.iterations
  and ctx.faithfulness >= ctx.min_faithfulness
  and ctx.perplexity < ctx.best_perplexity
```

The three clauses correspond to: budget remaining, faithfulness floor
not yet violated, perplexity strictly improved over the previous best
(no-op iterations terminate). `increment_iter` runs *only on the loop
edge* and updates `current_iter += 1` and
`best_perplexity = min(best_perplexity, perplexity)`.

`regrow_count > 0` is a literal-comparison guard on the `compress_done`
event; orca-lang's static checker treats it as a partition over the
event domain, so the verifier confirms one and only one branch fires.

## Action signatures

All actions take and return `ForgeContext`. None of them perform
control flow — they read inputs from the context, do their numerical
work, and write results back. Errors raise; the orca-lang runtime
maps the raise to the `error` event.

```python
def load_sae_and_corpus(ctx: ForgeContext) -> ForgeContext: ...
def compress_with_polygram(ctx: ForgeContext) -> ForgeContext: ...
def perform_regrowth(ctx: ForgeContext) -> ForgeContext: ...
def project_to_subspace(ctx: ForgeContext) -> ForgeContext: ...
def fine_tune_model(ctx: ForgeContext) -> ForgeContext: ...
def evaluate_faithfulness(ctx: ForgeContext) -> ForgeContext: ...
def increment_iter(ctx: ForgeContext) -> ForgeContext: ...
def rotate_for_next_iter(ctx: ForgeContext) -> ForgeContext: ...
def save_final_model(ctx: ForgeContext) -> ForgeContext: ...
def log_error(ctx: ForgeContext) -> ForgeContext: ...
```

`should_continue_loop` is the one non-action callable — it's a guard
predicate `(ForgeContext) -> bool`, registered in the machine's guard
table rather than the action table.

## Quantum-aware path (§7)

`--quantum-aware` flips `ctx.quantum_aware = True`. The flag is read
**only** by `compress_with_polygram`, which then calls Polygram's
`Compressor` with `confirmer="quantum_interference"` instead of the
default `decoder_geometry`. The FSM topology is identical; no new
states or transitions are introduced. q-orca-lang stays out of
sae-forge's import surface — it is reached transitively through
Polygram's `[behavioural]` extra, when and only when
`--quantum-aware` is passed.

## Open questions (deferred to future changes)

1. **Parallel compression strategies.** Should the FSM support a
   parallel region that runs `scale_aware_zero` and `scale_aware_merge`
   simultaneously, then keeps the winner? Defer: orca-lang's parallel
   regions are stable but the action layer would need a join with
   structural-scoring logic that is not yet specified. Track as
   `compare-compression-strategies-fsm`.
2. **Human-in-the-loop review state.** A `human_review` state between
   `evaluated` and `done`/`compressed` would let a researcher accept
   or reject each cycle. Defer: out of scope for v0.1, and adding it
   later is purely additive (one new state, two new transitions).
3. **Light fine-tune before regrowth.** Polygram's `Regrower` takes
   raw activation residuals; running a small fine-tune *before*
   regrowth might give cleaner residuals. Defer: this is a Polygram
   question, not a sae-forge orchestration question. Open it there
   if signal warrants.

## v0.1 implementation notes vs the original proposal

Two surprises during implementation, both affecting the original spec:

1. **Dependency name.** The proposal called for `orca-lang>=0.5`. That
   package does not exist on PyPI. The actual classical-orca Python
   runtime is `orca-runtime-python` (PyPI), module name
   `orca_runtime_python`. We pin `>=0.1.27` because earlier releases
   ship stubbed `_evaluate_guard` (always-true) and `_execute_action`
   (no-op). Both work in 0.1.27.

2. **Guard arithmetic is not parsed.** orca-runtime-python's parser
   silently mis-parses `ctx.field + 1 < ctx.other` as a null-check on
   `ctx.field`, which always passes. The original design.md guard
   `ctx.current_iter + 1 < ctx.iterations` would loop forever. The
   v0.1 implementation moves the loop-condition logic into the
   `evaluate_faithfulness` action, which writes a boolean
   `should_continue` field, and the FSM guards become the trivial
   `ctx.should_continue == true` / `== false`. This keeps the loop-
   termination logic in Python where comparisons + arithmetic are
   reliable, and the FSM only branches on a flat boolean. Update
   upstream is tracked separately.

3. **Compress / regrow / fine-tune are no-op pass-throughs in v0.1.**
   The actions update `current_sae_path` bookkeeping but don't yet
   call into Polygram's `Compressor` / `Regrower` (Polygram isn't on
   PyPI; only editable installs work). v0.2 swaps these for real
   calls. The byte-equivalence test holds because both orchestrators
   currently exercise the same projection-only path; v0.2 will need a
   weight-equivalence test that's tolerant to compression non-
   determinism.

## Why not Q-orca for this

The forge outer loop is classical: states are program states, not
quantum states. There is no superposition, no entanglement, no
unitary evolution. Using q-orca here would be a category error and
would force `[behavioural]`-tier dependencies (torch via q-orca's
runtime) into every FSM run. q-orca remains the right tool for
quantum-mechanical analyses of feature dictionaries, which are
Polygram's responsibility, not sae-forge's.
