# adaptive-regrow Specification

## Purpose

The `adaptive-regrow` capability defines an opt-in controller that
computes the per-cycle `effective_regrow_count` for the
`perform_regrowth` action in `BasisMachine`. The controller targets
a configured `n_features_target` based on the just-completed
compression's `n_features_kept`, bounded by
`[regrow_count, regrow_max]` and damped by `regrow_damping`.

This capability is opt-in (default `adaptive_regrow=False`); the
v0.2 fixed-`regrow_count` path remains the default and is preserved
byte-identically.

## ADDED Requirements

### Requirement: RegrowController is a deterministic pure function of its inputs

`saeforge.basis.RegrowController.next_count(...)` SHALL be a pure function of its five integer/float inputs (`n_features_kept`, `n_features_target`, `regrow_count`, `regrow_max`, `regrow_damping`). It SHALL NOT
read environment variables, system time, RNG state, or any
non-input. Two calls with identical arguments SHALL return identical
results.

The returned value SHALL satisfy
`regrow_count <= return_value <= regrow_max` for every valid input
combination (where "valid" means `regrow_max >= regrow_count >= 0`,
`n_features_target >= 0`, `regrow_damping ∈ [0.0, 1.0]`).

When `n_features_kept >= n_features_target` (the basis already
exceeds the target), the controller SHALL return `regrow_count` —
the v0.2 fallback. No growth pressure beyond the configured base.

#### Scenario: identical inputs return identical outputs

- **GIVEN** controller call `next_count(50, 100, 5, 32, 0.5)`
- **WHEN** the call is invoked twice with the same arguments
- **THEN** both invocations return the same int
- **AND** the value is in `[5, 32]`

#### Scenario: target reached returns the configured base

- **GIVEN** `n_features_kept = 150`, `n_features_target = 100`
- **WHEN** `next_count(150, 100, 5, 32, 0.5)` is called
- **THEN** the returned value is `5` (the `regrow_count` base)

#### Scenario: large gap is bounded by regrow_max

- **GIVEN** `n_features_kept = 0`, `n_features_target = 1000`,
  `regrow_max = 32`
- **WHEN** `next_count(0, 1000, 5, 32, 1.0)` is called (no damping)
- **THEN** the returned value is `32` (capped at `regrow_max`)

### Requirement: ForgePipeline validates adaptive_regrow knob coupling

`ForgePipeline.__post_init__` SHALL validate that when `adaptive_regrow=True` the dependent knobs are coherent. Specifically the constructor SHALL require BOTH:

- `regrow_max > regrow_count`
- `n_features_target > 0`

Failing either check SHALL raise `ValueError` whose message names
both fields and their current values. When `adaptive_regrow=False`,
the other three knobs (`regrow_max`, `n_features_target`,
`regrow_damping`) MAY be set to any value but SHALL be silently
ignored — they are inert when adaptation is disabled.

#### Scenario: adaptive without regrow_max raises ValueError

- **GIVEN** `ForgePipeline(adaptive_regrow=True, regrow_count=5,
  regrow_max=0, n_features_target=128)` is constructed
- **WHEN** `__post_init__` runs
- **THEN** `ValueError` is raised
- **AND** the message contains both `"regrow_max"` and
  `"regrow_count"`

#### Scenario: adaptive without n_features_target raises ValueError

- **GIVEN** `ForgePipeline(adaptive_regrow=True, regrow_count=5,
  regrow_max=32, n_features_target=0)` is constructed
- **WHEN** `__post_init__` runs
- **THEN** `ValueError` is raised
- **AND** the message contains `"n_features_target"`

#### Scenario: disabled toggle silently ignores other knobs

- **GIVEN** `ForgePipeline(adaptive_regrow=False, regrow_count=5,
  regrow_max=99, n_features_target=999, regrow_damping=0.7)` is
  constructed
- **WHEN** `__post_init__` runs
- **THEN** no error is raised
- **AND** the resulting ctx has `adaptive_regrow=False` and the
  other three fields are written to ctx but unused

### Requirement: adapt_and_regrow is byte-identical to perform_regrowth when adaptive_regrow is False

The composed action `saeforge.actions.adapt_and_regrow` SHALL
short-circuit on `ctx.get("adaptive_regrow") is False` (or the
field is missing) by returning the result of
`perform_regrowth(ctx, payload)` directly. No call to the
controller, no write to `effective_regrow_count`, no extra
`transitions_log` entry.

The byte-equivalence test
`test_imperative_and_fsm_byte_equivalent` SHALL pass without
modification under the default ctx (no adaptation knobs set).

#### Scenario: disabled toggle skips the controller entirely

- **GIVEN** ctx with `adaptive_regrow=False, regrow_count=5,
  regrow_max=32, n_features_target=128`
- **WHEN** `adapt_and_regrow(ctx, payload)` is called
- **THEN** the controller is NOT invoked (verified via mock)
- **AND** `effective_regrow_count` is NOT written to ctx
- **AND** the returned delta equals the delta from a direct
  `perform_regrowth(ctx, payload)` call on the same ctx

#### Scenario: byte-equivalence gate continues to pass

- **GIVEN** the existing `test_imperative_and_fsm_byte_equivalent`
  test setup (no adaptive knobs set)
- **WHEN** the test runs against the post-this-change tree
- **THEN** the imperative-path and FSM-path forged weights are
  byte-identical
- **AND** the action sequence in `transitions_log` is unchanged
  from v0.2

### Requirement: enabled toggle computes effective_regrow_count and logs both inner action names

`adapt_and_regrow` SHALL — when `ctx["adaptive_regrow"] is True` and the cold-start gate has passed (`current_feature_count` was written by a prior `compress_with_polygram`):

1. Call `_compute_effective_regrow_count(ctx)` which invokes the
   `RegrowController` and writes `effective_regrow_count` to ctx.
2. Append a `transitions_log` entry with action name
   `adapt_regrow_count` and `extras` containing the computed
   value, the gap (`n_features_target - n_features_kept`), and
   the configured `n_features_target`.
3. Call `perform_regrowth(ctx, payload)` which now reads
   `effective_regrow_count` from ctx (instead of `regrow_count`)
   and runs the regrow.
4. Append the regular `perform_regrowth` log entry.

The `transitions_log` SHALL therefore record exactly two entries
per regrow cycle: `adapt_regrow_count` then `perform_regrowth`,
in that order. Existing readers that index by action name see
one extra entry per regrow cycle.

#### Scenario: enabled cycle logs both action names in order

- **GIVEN** ctx with adaptation enabled and valid signals
  (`current_feature_count = 80, n_features_target = 128,
   regrow_count = 5, regrow_max = 32, regrow_damping = 0.5`)
- **WHEN** `adapt_and_regrow(ctx, payload)` is called
- **THEN** ctx contains `effective_regrow_count` (an int in
  `[5, 32]`)
- **AND** `transitions_log` contains exactly two new entries
- **AND** the first entry's `action` is `"adapt_regrow_count"`
- **AND** the second entry's `action` is `"perform_regrowth"`

### Requirement: cold-start cycle uses configured regrow_count verbatim

`adapt_and_regrow` SHALL detect the cold-start case (no prior compression has populated `current_feature_count`) and short-circuit to `perform_regrowth` using the configured `regrow_count` without invoking the controller. The first invocation in a forge run runs
*before* any `compress_with_polygram` has populated
`current_feature_count` (because the basis-loop transition order
is `compressed → regrown`, but the first compression must run
before the basis loop self-loops). The composed action SHALL
detect the cold-start case via
`ctx.get("current_feature_count") is None or 0` and short-circuit
to `perform_regrowth` with the configured `regrow_count`,
skipping the controller call.

This SHALL apply only on the first cycle. On every subsequent
cycle, `current_feature_count` is non-zero (written by the
preceding `compress_with_polygram`) and the controller runs
normally.

#### Scenario: first cycle skips the controller

- **GIVEN** ctx with adaptation enabled but
  `current_feature_count == 0` (cold start)
- **WHEN** `adapt_and_regrow(ctx, payload)` is called
- **THEN** the controller is NOT invoked
- **AND** the resulting regrow uses `regrow_count` from ctx
- **AND** no `adapt_regrow_count` entry is appended to
  `transitions_log` (only the `perform_regrowth` entry from
  the inner call)

### Requirement: two runs with identical seed and config produce byte-identical artifacts

Two `ForgePipeline.run()` invocations under `adaptive_regrow=True` with identical input seeds, identical ctx configuration, and identical input data SHALL produce byte-identical `forged/model.safetensors` files. This is the
v1 determinism guarantee.

This is NOT the same as the v0.2 byte-equivalence gate (which
compares imperative-path vs FSM-path under the *fixed-regrow*
default). The adaptive path is *not* expected to match the
fixed-regrow path — they produce different basis sizes by
design. The adaptive path is expected to match itself.

#### Scenario: deterministic forge under adaptive regrow

- **GIVEN** two `ForgePipeline` instances with identical configs:
  `adaptive_regrow=True, regrow_count=5, regrow_max=32,
   n_features_target=128, regrow_damping=0.5`
- **AND** the same `eval_input_ids` and same seed
- **WHEN** both are run end-to-end
- **THEN** the two `forged/model.safetensors` artifacts are
  byte-identical
- **AND** their final `current_feature_count` values match
- **AND** their `transitions_log` action sequences match

### Requirement: protect_top_k interaction is documented but not auto-managed

The implementation SHALL NOT auto-scale `protect_top_k` to track the basis size when `adaptive_regrow=True` is combined with a positive `protect_top_k`. The protected feature *count* SHALL remain constant across cycles even as the basis grows.

This requirement is non-normative for the v1 controller — it
exists to pin the documented coupling so users see it in the
spec. The right long-term fix is `protect_top_k_ratio`, tracked
as a separate change. The `docs/advanced-fsm-options.md` adaptive-regrow
subsection SHALL include a callout describing this coupling.

#### Scenario: protected count stays constant under basis growth

- **GIVEN** a forge run with `adaptive_regrow=True, protect_top_k=5`
- **WHEN** the basis grows from `current_feature_count=64` to
  `current_feature_count=128` across cycles
- **THEN** `len(ctx["protected_features"]) == 5` at every cycle
- **AND** the protected fraction has shrunk from 5/64 (≈7.8%) to
  5/128 (≈3.9%)
- **AND** no warning is logged (this is documented behavior, not
  an error)
