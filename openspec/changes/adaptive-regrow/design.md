# Design: adaptive-regrow

## Naming and field semantics

The proposal introduces three fields beyond the existing
`regrow_count`. To avoid the three-name confusion the original
draft had (`regrow_count` / `regrow_base` / `current_regrow_count`),
the names are pinned here:

| Field | Type | Default | Semantics |
|---|---|---|---|
| `regrow_count` | int | 0 | **Existing.** Configured base / fallback. Read by both v0.2 fixed path and v1 adaptive path. Never mutated. |
| `regrow_max` | int | 0 | **New.** Upper bound on the per-cycle regrow value when adaptation is on. `0` means "adaptation disabled" (the v0.2 default). |
| `n_features_target` | int | 0 | **New.** The configured target feature count. The controller grows toward this; `0` means "no target, no adaptation." |
| `regrow_damping` | float | 0.5 | **New.** Damping factor `∈ [0, 1]` applied to per-cycle adjustment. `1.0` = no damping (jump straight to target); `0.0` = no growth. |
| `effective_regrow_count` | int | unset | **New (per-cycle).** Written by `_compute_effective_regrow_count` before each regrow. Read by `perform_regrowth`. Cleared after each regrow cycle so the next cycle recomputes. |
| `adaptive_regrow` | bool | False | **New.** Master toggle. When False, every adaptive code path is an early-return — guarantees byte-equiv with v0.2. |

Crucially, the existing field `regrow_count` is preserved as the
configured input — it does NOT get renamed to `regrow_base`. This
lets every existing config, test, and CLI invocation continue to
work without modification. The ONLY ctx-field semantic change is
the addition of `effective_regrow_count`, which is read by
`perform_regrowth` only when `adaptive_regrow == True`.

## Signal source: `n_features_kept` only, for v1

Three candidate signals were considered:

1. **`n_features_kept`** (polygram-side, post-compression). The
   number of features the polygram `Compressor` retained. Direct
   measure of how much the basis shrank in this cycle.
2. **`scale_compression_ratio`** (polygram-side). The ratio of
   merged-norms / original-norms. Captures how much information
   the clustering preserved per kept feature.
3. **`recent_eval_losses` delta** (forge-side). Loss trajectory
   over the last N evals.

**v1 picks (1) exclusively.** Reasons:

- It's a *direct* measure of the controller's target. The
  controller's job is "make the basis size approach
  `n_features_target`"; `n_features_kept` is the basis size after
  the just-completed compression. Targeting it is a one-line
  formula.
- It's already in the `compress_with_polygram` action's
  `transitions_log` entry (we don't need to wire new fields).
- It's deterministic given the seed and the input SAE — no RNG
  in the polygram clustering once the seed is fixed.

`scale_compression_ratio` (2) is a richer signal but is
quality-shaped, not size-shaped — using it requires deciding on
a quality threshold, which is itself a tuning knob. Deferred to
the loss-based follow-up where loss-quality coupling is the
focus.

`recent_eval_losses` (3) is the most "intelligent" signal but
the *least* deterministic: the fine-tune step uses RNG-seeded
optimization, the eval input order matters, and bf16/fp16
accumulation noise shows up at the 4th decimal place. Forcing
determinism over loss signals would mean pinning the entire
fine-tune RNG, which the project explicitly does not want.
Deferred to `adaptive-regrow-loss`.

## The controller equation (v1: linear, damped, bounded)

```python
def next_count(
    n_features_kept: int,
    n_features_target: int,
    regrow_count: int,        # base/fallback
    regrow_max: int,           # bound
    regrow_damping: float,     # damping factor
) -> int:
    gap = max(0, n_features_target - n_features_kept)
    damped = int(round(gap * regrow_damping))
    return max(regrow_count, min(damped, regrow_max))
```

Properties pinned by the spec:

- **Bounds.** `regrow_count <= effective <= regrow_max` always.
- **Monotone in gap.** Larger gap → larger `effective` (until
  bounded by `regrow_max`).
- **Cold-start fallback.** When `n_features_kept` is unset
  (first cycle, before any compression has run), the controller
  returns `regrow_count`. The wrapper action checks
  `ctx.get("current_feature_count") is None` and short-circuits
  to `regrow_count` before calling the controller.
- **Disabled fallback.** When `adaptive_regrow=False` OR
  `regrow_max=0` OR `n_features_target=0`, the controller is
  not invoked; `effective_regrow_count` is never written; and
  `perform_regrowth` falls back to `regrow_count` exactly as in
  v0.2. Byte-equivalence is preserved by *not running the
  controller at all*, not by ensuring it computes the same
  value.

The damping factor is intentionally exposed because the
basis-loop self-loops `inner_refine_passes` times per shard. A
damping of `1.0` would jump to the target on the first pass; a
damping of `0.5` reaches the target asymptotically over several
passes. The default `0.5` is a heuristic — empirically chosen
in the "no oscillation, smooth growth" range — and the user can
tune it.

## FSM placement: composed action, not new state

Two options were considered for inserting the controller into
`BasisMachine`:

### Option A (rejected): new state `adapting` between compressed and regrown

```
compressed → adapting (compress_done, should_regrow, action=adapt_regrow_count)
adapting → regrown (adapt_done, action=perform_regrowth)
```

Pro: explicit in the Mermaid diagram; easier to reason about in
isolation.
Con: state count grows from 7 to 8 in `BasisMachine`; every test
that asserts the state set needs updating; the topology check
in `tests/fsm/test_topology.py` flags it; the diagram regen
becomes mandatory and reviewer-noticeable.

### Option B (picked): composed action `adapt_and_regrow`

```
compressed → regrown (compress_done, should_regrow, action=adapt_and_regrow)
```

The new action is a Python helper that internally calls
`_compute_effective_regrow_count` then `perform_regrowth`,
mirroring the `load_and_scan` pattern from `hierarchical-fsm`.
Both inner action names log to `transitions_log` so consumers
see the same shape they would under Option A.

Pro: zero topology change; the byte-equivalence test under
`adaptive_regrow=False` is trivially preserved (the composed
action with the controller short-circuited reduces to a pure
`perform_regrowth` call). One transition action rename in
`basis.orca.md`. One Mermaid label change (caught by drift CI).
Con: less visible in the diagram; readers have to know that
`adapt_and_regrow` is a composed helper.

**Picked: Option B.** The byte-equivalence gain is structural,
not operational — Option A would require either a no-op
`adapt_regrow_count` action under `adaptive_regrow=False` (which
adds a `transitions_log` entry that doesn't exist in v0.2) or a
guard-gated transition (which adds a guard expression, more
topology change). Option B's "skip the controller entirely when
disabled" is the cleaner v0.2-compatible path.

## Transition table delta on basis.orca.md

Exactly one row changes:

```diff
-| compressed | compress_done | should_regrow | regrown | perform_regrowth |
+| compressed | compress_done | should_regrow | regrown | adapt_and_regrow |
```

Every other row in `basis.orca.md` is unchanged. The action
table (`## actions`) gains one row:

```diff
+| adapt_and_regrow | (ctx) -> Context |
```

`perform_regrowth` stays in the action table (it's still called
by the composed helper, and remains the public action name in
documentation). The Mermaid diagram regenerates with the label
`/ adapt_and_regrow` on the affected edge; the drift CI fails
the build until the doc is regenerated.

## Determinism

The signal is `n_features_kept`, written by
`compress_with_polygram` from the polygram `CompressionReport`.
The polygram `Compressor` is deterministic given the seed and
the input SAE checkpoint (verified by polygram's own test suite,
and required by the existing `test_imperative_and_fsm_byte_equivalent`
test which compares safetensors bytes). The controller is a
pure function of integer inputs and a float damping factor —
itself deterministic. Therefore the per-cycle
`effective_regrow_count` is deterministic given the same seed
and config.

The byte-equivalence test under `adaptive_regrow=False` is
preserved because the controller is not invoked at all — the
composed action's first line is
`if not ctx.get("adaptive_regrow"): return perform_regrowth(ctx)`.

The byte-equivalence test under `adaptive_regrow=True` is NOT
expected to match the v0.2 fixed-regrow output (the basis size
will be different per cycle, hence the forged weights will
differ). This is by design — the adaptive path produces a
different basis than the fixed path. The relevant invariants
under `adaptive_regrow=True` are:

- Two runs with the same seed and config produce byte-identical
  forged weights (run-to-run reproducibility).
- The final `current_feature_count` is bounded:
  `n_features_target` *or* the v0.2 fixed-regrow upper bound,
  whichever is smaller. (Specifically, `regrow_max` caps each
  cycle, but `n_features_target` is the asymptote.)

Both invariants are pinned in `tests/fsm/test_adaptive_regrow.py`.

## Interaction with `protect_top_k`

Adaptive regrow grows the basis without touching the protected
set. The protected set is sized as an absolute count
(`protect_top_k`), so growing the basis from 100 to 200 features
while `protect_top_k = 5` shrinks the protected fraction from
5% to 2.5%. This is a real coupling, not a bug, but it can
surprise users who set `protect_top_k` once and walk away.

v1 documents this interaction in
`docs/advanced-fsm-options.md` but does not change the behavior.
The right long-term fix is `protect_top_k_ratio` (an additional
field that scales with the basis size), tracked as a separate
follow-up change. The `adaptive-regrow` capability spec
explicitly notes this interaction in a non-normative paragraph.

## Why this is internal-additive, not internal-replacing

The default behavior under `adaptive_regrow=False` is bit-for-bit
identical to v0.2 — same composed-action path, same controller
short-circuit, same `perform_regrowth` invocation, same
`regrow_count` consumption. The byte-equivalence gate verifies
this. The only externally-visible change for v0.2 users is the
one renamed action label in the Mermaid diagram, which is
internal documentation, not a behavior surface.

For users who opt in (`adaptive_regrow=True`), the behavior is
genuinely different — the forged basis grows toward
`n_features_target` instead of growing by `regrow_count` every
cycle. This is the whole point.

## Risks and rejected alternatives

### Rejected: implement as a new state `adapting`

Covered above (Option A). Topology drift cost outweighs
diagram-clarity benefit.

### Rejected: PID controller

PID over an int-valued setpoint suffers from quantization
bang-bang at the boundary. The damped linear controller is
already smooth enough for this domain. PID is the right tool
for continuous setpoints (e.g. learning rate); not the right
tool for `regrow_count`.

### Rejected: ML-based / learned controller

Adds a training surface inside the forge runtime. The
forge philosophy is "no learned components in the control
plane" — controllers should be auditable, deterministic, and
explainable. Linear-with-damping is.

### Risk: damping factor 0.5 might be wrong

Mitigated by exposing it as a tunable knob with a CLI flag.
Default is 0.5 because it's the smoothest "always grow,
asymptotically reach target" value across the synthetic
test scenarios. Real workloads may want lower (slower growth)
or higher (faster convergence to target). Documented in the
tuning section.

### Risk: signal lag

The controller reads `n_features_kept` from the *just-completed*
compression. By the time the regrow runs, that compression's
basis is already shaped. Adaptation responds with one-cycle
lag: cycle N's compression result drives cycle N's regrow,
which is then re-compressed in cycle N+1. This is fine for the
basis-loop pattern (it self-loops several times per shard) but
worth knowing for cross-shard tuning. Documented.

## Why land it now

Three queued changes (`adaptive-regrow`, `multi-objective-triggers`,
`adaptive-regrow-loss`) all want to insert logic at specific
points in the basis or stream loop. The hierarchical-fsm
refactor (PR #15) made each of these a single-machine touch
instead of a flat-graph re-partition. Landing `adaptive-regrow`
first validates the "single sub-machine touch" claim and gives
the other two a working precedent to follow.
