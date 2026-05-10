## Why

`forge-continual-learning-loop` (v0.2, PR #11) shipped a fixed
`regrow_count` integer that the `Regrower` action consumes after
each `compress_with_polygram` pass: grow the basis by exactly N
features, every cycle. It works, but it has two friction points
that show up immediately on real continual-learning runs:

1. **Per-domain tuning.** The "right" `regrow_count` for a
   small-vocab GPT-2 shard differs from the right value for a
   distribution-shifted shard later in the same run. Tuning by
   hand defeats the purpose of running multi-shard at all.
2. **No back-pressure.** A run that compressed too aggressively
   (low `scale_compression_ratio`, low `n_features_kept`) keeps
   regrowing the same fixed amount and underfits all the way
   through. A run that compressed gently keeps regrowing the same
   fixed amount and inflates uncontrollably across shards.

The newly-landed `hierarchical-fsm` (PR #15) made this trivially
easy to fix: `BasisMachine` is the right insertion point, the
basis-loop transition `compressed → regrown` is the right edge,
and the `transitions_log` already carries the polygram report
fields the controller needs. This change adds an optional
*adaptive* regrow path that consumes a single polygram-side
signal — `n_features_kept` — and targets a configured feature
count, bounded by `[regrow_count, regrow_max]`.

The shipped behavior under `adaptive_regrow=False` (default) is
**byte-identical** to v0.2. The byte-equivalence test continues
to be the load-bearing acceptance gate.

## What Changes

### Scope

Add an optional adaptive regrow controller to `BasisMachine`.
When `adaptive_regrow=True`, the controller computes
`effective_regrow_count` per cycle from the post-compression
polygram report and writes it to ctx; `perform_regrowth` reads
`effective_regrow_count` if present, falling back to the
configured `regrow_count` otherwise.

The fixed `regrow_count` stays as the configured base value and
remains the default behavior. **The `BasisMachine` topology
(state set + transition table) is unchanged** — adaptation is
implemented as a *composed action* on the existing
`compressed → regrown` transition, mirroring the `load_and_scan`
pattern from `hierarchical-fsm`. Zero topology drift; zero
Mermaid-diagram regen needed.

### New artifacts

- **`saeforge/basis.py` extension** — a new
  `RegrowController` class colocated with `FeatureBasis`. The
  controller is pure-Python, takes a polygram `CompressionReport`
  + the configured knobs, and returns the per-cycle
  `effective_regrow_count`. No torch, no IO, no state — fully
  deterministic given its inputs. (We do NOT introduce a
  `saeforge/basis/` package; the existing `basis.py` single-file
  layout is the convention.)
- **`saeforge/actions/__init__.py`: new `adapt_and_regrow`
  composed action** — runs `_compute_effective_regrow_count`
  (which calls `RegrowController.next_count(...)`) and then the
  existing `perform_regrowth`, threading the new
  `effective_regrow_count` ctx field between them. The
  `transitions_log` records `adapt_regrow_count` and
  `perform_regrowth` as two consecutive entries (preserving the
  v0.2 log shape that consumers depend on).
- **`tests/fsm/test_adaptive_regrow.py`** — controller unit
  tests (deterministic, bounds-respecting, target-tracking) +
  one integration test driving `BasisMachine` through several
  cycles with synthetic polygram reports + the byte-equivalence
  guarantee under `adaptive_regrow=False`.

### Modified artifacts

- **`saeforge/forge.py`** — `ForgePipeline` gains four new
  fields (all optional, all defaulting to v0.2 behavior):
  `adaptive_regrow: bool = False`, `regrow_max: int = 0`,
  `n_features_target: int = 0`,
  `regrow_damping: float = 0.5`. `_build_fsm_ctx` writes them to
  ctx alongside the existing `regrow_count`. Validation:
  `adaptive_regrow=True` requires `regrow_max > regrow_count`
  and `n_features_target > 0`.
- **`saeforge/machines/basis.orca.md`** — *one* transition
  action rename: the existing `compressed → regrown` transition's
  action changes from `perform_regrowth` to `adapt_and_regrow`.
  No state change, no guard change, no new edges. The Mermaid
  diagram regenerates with one label change; the drift CI
  (`tests/fsm/test_diagram_drift.py`) catches it and forces the
  doc update.
- **`saeforge/cli.py`** — new flags: `--adaptive-regrow`
  (boolean), `--regrow-max N`, `--n-features-target N`,
  `--regrow-damping FLOAT`. All optional; all behind the
  `adaptive_regrow` toggle.
- **`docs/advanced-fsm-options.md`** — new "Adaptive regrow"
  subsection under `### Basis loop (inner)` documenting the
  controller, the signal source (`n_features_kept` from the
  polygram `CompressionReport`), the targeting equation, and
  tuning guidance. The auto-generated Mermaid block updates to
  reflect the action rename.
- **`CHANGELOG.md`** — `## [Unreleased]` entry under
  `### Added`.

### CLI surface

```
sae-forge forge ... \
  --adaptive-regrow \
  --regrow-max 64 \
  --n-features-target 256 \
  --regrow-damping 0.5
```

The four flags are mutually-required: passing `--adaptive-regrow`
without `--regrow-max` and `--n-features-target` raises an
argparse error. `--regrow-damping` defaults to 0.5.

### Out of scope (deferred)

- **Loss-based signals.** `recent_eval_losses` exists in ctx
  but its determinism depends on the fine-tune RNG and the
  exact eval-input order. v1 uses the polygram-side
  `n_features_kept` exclusively because it is fully
  deterministic given the seed and the basis. Loss-based
  controllers tracked as a separate change
  (`adaptive-regrow-loss`).
- **Cross-shard regrow scheduling.** v1 is per-cycle (per
  basis-loop pass within a shard). Cross-shard scheduling
  (e.g. "grow more aggressively on shards 2+") is a follow-up.
- **Automatic `protect_top_k` adjustment.** Growing the basis
  while holding the protected set fixed shrinks the protected
  set's relative weight; this is a real coupling but the right
  fix is to expose `protect_top_k_ratio` (a fraction of the
  current basis size). Tracked as `protect-top-k-ratio`.
- **Learned controllers** (PID-style or ML-based). v1 ships
  the simple linear controller documented in `design.md`. PID
  tuning over an int-valued setpoint produces bang-bang
  behavior at the boundary; an ML controller adds a training
  surface we don't want in the forge runtime. Linear is the
  right v1.
- **Visualization / dashboard hooks.** The Mermaid diagram and
  `transitions_log` already surface `effective_regrow_count`
  per cycle. A live dashboard is a separate research
  investment.

## Capabilities

### New Capabilities

- **`adaptive-regrow`** — defines the controller's contract:
  signal source (`n_features_kept` from the polygram
  `CompressionReport`), targeting equation, bounds enforcement
  (`effective_regrow_count ∈ [regrow_count, regrow_max]`),
  cold-start fallback (first cycle uses `regrow_count`
  verbatim), and the deterministic-given-inputs guarantee.
  Includes the byte-equivalence-when-disabled scenario.

### Modified Capabilities

- **`forge-outer-loop`** (the capability now archived from
  `forge-continual-learning-loop`, augmented by the
  `hierarchical-fsm` MODIFIED delta) — one MODIFIED
  requirement reframes the basis-loop's `compressed → regrown`
  transition: action name changes from `perform_regrowth` to
  `adapt_and_regrow`. The transition's source, event, guard,
  and target are unchanged. The composed action's runtime
  semantics under `adaptive_regrow=False` are byte-identical
  to v0.2.

## Impact

- **No public API breakage.** `ForgePipeline.run()`,
  `run_machine(initial_context)`, the CLI surface, and the
  on-disk artifact layout are unchanged for the v0.2 default
  case (`adaptive_regrow=False`). New flags are additive.
- **Zero topology drift.** `BasisMachine`'s state set and
  transition graph are unchanged; only one transition's action
  name changes. The Mermaid diagram regenerates with one label
  change; the drift CI catches it.
- **`transitions_log` schema additive.** Entries from the new
  `adapt_regrow_count` action appear as a separate log entry
  before each `perform_regrowth` entry when adaptation runs.
  Existing readers that index by name see one extra entry per
  regrow cycle and are otherwise unaffected.
- **Test surface.** ~12 new tests in
  `tests/fsm/test_adaptive_regrow.py` (5 controller unit, 4
  integration, 3 byte-equivalence-when-disabled). All existing
  tests continue to pass — the byte-equivalence acceptance
  gate (`test_imperative_and_fsm_byte_equivalent`) is the
  primary check.

## Sequencing

- **Depends on:** `hierarchical-fsm` archiving (already on
  `main` after PR #15) and `forge-continual-learning-loop`
  archiving (already in flight). The capability deltas in
  this change assume both are present in `openspec/specs/`.
- **Independent of:** `forge-whisper-encoder` (architecture
  adapter work, no FSM intersection).
- **Single PR.** No staged rollout — the byte-equivalence gate
  passes or it doesn't, and a half-migrated tree has no
  meaningful intermediate state.
