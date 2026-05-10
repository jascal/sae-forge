## Why

`forge-continual-learning-loop` (v0.2, merged in PR #11) shipped a
single flat orca machine — `saeforge/machines/sae_forge.orca.md` —
that emulates three logically nested loops (Stream → Refine →
Basis) through ten states, eight guards, and a `(advance_stream,
should_continue, inner_refine_idx)` ctx tuple that the guard
expressions partition by hand. It works, the byte-equivalence net
holds, but the topology is *implicit*: a reader has to mentally
project the flat transition table back onto the three-level mental
model that `docs/advanced-fsm-options.md` describes in prose. The
guard names (`refine_same_shard`, `basis_loop_continue`,
`stream_advance`) are the only place the nesting shows up.

Three forces are pushing the implicit-nesting design past its
breaking point:

1. **Documentation drift.** The advanced-options doc describes a
   three-loop model. The machine file describes a flat ten-state
   graph. Every new knob (e.g. `inner_refine_passes`,
   `replay_ratio`) has to be threaded into both representations,
   and reviewers regularly catch the doc lagging the machine.
2. **Adaptive-regrow and multi-objective triggers** (the next two
   queued changes) both want to *insert* logic at a specific level
   of the nesting — adaptive-regrow operates inside the basis loop,
   multi-objective triggers operate at the stream-loop boundary.
   Inserting either into the flat machine means re-partitioning the
   guard space across all eight existing guards. A nested machine
   lets each change touch one sub-machine.
3. **Orca-runtime supports nesting natively.** The runtime already
   parses multi-machine files (separated by `---`) and supports
   `invoke: ChildMachine` in compound states (per the orca MCP
   server's documented syntax). Continuing to flatten by hand is
   leaving the runtime's most useful structural feature on the
   floor.

This change refactors the v0.2 flat machine into an explicit
three-machine hierarchy that mirrors the documented mental model
**without changing any runtime behavior, action signature, ctx
field semantics, or public API**. The
`test_imperative_and_fsm_byte_equivalent` net continues to be the
ground truth: it MUST pass before this change merges.

## What Changes

### Scope

Replace `saeforge/machines/sae_forge.orca.md` with three composed
machines under the same directory:

```
saeforge/machines/
  stream.orca.md      # outermost: shard handling, replay, task triggers
  refine.orca.md      # middle:    inner-refine-passes loop, eval
  basis.orca.md       # innermost: compress / regrow / project / fine-tune
```

The orchestrator (`saeforge/orchestrator.py`) loads the three files,
asks `orca_runtime_python` to compose them via its native
sub-machine invoke mechanism, and runs the resulting hierarchy. The
existing `run_machine(initial_context) -> dict` entry point is
unchanged; callers see the same in/out contract.

### New artifacts

- **`saeforge/machines/stream.orca.md`** — the outermost orca
  machine. Three states: `streaming` (compound; `invoke:
  RefineMachine`), `next_shard`, `done`. Carries the stream-loop
  ctx fields (`task_idx`, `n_tasks`, `task_trigger`,
  `tokens_seen_in_task`, `replay_*`, `advance_stream`).
- **`saeforge/machines/refine.orca.md`** — the middle machine.
  Four states: `entering` (calls `load_sae_and_corpus` and
  `scan_activations`), `refining` (compound; `invoke:
  BasisMachine`), `evaluating`, `exiting`. Carries the refine-loop
  ctx fields (`should_continue`, `iterations`, `current_iter`,
  `min_faithfulness`, `faithfulness`).
- **`saeforge/machines/basis.orca.md`** — the innermost machine.
  Six states: `compressed`, `regrown`, `projected`, `finetuned`,
  `done` (final), `failed` (final). Carries the basis-loop ctx
  fields (`inner_refine_idx`, `inner_refine_passes`,
  `regrow_count`, `protected_features`, `feature_usage`).
- **`saeforge/machines/__init__.py`** — small loader exposing
  `load_hierarchy() -> ComposedMachineDef` and the canonical event
  derivation extended for compound states. Strictly thinner than
  the user's draft `hierarchy.py`: composition is delegated to
  `orca_runtime_python`'s native multi-file parser; this file just
  reads the three resource paths and concatenates them with `---`
  separators before handing them to the runtime parser.
- **`saeforge/machines/visualize.py`** — Mermaid diagram emitter.
  Walks the parsed hierarchy and produces a single Mermaid
  `stateDiagram-v2` block with subgraphs per machine. Used by
  `docs/advanced-fsm-options.md` (rendered) and a new
  `sae-forge inspect --fsm-diagram` CLI flag (text output).
- **`tests/fsm/test_hierarchical.py`** — sub-machine isolation
  tests (each machine is independently runnable with a stubbed ctx),
  composition tests (the composed machine reaches the same final
  state as the v0.2 flat machine on identical input), and
  topology-checker tests (orca's verifier passes on each sub-machine
  and on the composed whole).

### Modified artifacts

- **`saeforge/orchestrator.py`** — `load_machine_definition()`
  becomes `load_machine_hierarchy()` returning the composed
  definition. `_derive_canonical_events` extended to handle
  compound states (a compound state's "next event" is the event
  fired when its invoked sub-machine reaches a final state).
  `run_machine(...)` signature unchanged.
- **`saeforge/actions/__init__.py`** — every action signature is
  unchanged. Each action gets a one-line docstring update naming
  which sub-machine owns it (stream / refine / basis). No behavior
  change. The `transitions_log` entries gain a `machine_path` field
  (e.g. `"stream/refine/basis"`) for debugging; consumers that
  ignore unknown keys are unaffected.
- **`saeforge/forge.py`** — `ForgePipeline._build_fsm_ctx` adds
  one new key, `_machine_path`, defaulting to `"stream"`. The
  field is set by the runtime as machines push/pop; `ForgePipeline`
  itself only initializes it. No public API change.
- **`docs/advanced-fsm-options.md`** — major rewrite around the
  hierarchy diagram. Per-machine knob tables replace the current
  flat ctx-field list. The hand-drawn ASCII art is replaced with
  the auto-generated Mermaid diagram from `visualize.py`.
- **`docs/architecture.md`** — new section "Three-machine
  hierarchy" sandwiched between the existing "Forge pipeline" and
  "Continual learning" sections.
- **`CHANGELOG.md`** — `## [Unreleased]` entry under `### Changed`:
  internal refactor of the FSM into a three-machine hierarchy; no
  behavior or API change.

### Deleted artifacts

- **`saeforge/machines/sae_forge.orca.md`** — the v0.2 flat machine
  is removed. The byte-equivalence test that loads it is updated to
  load the composed hierarchy. Since it's a package-internal
  resource (not documented as a stable surface), removal is not a
  breaking change.

### CLI surface

- **`sae-forge inspect --fsm-diagram`** — new flag; emits the
  Mermaid diagram to stdout. Inspired by the existing
  `--print-spec` flag and follows the same shape (read-only,
  no host download).

### Out of scope (deferred)

- **Parallel regions.** Orca supports orthogonal regions but the
  three forge loops are strictly nested — there's no useful
  parallelism to express today.
- **Dynamic machine swapping.** Loading a different basis machine
  per task is a research follow-up (`basis-strategy-swap`); this
  change wires the structure but does not exercise it.
- **Orca visual debugger integration.** The Mermaid emitter is
  enough for docs and CI artifacts. A live debugger UI is
  out-of-scope.
- **Migrating `forge-outer-loop-fsm` (v0.1).** The v0.1 spec
  documents a flat nine-state machine that no longer matches the
  shipped code. It will be archived/superseded by this change's
  delta plus the existing `forge-continual-learning-loop` archive
  flow — no separate migration code path.

## Capabilities

### New Capabilities

- **`hierarchical-fsm`** — the composed three-machine forge FSM.
  Defines: which states live in which machine, how compound states
  invoke sub-machines, how ctx fields are scoped per level, and how
  `transitions_log` records the machine path. The orca topology
  checker SHALL pass on each machine in isolation and on the
  composed whole.

### Modified Capabilities

- **`forge-outer-loop`** (the capability that
  `forge-continual-learning-loop` last modified; assumed archived
  before this change merges per sequencing note below) — the ten-
  state flat-topology requirements are reframed as a hierarchy
  with the same observable behavior. State names are preserved
  exactly so callers reading `transitions_log` see the same state
  identifiers. The MODIFIED requirements explicitly preserve the
  guard truth tables (`stream_advance`, `refine_same_shard`,
  `terminate_run` and the basis-loop guards), the action signatures,
  and the byte-equivalence safety net.

## Impact

- **No public API breakage.** `ForgePipeline.run()`,
  `run_machine(initial_context)`, the CLI surface, the YAML config
  schema, and the on-disk artifact layout are unchanged.
- **Zero runtime cost.** Orca compiles the hierarchy to the same
  flat reachable-state graph at parse time; the runtime traversal
  is structurally identical to the v0.2 flat machine. The
  byte-equivalence test gates this claim.
- **`transitions_log` schema additive change.** Entries gain a
  `machine_path` string field. Existing readers that index by name
  are unaffected; readers that re-serialize the log get one extra
  key.
- **Test surface.** ~15 new tests in `tests/fsm/test_hierarchical.py`
  (per-machine isolation × 3, composition × 3, topology-check × 3,
  ctx-scoping invariants × 3, Mermaid emit × 3). All existing FSM
  tests continue to pass without modification — they exercise the
  composed machine through `run_machine`, which sees the same
  state identifiers and the same final ctx.

## Sequencing

- **Depends on:** `forge-continual-learning-loop` archiving first
  (so `openspec/specs/forge-outer-loop/` exists as the base for
  this change's MODIFIED delta). If the archive ordering slips,
  rebase the delta against
  `openspec/changes/forge-continual-learning-loop/specs/forge-outer-loop/spec.md`
  instead.
- **Independent of:** `adaptive-regrow` and `multi-objective-triggers`
  (the two queued continual-learning extensions). Either can land
  before or after this refactor; landing this first is recommended
  because it gives both a much smaller surface to touch.
- **Single PR.** No staged rollout — the byte-equivalence test
  either passes or it doesn't, and a half-migrated tree has no
  meaningful intermediate state.
