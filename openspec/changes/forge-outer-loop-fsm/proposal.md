## Why

The v0 milestone closes with `forge-pipeline`, which ships an imperative
`ForgePipeline.run` orchestrator: load basis → project → assemble →
optional fine-tune → eval. That linear shape is fine for a single-pass
forge, but it has three known weaknesses:

1. **No formal control-flow contract.** Adding multi-cycle compression /
   regrowth would mean nesting `if iterations > 1:` inside `run()` and
   praying the loop terminates. There is no static check that, e.g., the
   regrowth path can't be entered when `regrow_count == 0`.
2. **No verifiable orchestration artifact.** A reviewer cannot inspect
   the imperative pipeline and prove "this can never re-enter
   `compressed` from `done`" without reading every branch.
3. **Scope assumes a pre-compressed input.** sae-forge currently treats
   the compressed SAE as exogenous. To run a *full* forge — including
   the compression and regrowth that produce the basis — we would need
   to call Polygram's `Compressor` / `Regrower` from inside `run()` with
   ad-hoc loop scaffolding.

[orca-lang](https://github.com/jascal/q-orca-lang) (the **classical**
FSM language, not q-orca's quantum extension) was built precisely for
this kind of problem. A `.orca.md` machine is statically verifiable, the
state graph is renderable, and per-state error transitions make
failure-mode coverage a checked property rather than a code-review
heuristic.

This change replaces the v0 imperative orchestrator with an orca-lang
FSM that drives the full forge — compression, optional regrowth,
projection, optional fine-tune, faithfulness eval, optional
multi-cycle iteration — with the heavy numerical work delegated to the
existing Python actions (`Compressor`, `Regrower`, `SubspaceProjector`,
`NativeModel`, faithfulness eval).

## What Changes

### Scope expansion (deliberate)

sae-forge's responsibility expands from "consume a compressed SAE,
forge a native model" to "drive the full compress → regrow → project →
fine-tune → eval loop, calling Polygram and the v0 components as
actions." Polygram remains the engine for compression and regrowth;
sae-forge supplies the orchestrator. This **removes** the implicit
assumption in v0 that the input is always already compressed — the FSM
accepts a raw SAE checkpoint and runs the full pipeline.

### New artifacts

- `saeforge/machines/sae_forge.orca.md` — the canonical FSM definition.
  Nine states (`init` initial; `loaded`, `compressed`, `regrown`,
  `projected`, `finetuned`, `evaluated`; `done` and `failed` final).
  Per-state `error` transitions to `failed`. `compress_done` event
  branches on `regrow_count > 0` to enter `regrown` or skip to
  `projected`. `evaluated` branches on `should_continue_loop()` to
  loop back to `compressed` or terminate.
- `saeforge/orchestrator.py` — Python runner that loads the machine via
  `importlib.resources`, binds the action table, owns the `Context`
  dataclass, and drives execution under the orca-lang Python runtime.
- `saeforge/actions/` — module containing the bound action functions
  (`load_sae_and_corpus`, `compress_with_polygram`, `perform_regrowth`,
  `project_to_subspace`, `fine_tune_model`, `evaluate_faithfulness`,
  `should_continue_loop`, `increment_iter`, `rotate_for_next_iter`,
  `save_final_model`, `log_error`). Each is `(Context) -> Context`.
- `saeforge/context.py` — the `ForgeContext` dataclass with the v0.1
  field set (see `design.md`).

### CLI surface

- `sae-forge forge --fsm` opts into the FSM-driven path. Default stays
  on the imperative `ForgePipeline.run` shipped in v0 to preserve the
  v0 acceptance test path. `--fsm` becomes the default in v0.2 once the
  FSM has shipped behind a release flag for one milestone.
- New flags: `--iterations N` (default 1), `--regrow-count N` (default
  0), `--target-feature-ratio F` (default 0.25), `--min-faithfulness F`
  (default 0.90), `--quantum-aware` (default false; surfaces inside
  `compress_with_polygram` only — see §7).

### Optional dependency

- New `[orca]` extra pinning the classical orca-lang runtime
  (`orca-lang>=0.5`, the first PyPI release exposing the Python FSM
  driver as `orca_lang.runtime`). The default forge path does not
  require it; importing `saeforge.orchestrator` does. Match the same
  lazy-import discipline as `[torch]` and `[polygram]`.

### Verification gate

- CI gains an `orca verify saeforge/machines/sae_forge.orca.md` step
  that runs whenever the file changes. Static verification failure
  blocks the merge — that is the whole point of choosing an FSM here.

## Capabilities

### New Capabilities

- `forge-outer-loop`: An orca-lang FSM that orchestrates the full forge
  pipeline (load → compress → optional regrow → project → optional
  fine-tune → evaluate → optional iterate → done|failed) with
  per-state error transitions, statically verifiable termination, and
  a Python-bound action table whose functions remain individually
  unit-testable.

### Modified Capabilities

- `forge-pipeline` (v0 capability from change 5): `ForgePipeline` gains
  an `orchestrator: Literal["imperative", "fsm"] = "imperative"` knob.
  When `"fsm"`, `run()` constructs the initial `ForgeContext` and
  delegates to `saeforge.orchestrator.run_machine(context)`. The
  imperative path is preserved unchanged for v0.1; v0.2 deprecates it.

## Impact

- New files only under `saeforge/machines/`, `saeforge/orchestrator.py`,
  `saeforge/actions/`, `saeforge/context.py`. No edits to the v0 core
  classes (`FeatureBasis`, `SubspaceProjector`, `NativeModel`).
- `pyproject.toml` gains `[orca]` extra; the `all` extra includes it.
- `AGENTS.md` is updated to reference the v0.1 change in the milestone
  list and to declare the orca-lang dep contract alongside the existing
  Polygram and torch contracts.
- `examples/forge_gpt2_toy.py` (the v0 change-5 deliverable) gains a
  sibling `examples/forge_gpt2_toy_fsm.py` that runs the same toy with
  `--fsm`. Both must converge to byte-identical forged weights for
  identical seeds — that equivalence is the spec's strongest scenario.
- No breaking changes. The default forge path is byte-identical to v0
  unless `--fsm` is passed.
