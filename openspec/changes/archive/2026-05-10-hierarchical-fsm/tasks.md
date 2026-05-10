## 1. Runtime probe (de-risk first)

- [ ] 1.1 Write a 30-line probe in `tests/fsm/test_runtime_compound.py` that builds a trivial two-machine fixture (`Outer { invoke: Inner }`) inline as a string, parses it via `orca_runtime_python.parse_orca_md`, runs it via `OrcaMachine`, and asserts the inner machine's `[final]` triggers the outer's transition. Goal: confirm the runtime supports compound states *before* migrating real code
- [ ] 1.2 If 1.1 fails: stop. File a bug against `orca-runtime-python`, park this change. If it passes: proceed
- [ ] 1.3 Document in `design.md` (open-questions section) which exact runtime API the orchestrator will call (`parse_orca_md` on concatenated text, vs. a hypothetical `parse_orca_files`); pick the one the runtime actually exposes today

## 2. Sub-machine definitions

- [ ] 2.1 New file `saeforge/machines/stream.orca.md` per design §"Stream machine". Five states (`init`, `streaming`, `next_shard`, `done`, `failed`). The `streaming` state declares `> invoke: RefineMachine` in its body
- [ ] 2.2 New file `saeforge/machines/refine.orca.md` per design §"Refine machine". Five states (`entering`, `refining`, `evaluating`, `exiting`, `failed`). The `refining` state declares `> invoke: BasisMachine`
- [ ] 2.3 New file `saeforge/machines/basis.orca.md` per design §"Basis machine". Six states (`compressed`, `regrown`, `projected`, `finetuned`, `done`, `failed`). All six basis-loop transitions copied verbatim from the v0.2 flat machine
- [ ] 2.4 Each `.orca.md` declares only the ctx fields it owns plus shared ctx (the orca runtime treats ctx as flat — declaring a field in any sub-machine makes it visible in all). Use comments to document scope-of-ownership
- [ ] 2.5 Run `orca verify` (via `mcp__orca__verify_machine`) on each of the three files in isolation. Each MUST verify clean. This is the topology-checker gate

## 3. Orchestrator

- [ ] 3.1 In `saeforge/orchestrator.py`, rename `load_machine_definition()` to `load_machine_hierarchy()`. Returns the composed parsed definition. Internal helper, no callers outside this file
- [ ] 3.2 The new loader reads all three `.orca.md` resources from `saeforge.machines` package data and concatenates them with a `\n---\n` separator before parsing
- [ ] 3.3 Extend `_derive_canonical_events` to handle compound states: a compound state's canonical event is the event the parent transition table fires when the child reaches `[final]` (read from the transition where the source equals the compound state name)
- [ ] 3.4 `run_machine(initial_context) -> dict` signature is unchanged. The driving loop logic is unchanged. Verify in `tests/test_orchestrator.py` that the existing tests pass without modification
- [ ] 3.5 Delete `saeforge/machines/sae_forge.orca.md` in the same commit as 3.1–3.4 (no transitional state)

## 4. ACTION_TABLE and action signatures

- [ ] 4.1 Action signatures in `saeforge/actions/__init__.py` are unchanged. Confirm by diff: only docstring updates and the new `machine_path` field in `_log` entries
- [ ] 4.2 Update `_log` to read `ctx.get("_machine_path", "stream")` and write it as `machine_path` on every transitions_log entry
- [ ] 4.3 Add a new helper `load_and_scan(ctx, payload)` that calls `load_sae_and_corpus` then `scan_activations` in sequence, merging the resulting dicts. This is the composed action invoked by RefineMachine's `entering → refining` transition (replaces today's two-state `loaded → activations_scanned` pair)
- [ ] 4.4 ACTION_TABLE registration: add `load_and_scan`. Keep `load_sae_and_corpus` and `scan_activations` registered for the topology checker (they remain referenced in the basis machine's failed-recovery flows in the unlikely future need); mark them `# composed via load_and_scan` in a comment

## 5. ForgePipeline / ctx wiring

- [ ] 5.1 In `saeforge/forge.py`, `_build_fsm_ctx` adds `_machine_path: "stream"` to the initial dict. No other ctx changes
- [ ] 5.2 The `_run_imperative` path (the byte-equivalence reference) is unchanged. It does NOT need to track `_machine_path` because its output is compared field-by-field against the FSM run in 6.1, and `_machine_path` is filtered out before comparison

## 6. Byte-equivalence acceptance gate

- [ ] 6.1 Update `test_imperative_and_fsm_byte_equivalent` to filter `machine_path` and `error_origin_machine` out of `transitions_log` entries and ctx before comparison. The action sequence (the `action` field of every entry) MUST match exactly. The final ctx scalar fields MUST match exactly. The `final_model_path` artifact MUST be byte-identical
- [ ] 6.2 If 6.1 fails: do not rebaseline. The hierarchy is wrong; fix it
- [ ] 6.3 Add a forced-error variant: inject a `RuntimeError` from `compress_with_polygram` and assert both the imperative path and the hierarchy reach `failed` with the same `error_message`. Additionally assert the FSM-only `ctx["error_origin_machine"] == "basis"` (no equivalent on the imperative side; checked on the FSM run only)
- [ ] 6.4 Add a dedicated `test_load_and_scan_ordering` (separate from the byte-equivalence test): assert that under `protect_top_k = 0` the `transitions_log` for `RefineMachine.entering` records exactly two entries — `load_sae_and_corpus` then `scan_activations` — in that order, and that `scan_activations`'s delta is empty (the v0.2 pass-through gating semantics). Then re-run with `protect_top_k = 5` and assert `scan_activations` writes `protected_features` to ctx. This pins the gating contract that the `load_and_scan` collapse preserves
- [ ] 6.5 Add a replay/shard byte-equivalence variant: a multi-shard run with `n_tasks = 2`, `replay_ratio = 0.25`, `replay_policy = "reservoir"`, and a fixed seed. Assert action sequence + final ctx scalars + artifact bytes match between imperative and FSM paths. This catches stream-loop hierarchy bugs that single-shard tests miss

## 7. Sub-machine isolation tests

- [ ] 7.1 New `tests/fsm/test_hierarchical.py`. Test that BasisMachine alone runs to completion under a stubbed ctx (provides minimal compress/regrow/project/finetune fixtures)
- [ ] 7.2 Test that RefineMachine alone runs to completion when BasisMachine is stubbed (mock the `> invoke: BasisMachine` to a single-state pass-through fixture)
- [ ] 7.3 Test that StreamMachine alone runs to completion when RefineMachine is stubbed
- [ ] 7.4 Test that the composed three-machine run reaches the same final ctx as the v0.2 flat machine on a fixed seed (this is the FSM half of 6.1 in isolation, useful when 6.1 fails to localize the bug)
- [ ] 7.5 StreamMachine multi-shard test: drive the machine through `n_tasks = 3` with `task_trigger = "labeled"`, stubbed RefineMachine (single-state pass-through that flips `ctx.advance_stream` per shard). Assert the machine reaches `done` with `task_idx == 2` and that `next_shard → streaming` re-entry fires exactly twice
- [ ] 7.6 Replay-buffer hierarchy test: with `replay_ratio = 0.25`, `replay_policy = "reservoir"`, `replay_buffer_size = 256`, drive StreamMachine for two shards and assert the buffer state ctx fields evolve identically to the v0.2 imperative path. This is the structural counterpart to 6.5's byte-equivalence check

## 8. Topology-checker tests

- [ ] 8.1 `tests/fsm/test_topology.py` — calls `mcp__orca__verify_machine` (or its Python-binding equivalent) on each sub-machine and asserts no errors. Failure messages must be visible in pytest output
- [ ] 8.2 Verify the composed hierarchy: the three concatenated machines parsed together MUST verify clean. Catches sub-machine-name typos in `invoke:` directives
- [ ] 8.3 Verify the guard truth tables: a small property test that enumerates all `(advance_stream, should_continue) ∈ {true, false}²` combinations and asserts exactly one of `stream_advance` / `refine_continue` / `terminate_run` (the new naming) fires per combination, matching the v0.2 truth table

## 9. Visualization

- [ ] 9.1 New `saeforge/machines/visualize.py` exposing `to_mermaid(hierarchy_def) -> str`
- [ ] 9.2 The emitter walks states, transitions, and `invoke:` references; produces a single `stateDiagram-v2` block with nested `state X { ... }` subgraphs per machine
- [ ] 9.3 Tests in `tests/fsm/test_visualize.py`: emitted Mermaid contains all 16 state names (5 + 5 + 6); contains all guard names from the three machines; renders without syntax errors when `mermaid-cli` is installed (graceful skip if not)
- [ ] 9.4 Embed the generated diagram in `docs/advanced-fsm-options.md`. Add a CI test `tests/fsm/test_diagram_drift.py` that calls `to_mermaid(load_machine_hierarchy())` and diffs the result against the committed Mermaid block in `docs/advanced-fsm-options.md` (extracted via a regex on the `\`\`\`mermaid` fence). The test fails on any drift — diagram is regenerated by re-running the helper and committing. This makes diagram-drift impossible to land

## 10. CLI

- [ ] 10.1 New flag `sae-forge inspect --fsm-diagram` in `saeforge/cli.py`. Mutually exclusive with `--print-spec` (argparse-level). Calls `to_mermaid(load_machine_hierarchy())` and prints to stdout
- [ ] 10.2 Tests in `tests/test_cli.py` cover the new flag's argparse contract and the smoke output (contains `stateDiagram-v2`)

## 11. Documentation

- [ ] 11.1 Major rewrite of `docs/advanced-fsm-options.md`. Sections: "Three-machine hierarchy" (top-level overview + Mermaid diagram); "Stream-machine knobs" (table of stream-scope ctx fields); "Refine-machine knobs"; "Basis-machine knobs". The flat-machine ASCII art is removed
- [ ] 11.2 New section in `docs/architecture.md` titled "FSM hierarchy" — short prose pointing readers to advanced-fsm-options.md for the diagram, plus a one-paragraph summary of the three-level mental model
- [ ] 11.3 `CHANGELOG.md` `## [Unreleased]` `### Changed` entry: "Internal: forge FSM refactored from a single ten-state flat machine into three composed sub-machines (stream / refine / basis). No public API or runtime behavior change. Adds `transitions_log[*].machine_path` for debugging"
- [ ] 11.4 `AGENTS.md` "FSM" subsection updated to point at the three .orca.md files and the visualizer; explicitly notes that all behavior changes go through the byte-equivalence test

## 12. OpenSpec scaffolding

- [x] 12.1 `openspec/changes/hierarchical-fsm/proposal.md`
- [x] 12.2 `openspec/changes/hierarchical-fsm/design.md` (this file's sibling)
- [x] 12.3 `openspec/changes/hierarchical-fsm/tasks.md` (this file)
- [x] 12.4 `openspec/changes/hierarchical-fsm/specs/hierarchical-fsm/spec.md` — ADDED requirements (new capability)
- [x] 12.5 `openspec/changes/hierarchical-fsm/specs/forge-outer-loop/spec.md` — MODIFIED requirements (reframes v0.2 ten-state flat machine as the hierarchy; preserves observable behavior)
- [ ] 12.6 Run `openspec validate hierarchical-fsm` locally; resolve any structural complaints before opening the PR

## 13. Validation matrix

- [ ] 13.1 Full `pytest` suite passes (existing + new) on Python 3.11 with the project's `[dev,intel,polygram,orca]` extras matrix
- [ ] 13.2 The byte-equivalence test (6.1) passes
- [ ] 13.3 `orca verify` passes on each sub-machine and on the composed file
- [ ] 13.4 The example scripts under `examples/` continue to run end-to-end on the user's 16GB Intel Mac configuration (gpt2-small + synthetic Llama smoke); no host-model download regression

## 14. Deferred follow-ups

- [ ] 14.1 **`adaptive-regrow`** — modifies BasisMachine only. This change makes that future work a one-machine touch instead of a flat-graph re-partition
- [ ] 14.2 **`multi-objective-triggers`** — modifies StreamMachine only. Same shape as 14.1
- [ ] 14.3 **`basis-strategy-swap`** — exercise dynamic sub-machine selection at runtime (load a different `basis_*.orca.md` per task). This change wires the structural support but does not exercise it
- [ ] 14.4 **Live orca debugger UI** — Mermaid is enough for docs; a live debugger is a separate research investment
