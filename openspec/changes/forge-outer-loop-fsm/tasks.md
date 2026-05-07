## 1. Dependency + extras

- [ ] 1.1 Add `[orca]` extra to `pyproject.toml` pinning `orca-lang>=0.5`; include it in `[all]`
- [ ] 1.2 Document the orca-lang dep contract in `AGENTS.md` (parallel to the polygram and torch contracts) — required for the FSM path, not for the imperative path
- [ ] 1.3 Add a CI matrix job that installs `[dev,orca]` and runs the orca-lang verifier against `saeforge/machines/sae_forge.orca.md`

## 2. ForgeContext + actions module

- [ ] 2.1 Add `saeforge/context.py` with the `ForgeContext` dataclass per `design.md` §"Context dataclass"
- [ ] 2.2 Add `saeforge/actions/__init__.py` re-exporting every action and the `should_continue_loop` guard
- [ ] 2.3 Add `saeforge/actions/load.py` implementing `load_sae_and_corpus(ctx)` — validates `ctx.sae_checkpoint` and `ctx.corpus_path` exist, sets `ctx.current_sae_path = ctx.sae_checkpoint`
- [ ] 2.4 Add `saeforge/actions/compress.py` implementing `compress_with_polygram(ctx)` — calls Polygram's `Compressor` with the chosen strategy; when `ctx.quantum_aware`, uses `confirmer="quantum_interference"` instead of `decoder_geometry`; writes `ctx.compressed_sae_path` and `ctx.current_feature_count`
- [ ] 2.5 Add `saeforge/actions/regrow.py` implementing `perform_regrowth(ctx)` — calls Polygram's `Regrower` against `ctx.compressed_sae_path`; writes `ctx.regrown_sae_path`
- [ ] 2.6 Add `saeforge/actions/project.py` implementing `project_to_subspace(ctx)` — loads the latest `ForgeBasis` from `ctx.regrown_sae_path or ctx.compressed_sae_path` and runs `SubspaceProjector.project_module(host_model)`; writes `ctx.projected_weights_path`
- [ ] 2.7 Add `saeforge/actions/finetune.py` implementing `fine_tune_model(ctx)` — torch fine-tune over `ctx.corpus_path`; writes `ctx.finetuned_model_path`. No-op shortcut when `ctx.iterations == 1` and a future `--no-finetune` flag is set (not in v0.1)
- [ ] 2.8 Add `saeforge/actions/evaluate.py` implementing `evaluate_faithfulness(ctx)` — runs the v0 `faithfulness_kl` against the host on a held-out slice; writes `ctx.faithfulness` and `ctx.perplexity`
- [ ] 2.9 Add `saeforge/actions/loop.py` implementing `should_continue_loop(ctx) -> bool`, `increment_iter(ctx)`, and `rotate_for_next_iter(ctx)` per `design.md` §"Guards"
- [ ] 2.10 Add `saeforge/actions/finalize.py` implementing `save_final_model(ctx)` (writes `ctx.final_model_path` under `ctx.output_dir`) and `log_error(ctx)` (writes structured error JSON next to the final model path)

## 3. Machine definition

- [ ] 3.1 Add `saeforge/machines/sae_forge.orca.md` with the nine states, the transition table from the proposal §4, and the action / guard tables that map names to `saeforge.actions.*`
- [ ] 3.2 Add per-state `error → failed` transitions on every state that performs side-effecting work (`loaded`, `compressed`, `regrown`, `projected`, `finetuned`, `evaluated`); `init` and the final states do not declare error transitions
- [ ] 3.3 Verify `orca verify saeforge/machines/sae_forge.orca.md` passes locally; fix any reported dead states or unreachable transitions before checking in

## 4. Orchestrator

- [ ] 4.1 Add `saeforge/orchestrator.py` exposing `run_machine(ctx: ForgeContext) -> ForgeResult`. Loads the machine via `importlib.resources.files("saeforge.machines") / "sae_forge.orca.md"`, binds the action and guard tables, and drives execution under `orca_lang.runtime`
- [ ] 4.2 Lazy-import `orca_lang.runtime` inside `run_machine`; raise the `[orca]`-extra ImportError from `saeforge.utils.lazy.require_extra` when missing
- [ ] 4.3 Emit a structured transition log (`ctx.output_dir / "transitions.jsonl"`) — one JSON line per transition with `from_state`, `to_state`, `event`, `guard`, `wall_clock_ms`, and `iter`
- [ ] 4.4 Map orca-lang exceptions to `ForgeResult.failed=True` with `error_message` populated from `ctx.error_message`

## 5. ForgePipeline integration

- [ ] 5.1 Add `orchestrator: Literal["imperative", "fsm"] = "imperative"` field to `ForgePipeline`
- [ ] 5.2 In `ForgePipeline.run`, when `orchestrator == "fsm"`, build the initial `ForgeContext` from the pipeline's fields and delegate to `saeforge.orchestrator.run_machine`; preserve the imperative path unchanged otherwise
- [ ] 5.3 Add `--fsm` CLI flag to `sae-forge forge`; wire `--iterations`, `--regrow-count`, `--target-feature-ratio`, `--min-faithfulness`, `--quantum-aware` through to the `ForgeContext`
- [ ] 5.4 Document the v0.1 → v0.2 migration: imperative is the default in v0.1, becomes deprecated in v0.2, removed in v1.0

## 6. Tests

- [ ] 6.1 Add `tests/test_forge_context.py` covering: dataclass defaults, field rotation by `rotate_for_next_iter`, `increment_iter` updates `current_iter` and `best_perplexity` correctly
- [ ] 6.2 Add `tests/test_forge_actions.py` with one unit test per action; use synthetic in-memory contexts and mock Polygram / torch entry points so the suite runs without `[polygram]` or `[torch]` installed
- [ ] 6.3 Add `tests/test_forge_orchestrator.py` covering: orca-lang `[orca]` extra missing → clear ImportError, single-pass forge with `iterations=1, regrow_count=0` reaches `done` in the expected state sequence, multi-pass with `iterations=3` reaches `done` after exactly three loop traversals when faithfulness improves each cycle, regression in faithfulness terminates early at the right state, an action raising in `compressed` reaches `failed` with `error_message` populated
- [ ] 6.4 Add `tests/test_forge_machine_static.py` that runs the orca-lang verifier programmatically (`orca_lang.verify_file`) against the shipped machine; this guards against regressions in the `.orca.md` file
- [ ] 6.5 Add `tests/test_fsm_imperative_equivalence.py` — runs the toy GPT-2 example under both `orchestrator="imperative"` and `orchestrator="fsm"` with the same seed and asserts byte-identical forged weights

## 7. Examples + docs

- [ ] 7.1 Add `examples/forge_gpt2_toy_fsm.py` mirroring `examples/forge_gpt2_toy.py` with `--fsm`
- [ ] 7.2 Update `README.md` "How it works" with a one-paragraph note on the FSM path; add a "FSM mode" subsection under "CLI" with the new flags
- [ ] 7.3 Update `AGENTS.md` to add the v0.1 milestone (`forge-outer-loop-fsm`) below the v0 list
- [ ] 7.4 Add `docs/research/forge-fsm-design.md` exporting the rendered state graph (graphviz DOT output of `orca render`)

## 8. OpenSpec scaffolding

- [x] 8.1 `openspec/changes/forge-outer-loop-fsm/proposal.md`
- [x] 8.2 `openspec/changes/forge-outer-loop-fsm/design.md`
- [x] 8.3 `openspec/changes/forge-outer-loop-fsm/tasks.md` (this file)
- [x] 8.4 `openspec/changes/forge-outer-loop-fsm/specs/forge-outer-loop/spec.md`
- [ ] 8.5 Run `openspec validate forge-outer-loop-fsm` and fix any reported issues
