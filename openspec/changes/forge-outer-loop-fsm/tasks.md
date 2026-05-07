## 1. Dependency + extras

- [x] 1.1 Add `[orca]` extra to `pyproject.toml` pinning `orca-runtime-python>=0.1.27` (PyPI distribution under the `orcalang` user; module name `orca_runtime_python`); include it in `[all]`. Wire `saeforge/machines/sae_forge.orca.md` into `[tool.hatch.build.targets.wheel.shared-data]` so it ships with the wheel
- [x] 1.2 Document the orca-runtime-python dep contract in `AGENTS.md` (parallel to the polygram and torch contracts)
- [ ] 1.3 Add a CI matrix job that installs `[dev,torch,orca]` and runs the FSM test suite — deferred to a CI-only follow-up

## 2. ForgeContext + actions module

- [x] 2.1 Use a plain `dict` as the context (orca-runtime-python's `OrcaMachine` mutates `self.context.update(result)`); document the field set in `saeforge/machines/sae_forge.orca.md` `## context` table
- [x] 2.2 Add `saeforge/actions/__init__.py` exporting `ACTION_TABLE` plus every action function
- [x] 2.3 `load_sae_and_corpus` validates `ctx["sae_checkpoint"]` exists, sets `current_sae_path`
- [x] 2.4 `compress_with_polygram` runs Polygram's `Compressor` when `ctx["validation_report_path"]` is set; pass-through otherwise. Writes the compression report at `<stem>_compression_report.json` so `FeatureBasis.from_polygram_checkpoint`'s auto-locator finds it
- [x] 2.5 `perform_regrowth` runs Polygram's `Regrower.from_compression_report` when `regrow_count > 0` AND a compression report is reachable; pass-through otherwise
- [x] 2.6 `project_to_subspace` does the real work: `FeatureBasis.from_polygram_checkpoint` → `SubspaceProjector.project_module` → `NativeModel.from_projected_weights` → `save_pretrained` to `output_dir/projected/`
- [x] 2.7 `fine_tune_model` runs N steps of LM cross-entropy training (AdamW) when `ctx["_finetune_input_ids"]` is set; pass-through otherwise. Logs the first / last step losses for monotonicity checks
- [x] 2.8 `evaluate_faithfulness` runs the existing `_kl_from_input_ids` against the carry-over host + native model; computes `perplexity = exp(kl)` and writes `should_continue` (the loop predicate, computed in Python — the idiomatic orca-lang pattern: guards stay flat boolean comparisons, complex predicates live in actions)
- [x] 2.9 `rotate_for_next_iter` increments `current_iter`, rotates `current_sae_path`, updates `best_perplexity`
- [x] 2.10 `save_final_model` writes the forged model to `output_dir/forged/` and a JSON summary to `forge_result.json`
- [x] 2.11 `log_error` populates `error_message`

## 3. Machine definition

- [x] 3.1 `saeforge/machines/sae_forge.orca.md` ships nine states, full `## context` table, four guards (`should_regrow`, `no_regrow`, `should_continue_loop` reading `ctx.should_continue == true`, `done_iterating` reading `ctx.should_continue == false`), and the canonical event/guard/action wiring from the proposal
- [x] 3.2 Per-state `error → failed` transitions on the six side-effecting states (`loaded`, `compressed`, `regrown`, `projected`, `finetuned`, `evaluated`)
- [x] 3.3 Verified by `test_machine_loads_and_has_nine_states` and `test_machine_has_required_guards` — orca-runtime-python's parser raises on dead states / undefined guards / malformed transitions, so passing parse means the topology is internally consistent

## 4. Orchestrator

- [x] 4.1 `saeforge/orchestrator.py` exposes `run_machine(initial_context)` that loads the machine via `importlib.resources.files("saeforge.machines") / "sae_forge.orca.md"`, registers every action from `ACTION_TABLE`, and drives execution under `OrcaMachine`
- [x] 4.2 Lazy-import `orca_runtime_python` inside `run_machine` and `load_machine_definition`; raise `[orca]`-extra ImportError via `saeforge.utils.lazy.require_extra` when missing
- [x] 4.3 Emit a structured transition log inside the context (`ctx["transitions_log"]`) — one entry per action with `action`, `wall_clock_ms`, and any action-specific extras
- [x] 4.4 Map action exceptions to the FSM `error` event via the `_step` wrapper; `error_message` populated from the exception class + message

## 5. ForgePipeline integration

- [x] 5.1 Add `orchestrator: str = "imperative"` field to `ForgePipeline` plus the new `iterations`, `regrow_count`, `quantum_aware` knobs
- [x] 5.2 Split `run_synthetic` into `_run_synthetic_imperative` and `_run_synthetic_fsm`; the FSM path serializes the basis to a temp safetensors so the FSM's checkpoint loader has something to read
- [x] 5.3 `_write_basis_as_checkpoint` preserves the basis dtype (float64) so the round-trip through `from_polygram_checkpoint` is byte-exact — required for the imperative/FSM byte-equivalence safety net
- [ ] 5.4 Wire `--fsm`, `--iterations`, `--regrow-count`, `--quantum-aware` to the `sae-forge forge` CLI — deferred to a follow-up cli-fsm-flags change

## 5b. Real-action tests (added after polygram shipped to PyPI)

- [x] 5b.1 `test_compress_action_runs_polygram_when_validation_report_provided`: writes uncompressed SAE → builds minimal `ValidationReport` → compress action produces compressed checkpoint with `n_features_kept < n_total`
- [x] 5b.2 `test_compress_action_passes_through_without_validation_report`: gating-by-input invariant
- [x] 5b.3 `test_fine_tune_action_reduces_loss`: 4 AdamW steps on random ids; final loss <= initial
- [x] 5b.4 `test_fine_tune_action_passes_through_without_input_ids`: gating-by-input invariant
- [x] 5b.5 `test_full_compress_then_forge_via_fsm`: end-to-end uncompressed-SAE → FSM compresses → projects → forges → eval; verifies forge_result.json carries `compress_mode == "polygram"` and the basis loader picked up the FSM-emitted compression report
- [x] 5b.6 New conftest fixture `synthetic_validation_report` builds a 2-pair-confirmed minimal `ValidationReport` JSON

## 6. Tests

- [x] 6.1 `test_machine_loads_and_has_nine_states`: state set equals the spec
- [x] 6.2 `test_machine_has_required_guards`: guard dict contains `should_regrow`, `no_regrow`, `should_continue_loop`
- [x] 6.3 `test_fsm_run_synthetic_end_to_end`: pipeline reaches `done`, writes the artifact tree, faithfulness KL non-negative
- [x] 6.4 `test_fsm_transitions_log_has_full_sequence`: action order is exactly `load → compress → project → finetune → evaluate → save_final` for the single-pass default
- [x] 6.5 `test_imperative_and_fsm_byte_equivalent`: SHA-256 of `forged/model.safetensors` equals between the two orchestrators
- [x] 6.6 `test_fsm_quantum_aware_topology_unchanged`: state set + transition set are byte-identical with `quantum_aware=True` vs `False`
- [x] 6.7 `test_fsm_orca_extra_missing_raises_actionable_import_error`: missing `[orca]` extra → `ImportError` whose message names `[orca]`

## 7. Examples + docs

- [ ] 7.1 Add `examples/forge_gpt2_toy_fsm.py` mirroring the imperative toy with `orchestrator="fsm"` — deferred (FSM path exercised by tests today)
- [x] 7.2 Update `AGENTS.md` orca-lang dep contract section: dep name is `orca-runtime-python`, not `orca-lang`
- [ ] 7.3 Update `README.md` "How it works" with the FSM mode subsection — deferred to a docs follow-up
- [ ] 7.4 Add `docs/research/forge-fsm-design.md` with a rendered state graph — deferred (orca-runtime-python doesn't ship `orca render`)

## 8. OpenSpec scaffolding

- [x] 8.1 `openspec/changes/forge-outer-loop-fsm/proposal.md` (updated for the dep-name correction)
- [x] 8.2 `openspec/changes/forge-outer-loop-fsm/design.md` (added the v0.1 implementation-notes section flagging the parser arithmetic gap and the dep-name change)
- [x] 8.3 `openspec/changes/forge-outer-loop-fsm/tasks.md` (this file)
- [x] 8.4 `openspec/changes/forge-outer-loop-fsm/specs/forge-outer-loop/spec.md`
