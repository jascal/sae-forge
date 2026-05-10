## 1. RegrowController class

- [ ] 1.1 New class `RegrowController` in `saeforge/basis.py` (NOT a new package — the existing single-file layout is the convention). Pure-Python, no torch import, no IO
- [ ] 1.2 Method `next_count(n_features_kept, n_features_target, regrow_count, regrow_max, regrow_damping) -> int` per design.md "controller equation"
- [ ] 1.3 Bounds invariant: returned value is always `regrow_count <= v <= regrow_max`. Asserted in unit test
- [ ] 1.4 Tests in `tests/fsm/test_adaptive_regrow.py::TestController`: deterministic-given-inputs, monotone-in-gap, bounds-respecting, damping-factor-effect, `gap=0` returns `regrow_count` (cold-start equivalent)

## 2. ForgePipeline knobs + ctx wiring

- [ ] 2.1 Add four new fields to `ForgePipeline` (`saeforge/forge.py`): `adaptive_regrow: bool = False`, `regrow_max: int = 0`, `n_features_target: int = 0`, `regrow_damping: float = 0.5`
- [ ] 2.2 `__post_init__` validation: when `adaptive_regrow=True`, require `regrow_max > regrow_count` AND `n_features_target > 0`. Clear `ValueError` naming both fields. When `adaptive_regrow=False`, the other three knobs MAY be set but are ignored (no validation)
- [ ] 2.3 `_build_fsm_ctx` writes the four fields to ctx. Existing `regrow_count` continues to be written verbatim — NOT renamed
- [ ] 2.4 Tests in `tests/test_forge_pipeline.py`: validation matrix (adaptive=True without regrow_max → raises; adaptive=False with regrow_max → silent; etc.)

## 3. adapt_and_regrow composed action

- [ ] 3.1 New helper `_compute_effective_regrow_count(ctx)` in `saeforge/actions/__init__.py`. Reads `current_feature_count`, `n_features_target`, `regrow_count`, `regrow_max`, `regrow_damping` from ctx. Calls `RegrowController.next_count(...)`. Writes `effective_regrow_count` to ctx and logs an `adapt_regrow_count` entry with `{value, gap, target}` extras
- [ ] 3.2 New action `adapt_and_regrow(ctx, payload)` in `saeforge/actions/__init__.py`. First line: `if not ctx.get("adaptive_regrow"): return perform_regrowth(ctx, payload)` — guarantees byte-equivalence under the disabled toggle. When enabled: call `_compute_effective_regrow_count`, then call `perform_regrowth`, merging deltas
- [ ] 3.3 Modify `perform_regrowth` to read `ctx.get("effective_regrow_count")` first, falling back to `ctx["regrow_count"]` when unset. Existing `regrow_count == 0` short-circuit unchanged
- [ ] 3.4 Add `adapt_and_regrow` to `ACTION_TABLE`. Keep `perform_regrowth` registered (still called by the composed helper; still the public action name in docs)
- [ ] 3.5 Tests in `tests/fsm/test_adaptive_regrow.py::TestComposedAction`: disabled-toggle path is byte-identical to direct `perform_regrowth` call; enabled path computes `effective_regrow_count` correctly; both inner action names log to `transitions_log` in order

## 4. BasisMachine transition rename

- [ ] 4.1 In `saeforge/machines/basis.orca.md`, change exactly one transition row: `compressed → regrown (compress_done, should_regrow, perform_regrowth)` → `... adapt_and_regrow`. No state change, no guard change, no new edges
- [ ] 4.2 Add `adapt_and_regrow` to the `## actions` table in `basis.orca.md`. Keep `perform_regrowth` listed (still referenced by tests + docs)
- [ ] 4.3 Run `tests/fsm/test_topology.py` — assert it still passes (state set unchanged, canonical events unchanged, guard truth table unchanged)

## 5. Mermaid diagram regen

- [ ] 5.1 Run `sae-forge inspect --fsm-diagram` after task 4. Confirm the only change in the output vs the committed diagram is the label on the affected edge
- [ ] 5.2 Paste the regenerated diagram into `docs/advanced-fsm-options.md` between the `<!-- BEGIN AUTO-GENERATED FSM DIAGRAM -->` / `END` markers
- [ ] 5.3 `tests/fsm/test_diagram_drift.py` — passes after 5.2; fails before 5.2 (deliberately — the drift CI is the gate that catches missed regenerations)

## 6. CLI

- [ ] 6.1 Add four new flags to `forge` subparser in `saeforge/cli.py`: `--adaptive-regrow` (boolean), `--regrow-max INT`, `--n-features-target INT`, `--regrow-damping FLOAT` (default 0.5)
- [ ] 6.2 Mutually-required group: `--adaptive-regrow` requires both `--regrow-max` and `--n-features-target`. Argparse-level error if missing
- [ ] 6.3 `_cmd_forge` threads the new flags into `ForgePipeline(...)` constructor
- [ ] 6.4 Tests in `tests/test_cli.py`: parser accepts the new flags; mutually-required validation rejects partial use

## 7. Byte-equivalence acceptance gate

- [ ] 7.1 The existing `test_imperative_and_fsm_byte_equivalent` MUST continue to pass with `adaptive_regrow=False` (the v0.2 default). This is the load-bearing gate
- [ ] 7.2 Add a new test `test_byte_equivalent_when_adaptive_regrow_disabled`: explicitly construct a pipeline with `adaptive_regrow=False, regrow_max=64, n_features_target=128` (the latter two should be ignored when adaptation is off) and assert the forged weights are byte-identical to a pipeline with the v0.2 minimal config (`adaptive_regrow=False, regrow_max=0, n_features_target=0`)
- [ ] 7.3 Add a determinism test `test_two_runs_same_seed_byte_identical_under_adaptive_regrow`: two runs with `adaptive_regrow=True` and identical seed/config produce byte-identical forged weights. Pins the v1 determinism guarantee

## 8. Integration test (synthetic growth scenario)

- [ ] 8.1 New test `test_adaptive_regrow_grows_smoothly_toward_target` in `tests/fsm/test_adaptive_regrow.py`. Concrete scenario: start with `current_feature_count = 100`, `n_features_target = 300`, `regrow_count = 5`, `regrow_max = 64`, `regrow_damping = 0.5`, `inner_refine_passes = 6`. Drive `BasisMachine` through 6 cycles with synthetic compression results that always preserve the basis size (n_features_kept = previous count + previous regrow). Assert: (a) `effective_regrow_count` per cycle is in `[5, 64]`; (b) the sequence is monotone non-increasing as the gap closes (controller damps as it approaches target, no overshoot); (c) `current_feature_count` after cycle 6 is in `[260, 300]` (close to target but not exceeding it)
- [ ] 8.2 New test `test_adaptive_regrow_respects_regrow_max`: synthetic scenario with `gap >> regrow_max`. Assert the controller never exceeds `regrow_max`
- [ ] 8.3 New test `test_adaptive_regrow_falls_back_to_regrow_count_when_target_reached`: synthetic scenario where `n_features_kept >= n_features_target` from the first cycle. Assert `effective_regrow_count == regrow_count` for every cycle

## 9. Documentation

- [ ] 9.1 New subsection "Adaptive regrow" under `### Basis loop (inner)` in `docs/advanced-fsm-options.md`. Required content:
  - **When to use it** (back-pressure on per-domain `regrow_count` tuning across multi-shard runs)
  - **The controller equation** (verbatim from `design.md`, including the cold-start fallback)
  - **Signal source** (`n_features_kept` from polygram `CompressionReport`; why not loss-based in v1)
  - **Tuning guidelines** — concrete table of common knob combinations and their growth profiles. At minimum: "start conservative" defaults (damping 0.5, regrow_max ≈ 0.2×target), "hit target faster" (damping → 1.0), "hit target slower / smoother" (damping → 0.25). Tie each to the synthetic test scenario in §8.1 so users can run a tiny example to see the curve
  - **The `protect_top_k` interaction caveat** (growing basis with fixed protected count shrinks protected fraction; reference the `protect-top-k-ratio` follow-up)
- [ ] 9.2 The auto-regenerated Mermaid block from task 5.2 lands in this commit. The diagram-drift CI gates the doc update
- [ ] 9.3 `CHANGELOG.md` `## [Unreleased]` `### Added` entry: "Adaptive regrow controller in BasisMachine. Opt-in via `--adaptive-regrow`. Defaults preserve byte-equivalence with v0.2 fixed-regrow path"
- [ ] 9.4 New section "Adaptive regrow" in `AGENTS.md` under the FSM heading, pointing readers at `docs/advanced-fsm-options.md` and the controller code in `saeforge/basis.py`
- [ ] 9.5 `docs/advanced-fsm-options.md`: explicit non-normative note about the `protect_top_k` interaction (growing basis with fixed protected count shrinks protected fraction). Reference the `protect-top-k-ratio` follow-up

## 10. OpenSpec scaffolding

- [x] 10.1 `openspec/changes/adaptive-regrow/proposal.md`
- [x] 10.2 `openspec/changes/adaptive-regrow/design.md`
- [x] 10.3 `openspec/changes/adaptive-regrow/tasks.md` (this file)
- [x] 10.4 `openspec/changes/adaptive-regrow/specs/adaptive-regrow/spec.md` (ADDED capability)
- [x] 10.5 `openspec/changes/adaptive-regrow/specs/forge-outer-loop/spec.md` (MODIFIED — one transition action rename)
- [ ] 10.6 Run `openspec validate adaptive-regrow --strict`; resolve any structural complaints before opening the PR

## 11. Validation matrix

- [ ] 11.1 Full `pytest` suite passes (existing + new) on Python 3.11 + 3.12 with `[dev,intel,polygram,orca]` extras
- [ ] 11.2 The byte-equivalence gate (7.1) passes
- [ ] 11.3 The determinism gate (7.3) passes
- [ ] 11.4 `orca verify` passes on `basis.orca.md` after the transition rename
- [ ] 11.5 `tests/fsm/test_diagram_drift.py` passes (i.e., the regenerated Mermaid block matches the live emit)
- [ ] 11.6 CLI flag combinations exercised in `tests/test_cli.py`

## 12. Deferred follow-ups

- [ ] 12.1 **`adaptive-regrow-loss`** — extend the controller to read loss-based signals (`recent_eval_losses` delta). Requires fine-tune RNG pinning to preserve determinism
- [ ] 12.2 **`protect-top-k-ratio`** — make `protect_top_k` an optional ratio (`protect_top_k_ratio`) of the current basis size, so growing the basis grows the protected set proportionally
- [ ] 12.3 **Cross-shard regrow scheduling** — per-shard `regrow_count_schedule` (e.g. "grow more aggressively on shards 2+")
- [ ] 12.4 **Visualization / dashboard hooks** — live plot of `effective_regrow_count` and `current_feature_count` per cycle
