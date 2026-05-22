# Implementation tasks

## 0. Pre-locks (blocking)

- [ ] 0.1 `add-multi-encoding-capability-sweep` (PRs #92-95) MUST be fully shipped (slice 5/N docs + v0.10.0 released) before this lands. This openspec layers on top.
- [ ] 0.2 The slice-4 acceptance gate's actual numbers (when in-flight run completes) inform whether the `top_1` mode's prediction is even plausible. If the 3 encodings have nearly-identical per-encoding recommendations on the pooled fixture, routing has nothing to route between; defer.

## 1. `saeforge/expert_routing.py` — new module

- [ ] 1.1 `ExpertRouter(nn.Module)` class with:
  - `__init__(forges, router_weights, importance_vector, mode, top_k)`.
  - `register_buffer` for `router_weights` and `importance_vector` (not Parameter; not learned in v1).
  - `forward(host_residual: Tensor)` dispatching to one of the three modes.
  - `_score(host_residual)` helper computing per-expert scores via einsum.
  - `_dispatch_top1(scores, host_residual)` / `_dispatch_topk(...)` / `_dispatch_weighted_soft(...)` per-mode helpers.
- [ ] 1.2 `RouterScoringCriterion` enum (`importance_weighted_norm` default + alternates: `subspace_energy`, `unweighted_sae_norm`, `lost_signal_minimization`). v1 ships only `importance_weighted_norm`; others scaffolded for future A/B comparison.
- [ ] 1.3 `compute_router_calibration(sae_state, forges, calibration_dataset) -> RouterCalibration` function. Returns `(router_weights, importance_vector)` pair ready to pass to ExpertRouter.
- [ ] 1.4 `ExpertRouter.from_progressive_sweep(history, sae_state, forges, calibration_dataset, mode='top_1')` classmethod constructor. Reads a `ProgressiveHistory` from a multi-encoding sweep; assembles the forge list + computes calibration + constructs the router.
- [ ] 1.5 `router.routing_decisions(host_residuals)` -> `dict[int, list[str]]` introspection method. Returns the per-input expert chosen, for diagnostic analysis. Doesn't run the chosen forge — just reports the routing.

## 2. Public surface + back-compat

- [ ] 2.1 Export `ExpertRouter`, `RouterScoringCriterion`, `compute_router_calibration` from `saeforge.__init__`.
- [ ] 2.2 Add to `saeforge.__all__`.
- [ ] 2.3 `test_public_surface_is_frozen` (in `tests/test_smoke.py`) updated with the new symbols.
- [ ] 2.4 ParetoFrontierRow, ProgressiveRecommendation, etc. unchanged — this openspec adds new top-level surface without modifying existing dataclasses.

## 3. Unit tests

- [ ] 3.1 `tests/test_expert_routing.py`:
  - `test_router_init_validates_shapes`: bad router_weights shape → ValueError.
  - `test_router_top1_picks_argmax`: synthesised forges + known router_weights → assert routed output equals the expected expert's output for each input.
  - `test_router_top_k_averages`: top_k=2 mode averages the two highest-scoring experts' outputs.
  - `test_router_weighted_soft_with_temperature`: weighted_soft mode with temperature → output is convex combination weighted by softmax(scores).
  - `test_calibration_computes_importance_vector`: known calibration set with synthetic labels → importance_vector matches expected (per-feature AUC - 0.5).
  - `test_router_round_trips_through_state_dict`: state_dict() / load_state_dict() preserves all buffers.
  - `test_router_moves_to_device`: `router.to('cuda')` if CUDA available; otherwise mark skip.

## 4. CLI surface

- [ ] 4.1 New `sae-forge route` subcommand. Flags:
  - `--frontier PATH` (required).
  - `--progressive-summary PATH` (required).
  - `--sae-checkpoints LABEL:PATH,LABEL:PATH,...` (required; matches the encodings the sweep ran).
  - `--calibration-config PATH` (required; YAML matching the existing CapabilityDataset.from_bio_sae schema).
  - `--eval-config PATH` (required; YAML; held-out evaluation set).
  - `--routing-mode {top_1, top_k, weighted_soft}` (default top_1).
  - `--top-k INT` (default 1; relevant when mode=top_k).
  - `--scoring-criterion {importance_weighted_norm, subspace_energy, unweighted_sae_norm, lost_signal_minimization}` (default importance_weighted_norm).
  - `--output-dir PATH` (required).
  - `--host HOST_ID`.
  - `--device DEV`.
- [ ] 4.2 `_cmd_route(args)` dispatch function:
  - Loads progressive_summary.json.
  - Validates that the supplied `--sae-checkpoints` match the encodings the sweep ran (warns on mismatch).
  - Constructs ExpertRouter via `from_progressive_sweep` + calibration set.
  - Runs the router on the evaluation set.
  - Emits `routing_results/per_input.jsonl` (one row per input: chosen expert + scores + routed retained_mauc estimate) + `routing_results/summary.json`.
- [ ] 4.3 Output summary includes side-by-side: routed retained_mauc / best-single-encoding retained_mauc / ensemble retained_mauc / routing cost vs ensemble cost.

## 5. Falsifiable acceptance gate

- [ ] 5.1 `tests/test_expert_routing_gate.py::test_routing_beats_single_best_on_pooled` (slow):
  - Bio-sae pooled fixture at n=5000 + 3 encodings (raw_slice + partition_q4 + partition_q8).
  - Calibration on first 1000 proteins; evaluation on remaining 4000.
  - Asserts `routed_top1_retained_mauc >= best_single_encoding_retained_mauc + 0.01`.
- [ ] 5.2 `tests/test_expert_routing_gate.py::test_routing_within_001_of_ensemble` (slow):
  - Same fixture; asserts `|routed_top1_retained_mauc - ensemble_average_retained_mauc| <= 0.01`.
- [ ] 5.3 `tests/test_expert_routing_gate.py::test_routing_cost_below_ensemble` (slow):
  - Same fixture; benchmarks routed wall time vs ensemble wall time; asserts `routed_wall_time <= 0.6 * ensemble_wall_time`.

All three slow tests must pass for this openspec to ship; if any falsifies, the writeup documents the failure mode per the openspec's three-outcome decision tree.

## 6. Documentation

- [ ] 6.1 README: new "Closed-form expert routing" subsection under the existing "Multi-encoding capability sweep" section. End-to-end CLI example + production-recommended mode + cost tradeoff table.
- [ ] 6.2 `docs/algorithm.md` §5: cross-reference to routing for users who need per-input expert selection.
- [ ] 6.3 CHANGELOG entry under `[Unreleased]`.

## 7. Bio-sae-side adoption (post-merge)

- [ ] 7.1 `bio-sae/scripts/route_pooled_ensemble.py`: end-to-end runner that wires up the bio-sae pooled fixture + the partition shadows + a calibration split + an evaluation split → routed retained_mauc.
- [ ] 7.2 Writeup under `bio-sae/docs/forge-capability-bottleneck.md` §5.7 (or new §6): the routed vs single-best vs ensemble comparison table. If routed > single_best AND ≈ ensemble: validates that per-input expert selection works on this substrate; if not: documents the negative result per the openspec's failure-mode framing.

## 8. Release

- [ ] 8.1 Version bump to v0.11.0 (new public surface: ExpertRouter + compute_router_calibration + sae-forge route). Minor bump because it's additive but introduces a meaningful new primitive.
- [ ] 8.2 Tag v0.11.0.
- [ ] 8.3 Bio-sae bumps pin.
