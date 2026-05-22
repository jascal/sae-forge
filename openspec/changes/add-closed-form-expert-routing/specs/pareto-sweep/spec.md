# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `saeforge.ExpertRouter` — closed-form per-input expert router

`saeforge.ExpertRouter` SHALL be a `torch.nn.Module` subclass with
this interface:

```python
class ExpertRouter(nn.Module):
    def __init__(
        self,
        forges: list[ForgedModule],
        router_weights: torch.Tensor,        # (K, n_features_full, d_model)
        importance_vector: torch.Tensor,     # (n_features_full,)
        mode: Literal["top_1", "top_k", "weighted_soft"] = "top_1",
        top_k: int = 1,
        temperature: float = 1.0,
    ): ...

    def forward(self, host_residual: torch.Tensor) -> torch.Tensor: ...

    def routing_decisions(self, host_residual: torch.Tensor) -> torch.Tensor:
        """Diagnostic method: returns chosen expert index per input
        without running the expert. Shape: (...)."""
```

Weights are stored as `register_buffer` (NOT `Parameter`) — fixed,
computed from calibration, not learned. Future trainability is
forward-compatible by promoting buffers to Parameters; v1 ships the
calibration-only variant.

### Requirement: Three routing modes

The `mode` argument SHALL control output construction per input:

- **`top_1`** (default): `argmax_E score_E`. Run only the chosen
  expert's forge. Output is that single forge's output.
- **`top_k`**: `topk_E score_E` for k = `top_k` (default 2). Run those
  k experts and AVERAGE their forge outputs. Cost: k × single-encoding.
- **`weighted_soft`**: `softmax(temperature · scores)` weighted
  combination of ALL K experts' outputs. Cost: K × single-encoding;
  equivalent to ensemble with input-conditional weights.

The mode SHALL be selectable at `__init__` and SHALL NOT be changed
after construction (frozen behaviour; users construct a new router
to change mode).

### Requirement: Scoring criterion

For mode `top_1` / `top_k` / `weighted_soft`, the per-expert score
SHALL be computed via the default `importance_weighted_norm`
criterion:

```
score_E = ||(importance_vector ⊙ SAE_encoder) · P_E · h||₂
```

Equivalently as a single batched op:

```
projections = einsum('kfd,...d->...kf', router_weights, h)
scores = (projections ** 2).sum(dim=-1).sqrt()
```

Where `router_weights[E] = importance_vector[:, None] * SAE_encoder @ P_E`.

The criterion SHALL be selectable via a `RouterScoringCriterion`
enum. v1 ships only the default; alternatives (`subspace_energy`,
`unweighted_sae_norm`, `lost_signal_minimization`) are reserved for
future A/B comparison.

### Requirement: `compute_router_calibration` function

`saeforge.compute_router_calibration(sae_state, forges, calibration_dataset) -> RouterCalibration`
SHALL:

1. Run the SAE encoder on host activations of the calibration_dataset.
2. Compute per-feature host-baseline AUC via Mann-Whitney rank-sum
   (the same machinery `sweep_pareto_capability` uses).
3. Return `RouterCalibration(router_weights, importance_vector)`:
   - `importance_vector[f] = max(per_feature_host_auc[f] - 0.5, 0.0)`
     (clamped non-negative; features with AUC < 0.5 in the symmetric
     convention contribute zero importance).
   - `router_weights[E, f, d] = importance_vector[f] *
     SAE_encoder[f, d] * P_E[d, d]` precomputed per expert.

Calibration cost SHALL be equivalent to ONE
`sweep_pareto_capability` host-baseline pass over the calibration
set — no per-cell forge runs are needed for calibration.

The function SHALL emit a stderr warning when the calibration
dataset is the same as (or overlaps with) the dataset the
multi-encoding sweep used — recommending a held-out calibration
split for production deployment.

### Requirement: `ExpertRouter.from_progressive_sweep` constructor

`ExpertRouter.from_progressive_sweep(history, sae_state, forges, calibration_dataset, mode="top_1")`
classmethod SHALL:

1. Validate that `history.recommendation.per_encoding_recommendations`
   is populated (multi-encoding sweep, not single).
2. Validate that the supplied `forges` list aligns with the
   encodings the sweep ran (label match).
3. Call `compute_router_calibration` to produce router_weights +
   importance_vector.
4. Return a fully-constructed `ExpertRouter` ready for inference.

### Requirement: `sae-forge route` CLI subcommand

The CLI SHALL ship a new `sae-forge route` subcommand:

```bash
sae-forge route \
    --frontier PATH \
    --progressive-summary PATH \
    --sae-checkpoints LABEL:PATH,LABEL:PATH,... \
    --calibration-config PATH \
    --eval-config PATH \
    --routing-mode {top_1, top_k, weighted_soft} \
    --top-k INT \
    --output-dir PATH \
    [--host HOST_ID] [--device DEV] [--scoring-criterion ENUM]
```

Output SHALL include:

- `routing_results/per_input.jsonl` — one row per evaluation-set
  input: `(input_id, chosen_expert_label, per_expert_scores,
  routed_retained_mauc_contribution)`.
- `routing_results/summary.json` — aggregated metrics: routed
  retained_mauc, best_single_encoding retained_mauc, ensemble
  retained_mauc, routed wall time, ensemble wall time, expert
  selection distribution.

Exit code:
- `0` if routing produces a result.
- `1` if calibration/evaluation produced 0 valid inputs (data error).
- `2` if config error (missing required arg, bad YAML, etc.).

### Requirement: Falsifiable acceptance gate

The change SHALL include three slow integration tests gated on
bio-sae fixtures:

1. **`test_routing_beats_single_best_on_pooled`**: routed_top1's
   retained_mauc on bio-sae pooled fixture at n=5000 (calibration
   on first 1000 proteins, evaluation on remaining 4000) with 3
   encodings SHALL be ≥ best-single-encoding retained_mauc by ≥
   0.01.
2. **`test_routing_within_001_of_ensemble`**: routed_top1's
   retained_mauc SHALL be within 0.01 absolute of ensemble-average
   retained_mauc on the same fixture.
3. **`test_routing_cost_below_ensemble`**: routed_top1's wall-time
   SHALL be ≤ 0.6× ensemble wall-time on the same fixture
   (validates real compute savings vs ensembling).

If any gate fails, the writeup documents the failure mode (per the
openspec's three-outcome decision tree) and the change is filed as
"shipped but unproven" pending a refined scoring criterion.

### Requirement: Public API surface additions

`saeforge.__all__` SHALL gain:
- `ExpertRouter`
- `RouterScoringCriterion`
- `RouterCalibration`
- `compute_router_calibration`

`test_public_surface_is_frozen` in `tests/test_smoke.py` SHALL be
updated to include these new symbols.

### Requirement: No modification of existing surface

This openspec adds NEW top-level public surface; it SHALL NOT
modify:
- `sweep_pareto_capability` / `sweep_pareto_capability_progressive`
  signatures.
- `ParetoFrontierRow` schema.
- `ProgressiveRecommendation` schema.
- The existing host-extraction or forge-extraction cache contracts.

The router is a downstream consumer of the existing surface; it
doesn't change anything those existing surfaces produce.
