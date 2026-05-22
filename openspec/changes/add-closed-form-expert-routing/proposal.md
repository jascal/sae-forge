# Closed-Form Expert Routing — diagnostic MoE without training

Add an `ExpertRouter` `nn.Module` that selects the best forge from a multi-encoding ensemble per-input, with **fixed weights computed from a one-time calibration step** (not learned via gradient descent). Routes via the dominant criterion `score_E = ||(importance_vector ⊙ SAE_encoder) · P_E · h||` — a closed-form alignment between expert E's subspace and the SAE's discriminative features. **The architecture-aware version of "best-of-ensemble selected without modifying expert projections"** that fits the closed-form forge contract.

## Why

The multi-encoding capability sweep (PRs #92-95) produces K forged modules, each with a different basis. The recommendation contract picks ONE encoding as the winner via cross-encoding tiebreaker. But for any given input, a *different* encoding might be the optimal forge:

- **partition_q4** is best for inputs whose discriminative signal lives in hierarchical-feature clusters.
- **partition_q8** is best for inputs whose signal lives in finer-grained clusters.
- **raw_slice** is best for inputs whose signal lives in the highest-row-norm features (the "easy" cases).

Today's wrapper commits to ONE encoding for ALL inputs after the sweep. Ensembling (run K forges in parallel) gives every input every encoding's output but pays K× inference cost. **Routing — pick one expert per input — sits between these extremes**: ~2× single-encoding cost, potentially close to ensemble quality.

The user's question (and constraint) from the 2026-05-22 chat: *"Is there an architecture for host model where the best of the ensemble experts is selected, without changing the underlying expert projection?"*

**Yes, and it's a small `nn.Module`.** No gradient training. No modification to forge weights. The routing function is a fixed-weight linear transformation + per-expert norm, computed from artifacts already on hand (SAE encoder weights, per-feature calibration mAUC).

## What

### `saeforge.ExpertRouter` — new `torch.nn.Module`

```python
class ExpertRouter(nn.Module):
    """Closed-form per-input expert routing across a multi-encoding
    forge ensemble.

    Weights are FIXED (computed from one-time calibration on a
    representative dataset). NOT learned via gradient descent.

    Future trainable-gating extension drops in by enabling
    `requires_grad=True` on the router_weights buffer — same
    nn.Module shape, but trained jointly with `add-progressive-
    finetune`'s gradient path.
    """

    def __init__(
        self,
        forges: list[ForgedModule],
        router_weights: torch.Tensor,        # (K, n_features_full, d_model)
        importance_vector: torch.Tensor,     # (n_features_full,) — fixed
        mode: Literal["top_1", "top_k", "weighted_soft"] = "top_1",
        top_k: int = 1,
    ): ...

    def forward(self, host_residual: torch.Tensor) -> torch.Tensor: ...
```

Three modes:

- **`top_1`** (default): per-input, pick the single best expert; run only that forge. Cheapest; one forge forward + the cheap scoring.
- **`top_k`** (k=2-3): pick top K experts; average their forge outputs. Compromise between routing and ensembling.
- **`weighted_soft`**: softmax-weighted combination of all K experts. Equivalent to ensembling with learned-style weights but computed from calibration; same cost as ensembling but with input-conditional weights.

### Pre-routing calibration

Computed once per (SAE, host, calibration_dataset) tuple. **No gradient descent.**

```python
def compute_router_calibration(
    sae_state: dict,
    forges: list[ForgedModule],
    calibration_dataset: CapabilityDataset,
) -> RouterCalibration:
    """
    For each SAE feature f, compute per_feature_importance[f] = the
    feature's host-baseline AUC minus 0.5 (how discriminative it is
    on the calibration labels — bounded in [0, 0.5] in the symmetric
    convention).

    Returns the (router_weights, importance_vector) pair ready to
    pass to ExpertRouter.__init__.
    """
```

Calibration cost: equivalent to ONE host-baseline mAUC pass on the calibration set. Trivial vs. a full sweep.

### `saeforge.ExpertRouter.from_progressive_sweep(history, ...)`

Constructor that takes a `ProgressiveHistory` from a multi-encoding sweep, the source SAE state, and a calibration dataset, and returns a fully-constructed `ExpertRouter` ready for inference.

### CLI: `sae-forge route`

```bash
sae-forge route \
    --frontier runs/.../frontier.jsonl \
    --progressive-summary runs/.../progressive_summary.json \
    --sae-checkpoints raw_slice:p1,partition_q4:p2,partition_q8:p3 \
    --calibration-config bio-pooled-calibration.yaml \
    --eval-config bio-pooled-eval.yaml \
    --routing-mode top_1 \
    --output runs/.../routing_results/
```

Stand-alone subcommand. Reads a completed multi-encoding sweep's outputs + a calibration dataset + an evaluation dataset; emits routing decisions + per-input expert assignments + ensemble-vs-routed-vs-best-single comparison table.

## Falsifiable acceptance gate

Three predictions, all measurable on bio-sae's pooled fixture at n=5000 with the three encodings from the slice 4 acceptance gate (raw_slice + partition_q4 + partition_q8):

| prediction | falsifies if |
|---|---|
| `retained_mauc(routed_top1)` ≥ `retained_mauc(best_single_encoding)` by ≥ 0.01 | routed retained_mauc within ±0.01 of best single → router didn't pick well; criterion needs revision |
| `retained_mauc(routed_top1)` is within 0.01 of `retained_mauc(ensemble_average)` | routed lags ensemble by more than 0.01 → routing isn't capturing the ensemble's benefit; the per-input optimal expert prediction is noisy |
| Inference cost(routed_top1) ≈ 2× inference cost(single encoding); strictly less than K× (= 3× here) | routed cost is K× → routing doesn't save compute over ensembling, defeating the purpose |

Three outcomes:
- **All three pass**: `add-closed-form-expert-routing` is validated; ship as production primitive.
- **Compute test passes but retained_mauc tests fail**: routing is cheap but doesn't actually select the right experts. Criterion needs revision (try alternates from design.md Decision 3). Document as "shipped but unproven" mirroring Wave C's pattern.
- **Compute test fails**: implementation bug; the scoring step is too expensive. Probably an einsum-shape issue.

## Why a `nn.Module` (vs pure Python function)

The user's question on this is exactly right. Three concrete benefits:

1. **Device portability.** `router.to('cuda')` works. A pure-Python function would need manual tensor handling at every device boundary.
2. **Composition.** `nn.Sequential(host, router, downstream_head)` glues into any pytorch pipeline. Hooks (`register_forward_hook`) work cleanly.
3. **ONNX export.** If anyone deploys a routed forge as part of a service, the entire pipeline (host → router → expert → SAE) exports as a single ONNX graph.
4. **Future learnability.** If `add-progressive-finetune` ever ships, the same `nn.Module` shape can be made trainable by setting `requires_grad=True` on the buffers and adding a gating loss. Closed-form today; learnable tomorrow — no architecture rework.

The router is in fact a tiny neural net:

- **Input**: host residual `(batch, d_model)`.
- **Per-expert scoring**: `(K, n_features_full, d_model)` weight tensor → batched einsum → `(batch, K, n_features_full)` projections.
- **Importance reweighting**: per-feature multiplication by `importance_vector`.
- **Norm reduction**: L2 norm per expert → `(batch, K)` scores.
- **Routing decision**: `argmax` (top_1) / `topk` / `softmax` (weighted_soft).
- **Expert dispatch**: index into the forge ensemble.

~80 lines of PyTorch including all three modes. Model size: K × n_features_full × d_model floats — ~4 MB for K=3, n_features=1024, d_model=320. Negligible.

## Scope (v1)

- **In:** `ExpertRouter` nn.Module + `compute_router_calibration` + `from_progressive_sweep` constructor + `sae-forge route` CLI + falsifiable gate.
- **Out:**
  - Learned gating (joint training of router_weights with experts). Gated on `add-progressive-finetune`; this openspec lays the architectural foundation by using an nn.Module shape that becomes trainable when fine-tune lands.
  - Cross-task routing (routing across DIFFERENT downstream-task SAEs). v1 is one-task at a time; multi-task is future research.
  - Per-token routing (currently routing is per-sequence/per-input — protein-level). Per-residue routing is a future extension once we know per-sequence routing works.
  - Hierarchical routing (route to a partition, then within that partition route to a specific encoding). Single-level only.

## What this is and isn't

**It IS:**
- A per-input router that picks the best-by-our-criterion expert.
- Closed-form (no training).
- ~50% cheaper than ensembling (K=3: routed is 2× single vs ensemble's 4×).
- A diagnostic MoE — multi-encoding with per-input expert selection.
- A real `nn.Module` with fixed weights; can be moved to GPU, exported to ONNX, composed in pytorch pipelines.

**It IS NOT:**
- Guaranteed to be better than the single best encoding. The routing criterion is a HEURISTIC; whether it correlates with held-out retained_mauc is the falsifiable gate.
- A learned gating function. v1 is calibration-derived only.
- An ensemble. The output is a single expert's output (or a sparse top-K combination).

## Related

- The multi-encoding ensemble this routes across: `add-multi-encoding-capability-sweep` (PRs #92-95).
- The unfit "warm-start" framing this counter-shapes again: `docs/proposals/warm-start-proposal-response.md` (PR #86).
- Future learnability hook: `add-progressive-finetune` (deferred).
- Diagnostic precursor (per-label-class breakdown answering "do encodings specialize?"): noted-only from the prior chat turn; if specialization is real, routing is well-motivated.
- The deferred ensemble openspec: `add-multi-encoding-ensemble-forge` (noted-only from prior turn; this openspec's `weighted_soft` mode is roughly that ensemble's residual-average mode).
