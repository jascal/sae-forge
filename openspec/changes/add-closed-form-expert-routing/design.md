## Context

The multi-encoding capability sweep (PRs #92-95) produces K forged modules + a winner-pick recommendation that commits to one encoding for all inputs. For inputs where a different encoding would have given higher retained_mauc, this leaves quality on the table.

Two ways out:
1. **Ensemble** — run all K forges in parallel. K× inference cost. Captures the best of every encoding but pays for all of them.
2. **Routing** — pick one expert per input. ~2× single-encoding cost (vs K× ensemble). Picks the right expert if the routing function is good.

The user's question from the 2026-05-22 chat constrains the architecture: routing **must not modify the underlying expert projection** + **must not require training** (per the recurring counter-shape pattern: closed-form forges, no gradient steps). This rules out joint MoE-style learned gating; it admits **closed-form gating derived from artifacts already on hand**.

The artifacts we have:
- K forged modules, each with a basis projection `P_E`.
- The SAE encoder weights (fixed).
- Per-feature host-baseline AUC, computed once per (SAE, host, dataset) tuple — already part of the existing capability sweep.

These three components compose into a fixed-weight routing function expressible as a small `nn.Module`. This openspec ships exactly that.

## Goals / Non-Goals

**Goals:**
- `ExpertRouter` nn.Module with three modes (top_1 / top_k / weighted_soft).
- One-time calibration step computing fixed router weights from artifacts already on hand.
- `from_progressive_sweep` constructor that wires up a router from a multi-encoding sweep's outputs.
- `sae-forge route` CLI subcommand.
- Falsifiable gate on bio-sae's pooled fixture: routed retained_mauc ≥ best single + 0.01 AND within 0.01 of ensemble.

**Non-Goals:**
- Learned gating. The router shape is forward-compatible (set requires_grad=True on the buffer and you have a trainable gate), but v1 is calibration-only.
- Per-residue routing. v1 routes per-sequence (per-protein). Residue-level routing is a future extension once sequence-level is validated.
- Cross-task routing (route across multiple SAEs). One-task per router instance.
- Hierarchical routing (route to a partition, then within partition route to encoding). Single-level only.

## Decisions

### Decision 1 — `nn.Module` shape, NOT a pure Python function

The router is implemented as a `torch.nn.Module` with fixed weights stored as `register_buffer`. Three concrete benefits:

1. **Device portability.** `.to('cuda')` Just Works. Pure Python needs manual tensor placement.
2. **Composition.** `nn.Sequential(host_extractor, router, downstream_head)` is the standard PyTorch composition pattern. Pure Python doesn't compose cleanly with hooks, gradients, or `torch.compile`.
3. **ONNX export.** Deploying a routed forge as part of an inference service exports cleanly via `torch.onnx.export`. Pure Python would need rewriting.
4. **Future learnability.** If `add-progressive-finetune` ships, the same nn.Module shape becomes trainable by setting `requires_grad=True` on the buffer + adding gating loss. The nn.Module is architecturally forward-compatible; pure Python isn't.

The "fixed weights" are stored as `register_buffer` (not `Parameter`) since they're computed from calibration, not learned. If a future change adds learnability, convert buffer → Parameter; same shape, same forward pass.

### Decision 2 — Scoring criterion: importance-weighted projected-residual norm

The score for expert E on input h is:

```
score_E = ||(importance_vector ⊙ SAE_encoder) · P_E · h||
```

Where:
- `h` is the host residual stream output (d_model dimensional).
- `P_E` is expert E's basis-projection (d_model × d_model projector onto the kept-feature subspace).
- `SAE_encoder` is the SAE's encoder matrix (n_features_full × d_model).
- `importance_vector` is per-feature host-baseline `AUC - 0.5` (positive for discriminative features, near 0 for noise).
- `⊙` is element-wise multiplication along the feature axis.

**Intuition**: weight each SAE feature's signal by how discriminative it is on the calibration labels; measure how much of that weighted signal expert E's subspace preserves. Pick the expert whose subspace best aligns with the SAE's discriminative directions for this specific input.

**Alternatives considered + rejected**:

- `||P_E · h||² / ||h||²` (subspace energy). Cheap but ignores label discrimination — favors any high-energy subspace regardless of relevance.
- `||SAE_encoder · P_E · h||²` (unweighted). Slightly better; favors high-norm-feature encodings. But over-fires on irrelevant features.
- `-||(I - P_E) · h||² weighted by importance` (minimize lost-relevant-signal). Mathematically equivalent under L2 to the chosen criterion. Just an inversion.

Configurable via `RouterScoringCriterion` enum so future revisions can A/B different criteria.

### Decision 3 — Three routing modes

```python
mode: Literal["top_1", "top_k", "weighted_soft"]
```

- **`top_1`** (default): `argmax_E score_E`. Run only the chosen expert's forge. ~2× single-encoding cost. Recommended production setting.
- **`top_k`**: `topk_E score_E` (k=2-3); run those K experts and AVERAGE their forge outputs. Compromise between routing and ensembling; K× cost for the chosen K.
- **`weighted_soft`**: `softmax(temperature · scores)` weighted average of all K experts. Cost ≡ ensemble cost; input-conditional weights instead of equal weights.

Default `top_1` matches the openspec's "best of ensemble is selected" framing. The other modes are exposed because:
- `top_k` is useful when the router can't reliably distinguish the best from the second-best (top_k=2 averages them; small-quality-loss insurance against bad routing).
- `weighted_soft` is the smoothest version and the natural drop-in for future learned gating (the softmax temperature becomes a learnable parameter).

### Decision 4 — Calibration pass dataset choice

The `compute_router_calibration` function takes a `CapabilityDataset`. Two natural choices:

1. **Same dataset as the multi-encoding sweep** — calibration data overlaps with the recommendation-selection data. Risk: routing overfits to the specific protein subset.
2. **Disjoint held-out dataset** — calibration on one slice, evaluation on another. Standard cross-validation hygiene.

**v1 recommends (2)** in documentation but doesn't enforce. Users running calibration on the same set the sweep used get a stderr warning + a `calibration_set_overlaps_with_sweep: bool` flag in the router's metadata. Forward-compatible with the `add-capability-benchmark-evaluation` openspec (noted-only) which would formalize a held-out dataset constructor.

Calibration cost: equivalent to ONE host-baseline mAUC pass on the calibration set. Trivial vs. a full sweep.

### Decision 5 — `router_weights` precomputed at construction time

The composite weight `(importance_vector ⊙ SAE_encoder) · P_E` is precomputed once per expert at construction time and cached in the buffer:

```python
self.register_buffer("router_weights", torch.stack([
    importance_vector[:, None] * SAE_encoder @ P_E
    for E in forges
], dim=0))  # shape: (K, n_features_full, d_model)
```

Forward pass becomes a single batched einsum + per-expert norm:

```python
scores = torch.einsum('kfd,...d->...kf', self.router_weights, h)
scores = (scores * self.importance_vector).norm(dim=-1)  # (..., K)
```

Trade-off: precomputation costs `K × n_features × d_model` storage (~4MB for K=3, n=1024, d_model=320 in fp32). Negligible vs. inference-time recomputation cost.

Optional `recompute_weights=True` constructor kwarg for users who'd rather pay the per-forward recomputation than the storage — useful for very large d_model where 4MB grows to 100+MB.

### Decision 6 — Falsifiable gate uses bio-sae's pooled fixture

The acceptance gate runs `routed(top_1)` vs `single_best` vs `ensemble_average` on bio-sae's pooled fixture at n=5000 with the 3 encodings from slice 4's gate. Three thresholds (from proposal.md):

- routed must beat single_best by ≥ 0.01.
- routed must be within 0.01 of ensemble.
- routed must be < K× single-encoding cost (real compute saving).

Rationale: this is the substrate where multi-encoding's PARTIAL_WIN was measured. If routing doesn't help here, it's unlikely to help on simpler substrates. Concentrated regime (bio-sae residue) where one encoding wins decisively is less interesting — routing doesn't add value if there's nothing to route between.

## Risks / Trade-offs

- **Scoring criterion may not predict held-out retained_mauc.** The importance-weighted projected-residual norm is a plausible heuristic but not theoretically guaranteed. The gate is falsifiable in this direction; we document the outcome honestly.
- **Per-expert importance vector requires labels.** Computed from calibration retained_mauc; impossible without labels. For label-free deployment (inference on new proteins without ground truth), the importance vector is "frozen" from calibration time — a reasonable production pattern but assumes calibration set's importance pattern generalizes.
- **Storage cost scales as K × n_features × d_model.** For K=10 encodings + n_features=4096 + d_model=2048, that's 320 MB. A real consideration for large SAEs. Mitigations: optional `recompute_weights=True` mode; alternative scoring criteria that don't need the full composite weight (e.g., subspace-energy norm doesn't need importance vector).
- **`weighted_soft` mode doesn't save compute over ensembling.** It just gives input-conditional weights. Worth shipping anyway since it's the natural learnable-gating drop-in.
- **No theoretical guarantee that closed-form gating beats single-best.** This is the empirical question. If the gate fails, document as "shipped but unproven" (mirroring Wave C). If it succeeds, validates the architecture for production use.
- **Cross-substrate transfer.** A router calibrated on bio-sae's pooled fixture may not route well on bio-sae's residue fixture. v1 documents this as "per-substrate calibration required"; future work could explore cross-substrate router transferability.
