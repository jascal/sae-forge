# Progressive Capability Sweep — smallest-n robust to data scale

Add a multi-stage capability-aware Pareto sweep that progressively grows protein count + narrows the active width set until the recommended optimum **stops shifting** as data is added. The recommendation contract becomes "smallest `target_n_features_kept` whose retained_mauc is stable across the last K stages of data scaling", not "argmax retained_mauc on a single sweep". The latter — what `sweep_pareto_capability` returns today — overfits to whatever subset of proteins happened to be in the eval sample.

## Why

Bio-sae's 2026-05-22 acceptance-gate runs surfaced the motivating signal directly:

| residue-feed sweep on `runs/uniref50_small/residue` | peak n | retained_mauc | retained_mauc at n=16 |
|---|---|---|---|
| n_proteins = 10 (writeup §3.1)  | 16  | 1.032 | 1.032 |
| n_proteins = 100 (this work)   | 48  | 1.045 | 1.033 |

Two things happened as protein count grew 10×:

1. **Peak shifted from n=16 to n=48.** The AUC estimator's variance tightened; widths that looked "tied for first" at 10 proteins separated cleanly at 100. Some marginal feature that didn't quite clear the 0.95 bar at low data crossed it once more data made the bar a sharper boundary.
2. **n=16's score stayed identical** (1.032 → 1.033). The denoising effect at small bases is robust to data scale; what's not robust is *which exact small basis* is the argmax.

A user running `sweep_pareto_capability` once on n=10 proteins would correctly identify the concentrated regime but would commit to n=16. A user running once on n=100 would commit to n=48. **Neither is wrong**, but neither is *stable* — the right answer is "any n in the plateau is fine, pick the smallest one that hasn't shifted out of the plateau under the data-scale stress test." Today's sweep wrapper has no way to express that.

The further-up-the-stack signal: bio-sae's spread-regime measurement (`runs/uniref50_n5000/pooled_w1024_k64`) is an inverse demonstration. Peak retained_mauc moved 0.932 (500 proteins, writeup §3.2) → 0.928 (1000 proteins, this work) at n=512 both times. **The peak position is data-scale-stable; the peak value is data-scale-stable; recommendation is therefore robust.** No need for progressive expansion in that regime — single sweep returns the right answer.

A progressive scheme self-decides whether more stages help. Concentrated regimes get more refinement; spread regimes converge in one stage.

## What

### 1. `sweep_pareto_capability_progressive(...)` — new wrapper

```python
from saeforge import sweep_pareto_capability_progressive

history = sweep_pareto_capability_progressive(
    sae_checkpoint=sae_path,
    host_model_id="facebook/esm2_t6_8M_UR50D",
    dataset=full_dataset,             # CapabilityDataset of ALL available data
    candidate_widths=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
    n_proteins_schedule=[10, 50, 200, 1000],  # fidelity axis
    retained_mauc_tolerance=0.005,    # peak-shift sensitivity
    plateau_tolerance=0.01,           # widths within X of peak are "plateau"
    min_plateau_widths=3,             # never prune below this many actives
    convergence_n_stages=2,           # peak stable across N stages → stop
    output_dir=Path("runs/progressive/"),
)

# history is a list[ProgressiveStageResult]; last stage's recommendation is
# the smallest target_n_features_kept that survived the final plateau.
print(history.recommendation)         # ProgressiveRecommendation
```

Each stage:

1. **Subsample** `dataset_full` to `n_proteins_schedule[stage]` proteins (cumulative; stage K+1 is a superset of stage K's subsample so the host-extraction cache is reusable).
2. **Sweep** `sweep_pareto_capability` over the currently active width set.
3. **Identify the plateau**: widths whose retained_mauc is within `plateau_tolerance` of the stage's peak. Always at least `min_plateau_widths` actives — flat plateaus shouldn't prune too aggressively.
4. **Expand neighbors**: add immediate-neighbor candidate widths around each plateau survivor (so we can refine resolution if the original grid was coarse). Neighbors come from `candidate_widths` — the sweep doesn't invent widths the user didn't request.
5. **Check convergence**: if the smallest plateau width and its retained_mauc haven't shifted across `convergence_n_stages` consecutive stages (within respective tolerances), stop. Otherwise advance.

### 2. `ProgressiveStageResult` + `ProgressiveRecommendation`

```python
@dataclass(frozen=True)
class ProgressiveStageResult:
    stage: int
    n_proteins: int
    active_widths: tuple[int, ...]
    rows: list[ParetoFrontierRow]      # the per-cell results from this stage
    plateau_widths: tuple[int, ...]    # widths within plateau_tolerance of peak
    peak_n: int                        # argmax retained_mauc on this stage
    peak_retained_mauc: float

@dataclass(frozen=True)
class ProgressiveRecommendation:
    target_n_features_kept: int        # the smallest stable-plateau width
    retained_mauc_vs_host: float
    stages_converged: int              # how many consecutive stages this width was a plateau-member
    converged: bool                    # convergence_n_stages reached?
    rationale: str                     # human-readable explanation
```

### 3. `sae-forge sweep-capability-progressive` CLI

```bash
sae-forge sweep-capability-progressive \
    --dataset-config bio-residue.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --candidate-widths 4,8,16,32,64,128,256,512,1024 \
    --schedule 10,50,200,1000 \
    --output-dir runs/progressive_residue/
```

Output: `frontier.jsonl` (cumulative across stages, with a new `stage` field on each row) + `progressive_summary.json` with the recommendation + convergence narrative.

### 4. `ParetoFrontierRow.stage`

One new optional field on `ParetoFrontierRow` so a progressive frontier's rows can be partitioned by stage. Default `None` — non-progressive rows carry no stage tag.

## Falsifiable acceptance gate

Two predictions, both reproducible against bio-sae's existing fixtures:

| fixture | predicted recommendation | falsifies if … |
|---|---|---|
| `runs/uniref50_small/residue` (concentrated, residue feed) | smallest-stable-plateau width ∈ [12, 64]; converges in ≤ 3 stages | recommendation is n=4 (collapsed) or n=256+ (no convergence) |
| `runs/uniref50_n5000/pooled_w1024_k64` (spread, pooled feed) | width = 512 ± 1 plateau bucket; converges in **1 stage** (single-shot sweep already stable) | takes > 2 stages to converge OR recommendation shifts by > 1 plateau bucket between stages |

Total compute SHALL be ≤ the equivalent full single-shot sweep at the largest n_proteins. Concentrated regime SHOULD be ~2-3× cheaper: stage 0's small-protein sweep prunes most widths cheaply.

## Why this is the right contract

The user's framing — "smallest n that's robust to data scale" — is exactly what this returns. Three concrete benefits over single-shot:

1. **Avoids picking a noise-driven argmax.** Single-shot at low n picks the width that happened to win that subsample's roll-of-the-dice. Progressive demands the width survive multiple subsamples.

2. **Compute-aware.** Spread regimes converge in 1 stage at the smallest n in the schedule — same cost as a single coarse sweep, but with the convergence-attestation as a free bonus diagnostic. Concentrated regimes refine adaptively where they need to.

3. **The convergence narrative IS the diagnostic.** A `ProgressiveRecommendation` that converges in 1 stage tells the user "data scale doesn't move this; you can trust it as-is". A recommendation that needed 4 stages to converge tells them "this substrate's optimum is genuinely close-call; small protein-pool changes might move it." The number is a substrate property worth reporting.

## What this does NOT solve

- **The structural ~9 % uniform-tax** on spread substrates. Progressive sweeping doesn't change the underlying mathematics — it just makes the recommendation more honest about how robust the answer is to sample size. Closing the 9 % gap is the algebra-side fix (orthonormalise basis / RMSNorm substitute) and is out of scope.
- **Cross-fixture transfer.** A recommendation that converges on substrate A says nothing about substrate B. The user runs the progressive sweep per (host, SAE, dataset) tuple they care about.
- **Choosing the schedule.** The schedule is a hyperparameter of the algorithm. Reasonable defaults (`[10, 50, 200, 1000]` for protein-scale substrates) ship in the CLI; users tune per-domain.

## Related

- Single-shot wrapper this composes: `saeforge.sweep_pareto_capability` (added by `add-downstream-capability-target` v0.8.0).
- Recommendation logic this generalises: `sae-forge recommend` (single-frontier predicate filtering; the progressive variant runs the recommendation inside the convergence loop).
- Successive halving / Hyperband: well-known in hyperparameter optimization. The novelty here is the *fidelity axis being protein count* (not training steps) and the *recommendation contract being stability across stages*, not best-at-final-budget.
- Bio-sae writeup: `bio-sae/docs/forge-capability-bottleneck.md` §3 — the two-regime characterisation that motivates the progressive design.
- Bio-sae empirical data: `bio-sae/runs/forge/acceptance_residue_n100/` + `bio-sae/runs/forge/acceptance_pooled_n1000/` — the 100-protein vs 1000-protein measurements that motivate "argmax shifts with data scale" as a real signal.
