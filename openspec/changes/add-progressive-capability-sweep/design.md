## Context

`add-downstream-capability-target` (shipped as v0.8.0 + v0.8.1) gives users a per-substrate Pareto frontier in retained-AUC space. The single-shot `sweep_pareto_capability` is the right answer when the recommendation question is "argmax retained_mauc on this exact eval sample". Bio-sae's empirical work showed it's NOT the right answer when the question is "what's the substrate's optimum I should commit to for production forging" — because the argmax depends on the eval sample's protein count, and bio-sae documented that dependence at both the writeup (§3.1: 10 proteins) and the higher-scale verification (this work: 100 proteins).

The pattern is well-known in hyperparameter optimization as **successive halving** (Karnin et al. 2013) / **Hyperband** (Li et al. 2017). The novel application is the fidelity axis: protein count, not training steps. That changes the per-stage cost model and the convergence criterion.

## Goals / Non-Goals

**Goals:**
- A new top-level wrapper `sweep_pareto_capability_progressive` that produces a *stable* recommendation, not an argmax-on-one-sample.
- Reuse the existing single-shot sweep machinery + host-extraction cache. The progressive wrapper is a loop over single-shot calls, not a separate forge path.
- Falsifiable acceptance gates against bio-sae's existing fixtures pinning specific convergence behaviour (residue regime: 2-3 stages; pooled regime: 1 stage).
- CLI surface (`sae-forge sweep-capability-progressive`) + a recommendation-emit step that surfaces the convergence narrative.
- Back-compat: the existing single-shot sweep + `sae-forge recommend` are unchanged.

**Non-Goals:**
- Bayesian-optimization-style continuous width selection. Widths come from the user's `candidate_widths` list; the progressive wrapper only PRUNES and EXPANDS-TO-NEIGHBORS within that list.
- Multi-fidelity confidence intervals. We don't compute AUC standard errors per cell; convergence is determined by *whether the argmax position is stable*, not by overlapping confidence intervals. (A future refinement could add CIs.)
- Per-width adaptive stage counts. The schedule is global; every stage advances every survivor. A future refinement could let each width spend more proteins independently.
- Sample-disjoint stages. v1 uses *cumulative* subsamples (stage K+1 ⊇ stage K) so the host-extraction cache survives across stages. Disjoint-sample variants (cross-validation-style) are a separate research question.

## Decisions

### Decision 1 — Cumulative subsamples (stage K+1 ⊇ stage K), not disjoint

Stage K's protein subsample is the first `n_proteins_schedule[K]` proteins of `dataset.sequences` (or a deterministic seeded sample). Stage K+1 takes the first `n_proteins_schedule[K+1]` — a strict superset.

**Why:**

- **Host-extraction cache reusability.** The cache keys on `(host_model_id, sequences_hash, aggregator, max_seq_len, feed)`. Different protein subsets ⇒ different hashes ⇒ different cache files. A disjoint-stage scheme writes a fresh cache per stage; a cumulative scheme reuses the previous stage's cache for the overlapping prefix and only extracts the new tail. Cache is by-row, so we make this work by keeping per-stage cache keys and per-stage extraction calls; the saving is amortised across cells WITHIN a stage, not across stages. **The benefit of cumulative isn't cache reuse — it's the next item.**
- **Statistical interpretability.** If stage 0 reported retained_mauc=1.032 on 10 proteins and stage 1 reports 1.045 on 100 proteins, the rise from 1.032 → 1.045 has a clean interpretation: "with 10× more data, we found marginal features that lifted forge_mauc relative to host." On a disjoint sample, the comparison is "different sample, different number — was the shift from data scale or from sample drift?" Cumulative makes the data-scale signal isolated.
- **The bio-sae acceptance test reproduces this.** The 100-protein measurement is a strict superset of the 10-protein measurement (both start from `uniref50_sample__n100_seed0.parquet`'s row 0); the shift in argmax position is therefore attributable to data scale, not sample-to-sample noise.

The trade-off: cumulative subsamples don't give true cross-validation guarantees. A held-out final stage (`dataset.sequences[-200:]`) could be added in a future revision for users who want "trained on 800, validated on 200". v1 doesn't ship that.

### Decision 2 — Plateau-based pruning, not top-K

Each stage identifies the *plateau* of widths within `plateau_tolerance` of the peak retained_mauc, then carries those + their immediate `candidate_widths` neighbours forward. Top-K (keep best K widths) is rejected for two reasons:

1. **Top-K is brittle on flat plateaus.** Bio-sae's residue regime shows 9 of 11 widths within 5 % of peak. Top-3 would prune 6 of those 9 — most of them won't get a chance to be the stable recommendation. Plateau-based pruning keeps the whole flat region while still trimming the cliff cells (n=4 at 0.96; n=256 at 0.76).
2. **Plateau identification gives a free side channel.** `ProgressiveStageResult.plateau_widths` is exactly the "any of these is fine, pick the smallest" set the recommendation contract returns. It's the same construct.

`min_plateau_widths` (default 3) guards against degenerate cases where one cell narrowly clears the tolerance.

### Decision 3 — Convergence on argmin-of-plateau, not argmax

The recommendation is *the smallest target_n_features_kept in the plateau*, not the absolute peak. Why:

- **The user's framing**: "smallest n that's robust to data scale." Argmax of retained_mauc is the *best* n; argmin of the plateau is the *cheapest stable* n.
- **Compute downstream**: a forged model at n=12 is ~25 % the parameters of n=48. If both sit in the plateau (both retained_mauc within 1 % of each other), pick n=12. The Pareto-optimal point on (capability, cost) is the small end of the plateau, not the peak of capability.
- **Stability**: small-n cells are more variance-sensitive than mid-n cells (fewer features = each one matters more). If a small-n cell sits in the plateau across multiple data-scale stages, that's a stronger stability claim than the absolute peak passing the same test. The contract rewards the stronger claim.

Concretely: convergence fires when **the smallest plateau-member width is unchanged for `convergence_n_stages` consecutive stages**, AND the retained_mauc gap between that width and the absolute peak stays within tolerance.

### Decision 4 — Neighbor expansion, not grid generation

After plateau pruning, the wrapper expands the active set with immediate-neighbour candidate widths. If `candidate_widths = [4, 8, 16, 32, 64, 128, 256, 512, 1024]` and the plateau after stage 0 is {16, 32, 64}, the next stage's active set is {8, 16, 32, 64, 128} (the plateau + each member's immediate neighbours in the candidate list).

**Why not generate new widths between candidates** (e.g., add n=24 between 16 and 32)?

- Users supply the candidate list because they have a budget / engineering opinion on which widths to consider. Generating widths the user didn't request adds them to the sweep without consent and breaks `target_n_features_kept` reproducibility across runs with different candidate lists.
- Doubling the grid resolution is the user's call. They can pass a denser candidate list (`[4, 8, 12, 16, 24, 32, 48, 64, 96, 128, ...]`) and let the plateau pruning concentrate compute there.
- Bio-sae's 100-protein run already used a denser grid (`[4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]`) and the plateau correctly picked n=48 as the peak. The progressive wrapper inherits whatever grid the user supplies; it doesn't second-guess the grid.

### Decision 5 — Schedule is global; per-stage budget is uniform across active widths

Stage K's budget is `len(active_widths_at_stage_K) × n_proteins_schedule[K]` forge cells. Every active width gets the same protein count. Why not adaptive (give borderline widths more proteins than cliff-cell widths)?

- Adaptive per-width budgets require AUC standard-error estimates per cell (to know which widths need more discrimination). v1 doesn't compute those.
- The naive uniform schedule already gives most-of-Hyperband's compute-savings: stage 0 prunes the cliff cells cheaply (n=4 and n=256 at 10 proteins), so stages 1+ don't pay for them.
- A future refinement can add per-width adaptive budgets without breaking the cumulative-schedule contract — the stages are still ordered low-to-high; the within-stage allocation is the new variable.

### Decision 6 — Recommendation contract: convergence is a hard requirement

If the convergence criterion doesn't fire within the supplied schedule, `ProgressiveRecommendation.converged = False` and `rationale` explains which stage's argmin-plateau-member differed from the previous. **The wrapper still returns a recommendation** (the last stage's argmin-plateau-member) but flags it as un-converged. CLI `sae-forge recommend` (when consuming a progressive frontier) checks the `converged` flag and warns prominently when False.

This avoids the failure mode where a user runs `sweep_pareto_capability_progressive` once with too-short a schedule, gets back a recommendation, doesn't notice the `converged=False`, and ships a non-stable forge. The CLI's `recommend` consumer raises if not converged unless `--accept-unconverged` is passed.

**Reviewer concern, telemetry, and falsifiable usage-rate claim.** PR #82's review flagged: "the convergence requirement is strict by default — probably the right call, but I'd want to see how often the `--accept-unconverged` flag actually gets used in practice." That's a measurable design hypothesis. To make it falsifiable:

1. **`progressive_summary.json` records the full convergence trajectory.** Every stage's `(stage, n_proteins, argmin_plateau_width, argmin_retained_mauc, plateau_size, neighbours_added)` is on disk. External benchmarking (counting un-converged ratios over a corpus of runs) needs no in-library telemetry — it's all in the artefact.

2. **Predicted usage rate.** With the default schedule `[10, 50, 200, 1000]` and `convergence_n_stages=2`, **the expected fraction of runs needing `--accept-unconverged` is ≤ 10 %** on representative substrates (the bio-sae two-fixture set + sm-sae / econ-sae's analogous fixtures once they adopt this). The 10 % is the falsifiable hypothesis. Reasoning:
   - Spread regimes converge in 1 stage (single-shot stability). 0 % un-converged expected here.
   - Concentrated regimes converge in 2-3 stages on bio-sae's residue fixture (writeup §3.1 + this work's n=100 verification). 0-20 % un-converged expected: most converge, some hit the schedule's tail on close-call sub-regimes.
   - Pathological substrates (no plateau exists; retained_mauc varies monotonically across all widths) will reliably exhaust the schedule. These exist but are rare in the SAE-fixture matrix.

3. **Fallback that's NOT `--accept-unconverged`.** Two opt-ins ship for users who don't want the strict default but also don't want to blanket-accept un-converged output:
   - `--convergence-n-stages 1`: declare convergence as soon as the last stage's argmin-plateau-member is plateau-stable on the *previous* stage. Looser but still data-scale-aware (vs. single-shot which doesn't even check).
   - `--schedule N` (single integer): degenerate to single-shot `sweep_pareto_capability` at protein count N. No convergence check; emits a progressive frontier with one stage, `converged=True` by definition. Documented as "I want the progressive frontier's reporting surface but not its strictness."

   These give users *informed* opt-outs, not just "trust me, the schedule failed but I'm shipping it anyway."

4. **Bio-sae-side reporting follow-up.** Once the progressive wrapper ships, bio-sae's `scripts/forge_capability_acceptance.py --progressive` re-runs both regimes and `runs/forge/progressive_*` carries the convergence trajectories. Sm-sae and econ-sae's analogous fixtures follow. After 6-month adoption, count the `--accept-unconverged` invocations across the three fixture repos against the predicted ≤ 10 %. Higher → schedule defaults are wrong (or `plateau_tolerance` is too tight); lower → strictness is well-calibrated.

## Risks / Trade-offs

- **The schedule is a hyperparameter.** Wrong schedule (too small, too large, wrong spacing) → wrong recommendation. Default `[10, 50, 200, 1000]` is bio-sae-calibrated; CLI documents the calibration source and recommends benchmarking the schedule on a representative substrate before committing.
- **Cumulative subsampling biases toward the beginning of `dataset.sequences`.** If the user's parquet is sorted (e.g., by protein length), stage 0's small subsample is unrepresentative. Mitigation: the wrapper documents this and recommends a pre-shuffle of the dataset. Could ship a `shuffle_seed` parameter that pre-shuffles, but that complicates the cache-key story; v1 defers.
- **The convergence threshold (`retained_mauc_tolerance`) is dataset-relative.** A 0.005 tolerance is generous on the spread regime (retained_mauc cluster within 0.05 there) but tight on the concentrated regime (retained_mauc cluster within 0.05 of each other). Users may want to tune per-dataset. The default is fine for the bio-sae fixtures; the CLI exposes it.
- **Memory: progressive cumulates rows across stages.** The output `frontier.jsonl` grows linearly in (#widths × #stages). Bio-sae's longest run would be 11 widths × 4 stages = 44 rows. Negligible at this scale; users running over many sub-substrates may want stage-cycling but that's not v1.
