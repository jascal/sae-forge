# Concise interpretability via distillation-by-routing

**Status:** methodology note + cross-fixture validation protocol.
**Module:** [`saeforge.isf`](../saeforge/isf.py) (`ensemble_route`, `recipe_auc_matrix`, `salience_headroom`, `capability_pareto`, `Recipe`).
**Empirical anchor:** the bio-sae motif-specialist line (PRs #2–#6) — numbers below.

## The thesis

A large SAE is a **substrate, not a dictionary.** The cheapest route to a
small, faithful interpretability model is not to prune one big SAE into a
smaller monolith — it's **distillation-by-routing**: send each concept to the
small specialist that reads it best, and fall back to a plain host readout for
the concepts that are already salient. The win is concentrated *exactly* on the
low-salience, relational concepts a reconstruction SAE systematically misses.

Four levers, isolated by controlled comparison (bio-sae, held-out synthetic
motifs, same split throughout):

| lever | controlled comparison | result |
|---|---|---|
| **Substrate dominates** | *same* occurrence-supervised objective on a weak from-scratch encoder vs strong ESM | occ-mAUC 0.761 → **0.998** |
| **Objective must align** | reconstruction vs occurrence-pooled supervision | 0 % → 6/6 motifs |
| **Metric is part of the experiment** | per-residue vs occurrence scoring of the *same* latents | 0 % → 0.998 |
| **Concision = routing** | best single recipe vs per-label routed ensemble | +0.021 lift, retained 1.035, 63 % beat host |

## The conditional — the salience heuristic (a rule of thumb, not a law)

A useful **rule of thumb** (observed across ≈2 fixtures so far — *not* a proven
law): specialisation's value tends to scale **inversely with target salience.**
On large, reconstruction-salient concepts a plain SAE already wins and a
specialist adds little (bio-sae's real-UniRef50 check: the unsupervised control
matched supervision 9/10 on large domains). So the methodology is **not**
"specialise everything" — it's:

> **diagnose salience → spend specialist budget where the host looks weak →
> route per concept.**

`saeforge.isf.salience_headroom(host_auc) = 1 − host_auc` is the cheap,
no-training diagnostic for *where to look*. Within bio-sae the routed lift
tracks it (motif tier host 0.893, headroom 0.107 → **+0.105**; salient
categorical host 0.951, headroom 0.049 → **+0.015**).

**Honest caveat — headroom is a prior, not a predictor.** The rule of thumb is
rough and already has a within-fixture exception: on econ-sae the *conjunctive*
tier had **low** headroom (0.072, host 0.928) yet got the **biggest** routing
lift (+0.033), because an objective-aligned specialist can beat the substrate
even where the substrate isn't weak. So a low headroom flags "the substrate
likely already encodes this" — it does **not** prove a specialist is useless.
Treat `salience_headroom` as a cheap *triage* signal, not a guarantee.

### Quantified — and it's weaker than it looked (`headroom_lift_analysis`)

We tried to upgrade the rule of thumb by measuring it. The result demotes it.
The trap: **`lift ≤ headroom` mechanically** (`ensemble_best ≤ 1`), so a raw
headroom→lift correlation is partly a ceiling artifact, not evidence of
prediction. `headroom_lift_analysis` de-confounds it with *fractional capture*
= `lift / headroom` ("of the room available, how much did a specialist
actually take?"):

| relationship | bio-sae (n=23) | econ-sae (n=32) | pooled (n=55) |
|---|---|---|---|
| **raw** headroom → lift (Pearson) | +0.59 | +0.01 | — |
| **de-confounded** headroom → frac-capture (Pearson) | +0.02 | **−0.49** | **−0.48** |

bio-sae's apparently-supportive +0.59 is **almost entirely the ceiling**: once
removed, the signal is **flat** (+0.02). In econ-sae it **inverts** (−0.49) —
specialists captured a *larger* fraction of the available room on *low*-headroom
labels. The reason is a confound inside `headroom` itself: high headroom (host
AUC near chance) often means **the concept is intrinsically hard**, not that an
easy specialist win is waiting — so even a specialist captures little of it.

**The defensible statement, then, is narrow:** `salience_headroom` is a valid
**necessary** condition (zero headroom ⇒ zero possible lift ⇒ don't bother) and
bounds the *maximum* achievable lift — but it is **not a sufficient predictor**
of *realized* lift, and high headroom can even signal "don't expect much."
"Specialise where headroom is high" survives only as "don't specialise where
headroom is zero." (Caveats: n is small and the bio/econ sign flip means this is
domain-dependent, not universal — itself a reason it's no law.)

## The methodology (recipe-agnostic, in `saeforge.isf`)

```python
from saeforge import recipe_auc_matrix, ensemble_route, salience_headroom

# recipes = [raw host, a polygram-tier basis, a supervised specialist, …]
A = recipe_auc_matrix([r.encode(X) for r in recipes], Y)   # (R, V) per-label AUC
route = ensemble_route(A, [r.name for r in recipes], host=0)
head  = salience_headroom(A[0])                            # where to specialise

route["ensemble_lift"]      # > 0 ⇔ the ensemble beats every single recipe (H-ISF headline)
route["retained"]           # ensemble mAUC / host mAUC
route["router_composition"] # which recipe owns how many concepts
```

A `Recipe` is anything with `name` + `encode(X) -> (N, d)`: a raw-host identity,
a supervised specialist (bio-sae P1), a polygram-tier slice, a regime-supervised
SAE (econ-sae Family G). The router is blind to how the latents were made — so
**encoding-family** diversity (H-ISF) and **objective-family** diversity
(supervised specialists) compose in the same ensemble.

## Cross-fixture validation protocol

The thesis is a property of *reading concepts out of representations*, not of any
domain — so it must hold across the three substrate fixtures (sm-sae gauge
symmetries / econ-sae double-entry / bio-sae biophysical), which already share a
ground-truth scoring convention. Run the same four steps on each fixture's hard
tier; the cases that *stress* the heuristic (low-headroom-but-helps, or
high-headroom-but-doesn't) teach more than the confirmations.

| fixture | hard-tier analogue | salience-heuristic prediction (falsifiable) |
|---|---|---|
| **sm-sae** | factorial particle features — salient | **low headroom expected** — conserved-quantity features sit at ~0 headroom (the substrate encodes them); probes the low-headroom end of the heuristic. |
| **econ-sae** | conjunctive-trap + regime tiers (has a regime-supervised Family G) | **strong lift, routed to the regime/conjunctive specialist** — direct analogue of bio-sae motifs. |
| **bio-sae** | planted motifs | **+0.105, 6/6** (done). |

Step 1 — `salience_headroom` per tier (no training): confirm the hard tiers sit
low for the host (high headroom) and the salient tiers near the ceiling.
Step 2 — train one aligned specialist on the *strong* substrate at the tier's
natural granularity. Step 3 — `recipe_auc_matrix` + `ensemble_route`. Step 4 —
check the lift lands on the high-headroom tier and is ~null on the salient one.

Status so far: bio-sae (anchor) + econ-sae (routing lift on the low-salience
tiers) support the rule of thumb; sm-sae shows the conserved-quantity features
sit at ~0 headroom (the predicted low end). The econ conjunctive tier is the
honest counterexample (low headroom, big lift) that keeps this a **heuristic**,
not a law.

## Tool division

- **sae-forge** — the methodology engine: `saeforge.isf` (this module) +
  `sweep_pareto_capability` (the params↔retained Pareto). Where the primitives
  live so every fixture imports the same ones.
- **n-orca** — recipe registry + verifier: each specialist ships a verified
  architecture doc (bio-sae already declared the JEPA backbone + Family G).
- **polygram** — geometry/audit: the router's per-label assignment *is* an
  `ExpertDictionary` partition; cancellation/interference checks the specialists
  encode orthogonal concepts (a genuinely concise model) rather than redundant
  ones, and the shared HEA encoding makes retained/lift comparable across
  fixtures.

The clean loop: **n-orca declares the specialists, sae-forge fits + routes +
Paretos them, polygram audits their geometry, the -sae fixtures validate
cross-domain with conservation laws as the safety net.**
