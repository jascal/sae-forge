# Two-basis forge (composition-subspace preserve)

A forge run projects every host weight through a single `FeatureBasis` — a
*residual / feature* basis whose rows read 1-operand **assertions** (token
identity, lexical class). The forge tax is that this preserves `mAUC` but
collapses `cov95`: the sharp monosemantic detectors smear, and — less visibly —
the residual directions the host *attention* reads and writes get dropped, so
the circuits built on them break while global KL barely moves.

Two-basis forge keeps two low-dimensional residual subspaces verbatim inside
the projection, with the Polygram basis carrying the orthogonal remainder:

- **`U_A` — assertion subspace** (`--assertion-preserve --assertion-k K`): the
  top-`K` sharpest (least-merged) atoms, kept verbatim → recovers `cov95`.
- **`U_C` — composition subspace** (`--composition-preserve`): per layer, the
  dominant directions of the host attention's read/write geometry — the top
  singular directions of `[W_Q^h | W_K^h]` (reads, `ln_1`-gain folded) and of
  `W_V^h W_O^h` (writes). Preserving `U_C` makes the forged QK/OV agree with
  the host on exactly the directions attention uses, so the circuits — and the
  multi-head idioms composed from them — stay faithful.

## Why it is a different axis from feature quality

Assertions are 1-operand objects (directions); rules are 2-operand bilinear
forms `M_h = W_Q^h W_K^h.T` and `OV_h = W_V^h W_O^h`. You cannot read a
2-argument relation off a 1-argument point — which is why no residual-feature
probe could read induction, and why `cov95` (a single-latent measure) is `0`
for *every* compositional function while the attention "macro" features read
them at `AUC ≈ 0.94–1.0` (the `lm-sae` χ-ladder). Residual-χ and
macro-legibility are orthogonal: the single basis can carry one, not both.

## The algebra

`U_C` is inserted as rows of the per-layer effective decoder `W_dec_eff`
(displacing the least-important atoms, so `n_features` is fixed — over-complete
Polygram bases cannot be orthonormalised to `n_features` rows). Then
`pinv(W_dec_eff) @ W_dec_eff` is the orthogonal projector onto a rowspace
containing `U_C`, so the forged query/key/value maps reproduce the host on
`span(U_C)` up to the global `scale_boost` the forge already applies. The
guarantee is exact for `scale_boost = 1` (well-conditioned bases) and verified
in `tests/test_two_basis_projector.py::test_augmented_preserves_host_QK_on_U_C`.

## Measuring it: circuit-faithfulness

Global KL is dominated by common, assertion-driven tokens and is nearly blind
to circuit breakage (induction-predictable tokens are a single-digit
percentage). `saeforge.eval.circuit_faithfulness` adds:

- `induction_predictable` / `in_context_repeat` token masks,
- `circuit_kl(host, forged, mask=…)` → masked vs complement KL,
- `assertion_cov95(forged_latents, oracle)` → monosemantic-detector fraction.

The shipping invariant is **induction-predictable KL(two-basis) ≤
single-basis** at a non-regressing global KL.

## Provenance and status

Motivated by laptop `lm-sae` results — GPT-2 + an exact lexical oracle, single
seed: the χ-ladder, the induction path-patch (removing the predecessor-write
directions collapses a downstream head's induction 36–56%), and the
weight-composition reads (static `comp_diag` predicts live induction edges at
ρ=0.86). It is **not** assumed to replicate at GPU scale on a production
Polygram SAE — the mechanism ships behind a default-off toggle. The
`dim(U_C ∩ basis)/dim(U_C)` overlap (in the run report under
`--circuit-faithfulness`) is self-diagnosing: a near-1.0 overlap means the
production basis already covers the composition subspace and the toggle
correctly does little.

## Deciding the defaults

`scripts/compare_single_vs_two_basis_gpt2.py` forges `gpt2` four ways
(single / assertion / composition / two-basis) over a `composition_rank` /
`assertion_k` sweep and emits the global KL, induction-predictable KL, the
preserved-dimension budget, the `U_C∩basis` overlap, and a Pareto plot. If
two-basis does not beat single-basis on induction KL at a non-regressing
global KL with conservative defaults, both toggles stay off and the design is
the dispositive negative result.

## Caveats / follow-ups

- `U_A` selection is a label-free sharpness proxy (least-merged atoms); a
  label-driven selection (with a `GroundTruthTarget` oracle) is deferred
  (`per-layer-assertion-atoms` / oracle-driven preserve).
- `U_C` extraction supports GPT-2 in v1; other architectures plug in via their
  adapter's head-geometry helper.
- MLP key-value composition, cross-layer circuit bundles, learned composition
  bridges, and `heads="induction-like"` auto-detection are out of scope (see
  `proposal.md`).
