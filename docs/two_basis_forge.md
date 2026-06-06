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
- **`U_C` — composition subspace** (`--composition-preserve`, default mode
  `writer-output`): the orthonormalised union of the **OV-output row spaces of
  the circuit's writer heads** — for each writer `(L, h)`, `rowspace(W_V^h
  W_O^h)`, the directions that head *writes* into the residual. Preserving it
  keeps the signal a downstream circuit reads intact, so the multi-head idiom
  (e.g. induction: predecessor-write → name-mover read) survives forging. The
  writer heads are chosen by `--composition-heads`: a behavioral preset
  (`prev-token` / `duplicate-token`, detected on the eval corpus by their Δ=1 /
  same-token-earlier attention), an explicit `L.H,…` list, or `all` for the
  legacy aggregate **reader-geometry** mode (`--composition-mode
  reader-geometry`) — the top singular directions of `[W_Q^h | W_K^h]` (reads)
  and `W_V^h W_O^h` (writes) per capture layer, which does **not** protect
  circuits and is kept only as an ablation.

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

## Why writer-output, not reader-geometry or attribution

`U_C` was redefined from the aggregate reader-layer geometry to the writer
heads' OV-output after an **alive single-layer GPT-2 forge** (`lm-sae`,
`two_basis_single_layer.py`) measured the induction-predictable KL *excess*
each candidate subspace removes:

| `U_C` candidate | what it preserves | induction excess removed |
|---|---|---|
| **writer-output** | `rowspace(W_V^h W_O^h)` of the predecessor-write heads | **−111%** (excess → below zero) |
| reader-geometry | aggregate `[W_Q\|W_K]` + OV of the capture layers | −6% (≈ no protection) |
| attribution (`∂loss/∂residual`) | top directions of the induction-loss gradient | **+14% worse** |

The circuit-faithfulness subspace and the loss-sensitivity subspace are nearly
**orthogonal** (`overlap ≈ 0.05`): **loss-sensitivity ≠ circuit-mechanism.** The
directions a loss gradient is most sensitive to are not the directions the
circuit actually moves signal along, so the cheap label-free attribution
shortcut does not protect the circuit — the writer heads must be identified
mechanistically (the idiom-library / `circuit_heads` detector is load-bearing).
Writer-output buys this at a small global-KL cost (the writer subspace and the
global-fidelity subspace differ), which the `--circuit-faithfulness` report
surfaces. Reader-geometry remains available as the documented null control.

The whole-model single-basis `ForgePipeline` cannot itself reproduce the
`excess ≤ 0` regression on 12-layer GPT-2 (a single basis shared across all
layers collapses to uniform output); that end-to-end validation lives in the
`lm-sae` single-layer alive forge. The sae-forge tests here verify the
**mechanism** — writer detection, the OV-output subspace, the preserve
invariant, and the pipeline wiring.

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
- `U_C` extraction and writer-head detection support GPT-2 in v1; other
  architectures plug in via their adapter's head-geometry helper.
- Writer detection ships two behavioral presets (`prev-token`,
  `duplicate-token`); richer auto-detection (e.g. full induction-circuit
  discovery) and explicit lists cover the rest. The detected writers — with
  their detection scores — are recorded in the run report.
- MLP key-value composition, cross-layer circuit bundles, and learned
  composition bridges are out of scope (see `proposal.md`).
