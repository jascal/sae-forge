## Why

A forge run today projects every host weight through a **single**
`FeatureBasis` (`saeforge/basis.py`) — the surviving decoder of a
Polygram-compressed SAE — and reconstructs the residual stream as its
projection onto that one subspace. That basis is a **residual /
feature basis**: its rows are directions `d_X` that read *assertions*
(token identity, lexical class, a present feature). The well-documented
forge tax is that this projection **preserves `mAUC` but collapses
`cov95`** — the forged latents stay linearly decodable in aggregate,
but the *sharp, single-latent monosemantic* detectors that made the
host interpretable are smeared.

Work in the sibling `lm-sae` instrument (GPT-2 + an exact lexical
oracle, laptop scale) localised *why*, and it is not a basis-quality
problem that more features or better tuning fixes. There are **two
different kinds of content in the residual stream**, and a residual
feature basis can only carry one of them:

1. **Assertions are 1-operand objects** — directions. An SAE factors
   them monosemantically. On the χ-ladder (`lm-sae/scripts/chi_ladder.py`)
   token identity has residual `cov95 = 0.90`.
2. **Rules / computation are 2-operand objects** — the bilinear forms
   `B_h[X,Y] = d_X·(W_Q^h W_K^hᵀ)·d_Y` (QK, the "match") and
   `V_h[Y,Z] = d_Y·(W_V^h W_O^h)·d_Z` (OV, the "move"). These are not
   directions; you cannot read a 2-argument relation off a 1-argument
   point. On the same χ-ladder *every* compositional function
   (in-context repeat, induction) has residual `cov95 = 0.00` while
   being read at `AUC ≈ 0.94–1.00` from the attention-circuit
   ("macro") features. Residual-χ and macro-legibility are
   **orthogonal axes**.

The single-basis forge tries to push both through one residual
dictionary. That has two distinct costs, and the second is currently
unmeasured:

- It **de-monosemanticises the assertions** (the classic `cov95`
  collapse).
- It **smears the composition-feeding directions** — the residual
  directions that *downstream* attention reads as its keys/values.
  When those are corrupted, the circuits built on them break. In
  `lm-sae` the induction circuit's predecessor-write directions are
  exactly such directions: surgically removing them from a downstream
  head's key path collapses that head's induction attention
  **36–56%** (`lm-sae/scripts/path_patch_induction.py`), and the
  static 2-head weight composition predicts the live edges at
  `ρ = 0.86` (`composition_probe.py`). These directions are **not**
  oracle-readable assertions, so even a `cov95`-perfect assertion
  preserve does not protect them.

This change adds an opt-in **two-basis forge**: preserve, verbatim,
*two* low-dimensional residual subspaces inside the projection — the
**assertion subspace** (sharp monosemantic atoms → recovers `cov95`)
and the **composition subspace** (the host attention's QK/OV
read+write geometry → keeps the macros, and therefore the circuits,
faithful). The Polygram basis continues to carry the remaining
residual variance. It also adds the **circuit-faithfulness** metric
the tax has been invisible to (KL restricted to circuit-driven
tokens, e.g. induction-predictable positions), so the mechanism can
be judged on the thing it targets rather than on global KL alone.

As with `hybrid-bridge-forge`, this proposal does **not** assume the
laptop `lm-sae` numbers replicate at GPU scale on a production SAE.
It ships the mechanism behind a default-off toggle, pins the
algebraic contract (which subspace is preserved and how), and adds
the GPT-2/Intel validation harness. If two-basis forge fails to beat
single-basis on the circuit-faithfulness metric with conservative
defaults, the toggle stays off and the design is the dispositive
negative result.

## What Changes

### Scope

Add an optional two-basis forge path to `ForgePipeline`. When
`composition_preserve=True` (and/or `assertion_preserve=True`), the
projection's *kept subspace* is augmented so that two extra
low-dimensional subspaces are reproduced **exactly** by the forged
weights, with the Polygram basis carrying the orthogonal remainder:

1. **Composition subspace `U_C`** — per capture layer, the principal
   subspace of the host attention's residual read/write geometry:
   the dominant left-singular directions of the stacked per-head
   `[W_Q^h | W_K^h]` (QK *reads* the residual) and of `W_V^h W_O^h`
   (OV *writes* the residual), optionally restricted to a supplied
   set of circuit heads. Preserving `U_C` makes the forged QK/OV
   bilinear forms agree with the host on the directions attention
   actually uses, so the macros — and the multi-head idioms composed
   from them — survive forging.
2. **Assertion subspace `U_A`** — the decoder directions of the
   top-`K_A` sharpest, most monosemantic SAE atoms, kept verbatim
   rather than merged by Polygram. This is the `lm-sae` P1 lever that
   recovers host `cov95`.

When both toggles are `False` (the default), zero new code paths
execute and the forged weights are **byte-identical** to pre-change.
The existing byte-equivalence gate
(`test_imperative_and_fsm_byte_equivalent`) remains the load-bearing
check.

### New artifacts

- **`saeforge/composition_subspace.py`** — new module.
  `CompositionSubspace` dataclass (per-layer orthonormal `U_C`,
  `d_model`, source-head list, singular-value tail logged for the
  rank choice) plus `extract_composition_subspace(host, *, layers,
  rank, heads="all") -> dict[int, np.ndarray]`. Pure-numpy on the
  host weight tensors; lazy-imports torch only to read parameters.
  ~150 lines.
- **`saeforge/augmented_basis.py`** — new module.
  `AugmentedBasis` wraps a `FeatureBasis` plus optional `U_A`
  (verbatim assertion atoms) and per-layer `U_C` (composition), and
  exposes the single contract the projector needs:
  `kept_subspace(layer) -> (W_dec_eff, preserve_mask)` returning an
  orthonormalised stacked subspace and the boolean mask of rows that
  must be reproduced exactly (vs. Polygram-merged). Validates
  `d_model` agreement and reports the preserved-dimension budget as a
  fraction of `d_model`. ~180 lines.
- **`saeforge/eval/circuit_faithfulness.py`** — new metric module.
  `circuit_kl(host, forged, tokens, *, circuit_mask)` returns
  `KL(host ‖ forged)` restricted to a boolean token mask, plus the
  `lm-sae`-style helpers to build the masks (`induction_predictable`,
  `in_context_repeat`). `assertion_cov95(forged_residual, oracle)`
  reuses the existing oracle-probe to report monosemantic-detector
  fraction on the forged residual. ~160 lines.
- **`tests/test_composition_subspace.py`** — `U_C` orthonormality;
  rank/budget reporting; the "preserve `U_C` ⇒ forged QK/OV agrees
  with host on `U_C` to tolerance" algebraic invariant on a tiny
  GPT-2.
- **`tests/test_augmented_basis.py`** — kept-subspace orthonormality;
  preserve-mask correctness; `d_model` mismatch raises;
  budget-fraction reporting.
- **`tests/integration/test_two_basis_forge_gpt2.py`** — end-to-end
  GPT-2 forge with `composition_preserve=True`: finite pre/post-FT
  KL, safetensors round-trip, and the **circuit-faithfulness
  invariant** — induction-predictable KL under two-basis forge is
  `≤` the single-basis baseline on the same bases/seed.
- **`scripts/compare_single_vs_two_basis_gpt2.py`** — one-shot
  comparison harness (not pytest). Single-basis vs assertion-only vs
  two-basis on `gpt2` with shared `n_features`/seeds; emits a table
  of (global KL, induction-predictable KL, assertion `cov95`,
  preserved-dim budget). This is the artifact that decides whether to
  default either toggle on.

### Modified artifacts

- **`saeforge/forge.py`** — `ForgePipeline` gains optional fields, all
  defaulting to v0 behaviour: `composition_preserve: bool = False`,
  `assertion_preserve: bool = False`,
  `composition_rank: int | None = None` (auto from a singular-value
  knee when `None`), `composition_heads: list | str = "all"`,
  `assertion_k: int = 0`. `__post_init__` builds the `AugmentedBasis`
  only when a toggle is on; otherwise `self.basis` is used unchanged.
- **`saeforge/projector.py`** — `SubspaceProjector.project_module`
  gains an optional `augmented: AugmentedBasis | None = None` kwarg.
  When `None`, the existing single-basis dispatch is byte-identical.
  When provided, each weight is projected through that layer's
  augmented kept subspace, and the preserve-mask rows are written
  back verbatim instead of through the Polygram merge.
- **`saeforge/cli.py`** — new `forge` flags:
  `--composition-preserve`, `--composition-rank N`,
  `--composition-heads {all,LIST}`, `--assertion-preserve`,
  `--assertion-k N`, `--circuit-faithfulness` (emit the new metric in
  the run report).
- **`saeforge/eval/__init__.py`** — export the circuit-faithfulness
  entry points so the sweep/report code can consume them.
- **`docs/`** — a `two_basis_forge.md` note: the two-kinds-of-content
  framing, the algebra, and a pointer to the comparison harness +
  the `lm-sae` provenance.
- **`CHANGELOG.md`** — `## [Unreleased]` `### Added`.

### Out of scope (deferred)

- **Non-attention composition (MLP key-value memories).** `U_C` here
  is the *attention* read/write geometry only. MLP neurons as
  key-value stores are a second composition surface; tracked as
  follow-up `mlp-composition-preserve`.
- **Learning the composition subspace.** `U_C` is read directly from
  host weights (SVD), not learned/fine-tuned. A learnable
  composition bridge is the `hybrid-bridge-forge` direction; the two
  compose but are not combined here.
- **Cross-layer composition routing.** The macro idioms span layers
  (induction = prev-token head ⊕ later induction head). v1 preserves
  each layer's local read/write geometry, which is sufficient to keep
  the per-edge composition faithful; explicit multi-layer
  circuit-subspace bundles are tracked as `circuit-subspace-bundle`.
- **Auto-selecting circuit heads.** v1 supports `heads="all"` (pool
  every head's geometry) or an explicit list. Behavioural
  head-discovery (which heads are induction/copy) is an analysis
  step, not a forge step; tracked as `circuit-head-autodetect`.

## Capabilities

### New Capabilities

- **`composition-subspace-preserve`** — defines the composition
  subspace `U_C` (extraction from host QK/OV geometry, orthonormality,
  rank/budget contract), the augmented kept-subspace contract (how
  `U_A`, `U_C`, and the Polygram basis stack and what
  "preserved verbatim" means algebraically), the
  preserve-when-disabled byte-equivalence scenario, and the
  circuit-faithfulness metric contract.

### Modified Capabilities

- **`subspace-projector`** — the existing
  `project_module covers every GPT-2 weight` requirement is MODIFIED
  to add the optional `augmented` dispatch arm. When unused,
  single-basis behaviour is preserved byte-identically.
- **`faithfulness-target`** — one ADDED requirement
  (`Circuit-restricted faithfulness KL`) reports circuit-restricted KL
  alongside the unchanged global KL, so the two-basis mechanism is
  judged on circuit fidelity, not only aggregate KL.

## Impact

- **No public API breakage.** Default `composition_preserve=False` /
  `assertion_preserve=False` leaves `ForgePipeline.run()`, the CLI,
  and on-disk artifacts byte-identical. All new fields/flags are
  additive.
- **No FSM topology change.** Augmented projection lives inside the
  existing `project_to_subspace` action — same state/guard/target —
  by passing the `AugmentedBasis` through ctx, exactly as
  `hybrid-bridge-forge` passes its bundle.
- **No `forge-forward-mode` change.** The forged model still runs
  `native_in_basis`; preserving `U_C` only changes *which* subspace
  the projection keeps, not how the forward pass executes.
- **Test surface.** ~30 new tests; the byte-equivalence gate passes
  unmodified.
- **Validation surface.** `scripts/compare_single_vs_two_basis_gpt2.py`
  (Intel/GPT-2, `lm-sae` oracle as ground truth) is the
  defaults-decision artifact.

## Sequencing

- **Depends on:** `subspace-projector` and `architecture-adapters`
  (both on `main`) for the projection algebra and the per-architecture
  weight walk; `faithfulness-target` for the KL reporting surface the
  circuit metric extends.
- **Orthogonal to:** `hybrid-bridge-forge` (multi-basis across
  *layers*; this is multi-subspace-of-different-*kind* within a
  layer). The two are composable — a hybrid run could preserve a
  composition subspace at each anchor — but v1 keeps them independent.
- **Single PR.** ~700 net LOC. The byte-equivalence gate plus the
  Intel/GPT-2 circuit-faithfulness integration test are the shipping
  criteria.
