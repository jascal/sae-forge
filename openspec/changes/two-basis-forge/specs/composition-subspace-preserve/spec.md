# composition-subspace-preserve Specification

## Purpose

The `composition-subspace-preserve` capability defines an opt-in forge
path that preserves, verbatim inside the projection, two low-dimensional
residual subspaces a single `FeatureBasis` cannot carry together: the
**assertion subspace** `U_A` (sharp monosemantic atoms → host `cov95`)
and the **composition subspace** `U_C` (the host attention's QK/OV
read+write geometry → faithful macros / circuits). The Polygram basis
carries the orthogonal remainder.

This capability pins: how `U_C` is extracted from host weights and what
"orthonormal / rank-bounded" means; the augmented kept-subspace contract
(`U_A`, `U_C`, basis remainder stacked, verbatim rows marked); the
byte-equivalence-when-disabled scenario; and the circuit-faithfulness
metric the mechanism is judged on.

It is opt-in (default `composition_preserve=False`,
`assertion_preserve=False`); the v0 single-basis path remains the
default and is preserved byte-identically.

## ADDED Requirements

### Requirement: composition subspace is the host QK/OV residual geometry

`extract_composition_subspace(host, *, layers, rank, heads, fold_ln1)`
SHALL, for each requested layer, return a `CompositionSubspace` whose
columns are an orthonormal basis of the union of (a) the top singular
directions of the stacked per-head query/key projections
`[W_Q^h | W_K^h]` over the selected `heads` (the residual directions
attention *reads*) and (b) the top singular directions of the stacked
per-head `W_V^h W_O^h` (the residual directions attention *writes*).
When `fold_ln1` is true the `ln_1.weight` gain SHALL be folded into the
residual side before the SVD. The returned `U` SHALL satisfy
`||UᵀU − I||_F < 1e-5`.

The preserved guarantee is algebraic: for any `Δr ∈ span(U)`, the forged
QK score `Δr · M_h^forged · Δr'` SHALL equal the host `Δr · M_h^host · Δr'`
for all `Δr' ∈ span(U)`, to projection tolerance — i.e. the macros are
exact on the preserved subspace.

#### Scenario: U_C is orthonormal and rank-bounded

- **GIVEN** a GPT-2 host and `rank=16`, `heads="all"`
- **WHEN** `extract_composition_subspace(host, layers=[6], rank=16)` runs
- **THEN** the returned `U` has shape `(768, r)` with `r <= 16`
- **AND** `||UᵀU − I||_F < 1e-5`

#### Scenario: QK macro is preserved on U_C

- **GIVEN** `U` from a host layer and that layer's head `h`
- **WHEN** the forged `M_h` is restricted to `span(U)`
- **THEN** it matches the host `M_h` restricted to `span(U)` to `1e-6`

#### Scenario: head restriction shrinks the source

- **GIVEN** `heads=[4, 11]` (a circuit-head subset)
- **WHEN** extraction runs
- **THEN** only those heads' projections contribute to the SVD inputs
- **AND** the reported `source_heads` equals `[4, 11]`

### Requirement: augmented kept subspace stacks preserve-rows first

`AugmentedBasis.kept_subspace(layer)` SHALL return
`(W_dec_eff, preserve_mask)` such that `U_A ∪ U_C[layer]` lie in
`rowspace(W_dec_eff)` and are reproduced verbatim, with `W_dec_eff`
keeping the SAME shape as `basis.W_dec` (`n_features` fixed) and the
Polygram basis carrying the remainder. `preserve_mask` SHALL be `True`
exactly on the `U_A ∪ U_C` rows.

Implementation note: because a production Polygram basis is
over-complete (`n_features > d_model`) and cannot be orthonormalised to
`n_features` rows without changing shapes, the shipped construction keeps
`n_features` fixed by **displacing the least-important (lowest decoder-norm)
atoms** with the verbatim `U_A`/`U_C` rows (rather than the literal
"orthonormalise the stack"). The contract — `U_A ∪ U_C ⊆ rowspace(W_dec_eff)`,
reproduced verbatim, shapes unchanged — is identical and is what the
projector guarantee relies on. All three sources SHALL share `d_model`; a mismatch SHALL raise
`ValueError` naming the mismatched source and the two `d_model` values.

#### Scenario: preserve mask marks exactly the verbatim rows

- **GIVEN** `U_A` with `K_A=8` rows and `U_C[6]` with rank `12`
- **WHEN** `kept_subspace(6)` is called
- **THEN** `preserve_mask` has exactly `20` `True` entries
- **AND** those entries index the leading orthonormalised rows

#### Scenario: d_model mismatch raises

- **GIVEN** `basis.W_dec` with `d_model=768` and a `U_C` with `d_model=1024`
- **WHEN** `AugmentedBasis(...)` is constructed
- **THEN** `ValueError` is raised
- **AND** the message contains `"d_model"`, `768`, and `1024`

### Requirement: disabled toggles are byte-identical to single-basis

When both `assertion_atoms is None` and `composition is None`,
`AugmentedBasis.kept_subspace(layer)` SHALL return
`(basis.W_dec, mask)` with `mask` all-`False`, and
`SubspaceProjector.project_module(host, augmented=that)` SHALL produce a
weight dict byte-identical to `project_module(host)` (no augmentation).
A forge run with both preserve toggles off SHALL pass the existing
`test_imperative_and_fsm_byte_equivalent` gate unmodified.

#### Scenario: null augmentation reproduces single-basis dict

- **GIVEN** a `tiny_gpt2` host and a committed single-basis reference dict
- **WHEN** `project_module(host, augmented=AugmentedBasis(basis, None, None))` runs
- **THEN** the output dict equals the reference dict key-for-key and value-for-value

### Requirement: circuit-faithfulness is reported alongside global KL

The run report under `--circuit-faithfulness` SHALL include
`KL(host ‖ forged)` restricted to a circuit token mask and to its
complement, for at least the `induction_predictable` mask, plus the
forged-residual assertion `cov95`. The shipping invariant SHALL be that
two-basis forge does not increase induction-predictable KL relative to
single-basis on matched bases and seed, and does not regress global KL
beyond the configured tolerance.

#### Scenario: induction-predictable KL is reported and separable

- **GIVEN** a forge run with `--circuit-faithfulness`
- **WHEN** the report is emitted
- **THEN** it contains `masked_kl`, `complement_kl`, and `n_masked` for the induction-predictable mask
- **AND** `circuit_kl` is `0.0` when forged logits equal host logits

#### Scenario: basis overlap is reported

- **GIVEN** a two-basis run on a production Polygram basis
- **WHEN** the report is emitted
- **THEN** it includes `dim(U_C ∩ S)/dim(U_C)` per layer
- **AND** a near-1.0 overlap is recorded as "basis already covers composition" rather than an error
