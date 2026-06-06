# composition-subspace-preserve Specification

## MODIFIED Requirements

### Requirement: composition subspace is the host QK/OV residual geometry

The composition subspace `U_C` SHALL be the orthonormalised union of the
**OV-output row spaces of the circuit's writer heads**, not the aggregate
read+write geometry of the reader/capture layers.

For each writer head `(L, h)`, `OV = W_V^h W_O^h` and its written
subspace is `rowspace(OV)` (dimension ≤ head_dim). `extract_writer_subspace`
SHALL stack the writer heads' `OV` matrices, take the top-`rank` right
singular vectors, and return an orthonormal `U` (`||UᵀU − I||_F < 1e-5`).
The forged QK/OV SHALL reproduce the host on `span(U)` to projection
tolerance (the existing augmented-basis preserve mechanism, unchanged).

The writer heads SHALL be selected by `composition_heads`: an explicit
`(layer, head)` list, or a behavioral preset (`"prev-token"` /
`"duplicate-token"`) resolved by `saeforge.circuit_heads` on the eval
corpus (top-`k` heads by Δ=1 / same-token-earlier attention). The legacy
aggregate reader-layer geometry SHALL remain available as
`mode="reader-geometry"` (`composition_heads="all"`) and SHALL carry a
docstring note that it does NOT protect circuits.

The rationale pinned by this requirement (from the alive single-layer
GPT-2 forge, the lm-sae consumer): preserving the writers' OV-output
removes the circuit-specific forge tax (−111% of the induction-predictable
KL excess), where the same-budget reader geometry does not (−6%); and the
label-free attribution subspace (`∂loss/∂residual`) is ~orthogonal to the
writer subspace (overlap 0.05) and does NOT protect the circuit — so the
writer identification is required, with no functional substitute.

#### Scenario: U_C is the writers' OV-output, orthonormal and rank-bounded

- **GIVEN** writer heads `[(4,11),(2,2)]` and `rank=16`
- **WHEN** `extract_writer_subspace(host, writer_heads=…, rank=16)` runs
- **THEN** the returned `U` has shape `(d_model, r)` with `r <= 16`
- **AND** `||UᵀU − I||_F < 1e-5`
- **AND** each writer head's `OV` output projected onto `span(U)` reproduces it to `1e-6`

#### Scenario: preset resolves to behavioral writer heads

- **GIVEN** `composition_heads="prev-token"` and an eval corpus
- **WHEN** the pipeline builds `U_C`
- **THEN** `saeforge.circuit_heads.identify` returns the top Δ=1 attention heads
- **AND** `U_C` is built from those heads' OV-output via `extract_writer_subspace`
- **AND** the run report records the identified writer heads

#### Scenario: reader-geometry is opt-in and documented as weaker

- **GIVEN** `composition_heads="all"` (or `mode="reader-geometry"`)
- **WHEN** `U_C` is built
- **THEN** the legacy aggregate reader-layer geometry path is used
- **AND** its docstring/report notes it does not protect circuits

### Requirement: disabled toggles are byte-identical to single-basis

When both `assertion_atoms is None` and `composition is None`,
`AugmentedBasis.kept_subspace(layer)` SHALL return
`(basis.W_dec, all-False mask)`, and a forge run with both preserve
toggles off SHALL be byte-identical to the single-basis path and pass the
`test_imperative_and_fsm_byte_equivalent` gate unmodified. This contract
is unchanged by the writer-output redefinition of `U_C`.

#### Scenario: writer-output change does not affect the disabled path

- **GIVEN** `composition_preserve=False` and `assertion_preserve=False`
- **WHEN** the forge runs
- **THEN** no writer identification or subspace extraction occurs
- **AND** the output is byte-identical to the single-basis forge
