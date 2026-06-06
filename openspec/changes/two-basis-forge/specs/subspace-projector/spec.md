# subspace-projector Specification

## MODIFIED Requirements

### Requirement: project_module covers every GPT-2 weight

`SubspaceProjector.project_module(host_model, *, attention_width="host", augmented=None)`
SHALL gain an optional `augmented: AugmentedBasis | None = None`
keyword argument. The argument SHALL default to `None`, in which case
the existing single-basis dispatch, the `D @ W` / `W @ E` projection
algebra, and the output dict shape contract (the full GPT-2 key/shape
table) are preserved byte-identically.

When `augmented is None` (the v0 default), the function's behavior,
output dict keys, and tensor shapes for every supported host
(`GPT2LMHeadModel`, `GPT2Model`, Llama-family, Gemma-2-family) SHALL
match the pre-change contract exactly. The
`test_imperative_and_fsm_byte_equivalent` byte-equivalence gate SHALL
continue to pass unmodified.

When `augmented is not None`, the function SHALL project each emitted
weight through that weight's layer kept subspace
(`augmented.kept_subspace(layer)`, layer attributed via the adapter's
`layer_index_for`) and SHALL write the rows marked by the returned
`preserve_mask` **verbatim from the host** instead of through the
Polygram-merged decode, so the assertion (`U_A`) and composition
(`U_C`) subspaces are reproduced exactly. Weights with no block layer
(`wte`, `wpe`, `lm_head`, `ln_f`) SHALL receive the assertion preserve
but no composition augmentation.

The returned dict SHALL have the same set of keys and the same per-key
tensor shapes as the single-basis path for the same host — augmentation
changes *which subspace is kept and which rows are verbatim*, never the
set of projected weights or their shapes.

#### Scenario: augmented=None preserves single-basis output

- **GIVEN** a `tiny_gpt2` fixture (n_embd=16, n_layer=2, n_head=4, vocab=100)
- **AND** an 8-feature `FeatureBasis` over the same `d_model=16`
- **WHEN** `SubspaceProjector(basis).project_module(host, augmented=None)` is called
- **THEN** the returned dict has the same keys as the pre-change call
- **AND** every per-key tensor is byte-identical to the pre-change output

#### Scenario: augmented path preserves host QK/OV on the composition subspace

- **GIVEN** an `AugmentedBasis` whose `U_C[ℓ]` is host layer `ℓ`'s QK/OV geometry
- **WHEN** `project_module(host, augmented=that)` is called
- **THEN** the forged `attn.c_attn` / `attn.c_proj` weights reproduce the host `M_h` and `OV_h` action on `span(U_C[ℓ])` to projection tolerance

#### Scenario: output keys and shapes are augmentation-invariant

- **GIVEN** a host and a `FeatureBasis`
- **WHEN** `project_module` is called with `augmented=None` and with a populated `AugmentedBasis`
- **THEN** both output dicts have the same set of keys
- **AND** every shared key has the same array shape in both
