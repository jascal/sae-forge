# subspace-projector Specification

## Purpose

Defines `SubspaceProjector` — pure-numpy projection helpers and the HF
GPT-2 weight walker. The projection algebra is documented in the module
docstring; the spec below pins the shape contracts and the GPT-2
parameter coverage.

## Requirements

### Requirement: Encode/decode are inverses on full row-rank bases

For any `FeatureBasis` whose `W_dec` is full row-rank (`rank == n_features`),
`projector.encode(projector.decode(z)) == z` for any `z` of shape
`(..., n_features)` within floating-point tolerance.

#### Scenario: 8-feature, 16-d basis round-trips

- **GIVEN** a `FeatureBasis` with shape `(8, 16)` and `scale_boost=1.0`
- **WHEN** `encode(decode(z))` is computed for a `(4, 8)` random `z`
- **THEN** the result equals `z` within `1e-9` absolute tolerance

### Requirement: Residual-input matrices project as `D @ W`

`project_residual_input(W)` SHALL transform a residual-input matrix
`W: (d_model, m)` into a basis-input matrix of shape `(n_features, m)`.
The identity `h_n @ project_residual_input(A) == h_d @ A` SHALL hold
for any `h_d` that lies in `span(W_dec)` (i.e., `h_d = h_n @ W_dec`).

#### Scenario: residual-input identity for h_d in span(D)

- **GIVEN** `h_n` shape `(3, n_features)`, `h_d = h_n @ W_dec`, and a
  random `A` of shape `(d_model, 12)`
- **WHEN** both `h_d @ A` and `h_n @ project_residual_input(A)` are
  computed
- **THEN** the two results are equal within `1e-9` absolute tolerance

### Requirement: Residual-output matrices project as `W @ E`

`project_residual_output(W)` SHALL transform `W: (m, d_model)` into
shape `(m, n_features)` by composing with the basis pseudoinverse
(equivalent to `encode(W)` over the second axis).

#### Scenario: residual-output shape

- **GIVEN** a basis with `n_features=8, d_model=16` and `W` of shape
  `(32, 16)`
- **WHEN** `project_residual_output(W)` is called
- **THEN** the result has shape `(32, 8)`

### Requirement: Bias and LN-aligned vectors project via the pseudoinverse

`project_residual_bias(b)` and `project_residual_aligned(v)` SHALL
both transform a `(d_model,)` vector into `(n_features,)` via
`v @ E`. The two helpers are semantically distinct (bias on residual
write vs LN scale/shift) but mathematically identical.

#### Scenario: shapes

- **GIVEN** a basis with `n_features=8, d_model=16` and a `(16,)` vector
- **WHEN** either helper is called
- **THEN** the result has shape `(8,)`

### Requirement: scale_boost amplifies encode linearly

A projector with `scale_boost=k` SHALL produce
`encode(x) == k * encode_with_unit_boost(x)`. `decode` and
`project_residual_input` SHALL be unaffected by `scale_boost` — only
encode-direction operations are amplified.

#### Scenario: scale_boost=2.5

- **GIVEN** the same basis, two projectors with `scale_boost=1.0` and
  `scale_boost=2.5` respectively
- **WHEN** the same input `x` is encoded by both
- **THEN** the second result equals `2.5` times the first within
  `1e-9` absolute tolerance

### Requirement: project_module covers every GPT-2 weight

For an HF `GPT2LMHeadModel` host model, `project_module` SHALL return a
dict containing exactly the following keys, each with the indicated
shape (where `f = n_features`, `d = n_embd`, `i = n_inner`,
`v = vocab_size`, `p = n_positions`):

```
transformer.wte.weight                 (v, f)
transformer.wpe.weight                 (p, f)
transformer.h.{0..n_layer-1}.ln_1.weight        (f,)
transformer.h.{...}.ln_1.bias                   (f,)
transformer.h.{...}.attn.c_attn.weight          (f, 3d)
transformer.h.{...}.attn.c_attn.bias            (3d,)
transformer.h.{...}.attn.c_proj.weight          (d, f)
transformer.h.{...}.attn.c_proj.bias            (f,)
transformer.h.{...}.ln_2.weight                 (f,)
transformer.h.{...}.ln_2.bias                   (f,)
transformer.h.{...}.mlp.c_fc.weight             (f, i)
transformer.h.{...}.mlp.c_fc.bias               (i,)
transformer.h.{...}.mlp.c_proj.weight           (i, f)
transformer.h.{...}.mlp.c_proj.bias             (f,)
transformer.ln_f.weight                (f,)
transformer.ln_f.bias                  (f,)
lm_head.weight                         (v, f)
```

For a bare `GPT2Model` (no LM head), the `lm_head.weight` key SHALL be
absent.

#### Scenario: tiny GPT-2 keyed exhaustively

- **GIVEN** a `GPT2LMHeadModel` with `n_embd=16, n_layer=2, n_head=4,
  vocab_size=100, n_positions=32` and a basis with `n_features=8,
  d_model=16`
- **WHEN** `project_module` is called
- **THEN** the dict has exactly the keys listed above
- **AND** every value has the shape listed above

### Requirement: Non-GPT-2 hosts raise NotImplementedError

`project_module` SHALL raise `NotImplementedError` for any
`host_model` that is not a `GPT2LMHeadModel` or `GPT2Model`. The
message SHALL contain `"GPT-2"` and the offending type's name.

#### Scenario: a non-HF object is rejected

- **GIVEN** a custom class `FakeBert`
- **WHEN** `project_module(FakeBert())` is called
- **THEN** `NotImplementedError` is raised whose message contains
  `"GPT-2"`

### Requirement: Lazy-import transformers, actionable error when missing

`project_module` SHALL lazy-import `transformers`. When `transformers`
is not installed, the call SHALL raise `ImportError` whose message
names the `[torch]` extra.
