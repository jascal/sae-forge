## MODIFIED Requirements

### Requirement: project_module covers every GPT-2 weight

`SubspaceProjector.project_module(host_model, *, attention_width="host", hybrid=None)` SHALL gain an
optional `hybrid: HybridBasisBundle | None = None` keyword
argument. The argument SHALL default to `None`, in which case the
existing single-basis dispatch and shape contract are preserved
byte-identically.

When `hybrid is None` (the v0 default), the function's behavior,
output dict keys, and tensor shapes for every supported host
(`GPT2LMHeadModel`, `GPT2Model`, Llama-family, Gemma-2-family)
SHALL match the pre-change contract exactly. The
`test_imperative_and_fsm_byte_equivalent` byte-equivalence gate
SHALL continue to pass unmodified.

When `hybrid is not None`, the function SHALL dispatch via
`saeforge.adapters._hybrid.walk_hybrid(host, adapter, self,
bundle=hybrid, attention_width=attention_width)`. The returned
dict SHALL have the same set of keys as the single-basis path for
the same host. The per-key tensor shapes SHALL match the single-
basis contract — using the `n_features` shared by all three bases
in the bundle (validated at bundle construction; see
`hybrid-bridge-forge` capability spec).

#### Scenario: hybrid=None preserves single-basis output

- **GIVEN** a `tiny_gpt2` fixture (n_embd=16, n_layer=2, n_head=4, vocab=100)
- **AND** an 8-feature `FeatureBasis` over the same `d_model=16`
- **WHEN** `SubspaceProjector(basis).project_module(host, hybrid=None)` is called
- **THEN** the returned dict has the same keys as the pre-change call
- **AND** every per-key tensor is byte-identical to the pre-change output

#### Scenario: hybrid dispatch returns the same key set

- **GIVEN** an untied `gpt2` host (n_layer=12) and a 3-basis bundle
  with `n_features=64` and shape-matching `d_model=768`
- **WHEN** `SubspaceProjector(basis_mid).project_module(host, hybrid=bundle)` is called
- **THEN** the returned dict has the same set of keys as the
  single-basis path on the same host
- **AND** the per-key shapes are the same as the single-basis path
  (with `f=64` substituted everywhere `n_features` appears in the
  shape contract)
