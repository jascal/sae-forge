## Why

`SubspaceProjector` is the v0 component that turns a `FeatureBasis`
plus a host model into a dictionary of projected weights. The
bootstrap change shipped the pure-numpy core methods (encode, decode,
shape helpers) and a `project_module(host_model)` stub. This change
implements the host walker for HF GPT-2.

Why GPT-2 first: it's the smoke target for the v0 milestone and the
toy host for the v0 forge example. Other architectures (Llama 2/3,
Gemma 2, Mistral) follow the same projection algebra but differ in
parameter naming and bias presence; landing them is a separate change
once the algebra is exercised end-to-end.

## What Changes

- Document the projection algebra in the module docstring: residual-input
  matrices use `D @ W` (where `D` is `W_dec`), residual-output matrices
  use `W @ E` (where `E` is the pseudoinverse), residual biases use
  `b @ E`, residual-aligned scale/shift vectors (LN γ, β) use the same
  pseudoinverse projection. Layer norm is not equivariant under linear
  projection — γ/β projection is a documented v0 lossy fallback,
  tracked by the forge-pipeline KL eval.
- Implement `project_residual_input`, `project_residual_output`,
  `project_residual_bias`, `project_residual_aligned`,
  `project_embed`, `project_unembed`, `project_qkv`, `project_mlp_in`,
  `project_mlp_out` as thin numpy wrappers.
- Implement `project_module(host_model)` for HF GPT-2:
  - Accepts `GPT2LMHeadModel` (returns lm_head weight) or `GPT2Model`
    (no lm_head).
  - Walks `transformer.{wte, wpe}`, every `transformer.h.{i}.{ln_1,
    attn.c_attn, attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj}`,
    `transformer.ln_f`, `lm_head` (when present).
  - Returns a dict keyed by HF parameter names whose values are
    `np.ndarray` projected weights, ready for
    `NativeModel.from_projected_weights`.
  - Lazy-imports `transformers`; raises a clear actionable
    `ImportError` naming the `[torch]` extra when missing.
  - Raises `NotImplementedError` for non-GPT-2 architectures.

## Capabilities

### New Capabilities

- `subspace-projector-core`: Pure-numpy projection helpers
  (`encode`, `decode`, `project_residual_input`, etc.) with documented
  shape contracts and a tested round-trip identity for full row-rank
  bases.
- `subspace-projector-gpt2-walker`: `project_module(host_model)` for
  `GPT2LMHeadModel` / `GPT2Model` — produces a flat dict of projected
  weights keyed by HF parameter names, with shapes matching the
  basis-width residual stream.

### Modified Capabilities

- `bootstrap`: the "scale_boost rejected when non-positive" scenario
  remains valid; no other bootstrap surface changes.

## Impact

- `saeforge/projector.py`: pure-numpy core helpers and the GPT-2
  walker. ~110 lines, all the projection algebra in one place.
- `tests/test_subspace_projector.py`: 9 tests covering encode/decode
  round-trip, shape contracts on every helper, scale-boost
  amplification, the residual-inverse identity (`h_n @ project(A) ==
  h_d @ A` when `h_d ∈ span(D)`), end-to-end GPT-2 walker shape
  audit, unsupported-architecture raise.
- `tests/conftest.py`: new `tiny_gpt2` fixture (16-embed, 2-layer,
  4-head, vocab 100) — small enough to construct in milliseconds.
- No public-API change beyond filling in the stub.
