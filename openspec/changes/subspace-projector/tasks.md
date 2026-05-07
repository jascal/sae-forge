## 1. Pure-numpy core

- [x] 1.1 Document the projection algebra in the module docstring with the four projection identities and the LN-not-equivariant caveat
- [x] 1.2 Implement `project_residual_input`, `project_residual_output`, `project_residual_bias`, `project_residual_aligned`, `project_embed`, `project_unembed`, `project_qkv`, `project_mlp_in`, `project_mlp_out` as thin numpy wrappers
- [x] 1.3 Keep `encode`/`decode` as the canonical entry points; everything else routes through them
- [x] 1.4 Wire `scale_boost` into `encode` only — projection-into-residual paths (decode, residual_input) are not amplified

## 2. GPT-2 host walker

- [x] 2.1 Implement `project_module(host_model)` accepting `GPT2LMHeadModel` and `GPT2Model`
- [x] 2.2 Walk wte, wpe, every block (ln_1, attn.c_attn, attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj), ln_f, lm_head; return a flat dict keyed by HF parameter names
- [x] 2.3 Lazy-import `transformers`; raise actionable `ImportError` naming the `[torch]` extra
- [x] 2.4 Raise `NotImplementedError` for non-GPT-2 architectures with a message naming the type

## 3. Tests

- [x] 3.1 Add `tiny_gpt2` fixture in conftest (n_embd=16, n_layer=2, n_head=4, vocab=100)
- [x] 3.2 Test encode/decode round-trip on full-rank basis
- [x] 3.3 Test shape contract on every helper
- [x] 3.4 Test `scale_boost` amplifies encode linearly
- [x] 3.5 Test the residual-inverse identity: `h_n @ project_residual_input(A) == h_d @ A` when `h_d ∈ span(D)`
- [x] 3.6 Test the GPT-2 walker produces every expected key with the right shape
- [x] 3.7 Test unsupported-architecture raise

## 4. OpenSpec scaffolding

- [x] 4.1 `openspec/changes/subspace-projector/proposal.md`
- [x] 4.2 `openspec/changes/subspace-projector/tasks.md` (this file)
- [x] 4.3 `openspec/changes/subspace-projector/specs/subspace-projector/spec.md`
