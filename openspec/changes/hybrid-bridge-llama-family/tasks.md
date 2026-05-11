## 1. LlamaTransformer bridge wiring

- [ ] 1.1 Add `_build_bridges(cfg)` static method to `LlamaTransformer` in `saeforge/adapters/llama.py`, mirroring the `ForgedGPT2.Transformer._build_bridges` shape. Returns `None` when `cfg.bridges` is False; returns an `nn.ModuleDict({"emb_mid": ..., "mid_lm": ...})` when True
- [ ] 1.2 Call `self.bridges = self._build_bridges(cfg)` at the end of `LlamaTransformer.__init__`
- [ ] 1.3 Update `LlamaTransformer.forward`: per-block loop applies `self.bridges["emb_mid"]` after block 0 and `self.bridges["mid_lm"]` after block `L-2`, gated by `len(self.layers) >= 3`. Matches the GPT-2 pattern exactly
- [ ] 1.4 No changes to `LlamaBlock`, `LlamaSelfAttention`, `SwiGLU_MLP`, or `ForgedLlama`; the wiring is contained to `LlamaTransformer`

## 2. Conftest fixtures

- [ ] 2.1 Bump `tiny_llama` fixture in `tests/conftest.py` from `num_hidden_layers=2` to `num_hidden_layers=4`. Confirm all existing tests using `tiny_llama` still pass
- [ ] 2.2 Add `tiny_qwen2_untied_4layer` fixture parallel to `tiny_gpt2_untied_4layer` — 4 layers, 16-dim residual, 4 heads, 2 KV heads, untied embeddings, qkv_bias=True (Qwen2 default)

## 3. Integration tests: Llama

- [ ] 3.1 New file `tests/integration/test_hybrid_bridge_llama.py`. Mirror the structure of `tests/integration/test_hybrid_bridge_gpt2.py`
- [ ] 3.2 `test_hybrid_forge_constructs_with_bridges`: build the forged module, assert `model.bridges` exists with `emb_mid` and `mid_lm` keys, assert state_dict contains `model.bridges.emb_mid.*` and `model.bridges.mid_lm.*` entries
- [ ] 3.3 `test_forward_pass_finite`: run a forward pass on the forged module, confirm output shape and finite values
- [ ] 3.4 `test_safetensors_round_trip`: save + load preserves bridge parameters bit-for-bit
- [ ] 3.5 `test_byte_equivalent_when_disabled`: `hybrid_bridge=False` on a Llama host produces the same state_dict as the pre-change path (no `bridges` keys)
- [ ] 3.6 `test_tied_embeddings_refused_at_run_time`: a tied-Llama host with `hybrid_bridge=True` raises the documented error
- [ ] 3.7 `TestZeroInitInversion`: `zero` init makes the `emb_mid` bridge output exactly zero; orthogonal init preserves Frobenius norm (matches the GPT-2 test pattern)

## 4. Integration tests: Qwen2

- [ ] 4.1 New file `tests/integration/test_hybrid_bridge_qwen2.py`. Same test set as Llama (smoke / round-trip / refusal / disabled / inversion), against `tiny_qwen2_untied_4layer`
- [ ] 4.2 Additionally confirm that `state_dict` contains both `model.bridges.*` keys AND the Qwen2 Q/K/V `bias` entries (validates the qkv_bias + bridges combination)

## 5. Cross-family registered_classes sanity

- [ ] 5.1 No change to `tests/test_architecture_adapters.py::test_registered_classes_contains_all_four_families`; the registry surface is unchanged

## 6. Docs

- [ ] 6.1 Append to `docs/hybrid_bridge_intel_gpt2.md`: a one-paragraph "Cross-family coverage" note pointing at the new Llama/Qwen2 integration tests as confirmation that the mechanism works beyond GPT-2
- [ ] 6.2 `CHANGELOG.md` `## [Unreleased]` `### Added` entry: "Hybrid-bridge insertion into the Llama-family native module forward path (Llama, Gemma-2, Qwen2). Closes the half-built state where `hybrid_bridge=True` was silently dropping bridges on non-GPT-2 hosts"

## 7. OpenSpec scaffolding

- [x] 7.1 `openspec/changes/hybrid-bridge-llama-family/proposal.md`
- [x] 7.2 `openspec/changes/hybrid-bridge-llama-family/design.md`
- [x] 7.3 `openspec/changes/hybrid-bridge-llama-family/tasks.md` (this file)
- [x] 7.4 `openspec/changes/hybrid-bridge-llama-family/specs/hybrid-bridge-llama-family/spec.md` (ADDED capability)
- [ ] 7.5 Run `openspec validate hybrid-bridge-llama-family --strict`; resolve any structural complaints

## 8. Validation matrix (pre-merge gates)

- [ ] 8.1 Full `pytest -q` passes on Python 3.11 + 3.12 with `[dev,intel,polygram,orca]` extras
- [ ] 8.2 The existing `tests/integration/test_hybrid_bridge_gpt2.py` continues to pass without modification (regression gate for GPT-2 path)
- [ ] 8.3 New Llama + Qwen2 integration tests pass
- [ ] 8.4 `ruff check saeforge tests examples` is clean
- [ ] 8.5 `openspec validate hybrid-bridge-llama-family --strict` passes

## 9. Deferred follow-ups

- [ ] 9.1 **Gemma-2 family integration test.** Mechanism works for Gemma-2 via the shared factory, but adding a real Gemma-2 hybrid integration test on Intel without weights cached doesn't add signal. Bundle with the T3 M4 Gemma-2-2B reproduction
- [ ] 9.2 **State-dict key normalization** (`forged-module-state-dict-normalization`). Unify the `transformer.bridges.*` vs `model.bridges.*` split. Out of scope here
- [ ] 9.3 **Cross-family comparison harness extension.** Extend `scripts/compare_single_vs_hybrid_gpt2.py` (rename to `compare_single_vs_hybrid.py`) to optionally target an untied Llama or Qwen2 host. Tracked as a separate ergonomic improvement
- [ ] 9.4 **`qwen3-dense-support`.** This change is the prerequisite. Once `hybrid-bridge-llama-family` is on `main`, the Qwen3 dense path can proceed without the silently-half-broken hybrid concern
