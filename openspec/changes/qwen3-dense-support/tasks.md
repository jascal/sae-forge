## 1. NativeModelConfig: qk_norm field

- [ ] 1.1 Add `qk_norm: bool = False` to `NativeModelConfig` in `saeforge/model.py`. Default preserves Llama / Gemma-2 / Qwen2 behavior
- [ ] 1.2 Add `"qwen3"` to `_SUPPORTED_FAMILIES`
- [ ] 1.3 Extend `_build_torch_module`'s Llama-family branch to include `"qwen3"` (routes to `build_llama_family_module`)

## 2. LlamaSelfAttention conditional q_norm/k_norm

- [ ] 2.1 In `saeforge/adapters/llama.py`, modify the `LlamaSelfAttention` class inside `_get_forged_llama_class`:
  - `__init__`: when `cfg.qk_norm` is True, construct `self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps or 1e-6)` and `self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps or 1e-6)`. When False (the default), set both to `None`
  - `forward`: after the Q/K reshape (`q = self.q_proj(x).view(...).transpose(-3, -2)` and the K analog), insert `if self.q_norm is not None: q = self.q_norm(q); k = self.k_norm(k)`. The change is two lines + the conditional
- [ ] 2.2 No changes to V projection, o_proj, attention mask, or scaled dot-product

## 3. Walker emits q_norm/k_norm pass-through

- [ ] 3.1 In `LlamaAdapter.walk` (`saeforge/adapters/llama.py`), inside the per-block loop, add a pass-through emit symmetric to the existing Q/K/V bias loop:
  ```python
  for qk in ("q_norm", "k_norm"):
      norm = getattr(block.self_attn, qk, None)
      if norm is not None:
          out[f"{prefix}.self_attn.{qk}.weight"] = to_numpy(norm.weight)
  ```
- [ ] 3.2 Guard is host-attribute-driven (`getattr(..., None) is not None`). Llama / Gemma-2 / Qwen2 hosts skip the emit; Qwen3 hosts emit
- [ ] 3.3 Confirm the existing `test_walker_rmsnorm_has_no_bias` (`tests/test_architecture_adapters.py`) still passes for Llama (no q_norm/k_norm submodules → no new keys)

## 4. Qwen3Adapter

- [ ] 4.1 New module `saeforge/adapters/qwen3.py`. `Qwen3Adapter(Qwen2Adapter)` with `family = "qwen3"`
- [ ] 4.2 Override `build_native_config`: call `super().build_native_config()`, then `replace(base, family="qwen3", qk_norm=qk_norm)` where `qk_norm` is detected from `host.model.layers[0].self_attn.q_norm is not None`
- [ ] 4.3 Walker is inherited from LlamaAdapter via Qwen2Adapter (no override)
- [ ] 4.4 Register `Qwen3ForCausalLM` in the `try`/`except` block at the bottom of the module — silent skip when transformers < 4.51 (no Qwen3)

## 5. Adapter registration

- [ ] 5.1 Append `from saeforge.adapters import qwen3 as _qwen3` to `saeforge/adapters/__init__.py`
- [ ] 5.2 Verify the import doesn't fail when `Qwen3ForCausalLM` isn't available — the module-level try/except handles it

## 6. Conftest fixture

- [ ] 6.1 New `tiny_qwen3_untied_4layer` fixture in `tests/conftest.py`, parallel to `tiny_qwen2_untied_4layer`. Uses `pytest.importorskip("transformers", minversion="4.51")` to skip gracefully on Intel installs
- [ ] 6.2 Config: 128-d residual, 4 layers, 4 heads, 2 KV heads (GQA), head_dim=32, intermediate=256, vocab=1024, max_pos=64, `tie_word_embeddings=False`

## 7. Unit tests: tests/test_qwen3_adapter.py (new)

- [ ] 7.1 Module-level guard: `pytest.importorskip("transformers", minversion="4.51")` — entire file skips if Qwen3 isn't available
- [ ] 7.2 `test_qwen3_dispatches_to_qwen3_adapter` — `adapter_for(tiny_qwen3).family == "qwen3"`
- [ ] 7.3 `test_qwen3_walker_emits_qk_norms` — every block has `model.layers.{i}.self_attn.q_norm.weight` and `model.layers.{i}.self_attn.k_norm.weight` in the walk output, each shape `(head_dim,)`
- [ ] 7.4 `test_qwen3_walker_omits_qkv_biases` — confirms the inherited bias auto-detection: Qwen3 hosts have no `q_proj.bias`, so the walker emits no `*.self_attn.q_proj.bias` keys
- [ ] 7.5 `test_qwen3_native_config_sets_qk_norm_true` — `build_native_config` returns a config with `family="qwen3"` and `qk_norm=True` and `qkv_bias=False`
- [ ] 7.6 `test_llama_qwen2_native_configs_keep_qk_norm_false` — sanity check that the new field defaults False for the other Llama-family hosts (regression gate)

## 8. Integration tests: tests/integration/test_hybrid_bridge_qwen3.py (new)

- [ ] 8.1 Module-level guard: `pytest.importorskip("transformers", minversion="4.51")`
- [ ] 8.2 Mirror the existing `test_hybrid_bridge_qwen2.py` shape:
  - `TestT0TinyQwen3Smoke` — bridges in state_dict (`model.bridges.emb_mid.*` / `model.bridges.mid_lm.*`); forward pass finite; safetensors round-trip preserves both bridges AND q_norm/k_norm weights bit-for-bit
  - `TestByteEquivalenceWhenDisabled` — `hybrid_bridge=False` leaves the state_dict without bridge keys; q_norm/k_norm weights are still present on the single-basis path
- [ ] 8.3 `TestQKNormCorrectness::test_q_norm_k_norm_in_state_dict` — explicit assertion that the forged module's state_dict contains the q_norm/k_norm weights with the right shape

## 9. End-to-end numerical correctness (deferred to T2 M4 smoke)

- [ ] 9.1 On the user's M4 box (transformers ≥ 4.51 confirmed available), run:
  ```
  python -c "from saeforge.adapters import adapter_for; from transformers import Qwen3ForCausalLM; ..."
  ```
  to load a real `Qwen3-0.6B`, build a 64-feature random basis, project + construct + run one forward pass, confirm logits are finite and shape-correct
- [ ] 9.2 Paste the M4 smoke output into the PR description before merge
- [ ] 9.3 Optionally extend `scripts/compare_single_vs_hybrid_gpt2.py` to accept `--host-model` and rerun the single-vs-hybrid comparison on `Qwen3-0.6B` for a one-time data point. Tracked as a separate ergonomic follow-up; not blocking

## 10. Docs

- [ ] 10.1 `CHANGELOG.md` `## [Unreleased]` `### Added` entry: "Qwen3 dense architecture adapter (Q/K per-head RMSNorm). Requires `transformers >= 4.51`; silently skipped under the `[intel]` extras which are capped at 4.49"
- [ ] 10.2 Update `docs/forge_layer_choice.md` (if it has an "Architecture coverage" table) — or add such a table — listing the supported families and their transformers-version requirements
- [ ] 10.3 Brief mention in `docs/hybrid_bridge_intel_gpt2.md` "Cross-family coverage" section noting Qwen3 is now wired and points at the new Qwen3 integration test

## 11. OpenSpec scaffolding

- [x] 11.1 `openspec/changes/qwen3-dense-support/proposal.md`
- [x] 11.2 `openspec/changes/qwen3-dense-support/design.md`
- [x] 11.3 `openspec/changes/qwen3-dense-support/tasks.md` (this file)
- [x] 11.4 `openspec/changes/qwen3-dense-support/specs/qwen3-dense-support/spec.md`
- [ ] 11.5 Run `openspec validate qwen3-dense-support --strict`; resolve any structural complaints

## 12. Pre-merge gates

- [ ] 12.1 `pytest -q` passes on the Intel surface (all Qwen3 tests skip cleanly via importorskip)
- [ ] 12.2 `ruff check saeforge tests examples` is clean
- [ ] 12.3 GitHub Actions CI green (test 3.11 + test 3.12, both skip Qwen3 tests since `[dev]` doesn't install transformers)
- [ ] 12.4 The existing Qwen2 / Llama / Gemma-2 / GPT-2 integration tests pass without modification (regression gate)
- [ ] 12.5 T2 M4 smoke output pasted into the PR description before merge (the load-bearing real-Qwen3 confirmation)

## 13. Deferred follow-ups

- [ ] 13.1 **`qwen3-moe-support`** — adds Q-router + per-expert SwiGLU decomposition, expert-pruning compression modes, routing-collapse diagnostics. Builds on this change
- [ ] 13.2 **T3 NVIDIA/CUDA validation** — community runs of the comparison harness on larger Qwen3 dense (4B, 8B) hosts. Add a section to `docs/hybrid_bridge_cuda_validation_request.md` (which tasks.md §11 of the parent change scoped) describing the Qwen3 setup
- [ ] 13.3 **Cross-family comparison harness extension** — rename and generalize `scripts/compare_single_vs_hybrid_gpt2.py` to accept `--host-model` and untie automatically if needed
- [ ] 13.4 **Removing the `[intel]` extras Qwen3 blind spot** — eventually drop the torch 2.2.2 / transformers <4.50 pin and replace with a CPU-only `[torch]` install. Bigger structural decision, out of scope here
