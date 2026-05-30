# Tasks — NanochatAdapter

## 1. Bring-up (no acceptance gate yet)

- [ ] 1.1. Create `saeforge/adapters/nanochat.py` with a `NanochatAdapter(ArchitectureAdapter)` class. `family = "nanochat"`.
- [ ] 1.2. Implement `walk()` per the Design weight inventory. Handle: wte, value_embeds, c_q/c_k/c_v separate, c_proj, mlp.c_fc/c_proj, ve_gate, lm_head, resid_lambdas/x0_lambdas. Skip cos/sin (buffers, regenerated).
- [ ] 1.3. Implement `build_native_config()` that maps lm-sae's `GPTConfig` dataclass fields to `NativeModelConfig`.
- [ ] 1.4. Implement `native_module_class()` returning a forged module that mirrors lm-sae's `GPT.forward`. Subclassing lm-sae's GPT is acceptable.
- [ ] 1.5. Implement `default_faithfulness_target()` returning `TokenCosineTarget`.
- [ ] 1.6. Lazy import of lm-sae's `GPT` inside `_register()`; skip registration silently if lm-sae isn't installed.
- [ ] 1.7. Register the import in `saeforge.adapters.__init__`.

## 2. Tests

- [ ] 2.1. `tests/test_nanochat_adapter.py::test_walk_keys` — instantiate a tiny lm-sae `GPT` (depth=1, n_embd=64, vocab=128), build a tiny `SubspaceProjector`, run `walk()`. Assert the output dict contains exactly the expected key set.
- [ ] 2.2. `tests/test_nanochat_adapter.py::test_forged_module_runs` — instantiate the forged module, run a forward on a (B=2, T=16) input, assert output shape `(B, T, n_features)` (or `(B, T, vocab)` if `lm_head` is in the walk).
- [ ] 2.3. `tests/test_nanochat_adapter.py::test_registry_dispatch` — `adapter_for(lm_sae_gpt)` returns a `NanochatAdapter` instance.
- [ ] 2.4. Skip-on-missing: all three tests `pytest.importorskip("lm_sae")`.

## 3. Acceptance gate

- [ ] 3.1. Run `ForgePipeline.run_synthetic` against an lm-sae layer-1 TopK SAE with token-identity probes (the 8192 BPE tokens themselves). Acceptance: mean retained-AUC > 0.55 over the probes that have at least 50 positives in the eval window. Absolute value doesn't matter; existence above chance does.
- [ ] 3.2. Run the same workflow as a tiny regression test (parametrised on B=4, T=128, num_features=128, num_eval_batches=16). Latency budget: under 60s on the CI machine.

## 4. lm-sae packaging coupling (cross-repo)

- [ ] 4.1. lm-sae renames the installable name from `lm-sae` (top-level modules) to `lm_sae` (importable package). The two files `train.py` and `prepare.py` become `lm_sae/train.py` and `lm_sae/prepare.py`. Update `pyproject.toml`'s `[tool.setuptools] py-modules` → `[tool.setuptools.packages.find]`. Sibling consumers (this adapter, future SAE training scripts) import from `lm_sae.train`.
- [ ] 4.2. Document the import-time gate (`if __name__ == "__main__":` wrap of the runtime block) as a stable contract in lm-sae's README. Any future agent-driven train.py edits must preserve it.

## 5. Deferred (separate proposals if needed)

- [ ] 5.1. GQA path (`n_kv_head < n_head`).
- [ ] 5.2. Sliding-window attention (if lm-sae re-enables it on Apple Silicon).
- [ ] 5.3. `attention_width="feature_native"` parity with GPT-2 adapter.
- [ ] 5.4. Step-budget primitive in `sweep_pareto_capability` (currently data-scale-based; trajectory experiments need step-scale).
