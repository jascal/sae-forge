## 1. Reproduce

- [x] 1.1 Add a GPU-guarded regression test (skip when `not torch.cuda.is_available()`): forge a width-mismatched basis, move the forged module to `cuda` while leaving the host on `cpu`, call `TokenCosineTarget.score`. Assert it currently raises the device-mismatch `RuntimeError` (then passes after the fix). Landed as `test_token_cosine_score_gpu_host_on_cpu` in `tests/test_esm2_adapter.py` (reuses the ESM helpers there; repo uses a flat `tests/` layout, not `tests/eval/`). Verified to raise the exact `mat2 is on cuda:0 … other tensors on cpu` error before the fix and pass after.

## 2. Fix

- [x] 2.1 In `saeforge/eval/targets/token_cosine.py::score`, changed the dtype-only cast of the host hidden state to `host_hidden[:, 1:-1, :].to(device=forged_hidden.device, dtype=forged_hidden.dtype)`, so `host_hidden @ basis_encode` and the downstream `host_flat`/`forged_flat` comparison share a device.
- [x] 2.2 Build `basis_encode` defensively as `forged_module.basis_encode.to(device=forged_hidden.device, dtype=forged_hidden.dtype)` so the fix holds even if the host-hidden device path changes later.
- [x] 2.3 Same-width path is device-safe: the host hidden state is moved onto `forged_hidden.device` *before* the shape branch, so `host_flat`/`forged_flat` are co-located whether or not the `basis_encode` projection runs. Pinned by `test_token_cosine_width_mismatch_runs_on_cpu` (CPU projection path) and the identity-basis same-width test.

## 3. Verify

- [x] 3.1 New GPU regression test passes; existing CPU token-cosine tests unchanged (identity-basis cosine still == 1.0). Full eval/faithfulness suite green: `pytest tests/test_esm2_adapter.py tests/test_evaluate_faithfulness_dispatch.py tests/test_faithfulness_target_protocol.py tests/test_audio_eval.py tests/test_downstream_capability_target.py tests/test_gt_alignment_target.py` → 80 passed, 2 skipped. `ruff check` clean.
- [ ] 3.2 Manual end-to-end: `scripts/forge_pipeline.py --mode polygram --device cuda` on an ESM-2 host (`esm2_t6_8M`) completes stage 4 without the device error. **Pending a sae-forge release + bio-sae env bump** — bio-sae imports `saeforge` from site-packages (the traceback is in `.venv/.../site-packages/saeforge/`), so it won't pick up this fix until sae-forge is published and bio-sae reinstalls. The unit-level regression (1.1) reproduces and fixes the identical `RuntimeError` in-tree.

## 4. Audit (added — answers "any other scripts affected?")

- [x] 4.1 Audited all five faithfulness targets + their delegated kernels. `TokenCosineTarget` is the only one that runs the host on `host.device` (token_cosine.py:76-77) and then mixes host/forged tensors — hence the only one with this bug. `KLTarget`→`_kl_from_input_ids` (forge.py:1592-1594) and `CosineTarget`→`cosine_faithfulness` (audio_eval.py:97-107) move *both* host and forged onto the eval `device`; `DownstreamCapabilityTarget` and `GroundTruthTarget` move forged to `device` and never read the host. The two dtype-only casts elsewhere (esm2.py:385, llama.py:484) operate within a single module forward (operands already co-located). No other site needs a fix.
