## 1. Reproduce

- [ ] 1.1 Add a GPU-guarded regression test in `tests/eval/test_token_cosine.py` (skip when `not torch.cuda.is_available()`): forge a width-mismatched basis, move the forged module to `cuda` while leaving the host on `cpu`, call `TokenCosineTarget.score`. Assert it currently raises the device-mismatch `RuntimeError` (then passes after the fix).

## 2. Fix

- [ ] 2.1 In `saeforge/eval/targets/token_cosine.py::score`, change the dtype-only cast of the host hidden state (the `host_hidden[:, 1:-1, :].to(forged_hidden.dtype)` line) to also set `device=forged_hidden.device`, so `host_hidden @ basis_encode` and the downstream `host_flat`/`forged_flat` comparison share a device.
- [ ] 2.2 Build `basis_encode` defensively as `forged_module.basis_encode.to(device=forged_hidden.device, dtype=forged_hidden.dtype)` so the fix holds even if the host-hidden device path changes later.
- [ ] 2.3 Confirm the same-width path (no `basis_encode` projection) is also device-safe — `host_flat` and `forged_flat` must be on one device before the cosine.

## 3. Verify

- [ ] 3.1 New GPU regression test passes; existing CPU token-cosine tests unchanged (byte-identical score on CPU).
- [ ] 3.2 Manual: `scripts/forge_pipeline.py --mode polygram --device cuda` on an ESM-2 host (`esm2_t6_8M`) completes stage 4 without the device error (the original repro from bio-sae's whole-loop work).
