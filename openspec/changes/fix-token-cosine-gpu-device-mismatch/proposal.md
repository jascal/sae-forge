## Why

`TokenCosineTarget.score` (`saeforge/eval/targets/token_cosine.py`) crashes with a
device mismatch whenever the forge runs on GPU while the host model stays on CPU —
the default path for `ForgePipeline(device="cuda", …)` driving an encoder host
(e.g. ESM-2):

```
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on
cuda:0, different from other tensors on cpu
  (token_cosine.py:105, host_hidden @ basis_encode)
```

**Root cause.** When the host and forged hidden states have different widths, the
target projects the host hidden state into the forged basis via the forged
module's `basis_encode` buffer:

```python
host_hidden  = host_hidden[:, 1:-1, :].to(forged_hidden.dtype)        # dtype only
basis_encode = forged_module.basis_encode.to(forged_hidden.dtype)     # dtype only
host_hidden  = host_hidden @ basis_encode                             # <-- mismatch
```

`host_hidden` is extracted on the host device (`host.device`, which is CPU when
the host isn't moved) and only ever cast to `forged_hidden.dtype`, never to
`forged_hidden.device`. `forged_module.basis_encode` lives on the forged module's
device (CUDA). The matmul therefore mixes a CPU operand with a CUDA operand.

**Impact.** Blocks `scripts/forge_pipeline.py --mode polygram --device cuda` (it
crashes in stage 4, *after* the compressed checkpoint is already written in stage
3) and any GPU forge that uses the token-cosine faithfulness target with a
width-mismatched basis. CPU runs are unaffected, which is why the existing
CPU-smoke tests don't catch it. Surfaced from bio-sae's whole-loop closure work
(forging ESM-2 on an RTX 5050).

## What Changes

### Scope

Align operand devices in `TokenCosineTarget.score` before the projection matmul
and the downstream flat-cosine comparison: move the host hidden state (and the
projected `host_flat`) onto the forged/eval device so every operand in the cosine
computation shares a device, regardless of where the host model was loaded.

### Modified Capabilities

- **token-cosine faithfulness scoring** — device-correct on GPU. The returned
  score on CPU is unchanged (byte-identical); no public API change.

Out of scope: the broader "cosine is the wrong faithfulness question" critique
(tracked separately). This change only makes the target *runnable* on GPU.
