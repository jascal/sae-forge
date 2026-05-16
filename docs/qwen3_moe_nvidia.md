# Qwen3-MoE forge — NVIDIA validation guide

This document covers running [`scripts/smoke_qwen3_moe.py`](../scripts/smoke_qwen3_moe.py)
to validate the Qwen3-MoE forge path against a real host on NVIDIA hardware.

## Why NVIDIA only

Qwen3-MoE 30B-A3B-Base is ~60GB at bf16. Neither the project's `[intel]`
extra (capped at `transformers<4.50`, no Qwen3 import) nor an M4 Apple
Silicon box (insufficient memory for the host) can run the real forge.

The M4 surface validates the *mechanism* via synthetic small-MoE
configs (`tests/integration/test_qwen3_moe_adapter.py` runs a 3-layer
4-expert top-2 host); NVIDIA is where the real Qwen3-30B-A3B-Base
host gets forged end-to-end.

## Hardware tiers

| Hardware | Status |
|---|---|
| A100/H100 ≥80GB single GPU | Comfortable. Host loads to GPU; forge runs in-place. |
| 2× A100-40GB | Works with `device_map="auto"` (default — script auto-shards). |
| 1× A100-40GB or RTX 4090 (24GB) | Requires aggressive CPU offload. Pass `--device-map balanced_low_0` or accept slow CPU-offloaded experts. |
| CPU only | Not supported. Script exits with code 2. |

## Quick start

```bash
# In a fresh venv with the [torch] extra:
pip install -e ".[torch,polygram,orca]"
python -c "import transformers; print(transformers.__version__)"   # must be >= 4.51

# Run the smoke (default Qwen3-30B-A3B-Base):
python scripts/smoke_qwen3_moe.py
```

Expected output (numbers depend on the host's actual config):

```
transformers: 4.5x.x
torch: 2.x.x  CUDA devices: 1
  cuda:0  NVIDIA A100-SXM4-80GB  80.0GB

Loading Qwen/Qwen3-30B-A3B-Base (device_map=auto, dtype=bf16)...
  hidden=<H>, layers=<L>, experts=<E>, top_k=<K>, moe_inter=<M>, vocab=<V>
  adapter family: qwen3_moe (expect: qwen3_moe)

Walking host through 256-feature basis...
  walker OK: <L> gate keys, <L*E> per-expert MLP key sets (matches <L> layers × <E> experts)
  native cfg: family=qwen3_moe, num_experts=<E>, num_experts_per_tok=<K>, moe_intermediate_size=<M>, qk_norm=True, qkv_bias=False

Building forged native module...
  forged module has Qwen3MoEMLP on all <L> layers (each with <E> experts) and q_norm/k_norm on every block

Running forward pass (16 tokens)...
  forward output shape: (1, 16, <V>)
  output is finite: True

SMOKE OK
```

Exit code `0` on success, `1` on assertion failure (wrong family,
missing modules, NaN/Inf logits), `2` on environment failure
(transformers <4.51, no CUDA, gated host model, OOM).

## Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--host-model <id>` | `Qwen/Qwen3-30B-A3B-Base` | Substitute any future Qwen3-MoE release. |
| `--n-features N` | `256` | Random basis size. Larger = more expressive but more memory. |
| `--device-map S` | `"auto"` | `"balanced"` for multi-GPU equal sharding; `"cpu"` for debug. |
| `--max-seq-len N` | `16` | Token count for the forward pass. Keep small to limit activation memory. |
| `--log-expert-utilization` | off | Enable routing-collapse diagnostic. See below. |
| `--seed N` | `0` | RNG seed for the random basis. |

## Diagnosing routing collapse

The hybrid-bridge mechanism and the basis projection both perturb the
residual stream feeding the router. If the projection significantly
shifts the gate's input distribution, the "which experts fire"
pattern may drift from the host's — *routing collapse*.

The `--log-expert-utilization` flag instruments both the host and the
forged module's routers on the same prompt and prints a per-layer
top-K agreement rate:

```
Logging expert utilization (top-K agreement vs host)...
  layer  0: top-K set agreement 92.0%
  layer  1: top-K set agreement 88.5%
  layer  2: top-K set agreement 85.0%
  ...
  layer 47: top-K set agreement 41.0%
    WARNING: low top-K agreement at layer 47; routing may be collapsing under projection
```

Interpretation:

- **≥80% agreement**: routing is faithful. The forged module fires
  approximately the same experts as the host for the same input.
- **50-80% agreement**: routing is drifting. Forge fine-tune should
  recover, but worth monitoring. Consider larger `--n-features`.
- **<50% agreement**: routing has collapsed for this layer. The forged
  basis at this layer's residual is no longer a faithful representation
  of the host's router-input distribution. Investigate:
  - Increase `--n-features`
  - Try the hybrid-bridge mode (three-basis with embed/mid/lm-head
    anchors) — `--hybrid-bridge --basis-embed ... --basis-lm-head ...`
  - Use a basis trained at this specific layer's residual

## Common failure modes

**`FAIL (env): Qwen3-MoE not available in this transformers install (... need >= 4.51)`**
Upgrade: `pip install -U 'transformers>=4.51'`. If you're on the
`[intel]` extras, you cannot upgrade past 4.49 (the torch 2.2.2 cap);
use a different machine with the `[torch]` extra.

**`FAIL (env): CUDA is not available`**
The host model is too large for CPU. Either get a CUDA-capable box or
substitute a smaller `--host-model` if/when one becomes available.

**`FAIL (env): host model ... requires HF auth (gated)`**
Run `huggingface-cli login` first. Qwen3 models are typically open
(not gated), but a future gated variant would trip this.

**`FAIL (env): OOM loading Qwen/Qwen3-30B-A3B-Base`**
Insufficient GPU memory. Use `--device-map balanced_low_0` to spread
across multiple GPUs, or accept CPU-offloaded experts (slower but
functional).

**`FAIL: expected family=qwen3_moe, got llama`**
The `Qwen3MoEAdapter` didn't register. Check that
`import saeforge.adapters.qwen3_moe` doesn't raise. Most likely cause:
your sae-forge install predates the qwen3-moe-support change.

**`FAIL: logits contain NaN/Inf`**
Likely a scale_boost issue with `n_features << d_model`. The script
uses `scale_boost="auto"` to handle this, but if it still triggers,
try a larger `--n-features` (e.g. 512 or 1024).

## What this script does NOT test

- Trained Polygram SAE bases (uses a random basis for plumbing
  validation; real bases are a downstream concern)
- Fine-tune dynamics (the forge fine-tune recipe is opt-in via
  `orchestrator="fsm"` and not part of the smoke surface)
- Sliding-window attention (the forged module uses standard causal;
  long-context drift is accepted as `ε_attn` per
  [`docs/algorithm.md`](algorithm.md) §5)
- Multi-GPU performance characteristics (use the standard PyTorch
  profiling tools for that)

## Reporting results

If you run this on NVIDIA hardware against `Qwen/Qwen3-30B-A3B-Base`,
please paste the output (especially the `--log-expert-utilization`
trace) into an issue on the sae-forge repo under the
`qwen3-moe-validation` label. This is how the project gathers
real-host evidence; it's especially valuable for tuning the
compression-mode defaults and detecting routing-collapse regressions.
