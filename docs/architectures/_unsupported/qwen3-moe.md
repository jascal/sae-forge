# qwen3-moe — not yet supported by n-orca v0.1

**Model:** [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)

Qwen3 mixture-of-experts decoder — n-orca v0 represents FFN as a dense block; a `MoeFeedForward` op (with `n_experts` and `top_k` parameters) would be the minimum required addition.

Once an adapter is added under `n_orca/hf/adapters/`, regenerate this
directory with:

```bash
.venv/bin/python scripts/generate_sibling_docs.py
```
