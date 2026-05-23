# whisper-base — not yet supported by n-orca v0.1

**Model:** [`openai/whisper-base`](https://huggingface.co/openai/whisper-base)

Encoder-decoder speech model — n-orca v0 does not yet model encoder-decoder topologies; would need a `WhisperAdapter` covering log-mel CNN stem + encoder + cross-attention decoder.

Once an adapter is added under `n_orca/hf/adapters/`, regenerate this
directory with:

```bash
.venv/bin/python scripts/generate_sibling_docs.py
```
