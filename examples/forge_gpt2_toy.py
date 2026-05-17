"""Toy GPT-2 forge against a synthetic 8-feature basis. CPU-friendly smoke target.

Build a tiny in-memory GPT-2 (16-embed, 2 layers, 4 heads, vocab 100), forge
it through a synthetic 8-feature basis, evaluate per-token KL on a held-out
input, and write the result tree to ``examples/output/toy/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main(output_dir: str | Path = "examples/output/toy/") -> dict:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    host_config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_inner=32,
    )
    host = GPT2LMHeadModel(host_config).eval()

    n_features = 8
    W_dec = rng.standard_normal((n_features, host_config.n_embd)).astype(np.float64)
    norms = np.linalg.norm(W_dec, axis=1)
    basis = FeatureBasis(
        kept_ids=np.arange(n_features),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )

    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(basis=basis, projector=projector, dtype="float32", device="cpu")

    eval_input_ids = torch.randint(0, host_config.vocab_size, (2, 16))
    result = pipeline.run_synthetic(host, output_dir, eval_input_ids=eval_input_ids)

    summary = {
        "n_params": result.n_params,
        "faithfulness": result.faithfulness,
        "faithfulness_target_name": result.faithfulness_target_name,
        # Back-compat: tests + downstream tooling still read
        # ``summary["faithfulness_kl"]``. Populated when the active
        # target is "kl"; null otherwise. Removed alongside
        # ForgeResult.faithfulness_kl.
        "faithfulness_kl": result.faithfulness if result.faithfulness_target_name == "kl" else None,
        "n_features": basis.n_features,
        "host_param_count": sum(p.numel() for p in host.parameters()),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
