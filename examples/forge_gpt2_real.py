"""Real GPT-2 forge against a synthetic 256-feature basis. Pipeline smoke test.

Runs the v0 imperative ForgePipeline end-to-end on the canonical 124M-param
GPT-2 from HuggingFace plus a *synthetic* random 256-feature basis. This
tells us the projector + native model + faithfulness eval all scale from
the toy 16-d case to production dimensions without integration bugs.

Faithfulness will be high (the basis is random, not a real SAE), but the
plumbing — weight projection at vocab=50257 / n_layer=12 / d=768, model
construction at hidden_size=256, KL eval against real tokenized prompts —
runs to completion.

Run:
    python examples/forge_gpt2_real.py [output_dir]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def main(output_dir: str | Path = "examples/output/gpt2_real/", n_features: int = 256) -> dict:
    import torch  # noqa: F401  (lazy-imported by saeforge.utils.lazy)

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    d_model = 768  # gpt2's n_embd
    print(f"building synthetic basis: n_features={n_features}, d_model={d_model}")
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float64)
    norms = np.linalg.norm(W_dec, axis=1)
    basis = FeatureBasis(
        kept_ids=np.arange(n_features),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )
    projector = SubspaceProjector(basis)

    eval_prompts = [
        "The quick brown fox jumps over the lazy",
        "In a hole in the ground there lived a",
        "It was the best of times, it was",
        "To be or not to be, that is the",
    ]

    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id="gpt2",
        eval_prompts=eval_prompts,
        dtype="float32",
        device="cpu",
    )

    print("forging (this loads gpt2 ~500MB on first run, projects every weight, runs KL eval)")
    t0 = time.monotonic()
    result = pipeline.run(output_dir)
    wall = time.monotonic() - t0

    summary = {
        "n_features": basis.n_features,
        "d_model": basis.d_model,
        "n_params_forged": result.n_params,
        "faithfulness_kl": result.faithfulness,
        "wall_clock_s": round(wall, 2),
        "output_dir": str(result.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/output/gpt2_real/"
    n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    main(out, n_features=n_features)
