"""Forge with the built-in :class:`saeforge.eval.GroundTruthTarget`.

This example shows how to gate a forge run on per-sample ground-truth
labels instead of LM perplexity. The setup is a 3-cluster
mixture-of-gaussians where each eval row carries a one-hot cluster ID;
``GroundTruthTarget`` then computes per-feature × per-label AUC, takes
the best-matching feature per label, averages, and returns that to the
FSM's faithfulness gate.

The example uses an explicit ``hidden_extractor=`` that returns a
clean projection of the cluster IDs into residual space. In real usage
you would omit the ``hidden_extractor=`` argument and the default
extractor would pull the forged model's residual stream — the score is
then a meaningful measurement of "does the forged model's residual
distinguish my fixture's labels?" rather than the saturated 1.0 this
demo produces by construction.

Runs in under a minute on a CPU laptop; no HF download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


def _build_tiny_gpt2():
    """A tiny torch GPT-2 — small enough to forge in a few seconds."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_inner=32,
    )
    return GPT2LMHeadModel(config).eval()


def _build_cluster_basis(d_model: int = 16, n_clusters: int = 3, seed: int = 0):
    """A synthetic SAE basis with one feature per cluster centroid.

    Returns ``(directions, basis)`` where ``directions`` is the
    ``(n_clusters, d_model)`` cluster-centroid matrix and ``basis`` is a
    :class:`saeforge.FeatureBasis` built directly from those rows.
    """
    from saeforge import FeatureBasis

    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_clusters, d_model)).astype(np.float64)
    norms = np.linalg.norm(directions, axis=1)
    basis = FeatureBasis(
        kept_ids=np.arange(n_clusters),
        W_dec=directions,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )
    return directions, basis


def _build_mixture_fixture(
    n_per_cluster: int = 32, n_clusters: int = 3, hidden_size: int = 16
):
    """Build labels + a hidden-state signal that AUCs near 1.0.

    Returns ``(labels, signal_tensor)`` where ``labels`` is ``(N, M)``
    one-hot cluster IDs and ``signal_tensor`` is ``(N, hidden_size)`` —
    the first ``M`` columns are the labels (perfect cluster signature),
    the remaining columns are gaussian noise. Per-feature AUC against
    the matching label column is then ~1.0.
    """
    import torch

    rng = np.random.default_rng(0)
    n = n_per_cluster * n_clusters
    cluster_ids = np.repeat(np.arange(n_clusters), n_per_cluster)
    labels = np.eye(n_clusters, dtype=np.float32)[cluster_ids]
    noise = rng.standard_normal((n, hidden_size - n_clusters)).astype(np.float32) * 0.01
    signal = np.concatenate([labels, noise], axis=1)
    return labels, torch.tensor(signal, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: Path | str | None = None) -> dict:
    import torch

    from saeforge import ForgePipeline, SubspaceProjector
    from saeforge.eval import GroundTruthTarget

    if output_dir is None:
        output_dir = Path("/tmp/sae-forge-gt-alignment")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    host = _build_tiny_gpt2()
    _directions, basis = _build_cluster_basis(
        d_model=host.config.n_embd, n_clusters=3
    )

    labels, signal_tensor = _build_mixture_fixture(
        n_per_cluster=32, n_clusters=3, hidden_size=host.config.n_embd
    )
    n_eval = labels.shape[0]

    # Custom extractor: return the cluster-signature signal directly.
    # This saturates AUC at 1.0 so the example is deterministic and
    # fast. In real usage you would omit `hidden_extractor=` entirely:
    #
    #     target = GroundTruthTarget(labels=labels, pool="mean")
    #
    # The default extractor duck-types `forged.torch_module.transformer`
    # (GPT-2 lineage) then `.model` (Llama / Gemma / Qwen lineage) and
    # returns the residual stream `(batch, seq, hidden_size)`. The
    # `pool="mean"` step then reduces across `seq`, and the reported
    # score is a meaningful measurement of how well the forged model's
    # residual distinguishes your fixture's labels — not the saturated
    # 1.0 the cluster-signature extractor produces here by construction.
    def _extractor(forged: Any, input_ids: Any) -> "torch.Tensor":
        return signal_tensor

    target = GroundTruthTarget(
        labels=labels,
        pool="mean",
        hidden_extractor=_extractor,
    )

    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        faithfulness=target,
        orchestrator="fsm",
    )

    eval_input_ids = torch.randint(0, host.config.vocab_size, (n_eval, 4))
    result = pipeline.run_synthetic(
        host, output_dir, eval_input_ids=eval_input_ids
    )

    summary = {
        "n_params": result.n_params,
        "faithfulness": result.faithfulness,
        "faithfulness_target_name": result.faithfulness_target_name,
        "n_features": basis.n_features,
        "n_eval": n_eval,
        "n_labels": int(labels.shape[1]),
        "output_dir": str(result.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
