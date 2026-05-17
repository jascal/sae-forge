"""Forge with a custom faithfulness target — GT-alignment on a synthetic fixture.

This example demonstrates :class:`saeforge.eval.faithfulness.FaithfulnessTarget`
end-to-end. It builds a tiny synthetic setup where the ground-truth
feature directions are known (a 3-cluster 2D-ish mixture lifted into the
16-dim residual stream), then forges a model whose SAE basis is those
same directions. The custom ``GTAlignmentTarget`` returns the mean
cosine similarity between the forged model's reconstructed feature
decoder weights and the known directions — a "did the forge recover
the planted features" score in ``[0, 1]``.

The point of the example is the protocol surface. Replace
``GTAlignmentTarget`` with any scorer that satisfies the
:class:`FaithfulnessTarget` protocol (``name``, ``better_when``,
``score(*, forged, host, ctx) -> (score, perplexity_analog)``) and the
rest of the wiring stays the same.

Runs in under a minute on a CPU laptop; no HF download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Custom target
# ---------------------------------------------------------------------------


class GTAlignmentTarget:
    """Mean per-feature absolute cosine similarity between the SAE basis
    the forge was built with and a known set of "ground-truth" feature
    directions.

    Constructor takes the GT directions and the basis decoder matrix
    (``W_dec``); ``score`` compares the two row-wise. Both arguments
    are stashed on the target at construction time — the protocol
    accepts ``ctx`` for cases where the inputs change per call, but the
    GT directions and the basis are static in this example.

    ``better_when="higher"`` — perfect alignment gives 1.0; orthogonal
    or zero-norm gives 0.0. Absolute cosine handles SAE feature signs
    being arbitrary (a sign-flipped feature decodes the same direction).
    """

    name = "gt_alignment"
    better_when = "higher"

    def __init__(self, gt_directions: np.ndarray, basis_W_dec: np.ndarray) -> None:
        self._gt_unit = _unit_rows(gt_directions)
        self._basis_unit = _unit_rows(basis_W_dec)

    def score(
        self,
        *,
        forged: Any,  # noqa: ARG002 — ignored; example compares basis directions
        host: Any,  # noqa: ARG002 — ignored; GT alignment doesn't need a teacher
        ctx: Mapping[str, Any],  # noqa: ARG002 — ignored; inputs stashed on self
    ) -> tuple[float, float]:
        n = min(self._gt_unit.shape[0], self._basis_unit.shape[0])
        cosines = (self._gt_unit[:n] * self._basis_unit[:n]).sum(axis=1)
        score = float(np.mean(np.abs(cosines)))
        return score, max(0.0, 1.0 - score)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


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


def _build_gt_basis(d_model: int = 16, n_clusters: int = 3, seed: int = 0):
    """Three "cluster centres" in residual space, treated as the
    planted feature directions. Returns the directions (n_clusters,
    d_model) and a FeatureBasis built directly from them.
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: Path | str | None = None) -> dict:
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    if output_dir is None:
        output_dir = Path("/tmp/sae-forge-gt-alignment")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    host = _build_tiny_gpt2()
    gt_directions, basis = _build_gt_basis(d_model=host.config.n_embd, n_clusters=3)

    projector = SubspaceProjector(basis)
    target = GTAlignmentTarget(gt_directions=gt_directions, basis_W_dec=basis.W_dec)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        faithfulness=target,
    )

    # A few random eval input_ids — the GT-alignment target ignores
    # them, but the pipeline machinery still wants something to feed
    # the model on the eval pass.
    eval_input_ids = torch.randint(0, host.config.vocab_size, (2, 8))
    result = pipeline.run_synthetic(host, output_dir, eval_input_ids=eval_input_ids)

    summary = {
        "n_params": result.n_params,
        "faithfulness": result.faithfulness,
        "faithfulness_target_name": result.faithfulness_target_name,
        "n_features": basis.n_features,
        "output_dir": str(result.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
