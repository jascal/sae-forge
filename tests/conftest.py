"""Shared fixtures for the test suite."""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def synthetic_compressed_sae(tmp_path):
    """Build a fake Polygram-compressed checkpoint + companion report.

    8-feature SAE with 16-dim residual stream. Cluster {2, 5} is collapsed
    onto representative 2 with merged_norm 1.5; row 5 is zeroed. Cluster
    {3, 7} is zeroed onto representative 3 with no merged_norm (zero
    strategy); row 7 is zeroed. Kept ids: [0, 1, 2, 3, 4, 6].
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(42)
    n_total = 8
    d_model = 16
    W_dec = rng.standard_normal((n_total, d_model)).astype(np.float32)
    # Zero the non-representative rows so loader's bookkeeping matches polygram's contract.
    W_dec[5] = 0.0
    W_dec[7] = 0.0
    # Rescale rep row 2 to merged_norm 1.5.
    rep_norm = np.linalg.norm(W_dec[2])
    if rep_norm > 0:
        W_dec[2] *= 1.5 / rep_norm

    checkpoint = tmp_path / "sae.compressed.safetensors"
    save_file({"W_dec": W_dec}, str(checkpoint))

    report = {
        "schema_version": 1,
        "source_checkpoint": "sae.safetensors",
        "source_checkpoint_sha256": "deadbeef",
        "output_checkpoint": str(checkpoint),
        "output_checkpoint_sha256": "feedface",
        "validation_report_dictionary_name": "synthetic",
        "validation_report_schema_version": 1,
        "strategy": "merge",
        "feature_ids": [2, 3, 5, 7],
        "clusters": [
            {
                "cluster_id": 0,
                "members": [2, 5],
                "representative": 2,
                "zeroed": [5],
                "cluster_norm_mean": 1.4,
                "cluster_norm_std": 0.1,
                "merged_norm": 1.5,
            },
            {
                "cluster_id": 1,
                "members": [3, 7],
                "representative": 3,
                "zeroed": [7],
                "cluster_norm_mean": 1.0,
                "cluster_norm_std": 0.05,
                "merged_norm": None,
            },
        ],
        "n_features_zeroed": 2,
        "n_features_kept": 6,
        "n_clusters": 2,
        "scale_compression_ratio": 0.92,
    }
    report_path = tmp_path / "sae.compressed_compression_report.json"
    report_path.write_text(json.dumps(report))

    return {
        "checkpoint": checkpoint,
        "report_path": report_path,
        "report": report,
        "W_dec_full": W_dec,
        "expected_kept_ids": [0, 1, 2, 3, 4, 6],
    }


@pytest.fixture
def tiny_synthetic_basis():
    """In-memory FeatureBasis for projector / model unit tests."""
    from saeforge import FeatureBasis

    rng = np.random.default_rng(0)
    n_kept = 8
    d_model = 16
    W_dec = rng.standard_normal((n_kept, d_model)).astype(np.float64)
    norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n_kept),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )


@pytest.fixture
def tiny_gpt2(monkeypatch):
    """A tiny torch GPT-2 — 16-dim residual, 2 layers, 4 heads, 100 vocab."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_inner=32,
    )
    model = GPT2LMHeadModel(config).eval()
    return model
