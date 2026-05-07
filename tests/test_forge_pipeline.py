"""Tests for ForgePipeline + faithfulness_kl + the toy example."""

from __future__ import annotations

import json

import pytest

from saeforge import ForgePipeline, NativeModel, SubspaceProjector


def test_run_synthetic_end_to_end(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "toy", eval_input_ids=eval_input_ids)

    assert isinstance(result.model, NativeModel)
    assert result.n_params > 0
    assert result.faithfulness_kl is not None
    assert result.faithfulness_kl >= 0.0
    assert (tmp_path / "toy" / "forged" / "config.json").is_file()
    assert (tmp_path / "toy" / "forged" / "model.safetensors").is_file()
    payload = json.loads((tmp_path / "toy" / "forge_result.json").read_text())
    assert payload["n_params"] == result.n_params


def test_run_requires_host_model_id_when_called_directly(tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        host_model_id=None,
    )
    with pytest.raises(ValueError, match="host_model_id"):
        pipeline.run(tmp_path / "out")


def test_faithfulness_kl_matches_when_forged_equals_host(tiny_gpt2):
    """Sanity check: KL(host || host) == 0. Constructs a forged model whose
    forward pass exactly matches the host by using an identity-like basis.
    """
    pytest.importorskip("torch")
    import numpy as np
    import torch

    from saeforge import FeatureBasis

    d_model = tiny_gpt2.config.n_embd
    identity_basis = FeatureBasis(
        kept_ids=np.arange(d_model),
        W_dec=np.eye(d_model, dtype=np.float64),
        merged_norms=np.ones(d_model),
        original_norms=np.ones(d_model),
        scale_compression_ratio=1.0,
    )
    projector = SubspaceProjector(identity_basis)
    pipeline = ForgePipeline(basis=identity_basis, projector=projector)
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, "/tmp/sae-forge-identity", eval_input_ids=input_ids)
    assert result.faithfulness_kl < 1e-3, f"identity-basis forge should be ~zero KL, got {result.faithfulness_kl}"


def test_faithfulness_kl_signature(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    forged = pipeline.run_synthetic(tiny_gpt2, tmp_path / "sig", eval_input_ids=input_ids)
    assert isinstance(forged.faithfulness_kl, float)
    assert forged.faithfulness_kl >= 0.0


def test_toy_example_runs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from examples.forge_gpt2_toy import main

    summary = main(output_dir=tmp_path / "toy")
    assert summary["n_features"] == 8
    assert summary["n_params"] > 0
    assert summary["faithfulness_kl"] is not None
