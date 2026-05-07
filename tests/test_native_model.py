"""Tests for NativeModel — config, construction, projection wiring, save/load."""

from __future__ import annotations

import pytest

from saeforge import NativeModel, SubspaceProjector
from saeforge.model import NativeModelConfig


def test_config_validates_qkv_inner_size_factorization():
    NativeModelConfig(
        hidden_size=8,
        qkv_inner_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
    )
    with pytest.raises(ValueError, match="qkv_inner_size"):
        NativeModelConfig(
            hidden_size=8,
            qkv_inner_size=15,  # not divisible by num_heads * head_dim
            num_layers=2,
            num_heads=4,
            head_dim=4,
            intermediate_size=32,
            vocab_size=100,
        )


def test_config_round_trip():
    config = NativeModelConfig(
        hidden_size=8,
        qkv_inner_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        max_position_embeddings=32,
    )
    payload = config.to_dict()
    assert NativeModelConfig.from_dict(payload) == config


def test_construct_with_synthetic_config():
    pytest.importorskip("torch")
    config = NativeModelConfig(
        hidden_size=8,
        qkv_inner_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = NativeModel(config)
    assert model.num_parameters() > 0


def test_forward_runs_on_random_input():
    torch = pytest.importorskip("torch")
    config = NativeModelConfig(
        hidden_size=8,
        qkv_inner_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = NativeModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    logits = model.forward(input_ids)
    assert logits.shape == (2, 16, 100)


def test_from_host_with_tiny_gpt2(tiny_gpt2, tiny_synthetic_basis):
    pytest.importorskip("torch")
    projector = SubspaceProjector(tiny_synthetic_basis)
    weights = projector.project_module(tiny_gpt2)
    config = NativeModelConfig(
        hidden_size=tiny_synthetic_basis.n_features,
        qkv_inner_size=tiny_gpt2.config.n_embd,
        num_layers=tiny_gpt2.config.n_layer,
        num_heads=tiny_gpt2.config.n_head,
        head_dim=tiny_gpt2.config.n_embd // tiny_gpt2.config.n_head,
        intermediate_size=tiny_gpt2.config.n_inner,
        vocab_size=tiny_gpt2.config.vocab_size,
        max_position_embeddings=tiny_gpt2.config.n_positions,
    )
    model = NativeModel.from_projected_weights(config, weights)
    assert model.num_parameters() > 0


def test_from_projected_weights_rejects_extra_keys(tiny_synthetic_basis):
    pytest.importorskip("torch")
    config = NativeModelConfig(
        hidden_size=tiny_synthetic_basis.n_features,
        qkv_inner_size=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        max_position_embeddings=32,
    )
    with pytest.raises(KeyError, match="no slot"):
        NativeModel.from_projected_weights(config, {"unknown_key": __import__("numpy").zeros((4,))})


def test_save_and_load_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    config = NativeModelConfig(
        hidden_size=8,
        qkv_inner_size=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        max_position_embeddings=16,
    )
    model = NativeModel(config)
    input_ids = torch.randint(0, 100, (1, 8))
    before = model.forward(input_ids)

    out = tmp_path / "forged"
    model.save_pretrained(out)
    assert (out / "config.json").is_file()
    assert (out / "model.safetensors").is_file()

    loaded = NativeModel.load_pretrained(out)
    after = loaded.forward(input_ids)
    assert torch.allclose(before, after, atol=1e-6)
