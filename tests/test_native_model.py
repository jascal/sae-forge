"""Tests for NativeModel — config, construction, projection wiring, save/load."""

from __future__ import annotations

import pytest

from saeforge import NativeModel, SubspaceProjector
from saeforge.model import NativeModelConfig


def test_config_validates_qkv_inner_size_factorization():
    NativeModelConfig(
        family="gpt2", hidden_size=8,
        qkv_inner_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
    )
    with pytest.raises(ValueError, match="qkv_inner_size"):
        NativeModelConfig(
            family="gpt2", hidden_size=8,
            qkv_inner_size=15,  # not divisible by num_heads * head_dim
            num_layers=2,
            num_heads=4,
            head_dim=4,
            intermediate_size=32,
            vocab_size=100,
        )


def test_config_round_trip():
    config = NativeModelConfig(
        family="gpt2", hidden_size=8,
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
        family="gpt2", hidden_size=8,
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
        family="gpt2", hidden_size=8,
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
        family="gpt2", hidden_size=tiny_synthetic_basis.n_features,
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
        family="gpt2", hidden_size=tiny_synthetic_basis.n_features,
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
        family="gpt2", hidden_size=8,
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


def test_save_and_load_round_trip_tied_embeddings(
    tmp_path, tiny_llama_tied, feature_basis_128_to_32
):
    """Regression test for the safetensors shared-storage crash.

    Before the fix, ``save_pretrained`` raised
    ``RuntimeError: Some tensors share memory ...`` whenever
    ``config.tied_embeddings`` was True (Gemma-2, tied Llama hosts):
    the ForgedLlama constructor aliases ``lm_head.weight`` to
    ``model.embed_tokens.weight``, but ``safetensors.torch.save_file``
    refuses to write tensors that share storage. The fix drops the
    aliased ``lm_head.weight`` from the saved state_dict and reconstructs
    the alias via the constructor on load.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for

    projector = SubspaceProjector(feature_basis_128_to_32)
    adapter = adapter_for(tiny_llama_tied)
    walk = adapter.walk(tiny_llama_tied, projector)
    config = adapter.build_native_config(
        tiny_llama_tied, feature_basis_128_to_32.n_features
    )
    assert config.tied_embeddings is True

    nm = NativeModel.from_projected_weights(config, walk)

    # Snapshot the forward output before save so we can verify the
    # round-trip reproduces it bit-for-bit.
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    nm._module.eval()
    with torch.no_grad():
        before = nm.forward(input_ids).clone()

    out = tmp_path / "forged_tied"
    nm.save_pretrained(out)
    assert (out / "model.safetensors").is_file()

    # The saved file must NOT contain lm_head.weight when tied — that's
    # the whole point of the fix.
    from safetensors.torch import load_file as _load_safetensors

    saved_state = _load_safetensors(str(out / "model.safetensors"))
    assert "lm_head.weight" not in saved_state
    assert "model.embed_tokens.weight" in saved_state

    loaded = NativeModel.load_pretrained(out)

    # The tied alias must be re-established by the constructor.
    assert (
        loaded._module.lm_head.weight.data_ptr()
        == loaded._module.model.embed_tokens.weight.data_ptr()
    )

    # Forward output is identical — no regression in the projected weights.
    loaded._module.eval()
    with torch.no_grad():
        after = loaded.forward(input_ids)
    assert torch.allclose(before, after, atol=1e-6)
