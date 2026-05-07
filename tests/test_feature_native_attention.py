"""Tests for v0.2 feature-native attention — opt-in mode where every dimension
of the forged model is k-wide, including attention internals.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from saeforge import FeatureBasis, ForgePipeline, NativeModel, SubspaceProjector
from saeforge.model import NativeModelConfig


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- Config validation ---------------------------------------------------


def test_config_default_is_host():
    config = NativeModelConfig(
        family="gpt2", hidden_size=8, qkv_inner_size=16, num_layers=2, num_heads=4,
        head_dim=4, intermediate_size=32, vocab_size=100,
    )
    assert config.attention_width == "host"


def test_config_feature_native_requires_qkv_inner_eq_hidden():
    with pytest.raises(ValueError, match="qkv_inner_size"):
        NativeModelConfig(
            family="gpt2", hidden_size=8, qkv_inner_size=16, num_layers=2, num_heads=4,
            head_dim=4, intermediate_size=32, vocab_size=100,
            attention_width="feature_native",
        )


def test_config_feature_native_requires_divisibility():
    with pytest.raises(ValueError, match="num_heads"):
        NativeModelConfig(
            family="gpt2", hidden_size=8, qkv_inner_size=8, num_layers=2, num_heads=3,
            head_dim=2, intermediate_size=32, vocab_size=100,
            attention_width="feature_native",
        )


def test_config_feature_native_valid():
    config = NativeModelConfig(
        family="gpt2", hidden_size=8, qkv_inner_size=8, num_layers=2, num_heads=4,
        head_dim=2, intermediate_size=32, vocab_size=100,
        attention_width="feature_native",
    )
    assert config.attention_width == "feature_native"
    assert config.qkv_inner_size == config.hidden_size


def test_config_rejects_unknown_attention_width():
    with pytest.raises(ValueError, match="attention_width"):
        NativeModelConfig(
            family="gpt2", hidden_size=8, qkv_inner_size=8, num_layers=2, num_heads=4,
            head_dim=2, intermediate_size=32, vocab_size=100,
            attention_width="bogus",
        )


# ---- Projector helpers ---------------------------------------------------


def test_project_residual_full_identity_basis():
    """When W_dec = I_d, project_residual_full(W) == W within tolerance."""
    d = 16
    basis = FeatureBasis(
        kept_ids=np.arange(d), W_dec=np.eye(d, dtype=np.float64),
        merged_norms=np.ones(d), original_norms=np.ones(d),
    )
    projector = SubspaceProjector(basis)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((d, d))
    out = projector.project_residual_full(W)
    assert np.allclose(out, W, atol=1e-9)


def test_project_residual_full_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    d = tiny_synthetic_basis.d_model
    n = tiny_synthetic_basis.n_features
    W = np.random.default_rng(1).standard_normal((d, d))
    out = projector.project_residual_full(W)
    assert out.shape == (n, n)


def test_project_residual_full_rejects_wrong_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    bad = np.random.default_rng(2).standard_normal((4, 8))
    with pytest.raises(ValueError, match="d_model"):
        projector.project_residual_full(bad)


def test_project_qkv_full_block_structure(tiny_synthetic_basis):
    """The (k, 3k) output's three k-wide blocks each equal project_residual_full
    of the corresponding (d, d) block of the input.
    """
    projector = SubspaceProjector(tiny_synthetic_basis)
    d = tiny_synthetic_basis.d_model
    n = tiny_synthetic_basis.n_features
    W = np.random.default_rng(3).standard_normal((d, 3 * d))
    out = projector.project_qkv_full(W)
    assert out.shape == (n, 3 * n)
    Wq, Wk, Wv = np.split(W, 3, axis=1)
    Oq, Ok, Ov = np.split(out, 3, axis=1)
    assert np.allclose(Oq, projector.project_residual_full(Wq), atol=1e-12)
    assert np.allclose(Ok, projector.project_residual_full(Wk), atol=1e-12)
    assert np.allclose(Ov, projector.project_residual_full(Wv), atol=1e-12)


def test_project_qkv_full_rejects_wrong_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    bad = np.random.default_rng(4).standard_normal((tiny_synthetic_basis.d_model, 8))
    with pytest.raises(ValueError, match=r"3\*d_model"):
        projector.project_qkv_full(bad)


# ---- project_module shape contract ---------------------------------------


def test_project_module_feature_native_shapes(tiny_gpt2, tiny_synthetic_basis):
    pytest.importorskip("torch")
    projector = SubspaceProjector(tiny_synthetic_basis)
    weights = projector.project_module(tiny_gpt2, attention_width="feature_native")
    n = tiny_synthetic_basis.n_features

    for i in range(tiny_gpt2.config.n_layer):
        prefix = f"transformer.h.{i}"
        assert weights[f"{prefix}.attn.c_attn.weight"].shape == (n, 3 * n)
        assert weights[f"{prefix}.attn.c_attn.bias"].shape == (3 * n,)
        assert weights[f"{prefix}.attn.c_proj.weight"].shape == (n, n)
        assert weights[f"{prefix}.attn.c_proj.bias"].shape == (n,)


def test_project_module_rejects_unknown_attention_width(tiny_gpt2, tiny_synthetic_basis):
    pytest.importorskip("torch")
    projector = SubspaceProjector(tiny_synthetic_basis)
    with pytest.raises(ValueError, match="attention_width"):
        projector.project_module(tiny_gpt2, attention_width="bogus")


# ---- End-to-end -----------------------------------------------------------


def test_feature_native_runs_end_to_end(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """Pipeline reaches done with feature_native; produces a working forged model."""
    pytest.importorskip("torch")
    import torch

    # n_features=8, num_heads=4 → head_dim=2 (divisibility holds)
    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        attention_width="feature_native",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "fn", eval_input_ids=eval_input_ids)
    assert isinstance(result.model, NativeModel)
    assert result.model.config.attention_width == "feature_native"
    assert result.model.config.qkv_inner_size == tiny_synthetic_basis.n_features
    assert result.faithfulness_kl is not None and result.faithfulness_kl >= 0.0


def test_feature_native_identity_basis_kl_is_zero(tiny_gpt2, tmp_path):
    """When W_dec = I_d_model, both-sides projection is identity → KL ≈ 0.

    This is the strongest correctness signal that the feature-native algebra
    is right at scale: D @ W @ E = I @ W @ I = W when k = d.
    """
    pytest.importorskip("torch")
    import torch

    d = tiny_gpt2.config.n_embd
    basis = FeatureBasis(
        kept_ids=np.arange(d), W_dec=np.eye(d, dtype=np.float64),
        merged_norms=np.ones(d), original_norms=np.ones(d),
    )
    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        attention_width="feature_native",
    )
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "fn-identity", eval_input_ids=input_ids)
    assert result.faithfulness_kl < 1e-3, (
        f"feature-native identity-basis forge should be ~zero KL, got {result.faithfulness_kl}"
    )


def test_feature_native_differs_from_host_on_nontrivial_basis(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """Regression check: host and feature_native produce different forged weights
    on a non-identity basis. If they were identical we'd suspect a no-op bug.
    """
    pytest.importorskip("torch")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    host_pipe = ForgePipeline(
        basis=tiny_synthetic_basis, projector=projector, attention_width="host"
    )
    host_result = host_pipe.run_synthetic(tiny_gpt2, tmp_path / "host", eval_input_ids=eval_input_ids)

    fn_pipe = ForgePipeline(
        basis=tiny_synthetic_basis, projector=projector, attention_width="feature_native"
    )
    fn_result = fn_pipe.run_synthetic(tiny_gpt2, tmp_path / "fn", eval_input_ids=eval_input_ids)

    host_weights = tmp_path / "host" / "forged" / "model.safetensors"
    fn_weights = tmp_path / "fn" / "forged" / "model.safetensors"
    assert _file_sha256(host_weights) != _file_sha256(fn_weights)
    # Param count differs because feature_native has smaller QKV (k vs host n_embd)
    assert fn_result.n_params != host_result.n_params


def test_feature_native_default_stays_host(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """ForgePipeline default attention_width is 'host' — preserves v0.1 behaviour."""
    pytest.importorskip("torch")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    assert pipeline.attention_width == "host"
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "default", eval_input_ids=eval_input_ids)
    assert result.model.config.attention_width == "host"
    # Host-mode preserves the v0 (k, 3 * host n_embd) c_attn shape
    expected_qkv_inner = tiny_gpt2.config.n_embd
    state = result.model.torch_module.state_dict()
    assert state["transformer.h.0.attn.c_attn.weight"].shape == (
        tiny_synthetic_basis.n_features, 3 * expected_qkv_inner,
    )


def test_cli_flag_present():
    from saeforge.cli import _build_parser

    parser = _build_parser()
    # Find the forge subparser
    forge_parser = None
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices and "forge" in action.choices:
            forge_parser = action.choices["forge"]
            break
    assert forge_parser is not None
    flag_names = []
    for act in forge_parser._actions:
        flag_names.extend(act.option_strings)
    assert "--feature-native-attention" in flag_names


def test_fsm_threads_attention_width(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """FSM orchestrator picks up attention_width from ctx and the project action honors it."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        attention_width="feature_native",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "fsm-fn", eval_input_ids=eval_input_ids)
    assert result.extras["final_state"] == "done"
    state = result.model.torch_module.state_dict()
    n = tiny_synthetic_basis.n_features
    assert state["transformer.h.0.attn.c_attn.weight"].shape == (n, 3 * n)
