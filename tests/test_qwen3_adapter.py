"""Qwen3 dense adapter unit tests.

Requires ``transformers >= 4.51`` (Qwen3 landed in 4.51). The whole file
skips gracefully on older installs — including the ``[intel]`` extra,
which is capped at ``<4.50`` to match torch 2.2.2 (the last x86_64 macOS
wheel).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers", minversion="4.51")


def test_qwen3_dispatches_to_qwen3_adapter(tiny_qwen3_untied_4layer):
    from saeforge.adapters import adapter_for

    adapter = adapter_for(tiny_qwen3_untied_4layer)
    assert adapter.family == "qwen3"


def test_registered_classes_contains_qwen3():
    from saeforge.adapters import registered_classes

    names = [c.__name__ for c in registered_classes()]
    assert "Qwen3ForCausalLM" in names
    # Existing families still present (regression gate)
    for n in ("GPT2LMHeadModel", "LlamaForCausalLM", "Gemma2ForCausalLM", "Qwen2ForCausalLM"):
        assert n in names, f"{n} missing from registered_classes after qwen3 registration"


def test_qwen3_walker_emits_q_norm_k_norm(tiny_qwen3_untied_4layer, feature_basis_128_to_32):
    from saeforge.adapters import adapter_for
    from saeforge.projector import SubspaceProjector

    projector = SubspaceProjector(feature_basis_128_to_32)
    walk = adapter_for(tiny_qwen3_untied_4layer).walk(tiny_qwen3_untied_4layer, projector)
    head_dim = tiny_qwen3_untied_4layer.config.head_dim
    for i in range(tiny_qwen3_untied_4layer.config.num_hidden_layers):
        for qk in ("q_norm", "k_norm"):
            key = f"model.layers.{i}.self_attn.{qk}.weight"
            assert key in walk, f"missing {key}"
            # Head-dim aligned, not residual-aligned (pass-through).
            assert walk[key].shape == (head_dim,), (
                f"{key} expected shape ({head_dim},) but got {walk[key].shape}"
            )


def test_qwen3_walker_omits_qkv_biases(tiny_qwen3_untied_4layer, feature_basis_128_to_32):
    """Qwen3 dense has no Q/K/V biases; the inherited auto-detection should produce no bias keys."""
    from saeforge.adapters import adapter_for
    from saeforge.projector import SubspaceProjector

    projector = SubspaceProjector(feature_basis_128_to_32)
    walk = adapter_for(tiny_qwen3_untied_4layer).walk(tiny_qwen3_untied_4layer, projector)
    for i in range(tiny_qwen3_untied_4layer.config.num_hidden_layers):
        for qkv in ("q_proj", "k_proj", "v_proj"):
            key = f"model.layers.{i}.self_attn.{qkv}.bias"
            assert key not in walk, f"unexpected bias key {key} in Qwen3 walk"


def test_qwen3_native_config_sets_qk_norm_true_and_qkv_bias_false(
    tiny_qwen3_untied_4layer, feature_basis_128_to_32
):
    from saeforge.adapters import adapter_for

    config = adapter_for(tiny_qwen3_untied_4layer).build_native_config(
        tiny_qwen3_untied_4layer, feature_basis_128_to_32.n_features
    )
    assert config.family == "qwen3"
    assert config.qk_norm is True
    assert config.qkv_bias is False


def test_llama_native_config_keeps_qk_norm_false(tiny_llama, feature_basis_128_to_32):
    """Regression gate: adding qk_norm doesn't leak into the Llama path."""
    from saeforge.adapters import adapter_for

    config = adapter_for(tiny_llama).build_native_config(
        tiny_llama, feature_basis_128_to_32.n_features
    )
    assert config.qk_norm is False


def test_qwen2_native_config_keeps_qk_norm_false(tiny_qwen2, feature_basis_128_to_32):
    """Regression gate: Qwen2 path is unaffected by Qwen3's qk_norm field."""
    from saeforge.adapters import adapter_for

    config = adapter_for(tiny_qwen2).build_native_config(
        tiny_qwen2, feature_basis_128_to_32.n_features
    )
    assert config.qk_norm is False
    # Qwen2 still detects qkv_bias=True (existing behavior)
    assert config.qkv_bias is True


def test_qwen3_forged_attention_has_q_norm_k_norm_modules(
    tiny_qwen3_untied_4layer, feature_basis_128_to_32
):
    """Confirm the forged native module's attention blocks construct the RMSNorm submodules."""
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    projector = SubspaceProjector(feature_basis_128_to_32)
    walk = adapter_for(tiny_qwen3_untied_4layer).walk(tiny_qwen3_untied_4layer, projector)
    config = adapter_for(tiny_qwen3_untied_4layer).build_native_config(
        tiny_qwen3_untied_4layer, feature_basis_128_to_32.n_features
    )
    model = NativeModel.from_projected_weights(config, walk)
    for layer in model.torch_module.model.layers:
        assert layer.self_attn.q_norm is not None
        assert layer.self_attn.k_norm is not None
        # head_dim-shaped weight parameter
        assert layer.self_attn.q_norm.weight.shape[-1] == config.head_dim


def test_llama_forged_attention_has_no_q_norm_k_norm_modules(
    tiny_llama, feature_basis_128_to_32
):
    """Regression gate: Llama forged attention is unchanged."""
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    projector = SubspaceProjector(feature_basis_128_to_32)
    walk = adapter_for(tiny_llama).walk(tiny_llama, projector)
    config = adapter_for(tiny_llama).build_native_config(
        tiny_llama, feature_basis_128_to_32.n_features
    )
    model = NativeModel.from_projected_weights(config, walk)
    for layer in model.torch_module.model.layers:
        assert layer.self_attn.q_norm is None
        assert layer.self_attn.k_norm is None
