"""Tests for the Whisper-encoder architecture adapter (§2.9 of the
``forge-whisper-encoder`` change).

Covers the walker shape audit (every projected key + shape), the
frozen-copy bit-for-bit invariant on the conv stem and positional
embeddings, the no-randomly-initialised-weights invariant for the
forged native module's parameter slots, the Whisper-encoder MHA
invariant (not GQA), and registry dispatch for both HF host classes
(``WhisperForConditionalGeneration`` and ``WhisperModel``).

The forward pass of ``ForgedWhisperEncoder`` is deferred to the §3
follow-up commit. These tests deliberately do not run a forward —
only the adapter's walk and the native module's parameter-slot
shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.adapters import adapter_for, registered_classes
from saeforge.model import NativeModel
from saeforge.projector import SubspaceProjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_keys(num_layers: int) -> set[str]:
    """The v0.4 walker keyset for a Whisper encoder with ``num_layers`` blocks."""
    keys: set[str] = {
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "embed_positions.weight",
        "layer_norm.weight",
        "layer_norm.bias",
    }
    for i in range(num_layers):
        prefix = f"layers.{i}"
        keys.update(
            {
                f"{prefix}.self_attn_layer_norm.weight",
                f"{prefix}.self_attn_layer_norm.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                # k_proj.bias intentionally absent — HF Whisper has none.
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.out_proj.bias",
                f"{prefix}.final_layer_norm.weight",
                f"{prefix}.final_layer_norm.bias",
                f"{prefix}.fc1.weight",
                f"{prefix}.fc1.bias",
                f"{prefix}.fc2.weight",
                f"{prefix}.fc2.bias",
            }
        )
    return keys


# ---------------------------------------------------------------------------
# Walker shape audit
# ---------------------------------------------------------------------------


class TestWalkerShape:
    def test_walker_emits_expected_keyset(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        projector = SubspaceProjector(feature_basis_64_to_32)
        walk = adapter_for(tiny_synthetic_whisper).walk(
            tiny_synthetic_whisper, projector
        )
        assert set(walk) == _expected_keys(num_layers=2)

    def test_walker_projected_shapes(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        # d_model=64, n_features=32, intermediate=128, n_mels=80,
        # max_source_positions=1500.
        projector = SubspaceProjector(feature_basis_64_to_32)
        walk = adapter_for(tiny_synthetic_whisper).walk(
            tiny_synthetic_whisper, projector
        )
        f = feature_basis_64_to_32.n_features
        d = 64
        i = 128
        p = 1500

        # Frozen-copied (not projected).
        assert walk["conv1.weight"].shape == (d, 80, 3)
        assert walk["conv1.bias"].shape == (d,)
        assert walk["conv2.weight"].shape == (d, d, 3)
        assert walk["conv2.bias"].shape == (d,)
        assert walk["embed_positions.weight"].shape == (p, d)

        # Per-block: project the residual side.
        # q/k/v_proj: HF Linear (out=d, in=d); project the in-axis → (d, f).
        for layer in range(2):
            prefix = f"layers.{layer}"
            assert walk[f"{prefix}.self_attn_layer_norm.weight"].shape == (f,)
            assert walk[f"{prefix}.self_attn_layer_norm.bias"].shape == (f,)
            assert walk[f"{prefix}.self_attn.q_proj.weight"].shape == (d, f)
            assert walk[f"{prefix}.self_attn.q_proj.bias"].shape == (d,)
            assert walk[f"{prefix}.self_attn.k_proj.weight"].shape == (d, f)
            assert walk[f"{prefix}.self_attn.v_proj.weight"].shape == (d, f)
            assert walk[f"{prefix}.self_attn.v_proj.bias"].shape == (d,)
            # out_proj writes residual: project first axis → (f, d).
            assert walk[f"{prefix}.self_attn.out_proj.weight"].shape == (f, d)
            assert walk[f"{prefix}.self_attn.out_proj.bias"].shape == (f,)
            assert walk[f"{prefix}.final_layer_norm.weight"].shape == (f,)
            assert walk[f"{prefix}.final_layer_norm.bias"].shape == (f,)
            assert walk[f"{prefix}.fc1.weight"].shape == (i, f)
            assert walk[f"{prefix}.fc1.bias"].shape == (i,)
            assert walk[f"{prefix}.fc2.weight"].shape == (f, i)
            assert walk[f"{prefix}.fc2.bias"].shape == (f,)
        assert walk["layer_norm.weight"].shape == (f,)
        assert walk["layer_norm.bias"].shape == (f,)


# ---------------------------------------------------------------------------
# Frozen-copy invariant
# ---------------------------------------------------------------------------


class TestFrozenCopy:
    def test_conv_stem_and_embed_positions_passthrough(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        projector = SubspaceProjector(feature_basis_64_to_32)
        walk = adapter_for(tiny_synthetic_whisper).walk(
            tiny_synthetic_whisper, projector
        )
        encoder = tiny_synthetic_whisper.encoder

        for key, host_tensor in [
            ("conv1.weight", encoder.conv1.weight),
            ("conv1.bias", encoder.conv1.bias),
            ("conv2.weight", encoder.conv2.weight),
            ("conv2.bias", encoder.conv2.bias),
            ("embed_positions.weight", encoder.embed_positions.weight),
        ]:
            host_arr = (
                host_tensor.detach().cpu().float().numpy().astype(np.float64)
            )
            np.testing.assert_array_equal(walk[key], host_arr)


# ---------------------------------------------------------------------------
# MHA invariant: Whisper encoder is not GQA
# ---------------------------------------------------------------------------


class TestMHA:
    def test_native_config_has_mha(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        adapter = adapter_for(tiny_synthetic_whisper)
        config = adapter.build_native_config(
            tiny_synthetic_whisper,
            feature_basis_64_to_32.n_features,
        )
        assert config.family == "whisper_encoder"
        assert config.output_kind == "encoder_states"
        assert config.vocab_size == 0
        assert config.hidden_size == feature_basis_64_to_32.n_features
        assert config.qkv_inner_size == 64
        assert config.num_layers == 2
        assert config.num_heads == 4
        assert config.head_dim == 16
        assert config.intermediate_size == 128
        # MHA: n_kv_heads == num_heads. No GQA on Whisper encoder.
        assert config.n_kv_heads == config.num_heads == 4
        assert config.max_position_embeddings == 1500


# ---------------------------------------------------------------------------
# No-randomly-initialised invariant
# ---------------------------------------------------------------------------


class TestNoRandomInit:
    def test_every_param_slot_reached_by_walk(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        adapter = adapter_for(tiny_synthetic_whisper)
        projector = SubspaceProjector(feature_basis_64_to_32)
        walk = adapter.walk(tiny_synthetic_whisper, projector)
        config = adapter.build_native_config(
            tiny_synthetic_whisper, feature_basis_64_to_32.n_features
        )
        nm = NativeModel.from_projected_weights(config, walk)

        walked = set(walk)
        unreachable = [
            name
            for name, _ in nm.torch_module.named_parameters()
            if name not in walked
        ]
        assert unreachable == [], (
            f"Forged Whisper encoder has parameter slots not populated by "
            f"the walk (would retain random init): {unreachable}"
        )


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


class TestRegistryDispatch:
    def test_whisper_model_resolves_to_adapter(self, tiny_synthetic_whisper):
        adapter = adapter_for(tiny_synthetic_whisper)
        assert adapter.family == "whisper_encoder"

    def test_for_conditional_generation_resolves_to_same_adapter(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers import (
            WhisperConfig,
            WhisperForConditionalGeneration,
            WhisperModel,
        )

        config = WhisperConfig(
            d_model=64,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            decoder_layers=1,
            decoder_attention_heads=1,
            decoder_ffn_dim=8,
            vocab_size=51865,
            num_mel_bins=80,
            max_source_positions=1500,
        )
        full = WhisperForConditionalGeneration(config).eval()
        enc_only = WhisperModel(config).eval()

        a1 = adapter_for(full)
        a2 = adapter_for(enc_only)
        # Same instance backs both class registrations.
        assert a1 is a2
        assert a1.family == "whisper_encoder"

    def test_registry_lists_both_whisper_classes(self):
        from transformers import WhisperForConditionalGeneration, WhisperModel

        registered = set(registered_classes())
        assert WhisperForConditionalGeneration in registered
        assert WhisperModel in registered

    def test_unregistered_error_lists_whisper_classes(self):
        class _FakeBert:
            pass

        with pytest.raises(NotImplementedError) as exc_info:
            adapter_for(_FakeBert())
        msg = str(exc_info.value)
        assert "_FakeBert" in msg
        assert "WhisperForConditionalGeneration" in msg
        assert "WhisperModel" in msg
