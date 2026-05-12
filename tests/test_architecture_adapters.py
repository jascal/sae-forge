"""Tests for the architecture-adapter registry and the bundled
GPT-2 / Llama / Gemma-2 adapters.

Covers tasks.md §7.1–§7.7 of multi-architecture-support: registry
dispatch, walker shape audits (incl. GQA), tied-embeddings paths,
the four-norm-per-block Gemma-2 layout, soft-cap config passthrough,
unregistered-architecture error, and the
no-randomly-initialised-weight invariant.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from saeforge import SubspaceProjector
from saeforge.adapters import (
    ArchitectureAdapter,
    adapter_for,
    register_adapter,
    registered_classes,
)
from saeforge.model import NativeModel


# ---------------------------------------------------------------------------
# §7.6 — registry dispatch + unregistered-architecture error
# ---------------------------------------------------------------------------


class TestRegistryDispatch:
    def test_gpt2_dispatches_to_gpt2_adapter(self, tiny_gpt2):
        adapter = adapter_for(tiny_gpt2)
        assert adapter.family == "gpt2"

    def test_llama_dispatches_to_llama_adapter(self, tiny_llama):
        adapter = adapter_for(tiny_llama)
        assert adapter.family == "llama"

    def test_gemma2_dispatches_to_gemma2_adapter(self, tiny_gemma2):
        adapter = adapter_for(tiny_gemma2)
        assert adapter.family == "gemma2"

    def test_qwen2_dispatches_to_qwen2_adapter(self, tiny_qwen2):
        adapter = adapter_for(tiny_qwen2)
        assert adapter.family == "qwen2"

    def test_registered_classes_contains_all_four_families(self):
        names = [c.__name__ for c in registered_classes()]
        assert "GPT2LMHeadModel" in names
        assert "LlamaForCausalLM" in names
        assert "Gemma2ForCausalLM" in names
        assert "Qwen2ForCausalLM" in names

    def test_unregistered_architecture_raises_with_actionable_message(self):
        class FakeBert:
            pass

        with pytest.raises(NotImplementedError) as excinfo:
            adapter_for(FakeBert())
        msg = str(excinfo.value)
        assert "FakeBert" in msg
        assert "Registered" in msg
        # Names every registered host class so the user sees what's supported.
        assert "GPT2LMHeadModel" in msg
        assert "LlamaForCausalLM" in msg
        assert "Gemma2ForCausalLM" in msg


# ---------------------------------------------------------------------------
# §7.3 / §7.4 — Llama walker keys and shapes (incl. GQA + tied embeddings)
# ---------------------------------------------------------------------------


class TestLlamaWalker:
    def test_walker_emits_expected_key_set(self, tiny_llama, feature_basis_128_to_32):
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_llama).walk(tiny_llama, projector)

        # Top-level + per-layer + final norm + lm_head (untied).
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for i in range(tiny_llama.config.num_hidden_layers):
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                expected.add(f"model.layers.{i}.self_attn.{proj_name}.weight")
            for mlp_name in ("gate_proj", "up_proj", "down_proj"):
                expected.add(f"model.layers.{i}.mlp.{mlp_name}.weight")
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                expected.add(f"model.layers.{i}.{norm_name}.weight")

        assert set(walk) == expected

    def test_walker_gqa_shapes(self, tiny_llama, feature_basis_128_to_32):
        # num_attention_heads=4, num_key_value_heads=2, head_dim=32 →
        # q_proj rows = 4 * 32 = 128; k/v_proj rows = 2 * 32 = 64.
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_llama).walk(tiny_llama, projector)

        n_features = feature_basis_128_to_32.n_features
        assert walk["model.layers.0.self_attn.q_proj.weight"].shape == (128, n_features)
        assert walk["model.layers.0.self_attn.k_proj.weight"].shape == (64, n_features)
        assert walk["model.layers.0.self_attn.v_proj.weight"].shape == (64, n_features)
        assert walk["model.layers.0.self_attn.o_proj.weight"].shape == (n_features, 128)

    def test_walker_swiglu_shapes(self, tiny_llama, feature_basis_128_to_32):
        # intermediate_size=256, hidden_size=128 → gate/up rows = 256;
        # down has shape (n_features, 256) post-projection.
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_llama).walk(tiny_llama, projector)

        n_features = feature_basis_128_to_32.n_features
        assert walk["model.layers.0.mlp.gate_proj.weight"].shape == (256, n_features)
        assert walk["model.layers.0.mlp.up_proj.weight"].shape == (256, n_features)
        assert walk["model.layers.0.mlp.down_proj.weight"].shape == (n_features, 256)

    def test_walker_rmsnorm_has_no_bias(self, tiny_llama, feature_basis_128_to_32):
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_llama).walk(tiny_llama, projector)
        # No `*.bias` keys for any RMSNorm — RMSNorm has no β.
        bias_keys = [k for k in walk if k.endswith(".bias")]
        assert bias_keys == []

    def test_tied_embeddings_omits_lm_head_from_walk(
        self, tiny_llama_tied, feature_basis_128_to_32
    ):
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_llama_tied).walk(tiny_llama_tied, projector)
        assert "lm_head.weight" not in walk
        # The native config emits tied_embeddings=True so the module
        # aliases lm_head.weight to embed_tokens.weight at construction.
        config = adapter_for(tiny_llama_tied).build_native_config(
            tiny_llama_tied, feature_basis_128_to_32.n_features
        )
        assert config.tied_embeddings is True

    def test_qwen2_walker_emits_qkv_biases(self, tiny_qwen2, feature_basis_128_to_32):
        """Qwen2 has Q/K/V biases (Llama/Gemma-2 don't); walker passes them through unprojected."""
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_qwen2).walk(tiny_qwen2, projector)
        for i in range(tiny_qwen2.config.num_hidden_layers):
            for qkv in ("q_proj", "k_proj", "v_proj"):
                key = f"model.layers.{i}.self_attn.{qkv}.bias"
                assert key in walk, f"missing {key}"

    def test_qwen2_native_config_sets_qkv_bias_true(self, tiny_qwen2, feature_basis_128_to_32):
        """Auto-detection from host's first-block q_proj.bias flips ``NativeModelConfig.qkv_bias`` on."""
        config = adapter_for(tiny_qwen2).build_native_config(
            tiny_qwen2, feature_basis_128_to_32.n_features
        )
        assert config.family == "qwen2"
        assert config.qkv_bias is True

    def test_llama_native_config_keeps_qkv_bias_false(self, tiny_llama, feature_basis_128_to_32):
        """Llama has no Q/K/V biases; auto-detection should leave the field False (backward compat)."""
        config = adapter_for(tiny_llama).build_native_config(
            tiny_llama, feature_basis_128_to_32.n_features
        )
        assert config.qkv_bias is False


# ---------------------------------------------------------------------------
# §7.5 — Gemma-2 four-norm-per-block + soft-cap config passthrough
# ---------------------------------------------------------------------------


class TestGemma2Walker:
    def test_walker_emits_four_norms_per_block(
        self, tiny_gemma2, feature_basis_128_to_32
    ):
        projector = SubspaceProjector(feature_basis_128_to_32)
        walk = adapter_for(tiny_gemma2).walk(tiny_gemma2, projector)

        for i in range(tiny_gemma2.config.num_hidden_layers):
            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                key = f"model.layers.{i}.{norm_name}.weight"
                assert key in walk, f"missing {key}"

    def test_softcap_surfaces_on_native_config(
        self, tiny_gemma2, feature_basis_128_to_32
    ):
        config = adapter_for(tiny_gemma2).build_native_config(
            tiny_gemma2, feature_basis_128_to_32.n_features
        )
        assert config.family == "gemma2"
        assert config.final_logit_softcap == 30.0
        assert config.attn_logit_softcap == 50.0


# ---------------------------------------------------------------------------
# §7.7 — no-randomly-initialised invariant on Llama / Gemma-2
# ---------------------------------------------------------------------------


class TestNoRandomInitSurvives:
    """For each family, every parameter slot in the forged native module
    must have a corresponding key in the adapter's walk — no parameter
    should retain its random initialisation.
    """

    def _audit(self, host, basis):
        projector = SubspaceProjector(basis)
        adapter = adapter_for(host)
        walk = adapter.walk(host, projector)
        config = adapter.build_native_config(host, basis.n_features)
        nm = NativeModel.from_projected_weights(config, walk)

        walked = set(walk)
        # When tied, lm_head.weight is aliased to embed_tokens; no walk
        # entry needed and from_projected_weights skips it.
        alias = {"lm_head.weight"} if config.tied_embeddings else set()

        unreachable = [
            name
            for name, _ in nm.torch_module.named_parameters()
            if name not in walked and name not in alias
        ]
        return unreachable

    def test_gpt2_no_unreachable_params(self, tiny_gpt2, tiny_synthetic_basis):
        # Need a basis matching the tiny_gpt2 d_model=16; tiny_synthetic_basis
        # is 16-d already.
        assert self._audit(tiny_gpt2, tiny_synthetic_basis) == []

    def test_llama_no_unreachable_params(
        self, tiny_llama, feature_basis_128_to_32
    ):
        assert self._audit(tiny_llama, feature_basis_128_to_32) == []

    def test_llama_tied_no_unreachable_params(
        self, tiny_llama_tied, feature_basis_128_to_32
    ):
        assert self._audit(tiny_llama_tied, feature_basis_128_to_32) == []

    def test_gemma2_no_unreachable_params(
        self, tiny_gemma2, feature_basis_128_to_32
    ):
        assert self._audit(tiny_gemma2, feature_basis_128_to_32) == []


# ---------------------------------------------------------------------------
# §7.8 — NativeModelConfig family-field validation
# ---------------------------------------------------------------------------


class TestNativeModelConfigFamily:
    def test_missing_family_raises(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(TypeError, match="family"):
            NativeModelConfig(  # type: ignore[call-arg]
                hidden_size=32, qkv_inner_size=32, num_layers=2,
                num_heads=4, head_dim=8, intermediate_size=64, vocab_size=100,
            )

    def test_unknown_family_rejected(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="family"):
            NativeModelConfig(
                family="not-real", hidden_size=32, qkv_inner_size=32,
                num_layers=2, num_heads=4, head_dim=8,
                intermediate_size=64, vocab_size=100,
            )

    def test_n_kv_heads_defaults_to_num_heads(self):
        from saeforge.model import NativeModelConfig

        config = NativeModelConfig(
            family="llama", hidden_size=32, qkv_inner_size=32,
            num_layers=2, num_heads=4, head_dim=8,
            intermediate_size=64, vocab_size=100,
        )
        assert config.n_kv_heads == 4

    def test_n_kv_heads_must_divide_num_heads(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="n_kv_heads"):
            NativeModelConfig(
                family="llama", hidden_size=32, qkv_inner_size=32,
                num_layers=2, num_heads=4, head_dim=8,
                intermediate_size=64, vocab_size=100,
                n_kv_heads=3,  # 4 % 3 != 0
            )


class TestNativeModelConfigOutputKind:
    """§1.6 — invalid-combination matrix for the v0.4 output_kind /
    vocab_size / whisper_encoder cross-constraints.
    """

    def _lm_kwargs(self, **overrides):
        base = dict(
            family="llama",
            hidden_size=32,
            qkv_inner_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=100,
        )
        base.update(overrides)
        return base

    def _whisper_kwargs(self, **overrides):
        base = dict(
            family="whisper_encoder",
            hidden_size=32,
            qkv_inner_size=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            intermediate_size=64,
            vocab_size=0,
            output_kind="encoder_states",
        )
        base.update(overrides)
        return base

    def test_lm_default_output_kind_is_logits(self):
        from saeforge.model import NativeModelConfig

        config = NativeModelConfig(**self._lm_kwargs())
        assert config.output_kind == "logits"

    def test_logits_requires_positive_vocab(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="vocab_size > 0"):
            NativeModelConfig(**self._lm_kwargs(vocab_size=0))

    def test_encoder_states_requires_zero_vocab(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="vocab_size == 0"):
            NativeModelConfig(**self._whisper_kwargs(vocab_size=100))

    def test_encoder_states_requires_whisper_family(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="whisper_encoder"):
            NativeModelConfig(
                **self._lm_kwargs(
                    output_kind="encoder_states", vocab_size=0
                )
            )

    def test_whisper_family_rejects_logits_output_kind(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="output_kind='encoder_states'"):
            NativeModelConfig(
                **self._whisper_kwargs(output_kind="logits")
            )

    def test_unknown_output_kind_rejected(self):
        from saeforge.model import NativeModelConfig

        with pytest.raises(ValueError, match="output_kind"):
            NativeModelConfig(**self._lm_kwargs(output_kind="probabilities"))

    def test_round_trip_preserves_output_kind(self):
        from saeforge.model import NativeModelConfig

        original = NativeModelConfig(**self._whisper_kwargs())
        round_tripped = NativeModelConfig.from_dict(original.to_dict())
        assert round_tripped.output_kind == "encoder_states"
        assert round_tripped.vocab_size == 0
        assert round_tripped.family == "whisper_encoder"

    def test_pre_v04_config_dict_defaults_to_logits(self):
        """A serialized config from before v0.4 lacks ``output_kind``;
        deserialising should fall through to the ``"logits"`` default."""
        from saeforge.model import NativeModelConfig

        payload = self._lm_kwargs()
        assert "output_kind" not in payload
        config = NativeModelConfig.from_dict(payload)
        assert config.output_kind == "logits"


# ---------------------------------------------------------------------------
# §7.6 (continued) — registry hygiene: registering a custom adapter works
# and the dispatcher honours first-match-wins.
# ---------------------------------------------------------------------------


class TestRegistryHygiene:
    def test_custom_adapter_dispatch(self):
        # Don't pollute the real registry — verify the public API works
        # with a one-off subclass + cleanup at the end.
        from saeforge.adapters import _REGISTRY

        class MyHost:
            pass

        class MyAdapter(ArchitectureAdapter):
            family = "gpt2"  # reuse a known family for the smoke test

            def walk(self, host, projector, *, attention_width="host"):
                return {}

            def build_native_config(self, host, n_features, *, attention_width="host"):
                raise NotImplementedError

            def native_module_class(self):
                raise NotImplementedError

        before = len(_REGISTRY)
        register_adapter(MyHost, MyAdapter())
        try:
            assert isinstance(adapter_for(MyHost()), MyAdapter)
        finally:
            # Restore registry — pop the entry we added.
            del _REGISTRY[before:]
