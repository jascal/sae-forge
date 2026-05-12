"""End-to-end hybrid-bridge forge on a tiny untied Llama.

Mirrors ``tests/integration/test_hybrid_bridge_gpt2.py``. Pins the
``hybrid-bridge-llama-family`` capability: when ``hybrid_bridge=True`` is
set on a Llama-family host, ``LlamaTransformer`` constructs ``BridgeModule``
instances and calls them at indices 0 and L-2 in the per-block forward
loop.

State-dict keys are prefixed ``model.bridges.*`` (Llama-family convention),
not ``transformer.bridges.*`` (GPT-2 convention). Both prefixes are
honest reflections of each host's HF naming; unifying is tracked as the
deferred ``forged-module-state-dict-normalization`` follow-up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _basis(*, n=32, d=128, seed=0):
    from saeforge.basis import FeatureBasis

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, d)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1),
        original_norms=np.linalg.norm(W, axis=1),
    )


def _hybrid_pipeline(host, *, n_features=32):
    from saeforge.bridges import BridgeConfig
    from saeforge.forge import ForgePipeline
    from saeforge.projector import SubspaceProjector

    d = host.config.hidden_size
    b_mid = _basis(n=n_features, d=d, seed=10)
    b_embed = _basis(n=n_features, d=d, seed=20)
    b_lm = _basis(n=n_features, d=d, seed=30)
    return ForgePipeline(
        basis=b_mid,
        projector=SubspaceProjector(b_mid),
        host_model_id="<offline>",
        hybrid_bridge=True,
        basis_embed=b_embed,
        basis_lm_head=b_lm,
        bridge_config=BridgeConfig(),
    )


def _build_forged_module(pipeline, host):
    from saeforge.model import NativeModel, _config_from_host

    bundle = pipeline._build_hybrid_bundle(host)
    weights = pipeline.projector.project_module(host, hybrid=bundle)
    config = _config_from_host(host, pipeline.basis.n_features)
    if bundle is not None:
        config.bridges = True
        config.bridge_init = pipeline.bridge_config.init
        config.bridge_nonlin = pipeline.bridge_config.nonlin
        config.bridge_pre_layernorm = pipeline.bridge_config.pre_layernorm
    return NativeModel.from_projected_weights(config, weights)


@pytest.fixture
def tiny_llama_tied_4layer():
    """Tied 4-layer Llama — used to exercise the tied-embedding refusal under hybrid."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config).eval()


class TestT0TinyLlamaSmoke:
    def test_hybrid_forge_constructs_with_bridges(self, tiny_llama):
        pipeline = _hybrid_pipeline(tiny_llama)
        model = _build_forged_module(pipeline, tiny_llama)
        # Bridges live on the inner LlamaTransformer (accessed as model.model).
        bridges = model.torch_module.model.bridges
        assert bridges is not None
        assert "emb_mid" in bridges
        assert "mid_lm" in bridges
        sd = model.torch_module.state_dict()
        assert any(k.startswith("model.bridges.emb_mid.") for k in sd)
        assert any(k.startswith("model.bridges.mid_lm.") for k in sd)

    def test_forward_pass_finite(self, tiny_llama):
        import torch

        pipeline = _hybrid_pipeline(tiny_llama)
        model = _build_forged_module(pipeline, tiny_llama)
        input_ids = torch.randint(0, 1024, (2, 8))
        with torch.no_grad():
            logits = model.forward(input_ids)
        assert logits.shape == (2, 8, 1024)
        assert torch.isfinite(logits).all()

    def test_safetensors_round_trip(self, tiny_llama, tmp_path: Path):
        import torch

        from saeforge.model import NativeModel

        pipeline = _hybrid_pipeline(tiny_llama)
        model = _build_forged_module(pipeline, tiny_llama)
        model.save_pretrained(tmp_path)
        loaded = NativeModel.load_pretrained(tmp_path)
        sd_orig = model.torch_module.state_dict()
        sd_loaded = loaded.torch_module.state_dict()
        assert set(sd_orig.keys()) == set(sd_loaded.keys())
        for k in sd_orig:
            assert torch.allclose(sd_orig[k], sd_loaded[k]), f"mismatch on {k}"


class TestTiedEmbeddingRefusal:
    def test_tied_llama_refused_at_run_time(self, tiny_llama_tied_4layer):
        pipeline = _hybrid_pipeline(tiny_llama_tied_4layer)
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            pipeline._build_hybrid_bundle(tiny_llama_tied_4layer)


class TestByteEquivalenceWhenDisabled:
    """`hybrid_bridge=False` produces a forged module byte-identical to the pre-this-change path."""

    def test_disabled_path_has_no_bridges_in_state_dict(self, tiny_llama):
        from saeforge.forge import ForgePipeline
        from saeforge.model import NativeModel, _config_from_host
        from saeforge.projector import SubspaceProjector

        b = _basis(n=32, d=128, seed=7)
        pipeline = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
        )
        weights = pipeline.projector.project_module(tiny_llama)
        config = _config_from_host(tiny_llama, b.n_features)
        model = NativeModel.from_projected_weights(config, weights)
        assert model.torch_module.model.bridges is None
        sd = model.torch_module.state_dict()
        assert not any(".bridges." in k for k in sd)


class TestZeroInitInversion:
    """Same algebraic claim as the GPT-2 suite: zero-init bridges destroy signal; orthogonal preserves it."""

    def test_zero_init_drives_block_output_to_zero(self, tiny_llama):
        import torch

        from saeforge.bridges import BridgeConfig
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b_mid = _basis(n=32, d=128, seed=1)
        pipeline = ForgePipeline(
            basis=b_mid,
            projector=SubspaceProjector(b_mid),
            host_model_id="<offline>",
            hybrid_bridge=True,
            basis_embed=_basis(n=32, d=128, seed=2),
            basis_lm_head=_basis(n=32, d=128, seed=3),
            bridge_config=BridgeConfig(init="zero", nonlin="none", pre_layernorm=False),
        )
        model = _build_forged_module(pipeline, tiny_llama)
        x = torch.randn(1, 4, 32)
        emb_mid = model.torch_module.model.bridges["emb_mid"]
        assert torch.allclose(emb_mid(x), torch.zeros_like(x))

    def test_orthogonal_init_preserves_frobenius_norm(self, tiny_llama):
        import torch

        from saeforge.bridges import BridgeConfig
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b_mid = _basis(n=32, d=128, seed=1)
        pipeline = ForgePipeline(
            basis=b_mid,
            projector=SubspaceProjector(b_mid),
            host_model_id="<offline>",
            hybrid_bridge=True,
            basis_embed=_basis(n=32, d=128, seed=2),
            basis_lm_head=_basis(n=32, d=128, seed=3),
            bridge_config=BridgeConfig(init="orthogonal", nonlin="none", pre_layernorm=False),
        )
        model = _build_forged_module(pipeline, tiny_llama)
        x = torch.randn(1, 4, 32)
        emb_mid = model.torch_module.model.bridges["emb_mid"]
        y = emb_mid(x)
        assert torch.allclose(torch.linalg.norm(y), torch.linalg.norm(x), atol=1e-4)
