"""Tests for the hybrid-bridge-forge implementation."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.basis import FeatureBasis
from saeforge.bridges import BridgeConfig, make_bridge
from saeforge.hybrid_basis import HybridBasisBundle


def _basis(*, n=8, d=16, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, d)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1),
        original_norms=np.linalg.norm(W, axis=1),
    )


# ---------------------------------------------------------------------------
# BridgeModule
# ---------------------------------------------------------------------------


class TestBridgeModule:
    def test_shape_preservation(self):
        pytest.importorskip("torch")
        import torch

        b = make_bridge(8, BridgeConfig())
        x = torch.randn(2, 7, 8)
        y = b(x)
        assert y.shape == x.shape

    def test_identity_init_no_ln_no_nonlin_reproduces_input(self):
        torch = pytest.importorskip("torch")
        b = make_bridge(
            8,
            BridgeConfig(init="identity", nonlin="none", pre_layernorm=False),
        )
        x = torch.randn(1, 1, 8)
        y = b(x)
        assert torch.allclose(y, x, atol=1e-6)

    def test_zero_init_outputs_zero(self):
        torch = pytest.importorskip("torch")
        b = make_bridge(
            8,
            BridgeConfig(init="zero", nonlin="none", pre_layernorm=False),
        )
        x = torch.randn(1, 1, 8)
        y = b(x)
        assert torch.allclose(y, torch.zeros_like(y))

    def test_orthogonal_init_frobenius_sqrt_n(self):
        torch = pytest.importorskip("torch")
        b = make_bridge(16, BridgeConfig(init="orthogonal"))
        # An orthogonal matrix of shape (n, n) has Frobenius norm sqrt(n).
        fro = torch.linalg.norm(b.linear.weight).item()
        assert abs(fro - 4.0) < 1e-4  # sqrt(16) = 4

    def test_train_false_freezes_parameters(self):
        pytest.importorskip("torch")
        b = make_bridge(8, BridgeConfig(train=False))
        for p in b.parameters():
            assert p.requires_grad is False

    def test_relu_activation_clamps_negatives(self):
        torch = pytest.importorskip("torch")
        b = make_bridge(
            8,
            BridgeConfig(init="identity", nonlin="relu", pre_layernorm=False),
        )
        x = torch.tensor([[[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]]])
        y = b(x)
        # Identity + ReLU: negatives clamped to 0.
        assert torch.allclose(y, torch.clamp(x, min=0.0), atol=1e-6)

    def test_n_features_too_small_raises(self):
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match="n_features must be >= 2"):
            make_bridge(1, BridgeConfig())

    def test_wrong_last_dim_raises_in_forward(self):
        torch = pytest.importorskip("torch")
        b = make_bridge(8, BridgeConfig())
        with pytest.raises(ValueError, match="expected last dim 8"):
            b(torch.randn(2, 3, 7))


# ---------------------------------------------------------------------------
# HybridBasisBundle
# ---------------------------------------------------------------------------


class TestHybridBasisBundle:
    def test_d_model_mismatch_raises(self):
        with pytest.raises(ValueError, match="d_model mismatch"):
            HybridBasisBundle(
                basis_embed=_basis(d=16, seed=0),
                basis_mid=_basis(d=16, seed=1),
                basis_lm_head=_basis(d=32, seed=2),
                n_layer=12,
            )

    def test_n_features_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_features mismatch"):
            HybridBasisBundle(
                basis_embed=_basis(n=8, seed=0),
                basis_mid=_basis(n=16, seed=1),
                basis_lm_head=_basis(n=8, seed=2),
                n_layer=12,
            )

    def test_too_few_layers_raises(self):
        with pytest.raises(ValueError, match="n_layer must be >= 3"):
            HybridBasisBundle(
                basis_embed=_basis(seed=0),
                basis_mid=_basis(seed=1),
                basis_lm_head=_basis(seed=2),
                n_layer=2,
            )

    def test_routing_gpt2_n_layer_12(self):
        bundle = HybridBasisBundle(
            basis_embed=_basis(seed=0),
            basis_mid=_basis(seed=1),
            basis_lm_head=_basis(seed=2),
            n_layer=12,
        )
        assert bundle.basis_for_layer(0) is bundle.basis_embed
        for i in range(1, 11):
            assert bundle.basis_for_layer(i) is bundle.basis_mid
        assert bundle.basis_for_layer(11) is bundle.basis_lm_head

    def test_out_of_range_raises(self):
        bundle = HybridBasisBundle(
            basis_embed=_basis(seed=0),
            basis_mid=_basis(seed=1),
            basis_lm_head=_basis(seed=2),
            n_layer=12,
        )
        with pytest.raises(IndexError):
            bundle.basis_for_layer(12)
        with pytest.raises(IndexError):
            bundle.basis_for_layer(-1)

    def test_boundaries_property(self):
        bundle = HybridBasisBundle(
            basis_embed=_basis(seed=0),
            basis_mid=_basis(seed=1),
            basis_lm_head=_basis(seed=2),
            n_layer=12,
        )
        assert bundle.boundaries == (0, 11)


# ---------------------------------------------------------------------------
# Hybrid routing through the projector
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_gpt2_untied():
    """A tiny GPT-2 (4 layers, untied embeddings) suitable for hybrid forging."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=4,
        n_head=4,
        n_inner=32,
        tie_word_embeddings=False,
    )
    return GPT2LMHeadModel(config).eval()


class TestProjectorHybridDispatch:
    def test_hybrid_none_byte_identical_to_single_basis(self, tiny_gpt2_untied):
        """Passing ``hybrid=None`` produces the same dict as omitting the kwarg."""
        from saeforge.projector import SubspaceProjector

        basis = _basis(n=8, d=16, seed=0)
        proj = SubspaceProjector(basis)
        out_a = proj.project_module(tiny_gpt2_untied, hybrid=None)
        out_b = proj.project_module(tiny_gpt2_untied)
        assert set(out_a.keys()) == set(out_b.keys())
        for k in out_a:
            np.testing.assert_array_equal(out_a[k], out_b[k])

    def test_hybrid_dispatch_same_keyset_as_single_basis(self, tiny_gpt2_untied):
        from saeforge.projector import SubspaceProjector

        bundle = HybridBasisBundle(
            basis_embed=_basis(seed=0),
            basis_mid=_basis(seed=1),
            basis_lm_head=_basis(seed=2),
            n_layer=tiny_gpt2_untied.config.n_layer,
        )
        proj = SubspaceProjector(bundle.basis_mid)
        out_hybrid = proj.project_module(tiny_gpt2_untied, hybrid=bundle)
        out_single = proj.project_module(tiny_gpt2_untied)
        assert set(out_hybrid.keys()) == set(out_single.keys())
        for k in out_hybrid:
            assert out_hybrid[k].shape == out_single[k].shape

    def test_hybrid_routes_blocks_to_correct_basis(self, tiny_gpt2_untied):
        """Block 0 → embed basis; blocks 1..L-2 → mid; L-1 → lm_head."""
        from saeforge.projector import SubspaceProjector

        # Use three identifiable bases by giving them very different seeds — the
        # walk-three-pick-by-key strategy means each region's output should
        # bit-match the single-basis walk for that basis.
        b_embed = _basis(seed=10)
        b_mid = _basis(seed=20)
        b_lm = _basis(seed=30)
        bundle = HybridBasisBundle(
            basis_embed=b_embed,
            basis_mid=b_mid,
            basis_lm_head=b_lm,
            n_layer=tiny_gpt2_untied.config.n_layer,
        )
        proj_mid = SubspaceProjector(b_mid)
        proj_embed = SubspaceProjector(b_embed)
        proj_lm = SubspaceProjector(b_lm)

        out_hybrid = proj_mid.project_module(tiny_gpt2_untied, hybrid=bundle)
        out_embed = proj_embed.project_module(tiny_gpt2_untied)
        out_mid = proj_mid.project_module(tiny_gpt2_untied)
        out_lm = proj_lm.project_module(tiny_gpt2_untied)

        L = tiny_gpt2_untied.config.n_layer
        # Block 0 keys → embed
        for k, v in out_hybrid.items():
            if ".h.0." in k:
                np.testing.assert_array_equal(v, out_embed[k]), f"key {k} should route to embed"
            elif f".h.{L-1}." in k:
                np.testing.assert_array_equal(v, out_lm[k]), f"key {k} should route to lm-head"
            elif any(f".h.{i}." in k for i in range(1, L - 1)):
                np.testing.assert_array_equal(v, out_mid[k]), f"key {k} should route to mid"
        # Non-block keys: wte/wpe → embed; ln_f/lm_head → lm-head
        np.testing.assert_array_equal(
            out_hybrid["transformer.wte.weight"], out_embed["transformer.wte.weight"]
        )
        np.testing.assert_array_equal(
            out_hybrid["transformer.wpe.weight"], out_embed["transformer.wpe.weight"]
        )
        np.testing.assert_array_equal(
            out_hybrid["transformer.ln_f.weight"], out_lm["transformer.ln_f.weight"]
        )
        np.testing.assert_array_equal(out_hybrid["lm_head.weight"], out_lm["lm_head.weight"])


# ---------------------------------------------------------------------------
# ForgePipeline validation
# ---------------------------------------------------------------------------


class TestForgePipelineValidation:
    def test_hybrid_without_basis_embed_raises(self):
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis(seed=0)
        with pytest.raises(ValueError, match="hybrid_bridge=True requires"):
            ForgePipeline(
                basis=b,
                projector=SubspaceProjector(b),
                hybrid_bridge=True,
                basis_embed=None,
                basis_lm_head=_basis(seed=2),
            )

    def test_hybrid_shape_mismatch_raises(self):
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis(n=8, seed=0)
        with pytest.raises(ValueError, match="n_features"):
            ForgePipeline(
                basis=b,
                projector=SubspaceProjector(b),
                hybrid_bridge=True,
                basis_embed=_basis(n=16, seed=1),
                basis_lm_head=_basis(n=8, seed=2),
            )

    def test_hybrid_disabled_with_extras_silently_accepted(self):
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis(seed=0)
        # Default hybrid_bridge=False; supplying the extras should NOT error.
        pipeline = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            basis_embed=_basis(seed=1),
            basis_lm_head=_basis(seed=2),
        )
        assert pipeline.hybrid_bridge is False
