"""Tests for SubspaceProjector — pure-numpy core + GPT-2 weight walker."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge import SubspaceProjector


def test_encode_decode_roundtrip_full_rank(tiny_synthetic_basis):
    """For full row-rank D, encode(decode(z)) == z."""
    projector = SubspaceProjector(tiny_synthetic_basis)
    rng = np.random.default_rng(7)
    z = rng.standard_normal((4, tiny_synthetic_basis.n_features))
    reconstructed = projector.encode(projector.decode(z))
    assert np.allclose(reconstructed, z, atol=1e-9)


def test_residual_input_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    W = np.random.default_rng(0).standard_normal((tiny_synthetic_basis.d_model, 32))
    out = projector.project_residual_input(W)
    assert out.shape == (tiny_synthetic_basis.n_features, 32)


def test_residual_output_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    W = np.random.default_rng(1).standard_normal((32, tiny_synthetic_basis.d_model))
    out = projector.project_residual_output(W)
    assert out.shape == (32, tiny_synthetic_basis.n_features)


def test_residual_bias_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    b = np.random.default_rng(2).standard_normal((tiny_synthetic_basis.d_model,))
    out = projector.project_residual_bias(b)
    assert out.shape == (tiny_synthetic_basis.n_features,)


def test_unembed_shape(tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    W = np.random.default_rng(3).standard_normal((100, tiny_synthetic_basis.d_model))
    out = projector.project_unembed(W)
    assert out.shape == (100, tiny_synthetic_basis.n_features)


def test_residual_inverse_consistency_full_rank(tiny_synthetic_basis):
    """For a residual-input matrix A and a residual-aligned vector h_d, projection commutes:
    (h_d @ A) @ E should equal (h_d @ E) @ (D @ A) @ E    (lossy under non-square)
    But for the identity round-trip h_d -> h_n -> h_d, projecting then deprojecting any
    matrix that consumes the residual leaves the bilinear form invariant on the basis-spanned
    subspace. We check the weaker, exact identity:
    h_n @ project_residual_input(A) == h_d @ A     when h_d == h_n @ D (i.e., h_d is in span(D)).
    """
    projector = SubspaceProjector(tiny_synthetic_basis)
    rng = np.random.default_rng(4)
    n = tiny_synthetic_basis.n_features
    d = tiny_synthetic_basis.d_model
    A = rng.standard_normal((d, 12))
    h_n = rng.standard_normal((3, n))
    h_d = h_n @ tiny_synthetic_basis.W_dec
    direct = h_d @ A
    via_basis = h_n @ projector.project_residual_input(A)
    assert np.allclose(direct, via_basis, atol=1e-9)


def test_scale_boost_amplifies_encode(tiny_synthetic_basis):
    p1 = SubspaceProjector(tiny_synthetic_basis, scale_boost=1.0)
    p2 = SubspaceProjector(tiny_synthetic_basis, scale_boost=2.5)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, tiny_synthetic_basis.d_model))
    assert np.allclose(p2.encode(x), 2.5 * p1.encode(x))


def test_project_module_gpt2(tiny_gpt2, tiny_synthetic_basis):
    projector = SubspaceProjector(tiny_synthetic_basis)
    weights = projector.project_module(tiny_gpt2)

    n_feat = tiny_synthetic_basis.n_features
    d_model = tiny_gpt2.config.n_embd
    n_inner = tiny_gpt2.config.n_inner
    vocab = tiny_gpt2.config.vocab_size
    max_pos = tiny_gpt2.config.n_positions

    assert weights["transformer.wte.weight"].shape == (vocab, n_feat)
    assert weights["transformer.wpe.weight"].shape == (max_pos, n_feat)

    for i in range(tiny_gpt2.config.n_layer):
        prefix = f"transformer.h.{i}"
        assert weights[f"{prefix}.ln_1.weight"].shape == (n_feat,)
        assert weights[f"{prefix}.ln_1.bias"].shape == (n_feat,)
        assert weights[f"{prefix}.attn.c_attn.weight"].shape == (n_feat, 3 * d_model)
        assert weights[f"{prefix}.attn.c_attn.bias"].shape == (3 * d_model,)
        assert weights[f"{prefix}.attn.c_proj.weight"].shape == (d_model, n_feat)
        assert weights[f"{prefix}.attn.c_proj.bias"].shape == (n_feat,)
        assert weights[f"{prefix}.ln_2.weight"].shape == (n_feat,)
        assert weights[f"{prefix}.mlp.c_fc.weight"].shape == (n_feat, n_inner)
        assert weights[f"{prefix}.mlp.c_fc.bias"].shape == (n_inner,)
        assert weights[f"{prefix}.mlp.c_proj.weight"].shape == (n_inner, n_feat)
        assert weights[f"{prefix}.mlp.c_proj.bias"].shape == (n_feat,)

    assert weights["transformer.ln_f.weight"].shape == (n_feat,)
    assert weights["lm_head.weight"].shape == (vocab, n_feat)


def test_project_module_unsupported_arch_raises(tiny_synthetic_basis):
    # ``project_module`` imports transformers eagerly; gate the test on
    # the [torch] extra so the no-extras install stays green.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    projector = SubspaceProjector(tiny_synthetic_basis)

    class FakeBert:
        pass

    # Multi-architecture-support: the dispatcher raises a registry-aware
    # error naming the offending type and the registered adapter set.
    with pytest.raises(NotImplementedError) as excinfo:
        projector.project_module(FakeBert())
    msg = str(excinfo.value)
    assert "FakeBert" in msg
    assert "Registered:" in msg
    # Whichever bundled adapters are loaded should appear in the list;
    # GPT2LMHeadModel is always there.
    assert "GPT2LMHeadModel" in msg
