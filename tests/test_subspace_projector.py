"""Tests for SubspaceProjector — pure-numpy core + GPT-2 weight walker."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from saeforge import SubspaceProjector
from saeforge.basis import FeatureBasis


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


# ---------------------------------------------------------------------------
# Regression tests for scale_boost='auto' + over-complete-basis warning.
#
# Empirical anchor: GPT-2 (d_model=768) with a 1024-feature Polygram-
# compressed basis required scale_boost ~ 0.25 for stable training; the
# default 1.0 produced activations that overflowed bf16 / saturated
# softmax. The fix adds an "auto" mode that returns
# min(1.0, d_model/n_features) for over-complete bases, plus a
# UserWarning when n>d and scale_boost=1.0 (the default footgun).
# ---------------------------------------------------------------------------


def _basis(n: int, d: int):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n, d)).astype(np.float32)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
        original_norms=np.linalg.norm(W, axis=1).astype(np.float32),
    )


class TestScaleBoostAuto:
    def test_auto_returns_one_when_n_lte_d(self):
        # Under-complete or square basis: identity-preserving default.
        proj = SubspaceProjector(_basis(8, 16), scale_boost="auto")
        assert proj.scale_boost == 1.0

    def test_auto_returns_d_over_n_when_overcomplete(self):
        # GPT-2-shaped over-complete: 768 / 1024 = 0.75.
        proj = SubspaceProjector(_basis(1024, 768), scale_boost="auto")
        assert proj.scale_boost == 768 / 1024

    def test_auto_at_exact_n_eq_d(self):
        # Boundary: n == d collapses to the under-complete branch (1.0).
        proj = SubspaceProjector(_basis(64, 64), scale_boost="auto")
        assert proj.scale_boost == 1.0

    def test_unknown_string_rejected(self):
        with pytest.raises(ValueError, match="positive float or 'auto'"):
            SubspaceProjector(_basis(8, 16), scale_boost="not-real")

    def test_explicit_numeric_passes_through(self):
        # Any positive float is accepted unchanged.
        proj = SubspaceProjector(_basis(1024, 768), scale_boost=0.25)
        assert proj.scale_boost == 0.25

    def test_negative_still_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            SubspaceProjector(_basis(8, 16), scale_boost=-1.0)


class TestOverCompleteWarning:
    def test_default_one_with_overcomplete_basis_warns(self):
        # n=1024 > d=768 + scale_boost=1.0 default → footgun warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SubspaceProjector(_basis(1024, 768))
        msgs = [str(w.message) for w in caught if "over-complete" in str(w.message)]
        assert len(msgs) == 1
        assert "n_features=1024" in msgs[0]
        assert "d_model=768" in msgs[0]
        assert "scale_boost" in msgs[0]
        # Empirical anchor named so the user can act.
        assert "0.25" in msgs[0]

    def test_under_complete_basis_does_not_warn(self):
        # n=8, d=16 + default 1.0 → no warning (this is the canonical
        # well-behaved case).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SubspaceProjector(_basis(8, 16))
        assert not any("over-complete" in str(w.message) for w in caught)

    def test_overcomplete_basis_with_explicit_scale_does_not_warn(self):
        # User picked an explicit value → they know what they're doing,
        # no scolding.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SubspaceProjector(_basis(1024, 768), scale_boost=0.25)
        assert not any("over-complete" in str(w.message) for w in caught)

    def test_overcomplete_basis_with_auto_does_not_warn(self):
        # "auto" resolved to 0.75 (< 1.0); not the default 1.0 so the
        # warning shouldn't fire.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SubspaceProjector(_basis(1024, 768), scale_boost="auto")
        assert not any("over-complete" in str(w.message) for w in caught)
