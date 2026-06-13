"""Tests for SubspaceProjector.encoder_override (change add-capability-trained-encoder, task 1.4)."""
import numpy as np
import pytest

from saeforge.basis import FeatureBasis
from saeforge.projector import SubspaceProjector


def _basis(n: int, d: int) -> FeatureBasis:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n, d)).astype(np.float32)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
        original_norms=np.linalg.norm(W, axis=1).astype(np.float32),
    )


def test_override_none_is_byte_identical():
    """encoder_override=None reproduces the existing pinv*scale encode exactly."""
    basis = _basis(n=8, d=4)  # under-complete: scale_boost=1.0, no footgun warning
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    proj = SubspaceProjector(basis=basis)
    expected = x @ basis.pseudoinverse() * proj.scale_boost
    np.testing.assert_array_equal(proj.encode(x), expected)


def test_override_equal_to_pinv_reproduces_default():
    """An override equal to pinv(W_dec)*scale reproduces the default encode (round-trip identity)."""
    basis = _basis(n=6, d=4)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((7, 4)).astype(np.float32)
    default = SubspaceProjector(basis=basis)
    E = (basis.pseudoinverse() * default.scale_boost).astype(np.float32)
    over = SubspaceProjector(basis=basis, encoder_override=E)
    np.testing.assert_allclose(over.encode(x), default.encode(x), rtol=1e-5, atol=1e-6)


def test_override_is_full_map_scale_not_reapplied():
    """With scale_boost != 1, the override is the full map (scale not double-applied)."""
    basis = _basis(n=16, d=4)  # over-complete
    rng = np.random.default_rng(3)
    E = rng.standard_normal((4, 16)).astype(np.float32)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    proj = SubspaceProjector(basis=basis, scale_boost=0.25, encoder_override=E)
    np.testing.assert_allclose(proj.encode(x), x @ E, rtol=1e-6)


@pytest.mark.parametrize("bad", [(16, 4), (4, 16, 1), (3, 8)])
def test_bad_shape_override_rejected(bad):
    """Wrong-shape override raises ValueError naming the expected (d_model, n_features)."""
    basis = _basis(n=8, d=4)  # expects (4, 8)
    E = np.zeros(bad, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(4, 8\)"):
        SubspaceProjector(basis=basis, encoder_override=E)


def test_project_residual_full_uses_override():
    """The v0.2 both-sides path stays consistent with the override."""
    basis = _basis(n=8, d=4)
    rng = np.random.default_rng(4)
    E = rng.standard_normal((4, 8)).astype(np.float32)
    W = rng.standard_normal((4, 4)).astype(np.float32)
    proj = SubspaceProjector(basis=basis, encoder_override=E)
    np.testing.assert_allclose(proj.project_residual_full(W), basis.W_dec @ W @ E, rtol=1e-5)
