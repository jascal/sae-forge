"""Tests for saeforge.augmented_basis — verbatim two-subspace preserve."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.augmented_basis import AugmentedBasis
from saeforge.composition_subspace import CompositionSubspace


def _comp(d_model, r, layer=0, seed=1):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d_model, r)))
    return {layer: CompositionSubspace(U=Q[:, :r], layer=layer, rank=r, source_heads="all", d_model=d_model)}


def test_null_augment_is_single_basis(tiny_synthetic_basis):
    ab = AugmentedBasis(tiny_synthetic_basis)
    W_eff, mask = ab.kept_subspace(0)
    assert np.array_equal(W_eff, tiny_synthetic_basis.W_dec)
    assert mask.dtype == bool and not mask.any()
    assert ab.preserved_dimension(0) == 0


def test_preserve_mask_marks_exactly_the_verbatim_rows(tiny_synthetic_basis):
    d = tiny_synthetic_basis.W_dec.shape[1]
    U_A = np.linalg.qr(np.random.default_rng(2).standard_normal((d, 2)))[0][:, :2].T  # (2, d)
    ab = AugmentedBasis(tiny_synthetic_basis, assertion_atoms=U_A, composition=_comp(d, 3))
    W_eff, mask = ab.kept_subspace(0)
    assert W_eff.shape == tiny_synthetic_basis.W_dec.shape  # n_features fixed
    assert int(mask.sum()) == 5  # 2 assertion + 3 composition
    assert ab.preserved_dimension(0) == 5
    # every verbatim direction appears as a preserved row
    V = np.concatenate([U_A, ab.composition[0].U.T], axis=0)
    preserved = W_eff[mask]
    for v in V:
        assert np.min(np.linalg.norm(preserved - v, axis=1)) < 1e-9


def test_composition_only_displaces_least_important_atoms(tiny_synthetic_basis):
    d = tiny_synthetic_basis.W_dec.shape[1]
    norms = np.linalg.norm(tiny_synthetic_basis.W_dec, axis=1)
    ab = AugmentedBasis(tiny_synthetic_basis, composition=_comp(d, 3))
    _, mask = ab.kept_subspace(0)
    # the 3 replaced rows are exactly the 3 lowest-norm atoms
    assert set(np.where(mask)[0]) == set(np.argsort(norms)[:3])


def test_d_model_mismatch_raises(tiny_synthetic_basis):
    d = tiny_synthetic_basis.W_dec.shape[1]
    with pytest.raises(ValueError, match="assertion_atoms"):
        AugmentedBasis(tiny_synthetic_basis, assertion_atoms=np.zeros((2, d + 1)))
    with pytest.raises(ValueError, match="d_model"):
        AugmentedBasis(tiny_synthetic_basis, composition=_comp(d + 1, 2))


def test_too_many_preserved_dims_raises(tiny_synthetic_basis):
    d = tiny_synthetic_basis.W_dec.shape[1]
    n_features = tiny_synthetic_basis.W_dec.shape[0]
    ab = AugmentedBasis(tiny_synthetic_basis, composition=_comp(d, min(d, n_features + 1)))
    if min(d, n_features + 1) > n_features:
        with pytest.raises(ValueError, match="exceeds basis n_features"):
            ab.kept_subspace(0)
