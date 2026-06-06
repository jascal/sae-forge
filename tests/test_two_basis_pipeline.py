"""Tests for ForgePipeline two-basis-forge knobs (task 5)."""

from __future__ import annotations

import pytest

from saeforge.forge import ForgePipeline
from saeforge.projector import SubspaceProjector


def _pipeline(basis, **kw):
    return ForgePipeline(basis=basis, projector=SubspaceProjector(basis), **kw)


def test_toggles_off_builds_no_augmented_basis(tiny_gpt2, tiny_synthetic_basis):
    p = _pipeline(tiny_synthetic_basis)
    assert p._build_augmented_basis(tiny_gpt2) is None


def test_composition_preserve_builds_per_layer_subspace_and_report(tiny_gpt2, tiny_synthetic_basis):
    p = _pipeline(tiny_synthetic_basis, composition_preserve=True, composition_rank=2)
    aug = p._build_augmented_basis(tiny_gpt2)
    assert aug is not None
    n_layer = tiny_gpt2.config.n_layer
    assert set(aug.composition) == set(range(n_layer))
    rep = p._last_augmented_report
    assert rep["d_model"] == tiny_synthetic_basis.W_dec.shape[1]
    for ell in range(n_layer):
        layer_rep = rep["layers"][ell]
        assert layer_rep["preserved_dim"] > 0
        assert 0.0 <= layer_rep["preserved_fraction"] <= 1.0
        assert 0.0 <= layer_rep["U_C_overlap_with_basis"] <= 1.0 + 1e-9


def test_assertion_preserve_selects_k_sharp_atoms(tiny_gpt2, tiny_synthetic_basis):
    p = _pipeline(tiny_synthetic_basis, assertion_preserve=True, assertion_k=3)
    atoms = p._select_assertion_atoms(3)
    assert atoms.shape == (3, tiny_synthetic_basis.W_dec.shape[1])
    aug = p._build_augmented_basis(tiny_gpt2)
    assert aug.assertion_atoms.shape[0] == 3


def test_preserve_and_hybrid_mutually_exclusive(tiny_synthetic_basis):
    with pytest.raises(ValueError, match="at most one"):
        _pipeline(tiny_synthetic_basis, composition_preserve=True, hybrid_bridge=True)


def test_host_wrapped_rejects_preserve(tiny_synthetic_basis):
    with pytest.raises(ValueError, match="host_wrapped"):
        _pipeline(tiny_synthetic_basis, forward_mode="host_wrapped", composition_preserve=True)
