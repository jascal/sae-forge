"""Tests for the two-basis projector arm (project_module(augmented=...))."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.augmented_basis import AugmentedBasis
from saeforge.composition_subspace import extract_composition_subspace
from saeforge.projector import SubspaceProjector


def test_augmented_none_is_byte_identical(tiny_gpt2, tiny_synthetic_basis):
    proj = SubspaceProjector(tiny_synthetic_basis)
    ref = proj.project_module(tiny_gpt2)
    null = proj.project_module(tiny_gpt2, augmented=AugmentedBasis(tiny_synthetic_basis))
    assert ref.keys() == null.keys()
    for k in ref:
        assert np.array_equal(ref[k], null[k]), k


def test_augmented_keys_and_shapes_invariant(tiny_gpt2, tiny_synthetic_basis):
    proj = SubspaceProjector(tiny_synthetic_basis)
    ref = proj.project_module(tiny_gpt2)
    comp = extract_composition_subspace(tiny_gpt2, layers=[0, 1], rank=2)
    d = tiny_synthetic_basis.W_dec.shape[1]
    U_A = np.linalg.qr(np.random.default_rng(0).standard_normal((d, 2)))[0][:, :2].T
    aug = AugmentedBasis(tiny_synthetic_basis, assertion_atoms=U_A, composition=comp)
    out = proj.project_module(tiny_gpt2, augmented=aug)
    assert out.keys() == ref.keys()
    for k in ref:
        assert out[k].shape == ref[k].shape, k


def test_augmented_preserves_host_QK_on_U_C(tiny_gpt2, tiny_synthetic_basis):
    """E @ W_eff = scale_boost · P_rowspace, so it is scale_boost·identity on U_C
    (U_C is inserted into the rowspace) — the circuit-faithfulness invariant."""
    proj = SubspaceProjector(tiny_synthetic_basis)  # n_features=8 <= d_model=16 -> scale_boost 1.0
    comp = extract_composition_subspace(tiny_gpt2, layers=[0], rank=2)
    aug = AugmentedBasis(tiny_synthetic_basis, composition=comp)
    W_eff, _ = aug.kept_subspace(0)
    E = np.linalg.pinv(W_eff) * proj.scale_boost
    EW = E @ W_eff
    U = comp[0].U
    for j in range(U.shape[1]):
        u = U[:, j]
        assert np.linalg.norm(EW @ u - proj.scale_boost * u) < 1e-9


def test_augmented_changes_only_composition_layers(tiny_gpt2, tiny_synthetic_basis):
    proj = SubspaceProjector(tiny_synthetic_basis)
    ref = proj.project_module(tiny_gpt2)
    comp = extract_composition_subspace(tiny_gpt2, layers=[0], rank=3)
    out = proj.project_module(tiny_gpt2, augmented=AugmentedBasis(tiny_synthetic_basis, composition=comp))
    # layer-0 attention changed (its kept subspace now carries U_C)
    assert not np.allclose(
        out["transformer.h.0.attn.c_attn.weight"], ref["transformer.h.0.attn.c_attn.weight"]
    )
    # a non-composition block (layer 1) and a non-block key are untouched
    assert np.array_equal(
        out["transformer.h.1.attn.c_attn.weight"], ref["transformer.h.1.attn.c_attn.weight"]
    )
    assert np.array_equal(out["transformer.wte.weight"], ref["transformer.wte.weight"])


def test_hybrid_and_augmented_mutually_exclusive(tiny_gpt2, tiny_synthetic_basis):
    proj = SubspaceProjector(tiny_synthetic_basis)
    with pytest.raises(ValueError, match="cannot be combined"):
        proj.project_module(
            tiny_gpt2, hybrid=object(), augmented=AugmentedBasis(tiny_synthetic_basis)
        )
