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
    # legacy reader-geometry path (composition_heads="all")
    p = _pipeline(tiny_synthetic_basis, composition_preserve=True, composition_rank=2,
                  composition_heads="all")
    aug = p._build_augmented_basis(tiny_gpt2)
    assert aug is not None
    n_layer = tiny_gpt2.config.n_layer
    assert set(aug.composition) == set(range(n_layer))
    rep = p._last_augmented_report
    assert rep["d_model"] == tiny_synthetic_basis.W_dec.shape[1]
    assert rep["composition_mode"] == "reader-geometry"
    for ell in range(n_layer):
        layer_rep = rep["layers"][ell]
        assert layer_rep["preserved_dim"] > 0
        assert 0.0 <= layer_rep["preserved_fraction"] <= 1.0
        assert 0.0 <= layer_rep["U_C_overlap_with_basis"] <= 1.0 + 1e-9


def test_writer_output_explicit_heads_builds_subspace_and_report(tiny_gpt2, tiny_synthetic_basis):
    """Explicit (layer, head) writers → writer-output U_C, replicated per layer, report lists them."""
    writers = [(0, 1), (1, 2)]
    p = _pipeline(tiny_synthetic_basis, composition_preserve=True, composition_rank=2,
                  composition_heads=writers, composition_mode="writer-output")
    aug = p._build_augmented_basis(tiny_gpt2)
    assert aug is not None
    n_layer = tiny_gpt2.config.n_layer
    assert set(aug.composition) == set(range(n_layer))
    # same writer-output subspace preserved at every layer
    U0 = aug.composition[0].U
    for ell in range(1, n_layer):
        assert aug.composition[ell].U.shape == U0.shape
    rep = p._last_augmented_report
    assert rep["composition_mode"] == "writer-output"
    assert rep["writer_heads"] == [[0, 1], [1, 2]]


def test_writer_output_preset_detects_heads(tiny_gpt2, tiny_synthetic_basis, monkeypatch):
    """A preset resolves writer heads from eval_prompts (detector wiring) and records them with scores."""
    import saeforge.circuit_heads as ch
    # pin the detector so the wiring test is deterministic (real detection is covered in
    # test_circuit_heads.py); the pipeline must consume the (layer, head, score) triples.
    monkeypatch.setattr(ch, "identify", lambda host, corpus, preset, **kw: [(0, 1, 0.42), (1, 3, 0.31)])
    p = _pipeline(tiny_synthetic_basis, composition_preserve=True, composition_rank=2,
                  composition_heads="prev-token", composition_mode="writer-output",
                  eval_prompts=["the cat sat on the mat and the cat ran"])
    # stub the calibration corpus so the wiring test never hits the network (the real path loads
    # the host's tokenizer via AutoTokenizer.from_pretrained); detection itself is monkeypatched.
    monkeypatch.setattr(p, "_calibration_corpus", lambda host: list(range(16)))
    aug = p._build_augmented_basis(tiny_gpt2)
    assert aug is not None
    rep = p._last_augmented_report
    assert rep["composition_mode"] == "writer-output"
    assert rep["writer_heads"] == [[0, 1], [1, 3]]
    assert rep["writer_scores"] == [0.42, 0.31]


def test_writer_output_preset_without_corpus_raises(tiny_gpt2, tiny_synthetic_basis):
    p = _pipeline(tiny_synthetic_basis, composition_preserve=True, composition_heads="prev-token")
    with pytest.raises(ValueError, match="calibration corpus"):
        p._build_augmented_basis(tiny_gpt2)


def test_bad_composition_mode_rejected(tiny_synthetic_basis):
    with pytest.raises(ValueError, match="composition_mode"):
        _pipeline(tiny_synthetic_basis, composition_mode="bogus")


def test_bad_composition_heads_rejected(tiny_synthetic_basis):
    with pytest.raises(ValueError, match="composition_heads"):
        _pipeline(tiny_synthetic_basis, composition_heads="not-a-preset")
    with pytest.raises(ValueError, match="composition_heads"):
        _pipeline(tiny_synthetic_basis, composition_heads=[(0, 1, 2)])


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


def test_composition_preserve_warns_experimental_unvalidated(tiny_synthetic_basis):
    # the writer-output U_C circuit-preservation claim was retracted; opting into composition_preserve
    # must warn that the feature is experimental/unvalidated (docs/two_basis_forge.md).
    with pytest.warns(UserWarning, match="EXPERIMENTAL and UNVALIDATED"):
        _pipeline(tiny_synthetic_basis, composition_preserve=True)


def test_assertion_only_does_not_trigger_composition_warning(tiny_synthetic_basis, recwarn):
    # assertion-preserve (cov95) is a separate, non-retracted claim — it must not emit the composition warning.
    _pipeline(tiny_synthetic_basis, assertion_preserve=True, assertion_k=3)
    assert not any("composition_preserve" in str(w.message) for w in recwarn.list)


def test_default_pipeline_emits_no_composition_warning(tiny_synthetic_basis, recwarn):
    _pipeline(tiny_synthetic_basis)  # both toggles off (the byte-identical default path)
    assert not any("composition_preserve" in str(w.message) for w in recwarn.list)
