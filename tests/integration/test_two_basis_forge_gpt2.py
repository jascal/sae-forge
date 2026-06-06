"""Integration: end-to-end two-basis forge on tiny GPT-2 (task 7.1)."""

from __future__ import annotations

import math

import pytest

from saeforge import ForgePipeline, NativeModel, SubspaceProjector


def _pipe(basis, **kw):
    return ForgePipeline(basis=basis, projector=SubspaceProjector(basis), **kw)


def test_two_basis_forge_end_to_end(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    import torch

    eval_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))
    pipe = _pipe(tiny_synthetic_basis, composition_preserve=True, composition_rank=2)
    res = pipe.run_synthetic(tiny_gpt2, tmp_path / "tb", eval_input_ids=eval_ids)

    assert isinstance(res.model, NativeModel)
    assert math.isfinite(res.faithfulness)
    assert (tmp_path / "tb" / "forged" / "model.safetensors").is_file()
    for p in res.model.torch_module.parameters():
        assert torch.isfinite(p).all()
    rep = pipe._last_augmented_report
    assert rep["layers"][0]["preserved_dim"] > 0
    assert 0.0 <= rep["layers"][0]["U_C_overlap_with_basis"] <= 1.0 + 1e-9


def test_two_basis_off_is_byte_identical_to_single_basis(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    import torch

    eval_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))
    base = _pipe(tiny_synthetic_basis)
    off = _pipe(tiny_synthetic_basis, composition_preserve=False, assertion_preserve=False)
    r1 = base.run_synthetic(tiny_gpt2, tmp_path / "a", eval_input_ids=eval_ids)
    r2 = off.run_synthetic(tiny_gpt2, tmp_path / "b", eval_input_ids=eval_ids)
    # toggles off -> augmented basis is None -> projection is byte-identical ->
    # identical forged model -> identical faithfulness.
    assert r1.faithfulness == r2.faithfulness
