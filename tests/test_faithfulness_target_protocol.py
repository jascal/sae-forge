"""Pluggable-faithfulness protocol tests.

Covers:

- :class:`FaithfulnessTarget` is runtime-checkable; both built-in
  targets satisfy it.
- :class:`KLTarget` and :class:`CosineTarget` round-trip the legacy
  helpers byte-identically on the same inputs.
- :func:`_default_target_for` maps families to the right built-in
  and raises a clean ValueError on unknown families.
- ``ForgePipeline(faithfulness=None, ...)`` produces a result whose
  ``faithfulness`` and ``faithfulness_target_name`` match
  ``ForgePipeline(faithfulness=KLTarget(), ...)`` — the load-bearing
  byte-identity property the spec pins.
"""

from __future__ import annotations

import pytest

from saeforge import ForgePipeline, SubspaceProjector
from saeforge.eval.faithfulness import FaithfulnessTarget
from saeforge.eval.targets import CosineTarget, KLTarget, _default_target_for


def test_kl_and_cosine_satisfy_protocol():
    assert isinstance(KLTarget(), FaithfulnessTarget)
    assert isinstance(CosineTarget(), FaithfulnessTarget)


def test_arbitrary_object_does_not_satisfy_protocol():
    assert not isinstance(object(), FaithfulnessTarget)


def test_kltarget_metadata():
    target = KLTarget()
    assert target.name == "kl"
    assert target.better_when == "lower"


def test_cosinetarget_metadata():
    target = CosineTarget()
    assert target.name == "cosine"
    assert target.better_when == "higher"


@pytest.mark.parametrize(
    "family,expected_name",
    [
        ("gpt2", "kl"),
        ("llama", "kl"),
        ("gemma2", "kl"),
        ("qwen2", "kl"),
        ("qwen3", "kl"),
        ("whisper_encoder", "cosine"),
    ],
)
def test_default_target_for_known_families(family, expected_name):
    assert _default_target_for(family).name == expected_name


@pytest.mark.parametrize("family", ["fictional", None, ""])
def test_default_target_for_unknown_family_raises(family):
    with pytest.raises(ValueError, match="unsupported family"):
        _default_target_for(family)


def test_kltarget_score_matches_legacy_kl_helper(tiny_gpt2, tiny_synthetic_basis):
    """KLTarget.score(...) MUST return the same float as the legacy
    ``_kl_from_input_ids`` helper on the same inputs — the v0.4
    behaviour contract.
    """
    pytest.importorskip("torch")
    import torch

    from saeforge.forge import _kl_from_input_ids

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, "/tmp/sae-forge-kl-parity", eval_input_ids=input_ids
    )

    legacy_kl = _kl_from_input_ids(result.model, tiny_gpt2, input_ids, device="cpu")
    target = KLTarget()
    score, perplexity = target.score(
        forged=result.model,
        host=tiny_gpt2,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )
    assert score == pytest.approx(legacy_kl, abs=1e-9)
    assert perplexity == pytest.approx(
        float(__import__("math").exp(legacy_kl)), rel=1e-9
    )


def test_kltarget_score_missing_input_ids_raises():
    target = KLTarget()
    with pytest.raises(KeyError, match="_eval_input_ids"):
        target.score(forged=object(), host=object(), ctx={})


def test_pipeline_default_matches_explicit_kltarget(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """``ForgePipeline(faithfulness=None)`` and
    ``ForgePipeline(faithfulness=KLTarget())`` MUST produce the same
    faithfulness score and ``faithfulness_target_name`` on the same
    fixture — the spec's byte-identity invariant.
    """
    pytest.importorskip("torch")
    import torch

    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    projector_a = SubspaceProjector(tiny_synthetic_basis)
    pipeline_a = ForgePipeline(basis=tiny_synthetic_basis, projector=projector_a)
    result_a = pipeline_a.run_synthetic(
        tiny_gpt2, tmp_path / "default", eval_input_ids=input_ids
    )

    projector_b = SubspaceProjector(tiny_synthetic_basis)
    pipeline_b = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector_b,
        faithfulness=KLTarget(),
    )
    result_b = pipeline_b.run_synthetic(
        tiny_gpt2, tmp_path / "explicit", eval_input_ids=input_ids
    )

    assert result_a.faithfulness == pytest.approx(result_b.faithfulness, abs=1e-9)
    assert result_a.faithfulness_target_name == "kl"
    assert result_b.faithfulness_target_name == "kl"
    assert result_a.n_params == result_b.n_params
