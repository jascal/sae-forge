"""End-to-end: ``ForgePipeline(faithfulness=<custom-target>)`` overrides
the family dispatch.

The "GTAlignmentTarget" defined inline is what
``examples/forge_with_gt_alignment.py`` ships in a fuller form. Here we
only need a target that:

- declares ``better_when="higher"`` to exercise that branch of the
  ``should_continue`` predicate,
- ignores ``host`` (proves the protocol allows it), and
- returns a deterministic ``(score, perplexity_analog)`` we can assert on.

The KL helper is patched to raise. The test passes iff the action layer
never consults it — i.e. the user-supplied target fully overrides the
family-dispatched default.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from saeforge import ForgePipeline, SubspaceProjector


class _ConstantTarget:
    """Returns a fixed score regardless of inputs; flags when called."""

    name = "constant"
    better_when = "higher"

    def __init__(self, score: float) -> None:
        self._score = score
        self.calls = 0

    def score(
        self,
        *,
        forged: Any,
        host: Any,
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        self.calls += 1
        return float(self._score), max(0.0, 1.0 - float(self._score))


def test_custom_target_overrides_family_dispatch_imperative(
    tiny_gpt2, tiny_synthetic_basis, tmp_path, monkeypatch
):
    """The imperative synthetic path with a custom target MUST call the
    target and MUST NOT call ``_kl_from_input_ids``.
    """
    pytest.importorskip("torch")
    import torch

    # Patch _kl_from_input_ids to raise. If the imperative path consults
    # it anywhere, the test fails immediately with this AssertionError.
    def _boom(*args, **kwargs):  # pragma: no cover — only hit on regression
        raise AssertionError(
            "_kl_from_input_ids was called even though faithfulness=<custom> "
            "was set on the pipeline"
        )

    monkeypatch.setattr("saeforge.forge._kl_from_input_ids", _boom)

    target = _ConstantTarget(score=0.42)
    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        faithfulness=target,
    )
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "custom", eval_input_ids=input_ids
    )

    assert target.calls == 1
    assert result.faithfulness == pytest.approx(0.42)
    assert result.faithfulness_target_name == "constant"


def test_custom_target_overrides_family_dispatch_fsm(
    tiny_gpt2, tiny_synthetic_basis, tmp_path, monkeypatch
):
    """The FSM synthetic path with a custom target MUST thread the
    target into the FSM ctx as ``_faithfulness_target`` and MUST NOT
    call ``_kl_from_input_ids``.
    """
    pytest.importorskip("torch")
    import torch

    def _boom(*args, **kwargs):  # pragma: no cover — only hit on regression
        raise AssertionError(
            "_kl_from_input_ids was called on the FSM path even though "
            "faithfulness=<custom> was set on the pipeline"
        )

    monkeypatch.setattr("saeforge.forge._kl_from_input_ids", _boom)

    target = _ConstantTarget(score=0.77)
    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        faithfulness=target,
        orchestrator="fsm",
    )
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "custom-fsm", eval_input_ids=input_ids
    )

    assert target.calls >= 1
    assert result.faithfulness == pytest.approx(0.77)
    assert result.faithfulness_target_name == "constant"


def test_higher_better_should_continue_predicate(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """A ``better_when="higher"`` target's score gates ``should_continue``
    with the positive ``score >= min_faithfulness`` convention (NOT the
    legacy KL-negation convention).

    Concretely: with iterations=2 (so the iter-budget gate is open) and
    min_faithfulness=0.5, a target returning 0.7 SHOULD continue; one
    returning 0.3 should NOT.
    """
    pytest.importorskip("torch")
    import torch

    from saeforge.actions import evaluate_faithfulness

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "higher", eval_input_ids=input_ids
    )

    def _run_with(score: float) -> bool:
        target = _ConstantTarget(score=score)
        ctx = {
            "_native_model": result.model,
            "_host_model": tiny_gpt2,
            "_eval_input_ids": input_ids,
            "_faithfulness_target": target,
            "iterations": 2,
            "current_iter": 0,
            "min_faithfulness": 0.5,
            "best_perplexity": float("inf"),
            "device": "cpu",
        }
        delta = evaluate_faithfulness(ctx)
        return bool(delta["should_continue"])

    assert _run_with(0.7) is True
    assert _run_with(0.3) is False
