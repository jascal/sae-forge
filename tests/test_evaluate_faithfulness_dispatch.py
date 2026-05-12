"""Tests for the family dispatch in
``saeforge.actions.evaluate_faithfulness`` (§5.5 of the
forge-whisper-encoder change).

The action splits on ``ctx['_native_model'].config.family``: LM
families (gpt2/llama/gemma2/qwen2/qwen3) go through the KL helper,
``whisper_encoder`` goes through the cosine helper. The dispatch
preserves byte-equivalence on the LM side (the FSM safety-net tests
in ``tests/test_forge_outer_loop_fsm.py`` enforce that
end-to-end); these tests exercise the branch behavior in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_native_model(family: str):
    """Stand-in for ``ctx['_native_model']`` carrying just a ``.config.family``."""
    config = SimpleNamespace(family=family)
    return SimpleNamespace(config=config)


# ---------------------------------------------------------------------------
# LM family routes to KL
# ---------------------------------------------------------------------------


class TestLMDispatch:
    def test_gpt2_calls_kl_helper(self):
        from saeforge.actions import evaluate_faithfulness

        ctx = {
            "_host_model": object(),
            "_native_model": _stub_native_model("gpt2"),
            "_eval_input_ids": object(),
            "iterations": 1,
            "current_iter": 0,
            "min_faithfulness": 0.0,
            "best_perplexity": float("inf"),
            "device": "cpu",
        }
        with patch(
            "saeforge.forge._kl_from_input_ids",
            return_value=0.123,
        ) as mock_kl:
            result = evaluate_faithfulness(ctx)

        mock_kl.assert_called_once()
        assert result["faithfulness"] == pytest.approx(0.123)
        # perplexity = exp(KL)
        import math

        assert result["perplexity"] == pytest.approx(math.exp(0.123))

    def test_llama_routes_through_kl(self):
        from saeforge.actions import evaluate_faithfulness

        ctx = {
            "_host_model": object(),
            "_native_model": _stub_native_model("llama"),
            "_eval_input_ids": object(),
            "iterations": 1,
            "current_iter": 0,
            "min_faithfulness": 0.0,
            "best_perplexity": float("inf"),
            "device": "cpu",
        }
        with patch("saeforge.forge._kl_from_input_ids", return_value=0.05) as m:
            result = evaluate_faithfulness(ctx)
        m.assert_called_once()
        assert result["faithfulness"] == pytest.approx(0.05)

    def test_missing_native_model_falls_to_lm_path_with_zero_kl(self):
        """Pre-loading or bootstrap context paths where ``_native_model`` is
        absent SHALL not raise; the action behaves as the v0.3 KL path with
        ``kl = 0.0`` (the v0.3 default-faith fallback)."""
        from saeforge.actions import evaluate_faithfulness

        ctx = {
            "iterations": 1,
            "current_iter": 0,
            "min_faithfulness": 0.0,
            "best_perplexity": float("inf"),
        }
        result = evaluate_faithfulness(ctx)
        assert result["faithfulness"] == 0.0
        # exp(0) = 1.0
        assert result["perplexity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# whisper_encoder family routes to cosine
# ---------------------------------------------------------------------------


class TestWhisperEncoderDispatch:
    def _whisper_ctx(self, **overrides):
        ctx = {
            "_host_model": object(),
            "_native_model": _stub_native_model("whisper_encoder"),
            "_eval_audio_features": object(),
            "iterations": 1,
            "current_iter": 0,
            "min_faithfulness": 0.0,
            "best_perplexity": float("inf"),
            "device": "cpu",
        }
        ctx.update(overrides)
        return ctx

    def test_whisper_calls_cosine_helper(self):
        from saeforge.actions import evaluate_faithfulness

        with patch(
            "saeforge.audio_eval.cosine_faithfulness",
            return_value=0.87,
        ) as mock_cos:
            result = evaluate_faithfulness(self._whisper_ctx())

        mock_cos.assert_called_once()
        assert result["faithfulness"] == pytest.approx(0.87)
        # perplexity analog = 1 - cosine
        assert result["perplexity"] == pytest.approx(0.13)

    def test_whisper_does_not_call_kl_helper(self):
        from saeforge.actions import evaluate_faithfulness

        with patch("saeforge.forge._kl_from_input_ids") as mock_kl, patch(
            "saeforge.audio_eval.cosine_faithfulness", return_value=0.5
        ):
            evaluate_faithfulness(self._whisper_ctx())

        mock_kl.assert_not_called()

    def test_whisper_min_faithfulness_threshold_passes_when_cosine_above(self):
        """min_faith=0.95 with cosine=0.97 → cosine ≥ min_faith.
        Combined with iterations=2 and best_perplexity=infinite → should_continue=True.
        """
        from saeforge.actions import evaluate_faithfulness

        ctx = self._whisper_ctx(
            min_faithfulness=0.95,
            iterations=2,
            current_iter=0,
            best_perplexity=float("inf"),
        )
        with patch(
            "saeforge.audio_eval.cosine_faithfulness", return_value=0.97
        ):
            result = evaluate_faithfulness(ctx)
        assert result["should_continue"] is True

    def test_whisper_min_faithfulness_threshold_fails_when_cosine_below(self):
        """min_faith=0.95 with cosine=0.80 → cosine < min_faith → should_continue=False."""
        from saeforge.actions import evaluate_faithfulness

        ctx = self._whisper_ctx(
            min_faithfulness=0.95,
            iterations=2,
            current_iter=0,
            best_perplexity=float("inf"),
        )
        with patch(
            "saeforge.audio_eval.cosine_faithfulness", return_value=0.80
        ):
            result = evaluate_faithfulness(ctx)
        assert result["should_continue"] is False

    def test_whisper_progress_check_against_best_perplexity(self):
        """For encoder, perplexity = 1 - cosine. If 1 - cosine ≥ best_perp,
        should_continue is False (no progress to make).
        """
        from saeforge.actions import evaluate_faithfulness

        # cosine=0.9 → perplexity=0.1; best=0.05 → 0.1 > 0.05, no progress.
        ctx = self._whisper_ctx(
            iterations=2, current_iter=0, best_perplexity=0.05
        )
        with patch(
            "saeforge.audio_eval.cosine_faithfulness", return_value=0.9
        ):
            result = evaluate_faithfulness(ctx)
        assert result["should_continue"] is False

    def test_whisper_uses_precomputed_states_when_present(self):
        """When ctx['_eval_encoder_states'] is set, evaluate_faithfulness
        passes them through to cosine_faithfulness's precomputed kwarg
        so the host forward is skipped inside the FSM."""
        from saeforge.actions import evaluate_faithfulness

        precaptured = object()
        ctx = self._whisper_ctx()
        ctx["_eval_encoder_states"] = precaptured

        with patch(
            "saeforge.audio_eval.cosine_faithfulness", return_value=0.8
        ) as mock_cos:
            evaluate_faithfulness(ctx)

        # The mock SHOULD be called with precomputed_host_states matching
        # the ctx field. Inspect the call kwargs.
        assert mock_cos.call_count == 1
        _, kwargs = mock_cos.call_args
        assert kwargs["precomputed_host_states"] is precaptured

    def test_whisper_missing_audio_features_returns_zero_cosine(self):
        """If ``_eval_audio_features`` is absent (bootstrap path), the
        action SHALL not raise; faithfulness defaults to 0.0."""
        from saeforge.actions import evaluate_faithfulness

        ctx = {
            "_host_model": object(),
            "_native_model": _stub_native_model("whisper_encoder"),
            # no _eval_audio_features
            "iterations": 1,
            "current_iter": 0,
            "min_faithfulness": 0.0,
            "best_perplexity": float("inf"),
        }
        result = evaluate_faithfulness(ctx)
        assert result["faithfulness"] == 0.0
        assert result["perplexity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Logged transition entry carries the family-aware fields
# ---------------------------------------------------------------------------


class TestLoggedEntry:
    def test_log_entry_records_dispatch_outcome(self):
        from saeforge.actions import evaluate_faithfulness

        ctx = {
            "_host_model": object(),
            "_native_model": _stub_native_model("whisper_encoder"),
            "_eval_audio_features": object(),
            "iterations": 1,
            "current_iter": 0,
            "min_faithfulness": 0.0,
            "best_perplexity": float("inf"),
            "transitions_log": [],
        }
        with patch(
            "saeforge.audio_eval.cosine_faithfulness", return_value=0.5
        ):
            evaluate_faithfulness(ctx)

        log = ctx["transitions_log"]
        last = log[-1]
        assert last["action"] == "evaluate_faithfulness"
        assert last["faithfulness"] == pytest.approx(0.5)
        assert last["perplexity"] == pytest.approx(0.5)
