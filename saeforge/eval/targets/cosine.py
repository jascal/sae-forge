"""CosineTarget — per-frame cosine similarity for Whisper-encoder forges.

Implements the :class:`saeforge.eval.faithfulness.FaithfulnessTarget`
protocol with ``better_when="higher"``. The score is bit-equal to
``cosine_faithfulness(forged, host, audio_features, ...)``; the
perplexity analog is ``max(0, 1 - score)``, matching the v0.4
whisper-encoder evaluator.

Reads three ctx keys:

- ``_eval_audio_features`` (required): mel-spectrogram input to the
  forged encoder.
- ``_eval_encoder_states`` (optional): pre-captured host encoder
  output. When present, skips the host forward.
- ``device``: torch device string, defaults to ``"cpu"``.
"""

from __future__ import annotations

from typing import Any, Mapping


class CosineTarget:
    """Per-frame cosine-similarity faithfulness scorer for Whisper encoders."""

    name = "cosine"
    better_when = "higher"

    def score(
        self,
        *,
        forged: Any,
        host: Any,
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        from saeforge.audio_eval import cosine_faithfulness

        try:
            audio_features = ctx["_eval_audio_features"]
        except KeyError as exc:  # pragma: no cover — explicit message in raise
            raise KeyError(
                "CosineTarget.score requires ctx['_eval_audio_features'] "
                "(mel-spectrogram input). Populate it on the pipeline via "
                "eval_audio_features."
            ) from exc
        if audio_features is None:
            raise KeyError(
                "CosineTarget.score requires ctx['_eval_audio_features'] to "
                "be non-None"
            )

        precomputed = ctx.get("_eval_encoder_states")
        device = ctx.get("device", "cpu")
        if precomputed is None and host is None:
            # Same defensive zero the v0.4 _evaluate_whisper_encoder used
            # when neither a host nor a pre-capture is available.
            cosine = 0.0
        else:
            cosine = cosine_faithfulness(
                forged,
                host,
                audio_features,
                precomputed_host_states=precomputed,
                device=device,
            )
        perplexity = max(0.0, 1.0 - cosine)
        return float(cosine), float(perplexity)
