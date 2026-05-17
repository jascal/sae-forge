"""Built-in faithfulness targets and the family-based default policy.

The :class:`~saeforge.eval.faithfulness.FaithfulnessTarget` protocol
lives in :mod:`saeforge.eval.faithfulness` alongside the existing
``faithfulness_kl`` function it generalises. This sub-package houses
the two built-in implementations:

- :class:`KLTarget` — LM-family default; per-token KL.
- :class:`CosineTarget` — Whisper-encoder default; per-frame cosine.

:func:`_default_target_for` is the family-dispatch policy the
``evaluate_faithfulness`` action falls back to when the user has not
passed an explicit ``faithfulness=`` to :class:`~saeforge.ForgePipeline`.
"""

from __future__ import annotations

from saeforge.eval.faithfulness import FaithfulnessTarget
from saeforge.eval.targets.cosine import CosineTarget
from saeforge.eval.targets.kl import KLTarget

__all__ = [
    "CosineTarget",
    "KLTarget",
    "_default_target_for",
]


_LM_FAMILIES = frozenset({"gpt2", "llama", "gemma2", "qwen2", "qwen3"})


def _default_target_for(family: str | None) -> FaithfulnessTarget:
    """Return the default target for a forged-model family.

    LM families (``gpt2`` / ``llama`` / ``gemma2`` / ``qwen2`` /
    ``qwen3``) map to :class:`KLTarget`. ``whisper_encoder`` maps to
    :class:`CosineTarget`. Any other family — including ``None`` —
    raises :class:`ValueError` whose message names the offending
    family and the supported set, mirroring the v0.4
    ``evaluate_faithfulness`` action's behaviour on unknown families.
    """
    if family == "whisper_encoder":
        return CosineTarget()
    if family in _LM_FAMILIES:
        return KLTarget()
    supported = sorted(_LM_FAMILIES | {"whisper_encoder"})
    raise ValueError(
        f"_default_target_for: unsupported family {family!r}. "
        f"Supported: {supported}. Pass an explicit "
        "ForgePipeline(faithfulness=...) target to override."
    )
