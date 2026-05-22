"""Built-in faithfulness targets and the family-based default policy.

The :class:`~saeforge.eval.faithfulness.FaithfulnessTarget` protocol
lives in :mod:`saeforge.eval.faithfulness` alongside the existing
``faithfulness_kl`` function it generalises. This sub-package houses
the three built-in implementations:

- :class:`KLTarget` — LM-family default; per-token KL.
- :class:`CosineTarget` — Whisper-encoder default; per-frame cosine.
- :class:`GroundTruthTarget` — fixture-specific opt-in; per-feature
  × per-label AUC against a binary label matrix. Never a family
  default; passed via ``ForgePipeline(faithfulness=...)``.

:func:`_default_target_for` is the family-dispatch policy the
``evaluate_faithfulness`` action falls back to when the user has not
passed an explicit ``faithfulness=`` to :class:`~saeforge.ForgePipeline`.
"""

from __future__ import annotations

from saeforge.eval.faithfulness import FaithfulnessTarget
from saeforge.eval.targets.cosine import CosineTarget
from saeforge.eval.targets.gt_alignment import GroundTruthTarget
from saeforge.eval.targets.kl import KLTarget
from saeforge.eval.targets.token_cosine import TokenCosineTarget

__all__ = [
    "CosineTarget",
    "GroundTruthTarget",
    "KLTarget",
    "TokenCosineTarget",
    "_default_target_for",
]


def _default_target_for(family: str | None) -> FaithfulnessTarget:
    """Return the default target for a forged-model family.

    Dispatches via the architecture-adapter registry: each adapter's
    :meth:`~saeforge.adapters.base.ArchitectureAdapter.default_faithfulness_target`
    declares what scorer the family defaults to. LM-family adapters
    inherit the ABC's :class:`KLTarget` default; the Whisper-encoder
    adapter overrides to :class:`CosineTarget`. Unknown families —
    including ``None`` — raise :class:`ValueError` whose message names
    the offending family and the registered set.

    Before the world-model-protocol refactor this dispatch lived as a
    hardcoded ``_LM_FAMILIES`` frozenset; the new path is byte-
    identical on the bundled families and additionally picks up any
    family registered by third-party adapters. One intentional
    behavioural change: ``qwen3_moe`` previously raised here (it was
    missing from ``_LM_FAMILIES``) and now returns ``KLTarget()`` like
    its sibling LM families.
    """
    from saeforge.adapters import _REGISTRY, adapter_for_family

    try:
        adapter = adapter_for_family(family)
    except ValueError as exc:
        registered = sorted({a.family for _, a in _REGISTRY})
        raise ValueError(
            f"_default_target_for: unsupported family {family!r}. "
            f"Registered: {registered}. Pass an explicit "
            "ForgePipeline(faithfulness=...) target to override."
        ) from exc
    return adapter.default_faithfulness_target()
