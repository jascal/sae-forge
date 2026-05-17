"""KLTarget — per-token KL between forged and host logits.

Implements the :class:`saeforge.eval.faithfulness.FaithfulnessTarget`
protocol with ``better_when="lower"``. The score it returns is bit-equal
to ``_kl_from_input_ids(forged, host, input_ids, device=device)``; the
perplexity analog is ``exp(score)``, matching the v0.4 LM evaluator.

Reads two ctx keys:

- ``_eval_input_ids``: pre-tokenised eval prompts (the FSM-side fast
  path; populated by ``ForgePipeline._run_real_fsm`` /
  ``_run_synthetic_fsm``).
- ``device``: torch device string, defaults to ``"cpu"``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class KLTarget:
    """Per-token KL faithfulness scorer."""

    name = "kl"
    better_when = "lower"

    def score(
        self,
        *,
        forged: Any,
        host: Any,
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        from saeforge.forge import _kl_from_input_ids

        try:
            input_ids = ctx["_eval_input_ids"]
        except KeyError as exc:  # pragma: no cover — explicit message in raise
            raise KeyError(
                "KLTarget.score requires ctx['_eval_input_ids'] (pre-tokenised "
                "eval prompts). Populate it on the pipeline via eval_prompts "
                "or pass it through the FSM ctx."
            ) from exc
        if input_ids is None:
            raise KeyError(
                "KLTarget.score requires ctx['_eval_input_ids'] to be non-None"
            )

        device = ctx.get("device", "cpu")
        kl = _kl_from_input_ids(forged, host, input_ids, device=device)
        # exp(KL) matches the v0.4 _evaluate_lm perplexity analog; clamp
        # negative KL noise (fp32 round-off near zero) to keep the analog
        # finite and positive.
        perplexity = math.exp(kl) if kl >= 0 else math.inf
        return float(kl), float(perplexity)
