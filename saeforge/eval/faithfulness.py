"""Faithfulness KL — token-level KL divergence between a forged model and its host.

This module also defines :class:`FaithfulnessTarget`, the protocol that
generalises the loop-gating faithfulness signal beyond hard-coded KL.
Built-in implementations live in :mod:`saeforge.eval.targets`.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from saeforge.utils.lazy import require_extra


@runtime_checkable
class FaithfulnessTarget(Protocol):
    """Protocol for pluggable loop-gating faithfulness scorers.

    A target is a small adapter that turns a forged-vs-host comparison
    into a scalar score the FSM's refine loop can gate on. ``KLTarget``
    and ``CosineTarget`` (in :mod:`saeforge.eval.targets`) are the two
    built-in implementations; users can supply their own by satisfying
    this protocol.

    Members
    -------
    name:
        Short stable slug (e.g. ``"kl"``, ``"cosine"``, ``"gt_alignment"``).
        Surfaces as ``ForgeResult.faithfulness_target_name`` and in the
        FSM transitions-log entry. SHOULD be lowercase snake_case or
        kebab-case so downstream JSON / metadata consumers can match
        without quoting surprises.
    better_when:
        ``"higher"`` if larger scores indicate higher faithfulness
        (cosine, GT-alignment, probe accuracy), ``"lower"`` if smaller
        scores do (KL, MSE). The FSM's ``min_faithfulness`` predicate
        consults this field per call.
    score:
        Called as ``score(forged=..., host=..., ctx=...)`` and returns
        ``(score, perplexity_analog)`` where ``perplexity_analog`` is a
        positive-real quantity the FSM's ``perplexity < best_perplexity``
        progress check consumes. Convention:

        - ``better_when == "lower"``: ``perplexity_analog`` is a
          monotonically *increasing* function of the score
          (canonical: ``exp(score)`` for KL).
        - ``better_when == "higher"``: ``perplexity_analog`` is a
          monotonically *decreasing* function of the score
          (canonical: ``1 - score`` for cosine, clamped at 0).

    Notes for implementers
    ----------------------
    - The ``host`` argument MAY be ignored. Targets that don't consult
      a teacher (GT-alignment, monosemanticity, probe accuracy reading
      a cached probe) SHOULD accept ``host`` for protocol conformance
      but SHOULD NOT move it or run a forward through it. sae-forge
      still loads the host on the FSM path today; a future
      ``requires_host`` opt-out is tracked as a follow-up.
    - Implementations SHALL NOT mutate ``ctx``. They MUST raise a
      ``KeyError`` (or ``ValueError``) naming the expected key if a
      required ctx field is missing — silent zero-score returns from
      missing inputs are a debugging hazard.
    - Third-party targets SHOULD namespace their ctx keys with a
      module-specific prefix (e.g. ``_myorg_input_ids``,
      ``_gt_alignment_inputs``) to avoid clashes with sae-forge
      built-ins, which use the ``_eval_*`` prefix.
    """

    name: str
    better_when: Literal["higher", "lower"]

    def score(
        self,
        *,
        forged: Any,
        host: Any,
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        ...


def faithfulness_kl(
    forged_model,
    host_model,
    prompts: list[str],
    *,
    tokenizer=None,
    max_length: int = 32,
    device: str = "cpu",
) -> float:
    """Mean per-token KL(host || forged) across ``prompts``.

    ``forged_model`` is a sae-forge ``NativeModel``; ``host_model`` is the HF
    model that was projected. Both must use the same tokenizer (passed via
    ``tokenizer`` or auto-loaded from the host's config when available).
    Returns the per-token KL averaged across all prompts and positions.
    """
    torch = require_extra("torch", "torch")
    F = torch.nn.functional

    if tokenizer is None:
        transformers = require_extra("transformers", "torch")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            host_model.config._name_or_path or "gpt2"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    forged_module = forged_model.torch_module.to(device).eval()
    host_module = host_model.to(device).eval()

    with torch.no_grad():
        forged_logits = forged_module(input_ids)
        host_out = host_module(input_ids=input_ids, attention_mask=attention_mask)
        host_logits = host_out.logits if hasattr(host_out, "logits") else host_out[0]

    log_q = F.log_softmax(forged_logits, dim=-1)
    log_p = F.log_softmax(host_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    masked = kl * attention_mask
    n_tokens = attention_mask.sum().clamp(min=1)
    return float((masked.sum() / n_tokens).item())
