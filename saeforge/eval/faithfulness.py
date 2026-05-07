"""Faithfulness KL — token-level KL divergence between a forged model and its host."""

from __future__ import annotations


def faithfulness_kl(
    forged_model,
    host_model,
    prompts: list[str],
    *,
    max_new_tokens: int = 16,
) -> float:
    """Mean per-token KL(host || forged) across ``prompts``.

    Lazy-imports torch + transformers; requires the ``[torch]`` extra.
    """
    raise NotImplementedError(
        "faithfulness_kl is the change-5 deliverable; "
        "see openspec/changes/forge-pipeline/proposal.md."
    )
