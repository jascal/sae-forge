"""GroundTruthTarget — per-feature × per-label AUC for label-rich fixtures.

Implements the :class:`saeforge.eval.faithfulness.FaithfulnessTarget`
protocol with ``better_when="higher"``. The score is the mean over
labels of the best matching feature's AUC; the perplexity analog is
``max(0, 1 - score)``, mirroring :class:`CosineTarget`'s convention
for ``better_when="higher"`` scorers.

Reads two ctx keys:

- ``_eval_input_ids`` (required): pre-tokenised eval prompts (the
  same key consumed by :class:`KLTarget`). Built-in targets share
  the ``_eval_*`` ctx namespace.
- ``device`` (optional): torch device string, defaults to ``"cpu"``.

``host`` is accepted on ``score(...)`` for protocol conformance but
is never consulted. See the protocol docstring at
``saeforge/eval/faithfulness.py:55-60`` for the host-MAY-be-ignored
carve-out; the deferred ``requires_host`` opt-out is what eventually
lets the FSM skip the upstream host forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np
from scipy.stats import rankdata

from saeforge.utils import require_extra

if TYPE_CHECKING:  # pragma: no cover
    import torch


class GroundTruthTarget:
    """Ground-truth feature-alignment faithfulness scorer.

    Pools the forged residual stream across the sequence axis and
    computes per-feature × per-label AUC against a binary label
    matrix. The reported score is ``mean(max_over_features(AUC))``:
    for each label column, the best-matching feature's AUC, averaged
    across label columns.

    Parameters
    ----------
    labels:
        ``(N, M)`` binary-castable label matrix. Row ``i`` is the
        label vector for eval row ``i``; column ``j`` is one binary
        category. Coerced to ``float`` at construction time; no torch
        import happens.
    scorer:
        Scoring rule. Only ``"auc"`` ships in v1. The parameter is a
        forward-compatibility hook for future scorers (Pearson,
        Spearman, monosemanticity); supplying any other value raises
        :class:`ValueError`.
    pool:
        Cross-sequence reduction applied to the residual tensor
        before AUC. ``"mean"`` matches the sm-sae default and is the
        recommended choice for residual-stream features; ``"max"``
        targets max-activation features; ``"last"`` targets
        last-token-conditioned features.
    hidden_extractor:
        Optional ``(forged, input_ids) -> tensor`` hook. When
        ``None``, the default extractor tries
        ``forged.torch_module.transformer(input_ids)`` (GPT-2 shape)
        then ``forged.torch_module.model(input_ids)`` (Llama / Gemma
        / Qwen shape). If neither attribute exists, a
        :class:`RuntimeError` naming ``hidden_extractor=`` is raised.
        User-supplied extractors MAY pre-pool to 2D; pooling is
        skipped when the returned tensor is already 2D.
    """

    name = "gt_alignment"
    better_when = "higher"

    def __init__(
        self,
        labels: "np.ndarray | Any",
        *,
        scorer: Literal["auc"] = "auc",
        pool: Literal["mean", "max", "last"] = "mean",
        hidden_extractor: Callable[..., "torch.Tensor"] | None = None,
    ) -> None:
        labels_arr = np.asarray(labels, dtype=float)
        if labels_arr.ndim != 2:
            raise ValueError(
                f"GroundTruthTarget(labels=...) expects a 2D array; got "
                f"shape {labels_arr.shape!r} (ndim={labels_arr.ndim})."
            )
        if labels_arr.shape[0] < 1 or labels_arr.shape[1] < 1:
            raise ValueError(
                "GroundTruthTarget(labels=...) expects shape (N, M) with "
                f"N>=1 and M>=1; got shape {labels_arr.shape!r}."
            )
        if scorer != "auc":
            raise ValueError(
                f"GroundTruthTarget(scorer={scorer!r}) is not supported. "
                "Supported scorers in v1: {'auc'}."
            )
        if pool not in ("mean", "max", "last"):
            raise ValueError(
                f"GroundTruthTarget(pool={pool!r}) is not supported. "
                "Supported pool strategies: ('mean', 'max', 'last')."
            )
        self.labels = labels_arr
        self.scorer = scorer
        self.pool = pool
        self.hidden_extractor = hidden_extractor

    def score(
        self,
        *,
        forged: Any,
        host: Any,  # noqa: ARG002 — accepted for protocol conformance, never read
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        torch = require_extra("torch", "torch")

        try:
            input_ids = ctx["_eval_input_ids"]
        except KeyError as exc:
            raise KeyError(
                "GroundTruthTarget.score requires ctx['_eval_input_ids'] "
                "(pre-tokenised eval prompts). Populate it on the pipeline "
                "via eval_prompts or pass it through the FSM ctx."
            ) from exc
        if input_ids is None:
            raise KeyError(
                "GroundTruthTarget.score requires ctx['_eval_input_ids'] "
                "to be non-None."
            )

        n_inputs = int(input_ids.shape[0])
        if self.labels.shape[0] != n_inputs:
            raise ValueError(
                f"GroundTruthTarget.score: labels.shape[0]="
                f"{self.labels.shape[0]} does not match input_ids.shape[0]="
                f"{n_inputs}. The label matrix must be row-aligned with "
                "the eval set."
            )

        device = ctx.get("device", "cpu")
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(device)

        extractor = self.hidden_extractor or _default_hidden_extractor
        with torch.no_grad():
            hidden = extractor(forged, input_ids)

        if hidden.ndim == 3:
            if self.pool == "mean":
                hidden = hidden.mean(dim=1)
            elif self.pool == "max":
                hidden = hidden.max(dim=1).values
            else:  # "last"
                hidden = hidden[:, -1, :]

        scores = hidden.detach().cpu().numpy().astype(float, copy=False)
        auc = _pairwise_auc(scores, self.labels)
        mean_best_auc = float(auc.max(axis=0).mean())
        return mean_best_auc, max(0.0, 1.0 - mean_best_auc)


def _default_hidden_extractor(forged: Any, input_ids: "torch.Tensor") -> "torch.Tensor":
    """Duck-typed residual-stream extractor for the bundled LM-shape forges.

    Tries ``forged.torch_module.transformer(input_ids)`` (GPT-2 lineage)
    then ``forged.torch_module.model(input_ids)`` (Llama / Gemma2 /
    Qwen2 / Qwen3 / Qwen3_moe). The first attribute present wins. If
    neither exists, raises :class:`RuntimeError` naming
    ``hidden_extractor=`` so the caller knows the escape hatch.
    """
    module = forged.torch_module
    if hasattr(module, "transformer"):
        hidden = module.transformer(input_ids)
        return hidden.detach().cpu()
    if hasattr(module, "model"):
        hidden = module.model(input_ids)
        return hidden.detach().cpu()
    raise RuntimeError(
        "GroundTruthTarget could not locate a residual-stream attribute "
        f"on forged.torch_module (type={type(module).__name__}). Tried "
        "`.transformer` (GPT-2 shape) and `.model` (Llama shape). Pass "
        "hidden_extractor=... explicitly when scoring against an exotic "
        "forge."
    )


def _pairwise_auc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-feature × per-label AUC via rank-sum.

    ``scores`` is ``(N, F)``; ``labels`` is ``(N, M)`` binary-castable.
    Returns an ``(F, M)`` AUC matrix.

    Uses :func:`scipy.stats.rankdata` with ``method="average"`` so the
    result is bit-equal to :func:`sklearn.metrics.roc_auc_score`
    per (feature, label) pair (modulo floating-point noise). Average-
    rank ties handling is the convention sklearn uses internally;
    swapping in ordinal ranks (``np.argsort(np.argsort(...))``) silently
    drifts on tie-heavy fixtures like discrete-cluster labels.

    Degenerate label columns (all-positive or all-negative) get AUC
    ``0.5`` (chance) with no warning — synthetic fixtures hit this
    often enough that a warning would be noise.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = scores.shape[0]
    ranks = rankdata(scores, axis=0, method="average")  # (N, F)
    n_pos = labels.sum(axis=0)                          # (M,)
    n_neg = n - n_pos
    sum_pos = ranks.T @ labels                          # (F, M)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    degenerate = (n_pos == 0) | (n_neg == 0)
    if degenerate.any():
        auc[:, degenerate] = 0.5
    return auc
