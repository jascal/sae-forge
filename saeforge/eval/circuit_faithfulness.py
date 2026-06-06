"""Circuit-restricted faithfulness — KL on circuit-driven tokens + assertion cov95.

Global ``KL(host ‖ forged)`` is dominated by the common, assertion-driven
next-token mass and is nearly blind to circuit breakage: induction-predictable
tokens are a single-digit percentage of tokens. A forge mechanism that targets
*circuit* fidelity (two-basis composition preserve) must be judged on the
masked KL, not only the aggregate.

This module provides the circuit token masks (ported from the ``lm-sae``
rung-3 analysis), the restricted KL, and the forged-residual assertion
``cov95`` (the monosemantic-detector fraction). Pure-numpy; logits may be
passed as numpy arrays or torch tensors.

See ``openspec/specs/faithfulness-target`` (Circuit-restricted faithfulness KL).
"""

from __future__ import annotations

import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def induction_predictable(token_ids) -> np.ndarray:
    """Boolean mask: position ``t`` whose correct next token equals what followed
    the current token's previous in-context occurrence (the induction target).

    ``pred[t]`` marks the position *being predicted* — i.e. the model predicts
    ``token_ids[t]`` from the context up to ``t-1`` and induction would get it
    right. ``pred[0] = pred[1] = False``.
    """
    c = list(token_ids)
    n = len(c)
    pred = np.zeros(n, dtype=bool)
    for t in range(2, n):
        prev = c[t - 1]
        ps = [p for p in range(t - 1) if c[p] == prev]
        if ps and c[ps[-1] + 1] == c[t]:
            pred[t] = True
    return pred


def in_context_repeat(token_ids) -> np.ndarray:
    """Boolean mask: position ``t`` whose token already appeared earlier in-context."""
    c = list(token_ids)
    n = len(c)
    rep = np.zeros(n, dtype=bool)
    seen: set = set()
    for t in range(n):
        if c[t] in seen:
            rep[t] = True
        seen.add(c[t])
    return rep


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def circuit_kl(host_logits, forged_logits, *, mask) -> dict:
    """KL(host ‖ forged) split into the ``mask`` tokens and their complement.

    ``host_logits`` / ``forged_logits`` are ``(..., seq, vocab)``; ``mask`` is a
    boolean array broadcastable to ``(..., seq)``. Returns ``masked_kl``,
    ``complement_kl``, ``n_masked``, and ``global_kl``.
    """
    lp = _log_softmax(_to_np(host_logits))
    lq = _log_softmax(_to_np(forged_logits))
    p = np.exp(lp)
    kl = (p * (lp - lq)).sum(-1)            # (..., seq)
    kl_flat = kl.reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.shape != kl_flat.shape:
        raise ValueError(f"mask shape {m.shape} does not match per-token KL shape {kl_flat.shape}")
    n_masked = int(m.sum())
    masked_kl = float(kl_flat[m].mean()) if n_masked else 0.0
    complement_kl = float(kl_flat[~m].mean()) if (~m).any() else 0.0
    return {
        "masked_kl": masked_kl,
        "complement_kl": complement_kl,
        "n_masked": n_masked,
        "global_kl": float(kl_flat.mean()),
    }


def _auc_per_feature(features: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Mann–Whitney AUC of each feature column for a binary ``label``; ``(K,)``."""
    N, K = features.shape
    pos = label.astype(bool)
    npos = int(pos.sum())
    nneg = N - npos
    if npos == 0 or nneg == 0:
        return np.full(K, np.nan)
    ranks = np.argsort(np.argsort(features, axis=0), axis=0).astype(np.float64) + 1.0
    sum_pos = ranks[pos].sum(0)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def assertion_cov95(forged_latents, oracle, *, thresh: float = 0.95) -> dict:
    """Monosemantic-detector fraction of the forged residual against an oracle.

    ``forged_latents`` is ``(N, K)`` (basis-coordinate activations of the forged
    residual); ``oracle`` is ``(N, L)`` binary label columns. For each label,
    the best single-latent AUC is taken; ``cov95`` is the fraction of labels
    with best AUC ``>= thresh`` — the ``lm-sae`` cov95 on the forged residual.
    """
    F = _to_np(forged_latents)
    Y = _to_np(oracle)
    if Y.ndim == 1:
        Y = Y[:, None]
    best = np.array([np.nanmax(np.abs(_auc_per_feature(F, Y[:, j]) - 0.5)) + 0.5
                     for j in range(Y.shape[1])])
    return {
        "cov95": float(np.nanmean(best >= thresh)),
        "mean_best_auc": float(np.nanmean(best)),
        "n_labels": int(Y.shape[1]),
    }
