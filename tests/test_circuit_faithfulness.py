"""Tests for saeforge.eval.circuit_faithfulness."""

from __future__ import annotations

import numpy as np

from saeforge.eval import (
    assertion_cov95,
    circuit_kl,
    in_context_repeat,
    induction_predictable,
)


def test_induction_predictable_fixture():
    # ... 1 2 1 2 3 : at t=3 (predicting the 2nd "2"), prev token "1" recurred
    # at p=0 and was followed by "2" == token[3] -> True. No other position qualifies.
    ids = [1, 2, 1, 2, 3]
    assert induction_predictable(ids).tolist() == [False, False, False, True, False]


def test_in_context_repeat_fixture():
    ids = [1, 2, 1, 2, 3]
    assert in_context_repeat(ids).tolist() == [False, False, True, True, False]


def test_circuit_kl_zero_for_identical():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((1, 8, 20))
    mask = np.zeros((1, 8), bool)
    mask[0, ::2] = True
    out = circuit_kl(logits, logits.copy(), mask=mask)
    assert out["masked_kl"] == 0.0
    assert out["complement_kl"] == 0.0
    assert out["global_kl"] == 0.0
    assert out["n_masked"] == 4


def test_circuit_kl_separates_mask_and_complement():
    rng = np.random.default_rng(1)
    host = rng.standard_normal((6, 12))
    forged = host.copy()
    # corrupt only the masked positions -> masked_kl > complement_kl (== 0)
    mask = np.array([True, False, True, False, False, False])
    forged[mask] += rng.standard_normal((mask.sum(), 12))
    out = circuit_kl(host, forged, mask=mask)
    assert out["masked_kl"] > 0.0
    assert out["complement_kl"] == 0.0
    assert out["n_masked"] == 2


def test_assertion_cov95_detects_separable_label():
    rng = np.random.default_rng(2)
    N = 200
    label = (rng.random(N) < 0.4).astype(int)
    # one latent perfectly separates the label, the rest are noise
    sep = label + rng.normal(0, 0.01, N)
    noise = rng.standard_normal((N, 4))
    feats = np.column_stack([sep, noise])
    out = assertion_cov95(feats, label[:, None])
    assert out["n_labels"] == 1
    assert out["mean_best_auc"] > 0.99
    assert out["cov95"] == 1.0
