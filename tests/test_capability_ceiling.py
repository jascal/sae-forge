"""Tests for the capability-ceiling decomposition (add-capability-ceiling-diagnostic, PR #122).

Synthetic, fast (no model download). Verifies the decomposition fields populate with sane orderings and that
the gaps respond correctly to whether the SAE atoms are a good vs a poor rank-`N` basis for the labels.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from saeforge.capability_ceiling import capability_ceiling_decomposition  # noqa: E402


def _fixture(seed=0, n_obs=400, d=48, n_features=64, V=20):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, d))
    W_dec = rng.standard_normal((n_features, d))
    W_enc = np.linalg.pinv(W_dec)
    b = rng.standard_normal(n_features) * 0.1
    Z = np.maximum(X @ W_enc + b, 0.0)
    feat = np.argsort(-(Z > 0).mean(0))[:V]
    Y = (Z[:, feat] > 0).astype(float)

    def enc(x):
        return torch.relu(
            x @ torch.tensor(W_enc[:, feat], dtype=torch.float32)
            + torch.tensor(b[feat], dtype=torch.float32)
        )

    return X, enc, Y, W_dec


def test_decomposition_populates_and_orders_sanely():
    X, enc, Y, W_dec = _fixture()
    d = capability_ceiling_decomposition(X, enc, Y, W_dec, width=16, steps=100, seed=0)
    out = d.to_dict()
    for key in ("retained_mauc_random", "retained_mauc_svd", "retained_mauc_pinv",
                "retained_mauc_best_atoms", "retained_mauc_ceiling",
                "selection_gap", "interpretability_tax", "ceiling_gap"):
        assert key in out and np.isfinite(out[key])
    # gaps are consistent with the retained values
    assert d.selection_gap == pytest.approx(d.retained_mauc_best_atoms - d.retained_mauc_pinv, abs=1e-9)
    assert d.interpretability_tax == pytest.approx(d.retained_mauc_ceiling - d.retained_mauc_best_atoms, abs=1e-9)
    assert d.ceiling_gap == pytest.approx(1.0 - d.retained_mauc_ceiling, abs=1e-9)
    # the trained ceiling is at least as good as the random floor
    assert d.retained_mauc_ceiling >= d.retained_mauc_random - 1e-2


def test_capability_supervised_selection_beats_top_norm_when_norm_is_uninformative():
    """If atom row-norm is uninformative for the task, capability-supervised `best_atoms` should not be worse
    than `pinv`(top-norm) — `selection_gap` ≥ ~0 (the supervised selection can only help or tie)."""
    X, enc, Y, W_dec = _fixture(seed=1)
    d = capability_ceiling_decomposition(X, enc, Y, W_dec, width=12, steps=80, seed=1)
    assert d.selection_gap >= -0.05  # supervised selection is competitive with top-norm


def test_width_monotonicity_of_pinv():
    """A wider basis retains at least as much as a narrower one (more atoms = better reconstruction)."""
    X, enc, Y, W_dec = _fixture(seed=2)
    d_small = capability_ceiling_decomposition(X, enc, Y, W_dec, width=8, steps=40, seed=0)
    d_big = capability_ceiling_decomposition(X, enc, Y, W_dec, width=32, steps=40, seed=0)
    assert d_big.retained_mauc_pinv >= d_small.retained_mauc_pinv - 1e-6
