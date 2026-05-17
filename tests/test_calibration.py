"""Tests for ``saeforge.calibration`` — pure-numpy helpers.

The ``transformers``-dependent loaders (``load_calibration_corpus``,
``load_host_unembed``) are exercised by the live MBP smoke gate documented
in the change proposal, not unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from saeforge import SubspaceProjector
from saeforge.basis import FeatureBasis
from saeforge.calibration import (
    ANOMALOUS_TOKEN_IDS,
    compute_forged_logit_std,
    compute_host_logit_std,
    top1_is_anomalous,
)


def _identity_basis(d: int) -> FeatureBasis:
    """Identity orthonormal basis with ``n_features == d_model``."""
    eye = np.eye(d, dtype=np.float64)
    norms = np.ones(d, dtype=np.float64)
    return FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64),
        W_dec=eye,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )


def test_anomalous_token_set_seeded():
    """GPT-2 anomalous token set includes the canonical SolidGoldMagikarp ID."""
    assert "gpt2" in ANOMALOUS_TOKEN_IDS
    assert 36174 in ANOMALOUS_TOKEN_IDS["gpt2"]  # SolidGoldMagikarp


def test_compute_host_logit_std_shape_and_value():
    """Synthetic (n_tokens, d_model) input + unembed produces a finite scalar."""
    rng = np.random.default_rng(0)
    host_acts = rng.standard_normal((8, 16)).astype(np.float64)
    host_unembed = rng.standard_normal((100, 16)).astype(np.float64)
    std = compute_host_logit_std(host_acts, host_unembed)
    assert np.isfinite(std)
    assert std > 0.0


def test_compute_forged_logit_std_matches_host_at_identity_sb1():
    """Identity-orthonormal basis with sb=1.0 round-trips logits exactly."""
    rng = np.random.default_rng(1)
    d = 8
    basis = _identity_basis(d)
    projector = SubspaceProjector(basis=basis, scale_boost=1.0)
    host_acts = rng.standard_normal((32, d)).astype(np.float64)
    host_unembed = rng.standard_normal((50, d)).astype(np.float64)

    host_std = compute_host_logit_std(host_acts, host_unembed)
    forged_std = compute_forged_logit_std(host_acts, projector, host_unembed)
    assert forged_std == pytest.approx(host_std, rel=1e-9)


def test_compute_forged_logit_std_scales_with_scale_boost():
    """Forged std scales linearly with scale_boost on identity basis."""
    rng = np.random.default_rng(2)
    d = 8
    basis = _identity_basis(d)
    host_acts = rng.standard_normal((32, d)).astype(np.float64)
    host_unembed = rng.standard_normal((50, d)).astype(np.float64)
    host_std = compute_host_logit_std(host_acts, host_unembed)

    p_quarter = SubspaceProjector(basis=basis, scale_boost=0.25)
    forged_quarter = compute_forged_logit_std(host_acts, p_quarter, host_unembed)
    assert forged_quarter == pytest.approx(0.25 * host_std, rel=1e-9)


def test_top1_is_anomalous_detects_majority_anomaly():
    """When the unembed favours an anomalous-token row, top-1 is anomalous."""
    rng = np.random.default_rng(3)
    d = 8
    basis = _identity_basis(d)
    projector = SubspaceProjector(basis=basis, scale_boost=1.0)
    host_acts = rng.standard_normal((32, d)).astype(np.float64)

    # Unembed where row 36174 (SolidGoldMagikarp) has a huge magnitude:
    # the round-tripped logits will pick it as argmax everywhere.
    host_unembed = 0.01 * rng.standard_normal((40000, d)).astype(np.float64)
    host_unembed[36174] = host_acts.mean(axis=0) * 1e3  # dominant direction

    anomalous = ANOMALOUS_TOKEN_IDS["gpt2"]
    assert top1_is_anomalous(host_acts, projector, host_unembed, anomalous) is True


def test_top1_is_anomalous_clean_negative():
    """A random unembed with no anomalous-token bias does not trigger."""
    rng = np.random.default_rng(4)
    d = 8
    basis = _identity_basis(d)
    projector = SubspaceProjector(basis=basis, scale_boost=1.0)
    host_acts = rng.standard_normal((32, d)).astype(np.float64)
    # Small unembed table (1000 rows) — none overlap the gpt2 anomalous IDs
    # which all sit above 30000.
    host_unembed = rng.standard_normal((1000, d)).astype(np.float64)

    anomalous = ANOMALOUS_TOKEN_IDS["gpt2"]
    assert top1_is_anomalous(host_acts, projector, host_unembed, anomalous) is False


def test_compute_host_logit_std_rejects_shape_mismatch():
    rng = np.random.default_rng(5)
    host_acts = rng.standard_normal((8, 16))
    host_unembed = rng.standard_normal((100, 32))  # d_model mismatch
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_host_logit_std(host_acts, host_unembed)
