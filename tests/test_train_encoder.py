"""Tests for saeforge.training.train_encoder (change add-capability-trained-encoder, task 2.5)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from saeforge.basis import FeatureBasis  # noqa: E402
from saeforge.training import EncoderCalibrationReport, train_encoder  # noqa: E402


def _basis(n, d, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64), W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
        original_norms=np.linalg.norm(W, axis=1).astype(np.float32),
    )


def _relu_enc(W_enc, b_enc):
    Wt = torch.tensor(W_enc, dtype=torch.float32)
    bt = torch.tensor(b_enc, dtype=torch.float32)
    return lambda x: torch.relu(x @ Wt + bt)


def _labels_from(host_encoder, X, n_labels, seed=0):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        Z = host_encoder(torch.tensor(X, dtype=torch.float32)).numpy()
    dims = rng.choice(Z.shape[1], size=n_labels, replace=False)
    Y = np.stack([(Z[:, d] > np.quantile(Z[:, d], 0.67)).astype(np.float64) for d in dims], axis=1)
    return Y


def test_lossless_basis_no_spurious_gain():
    """Square invertible basis ⇒ pinv reconstructs exactly ⇒ trained ties pinv (delta ≈ 0)."""
    d, n, L = 6, 6, 10  # n == d, W_dec invertible ⇒ lossless
    rng = np.random.default_rng(1)
    basis = _basis(n, d, seed=1)
    X = rng.standard_normal((200, d)).astype(np.float64)
    enc = _relu_enc(rng.standard_normal((d, L)) / np.sqrt(d), -0.1 * np.abs(rng.standard_normal(L)))
    Y = _labels_from(enc, X, n_labels=4)
    E, rep = train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y,
                           steps=100, lr=5e-3, seed=1)
    assert isinstance(rep, EncoderCalibrationReport)
    assert E.shape == (d, n)  # matched capacity
    assert rep.retained_mauc_pinv_baseline == pytest.approx(1.0, abs=0.02)
    assert rep.delta_heldout == pytest.approx(0.0, abs=0.02)  # no spurious gain
    assert not rep.overfit_flag


def test_bottleneck_nonlinear_trained_beats_pinv():
    """Genuine bottleneck + nonlinear encoder ⇒ trained held-out > pinv baseline, no overfit."""
    d, n, L = 32, 10, 24
    rng = np.random.default_rng(2)
    basis = _basis(n, d, seed=2)
    X = (rng.standard_normal((600, d)) * (1 + 2 * (np.arange(d) < d // 3))).astype(np.float64)
    enc = _relu_enc(rng.standard_normal((d, L)) / np.sqrt(d), -0.2 * np.abs(rng.standard_normal(L)))
    Y = _labels_from(enc, X, n_labels=6)
    E, rep = train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y,
                           steps=400, lr=5e-3, seed=2)
    assert rep.delta_heldout > 0.01          # trained recovers capability pinv leaves on the table
    assert rep.retained_mauc_trained >= rep.retained_mauc_pinv_baseline
    assert not rep.overfit_flag              # held-out generalization, not memorization


def test_returned_encoder_never_below_baseline():
    """Early-stop protection: the RETURNED E's held-out retained-mAUC >= the pinv baseline, always."""
    d, n, L = 24, 8, 16
    rng = np.random.default_rng(3)
    basis = _basis(n, d, seed=3)
    X = rng.standard_normal((120, d)).astype(np.float64)
    enc = _relu_enc(rng.standard_normal((d, L)) / np.sqrt(d), np.zeros(L))
    Y = _labels_from(enc, X, n_labels=5)
    E, rep = train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y,
                           steps=300, lr=1e-2, seed=3)
    assert rep.retained_mauc_trained >= rep.retained_mauc_pinv_baseline - 1e-6
    assert isinstance(rep.overfit_flag, bool)


def test_overfit_flag_fires_on_memorized_fit():
    """A tiny random-label supervised run: last-step E memorizes the fit set while held-out collapses
    ⇒ overfit_flag=True, yet the RETURNED (early-stop-protected) E stays at the baseline floor."""
    d, n, V = 16, 6, 5
    rng = np.random.default_rng(1)
    basis = _basis(n, d, seed=1)
    X = rng.standard_normal((28, d)).astype(np.float64)
    enc = _relu_enc(rng.standard_normal((d, V)) / np.sqrt(d), np.zeros(V))
    Y = (rng.random((28, V)) > 0.5).astype(np.float64)  # labels uncorrelated with host latents
    E, rep = train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y,
                           objective="supervised", steps=600, lr=2e-2, holdout_frac=0.4,
                           patience=600, eval_every=20, seed=1)
    assert rep.overfit_flag is True
    assert rep.retained_mauc_trained >= rep.retained_mauc_pinv_baseline - 1e-6  # protection still holds


def test_supervised_path_runs():
    """objective='supervised' (BCE, encoder width == n_labels) returns a valid report."""
    d, n, V = 20, 8, 6
    rng = np.random.default_rng(4)
    basis = _basis(n, d, seed=4)
    X = rng.standard_normal((150, d)).astype(np.float64)
    enc = _relu_enc(rng.standard_normal((d, V)) / np.sqrt(d), np.zeros(V))  # width == V
    Y = _labels_from(enc, X, n_labels=V)
    E, rep = train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y,
                           objective="supervised", steps=80, lr=5e-3, seed=4)
    assert E.shape == (d, n)
    assert 0.0 <= rep.retained_mauc_trained <= 1.0
    assert rep.objective == "supervised"


def test_bad_holdout_frac_raises():
    basis = _basis(6, 4)
    X = np.zeros((10, 4))
    Y = np.zeros((10, 3))
    enc = _relu_enc(np.zeros((4, 5)), np.zeros(5))
    with pytest.raises(ValueError, match="holdout_frac"):
        train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y, holdout_frac=1.5)
