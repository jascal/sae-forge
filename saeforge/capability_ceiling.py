"""Capability-ceiling diagnostic (change ``add-capability-ceiling-diagnostic``, corrected by PR #122).

Decompose, at a compressed rank ``N``, the forge tax measured by **retained-mAUC** (an **encoder-side** metric:
does the SAE encoder still recover the downstream features through a rank-`N` bottleneck) into four reference
points + three gaps. Crucially — and this is the PR #122 correction — every baseline uses **activation /
encoder geometry**, NOT the readout subspace: readout-alignment is decode-specific and *harmful* on an
encoder-side metric (polygram ``add-readout-aligned-geometry-profile``, archived). R2's *principle* (a trained
subspace beats the closed-form one) transfers; its *basis* does not.

Per width ``N`` (all activation-level, same labels + held-out items):

- ``retained_mauc_random``  — mean over k random rank-`N` projections (floor).
- ``retained_mauc_svd``     — top-`N` **activation-PCA** subspace (encoder-side frozen-linear reference).
- ``retained_mauc_pinv``    — ``pinv``(top-`N`-by-norm SAE atoms) — today's interpretable basis (ships).
- ``retained_mauc_best_atoms`` — ``pinv``(best-`N` SAE atoms by **capability-supervised** selection).
- ``retained_mauc_ceiling`` — a **trained** rank-`N` subspace (init activation-PCA), the oracle (never shipped).

Gaps: ``selection_gap = best_atoms − pinv`` (fixable by atom selection), ``interpretability_tax = ceiling −
best_atoms`` (intrinsic cost of SAE atoms), ``ceiling_gap = 1.0 − ceiling`` (a measured gap at rank `N`,
achievability OPEN; the ceiling is empirical — a lower bound on the intrinsic cost).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from saeforge.training.encoder import _mauc, _np_encoder


@dataclass
class CeilingDecomposition:
    width: int
    retained_mauc_random: float
    retained_mauc_svd: float
    retained_mauc_pinv: float
    retained_mauc_best_atoms: float
    retained_mauc_ceiling: float
    selection_gap: float        # best_atoms - pinv  (fixable by selection)
    interpretability_tax: float  # ceiling - best_atoms (intrinsic to SAE atoms)
    ceiling_gap: float          # 1.0 - ceiling (measured gap at rank N; achievability open)
    overfit_flag: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _retained(recon: np.ndarray, sae_encoder, Y: np.ndarray, host_m: float) -> float:
    Z = _np_encoder(sae_encoder, recon)
    return _mauc(Z, Y) / host_m if host_m > 0 else 0.0


def _proj_onto(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal projection of rows of X onto span(columns of B). B: (d, N)."""
    q, _ = np.linalg.qr(B)                  # (d, N) orthonormal
    return (X @ q) @ q.T


def _activation_pca(X: np.ndarray, n: int) -> np.ndarray:
    """Top-`n` principal directions of the activations (encoder-side frozen-linear basis). Returns (d, n)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    vt = np.linalg.svd(Xc, full_matrices=False)[2]  # (min(N_obs,d), d)
    return vt[: n].T


def _capability_supervised_order(X: np.ndarray, W_dec: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Order SAE atoms by how well their single-atom projection discriminates the labels (encoder-side,
    capability-supervised). Score atom j = max-over-labels AUC of (X @ W_dec[j]) vs Y. Descending argsort."""
    from saeforge.training.encoder import _auc_matrix

    a = X @ W_dec.T                          # (N_obs, n_features) per-atom projections
    auc = _auc_matrix(a, Y)                  # (n_features, V)
    score = auc.max(axis=1)                  # best label AUC per atom
    return np.argsort(-score)


def _train_ceiling(
    X: np.ndarray, sae_encoder, Y: np.ndarray, n: int, *, steps: int, lr: float, seed: int,
    holdout_frac: float = 0.3, patience: int = 30, eval_every: int = 5,
) -> tuple[float, bool]:
    """Best rank-`N` subspace for retained-mAUC: train an orthonormal projection B (init activation-PCA),
    distilling the SAE latents of the projected activations to the host's, held-out + overfit-guarded.
    Returns (held-out retained_mauc_ceiling, overfit_flag)."""
    import torch

    N_obs, d = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_obs)
    nh = max(1, int(round(N_obs * holdout_frac)))
    held, fit = perm[:nh], perm[nh:]

    B0 = _activation_pca(X[fit], n)          # (d, n)
    Xt = torch.tensor(X[fit], dtype=torch.float32)
    B = torch.tensor(B0, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        target = sae_encoder(Xt).detach()
    opt = torch.optim.Adam([B], lr=lr)

    def retained_np(B_np, idx) -> float:
        q, _ = np.linalg.qr(B_np)
        recon = (X[idx] @ q) @ q.T
        host_m = _mauc(_np_encoder(sae_encoder, X[idx]), Y[idx])
        return _mauc(_np_encoder(sae_encoder, recon), Y[idx]) / host_m if host_m > 0 else 0.0

    base_held = retained_np(B0, held)
    base_fit = retained_np(B0, fit)
    best_held, best_B, last_B = base_held, B0.copy(), B0.copy()
    no_improve = 0
    for step in range(steps):
        opt.zero_grad()
        q, _ = torch.linalg.qr(B)
        recon = (Xt @ q) @ q.T
        pred = sae_encoder(recon)
        loss = 1.0 - torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()
        loss.backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            last_B = B.detach().cpu().numpy().astype(np.float64)
            h = retained_np(last_B, held)
            if h > best_held + 1e-9:
                best_held, best_B, no_improve = h, last_B.copy(), 0
            else:
                no_improve += eval_every
                if no_improve >= patience:
                    break
    overfit = (retained_np(last_B, fit) > base_fit + 1e-6) and (retained_np(last_B, held) < base_held - 1e-6)
    return float(retained_np(best_B, held)), bool(overfit)


def capability_ceiling_decomposition(
    host_acts: np.ndarray,
    sae_encoder,
    labels: np.ndarray,
    W_dec_full: np.ndarray,
    width: int,
    *,
    steps: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    n_random: int = 4,
) -> CeilingDecomposition:
    """Compute the activation-level capability-ceiling decomposition at rank ``width``. All baselines use
    activation/encoder geometry (PR #122). See module docstring."""
    X = np.asarray(host_acts, dtype=np.float64)
    Y = np.asarray(labels, dtype=np.float64)
    W = np.asarray(W_dec_full, dtype=np.float64)
    n = int(width)
    host_m = _mauc(_np_encoder(sae_encoder, X), Y)

    # pinv(top-N-by-norm atoms) — the shipped interpretable basis.
    norms = np.linalg.norm(W, axis=1)
    top_norm = np.argsort(-norms)[:n]
    Wn = W[top_norm]
    pinv = _retained((X @ np.linalg.pinv(Wn)) @ Wn, sae_encoder, Y, host_m)

    # best_atoms — capability-supervised selection.
    best_ids = _capability_supervised_order(X, W, Y)[:n]
    Wb = W[best_ids]
    best_atoms = _retained((X @ np.linalg.pinv(Wb)) @ Wb, sae_encoder, Y, host_m)

    # svd — activation-PCA reference.
    svd = _retained(_proj_onto(X, _activation_pca(X, n)), sae_encoder, Y, host_m)

    # random floor.
    rng = np.random.default_rng(seed + 12345)
    rand = float(np.mean([
        _retained(_proj_onto(X, rng.standard_normal((X.shape[1], n))), sae_encoder, Y, host_m)
        for _ in range(n_random)
    ]))

    # ceiling — trained subspace (oracle).
    ceiling, overfit = _train_ceiling(X, sae_encoder, Y, n, steps=steps, lr=lr, seed=seed)

    return CeilingDecomposition(
        width=n, retained_mauc_random=rand, retained_mauc_svd=svd, retained_mauc_pinv=pinv,
        retained_mauc_best_atoms=best_atoms, retained_mauc_ceiling=ceiling,
        selection_gap=best_atoms - pinv, interpretability_tax=ceiling - best_atoms,
        ceiling_gap=1.0 - ceiling, overfit_flag=overfit,
    )
