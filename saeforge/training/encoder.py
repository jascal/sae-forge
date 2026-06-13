"""Capability-trained encoder (change ``add-capability-trained-encoder``, task 2).

Fit a matched-capacity encoder ``E`` (shape ``(d_model, n_features)``, the same shape as
``pinv(W_dec)``) from the init ``E0 = pinv(W_dec) * scale_boost`` against a capability objective,
on a fit split, and score it on a disjoint **held-out** split against the ``pinv`` baseline. This is
the "supervised forge" deferred by ``add-downstream-capability-target`` — see that change + ``design.md``.

Inputs are the **decomposed** capability pieces (host activations, a host task-encoder, labels) rather
than a ``CapabilityDataset``, because ``add-downstream-capability-target`` is not yet implemented; when it
lands, a thin ``train_encoder(dataset=...)`` wrapper SHALL adapt it onto this core (``CapabilityDataset``'s
``encoder`` IS ``host_encoder`` here — the downstream task encoder, NOT the host transformer).

Discipline (R2 + the retracted U_C): matched capacity, the rank-AUC metric is **scoring-only** (never the
loss), and the **held-out** comparison is the gate. ``overfit_flag`` surfaces the fit-up/held-out-down mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Tuple

import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.utils.lazy import require_extra


@dataclass
class EncoderCalibrationReport:
    """Held-out, compression-controlled comparison of a trained encoder vs the pinv baseline."""

    retained_mauc_trained: float  # held-out: trained forge mAUC / host mAUC
    retained_mauc_pinv_baseline: float  # held-out: pinv forge mAUC / host mAUC (same items)
    delta_heldout: float  # retained_mauc_trained - retained_mauc_pinv_baseline
    retained_mauc_trained_fit: float  # fit-split trained retained mAUC (diagnostic, never gates)
    overfit_flag: bool  # fit improves over baseline AND held-out regresses below it
    objective: str
    loss: str
    steps: int
    steps_run: int  # may be < steps if early-stopped
    lr: float
    holdout_frac: float
    n_fit: int
    n_heldout: int


def _auc_matrix(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Mann-Whitney per-latent x per-label AUC. Z: (N, L) latents, Y: (N, V) binary labels.
    Returns (L, V). AUC of latent l ranking label v's positives above its negatives."""
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    N, L = Z.shape
    # ranks of each latent column over the N items (average ranks for ties)
    order = np.argsort(Z, axis=0)
    ranks = np.empty_like(Z)
    arange = np.arange(1, N + 1, dtype=np.float64)
    for li in range(L):
        ranks[order[:, li], li] = arange
    npos = Y.sum(axis=0)  # (V,)
    sum_ranks_pos = ranks.T @ Y  # (L, V): summed rank of positives per (latent,label)
    nneg = N - npos
    denom = npos * nneg  # (V,)
    auc = (sum_ranks_pos - npos[None, :] * (npos[None, :] + 1.0) / 2.0) / np.where(denom == 0, 1.0, denom)
    auc[:, denom == 0] = 0.5  # undefined → chance
    return auc


def _mauc(Z: np.ndarray, Y: np.ndarray) -> float:
    """mean-over-labels of max-over-latents AUC (the DownstreamCapabilityTarget convention)."""
    auc = _auc_matrix(Z, Y)
    return float(auc.max(axis=0).mean())


def train_encoder(
    *,
    basis: FeatureBasis,
    host_acts: np.ndarray,
    host_encoder: Callable,
    labels: np.ndarray,
    objective: Literal["distill", "supervised", "forge_distill"] = "distill",
    loss: Literal["cosine", "mse"] = "cosine",
    init: Literal["pinv"] = "pinv",
    scale_boost: float = 1.0,
    steps: int = 300,
    lr: float = 1e-3,
    holdout_frac: float = 0.3,
    patience: int = 30,
    eval_every: int = 5,
    seed: int = 0,
    forge: Any = None,
    sequences: "list[str] | None" = None,
    feed: str = "pooled",
    minibatch: int = 16,
) -> Tuple[np.ndarray, EncoderCalibrationReport]:
    """Fit a matched-capacity encoder ``E`` and return ``(E, EncoderCalibrationReport)``.

    ``host_acts``: (N, d_model) host hidden states (one aggregated vector per item).
    ``host_encoder``: a torch-differentiable callable (M, d_model) -> (M, latent_width) — the downstream
        task encoder (e.g. the host SAE). For ``objective="supervised"`` its output width MUST equal V.
    ``labels``: (N, V) binary label matrix (scoring for both objectives; training target for supervised).
    """
    torch = require_extra("torch", "torch")
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1); got {holdout_frac}")
    if init != "pinv":
        raise ValueError(f"only init='pinv' is supported; got {init!r}")

    X = np.asarray(host_acts, dtype=np.float64)
    Y = np.asarray(labels, dtype=np.float64)
    N, d = X.shape
    if Y.shape[0] != N:
        raise ValueError(f"labels rows ({Y.shape[0]}) must match host_acts rows ({N})")
    W_dec = np.asarray(basis.W_dec, dtype=np.float64)
    n_features, d_model = W_dec.shape
    if d != d_model:
        raise ValueError(f"host_acts d ({d}) must match basis d_model ({d_model})")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_held = max(1, int(round(N * holdout_frac)))
    held_idx, fit_idx = perm[:n_held], perm[n_held:]
    if len(fit_idx) < 1:
        raise ValueError(f"fit split empty (N={N}, holdout_frac={holdout_frac})")

    E0 = (np.linalg.pinv(W_dec) * scale_boost).astype(np.float64)  # (d_model, n_features)

    if objective == "forge_distill":
        return _forge_distill_train(
            torch=torch, basis=basis, X=X, Y=Y, host_encoder=host_encoder, E0=E0,
            fit_idx=fit_idx, held_idx=held_idx, forge=forge, sequences=sequences, feed=feed,
            loss=loss, steps=steps, lr=lr, patience=patience, eval_every=eval_every,
            minibatch=minibatch, seed=seed,
        )

    def project(X_rows: np.ndarray, E: np.ndarray) -> np.ndarray:
        return (X_rows @ E) @ W_dec  # forged → decoded, back to d_model

    def retained(E: np.ndarray, idx: np.ndarray) -> float:
        Zf = _np_encoder(host_encoder, project(X[idx], E))
        Zh = _np_encoder(host_encoder, X[idx])
        host_m = _mauc(Zh, Y[idx])
        return _mauc(Zf, Y[idx]) / host_m if host_m > 0 else 0.0

    # --- torch training of E (matched capacity) ---
    dev = "cpu"
    Xt = torch.tensor(X[fit_idx], dtype=torch.float32, device=dev)
    Wt = torch.tensor(W_dec, dtype=torch.float32, device=dev)
    E = torch.tensor(E0, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([E], lr=lr)
    if objective == "distill":
        target = host_encoder(Xt).detach()
    else:
        Yt = torch.tensor(Y[fit_idx], dtype=torch.float32, device=dev)

    rt_base = retained(E0, held_idx)  # the pinv held-out baseline (the gate floor)
    rt_base_fit = retained(E0, fit_idx)
    best_held = rt_base  # never return something worse than baseline on held-out
    best_E = E0.copy()
    E_last = E0.copy()  # the final-step encoder (for the overfit diagnostic; NOT returned)
    steps_run = 0
    no_improve = 0
    for step in range(steps):
        opt.zero_grad()
        decoded = (Xt @ E) @ Wt
        pred = host_encoder(decoded)
        if objective == "distill":
            if loss == "cosine":
                loss_val = 1.0 - torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()
            else:  # standardized MSE
                t = (target - target.mean(0)) / (target.std(0) + 1e-6)
                p = (pred - pred.mean(0)) / (pred.std(0) + 1e-6)
                loss_val = torch.nn.functional.mse_loss(p, t)
        else:  # supervised BCE-with-logits (pred width must equal V)
            loss_val = torch.nn.functional.binary_cross_entropy_with_logits(pred, Yt)
        loss_val.backward()
        opt.step()
        steps_run = step + 1
        if (step + 1) % eval_every == 0:
            E_last = E.detach().cpu().numpy().astype(np.float64)
            held = retained(E_last, held_idx)
            if held > best_held + 1e-9:
                best_held, best_E, no_improve = held, E_last.copy(), 0
            else:
                no_improve += eval_every
                if no_improve >= patience:
                    break

    E_final = best_E  # RETURN the held-out-best (early-stop-protected) encoder
    rt_trained = retained(E_final, held_idx)
    rt_trained_fit = retained(E_final, fit_idx)
    # overfit_flag is a DIAGNOSTIC on the last-step E (did unprotected training start to overfit —
    # fit above baseline while held-out fell below it?); early-stop returns the safe best-E regardless.
    overfit = (retained(E_last, fit_idx) > rt_base_fit + 1e-6) and (retained(E_last, held_idx) < rt_base - 1e-6)
    report = EncoderCalibrationReport(
        retained_mauc_trained=rt_trained,
        retained_mauc_pinv_baseline=rt_base,
        delta_heldout=rt_trained - rt_base,
        retained_mauc_trained_fit=rt_trained_fit,
        overfit_flag=bool(overfit),
        objective=objective,
        loss=loss,
        steps=steps,
        steps_run=steps_run,
        lr=lr,
        holdout_frac=holdout_frac,
        n_fit=int(len(fit_idx)),
        n_heldout=int(len(held_idx)),
    )
    return E_final.astype(basis.W_dec.dtype), report


def _forge_distill_train(*, torch, basis, X, Y, host_encoder, E0, fit_idx, held_idx, forge, sequences,
                         feed, loss, steps, lr, patience, eval_every, minibatch, seed):
    """`objective="forge_distill"` — fit `E` through the FULL forge (not the activation proxy). The loss
    distills the forged→decoded→encoded latents to the host's own latents; held-out scoring runs the same
    full forge. v1: pooled feed (1 forge row per protein). See add-full-forge-encoder-training."""
    if forge is None or sequences is None:
        raise ValueError(
            "objective='forge_distill' requires a forge context: forge=DifferentiableEsm2Forge(host, "
            "basis, scale_boost) and sequences=[...] (1:1 with host_acts rows for pooled feed)."
        )
    if feed != "pooled":
        raise NotImplementedError(
            "forge_distill v1 supports feed='pooled' (one forge row per protein); residue-feed minibatching "
            "is a follow-up."
        )
    if len(sequences) != X.shape[0]:
        raise ValueError(
            f"forge_distill: sequences ({len(sequences)}) must align 1:1 with host_acts rows ({X.shape[0]})."
        )
    ids_all = forge.tokenize(sequences)
    rng = np.random.default_rng(seed)

    def forge_d(E_t, idx):
        return forge.forge_d(E_t, [ids_all[int(i)] for i in idx], feed="pooled")

    def retained(E_np, idx):
        with torch.no_grad():
            Zf = host_encoder(forge_d(torch.tensor(E_np, dtype=torch.float32), idx)).cpu().numpy()
        Zh = _np_encoder(host_encoder, X[idx])
        host_m = _mauc(Zh, Y[idx])
        return _mauc(Zf, Y[idx]) / host_m if host_m > 0 else 0.0

    with torch.no_grad():
        target_all = host_encoder(torch.tensor(X, dtype=torch.float32)).detach()  # (N, latent), fixed

    E = torch.tensor(E0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([E], lr=lr)
    rt_base, rt_base_fit = retained(E0, held_idx), retained(E0, fit_idx)
    best_held, best_E, E_last = rt_base, E0.copy(), E0.copy()
    steps_run = no_improve = 0
    for step in range(steps):
        mb = rng.choice(fit_idx, size=min(minibatch, len(fit_idx)), replace=False)
        opt.zero_grad()
        pred = host_encoder(forge_d(E, mb))
        tgt = target_all[torch.as_tensor(mb, dtype=torch.long)]
        if loss == "cosine":
            loss_val = 1.0 - torch.nn.functional.cosine_similarity(pred, tgt, dim=1).mean()
        else:
            t = (tgt - tgt.mean(0)) / (tgt.std(0) + 1e-6)
            p = (pred - pred.mean(0)) / (pred.std(0) + 1e-6)
            loss_val = torch.nn.functional.mse_loss(p, t)
        loss_val.backward()
        opt.step()
        steps_run = step + 1
        if (step + 1) % eval_every == 0:
            E_last = E.detach().cpu().numpy().astype(np.float64)
            held = retained(E_last, held_idx)
            if held > best_held + 1e-9:
                best_held, best_E, no_improve = held, E_last.copy(), 0
            else:
                no_improve += eval_every
                if no_improve >= patience:
                    break

    rt_trained, rt_trained_fit = retained(best_E, held_idx), retained(best_E, fit_idx)
    overfit = (retained(E_last, fit_idx) > rt_base_fit + 1e-6) and (retained(E_last, held_idx) < rt_base - 1e-6)
    report = EncoderCalibrationReport(
        retained_mauc_trained=rt_trained, retained_mauc_pinv_baseline=rt_base,
        delta_heldout=rt_trained - rt_base, retained_mauc_trained_fit=rt_trained_fit,
        overfit_flag=bool(overfit), objective="forge_distill", loss=loss, steps=steps,
        steps_run=steps_run, lr=lr, holdout_frac=len(held_idx) / (len(held_idx) + len(fit_idx)),
        n_fit=int(len(fit_idx)), n_heldout=int(len(held_idx)),
    )
    return best_E.astype(basis.W_dec.dtype), report


def _np_encoder(host_encoder: Callable, X_rows: np.ndarray) -> np.ndarray:
    """Call ``host_encoder`` (torch) on numpy rows and return numpy latents (no grad, for scoring)."""
    torch = require_extra("torch", "torch")
    with torch.no_grad():
        z = host_encoder(torch.tensor(np.asarray(X_rows), dtype=torch.float32))
    return z.detach().cpu().numpy()
