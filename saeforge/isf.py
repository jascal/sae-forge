"""ISF — per-label routing over a diverse ensemble of small specialists.

*Concise interpretability via distillation-by-routing.* A large SAE is a
**substrate, not a dictionary**: the cheapest way to a small, faithful
interpretability model is not to prune one big SAE, but to route each concept
to the small specialist that reads it best, and fall back to a plain host
readout for the concepts that are already salient.

This module is the recipe-agnostic core of that methodology, factored out of
the bio-sae motif-specialist line so econ-sae / sm-sae (and anyone with an SAE
+ a labelled fixture) can pick it up unchanged. The pieces:

``Recipe``            a named encoder (raw host, supervised specialist, a
                      polygram-tier basis, …) — anything with ``encode(X) -> Z``.
``recipe_auc_matrix`` per-recipe × per-label best-latent AUC, using the same
                      Mann-Whitney kernel sae-forge's capability sweep uses.
``ensemble_route``    the router ``R[v] = argmax_m AUC[m, v]`` + the H-ISF
                      headline metrics (ensemble lift over best single recipe,
                      retained vs host, fraction of labels that beat host).
``salience_headroom`` the cheap, no-training diagnostic ``1 − host_auc`` that
                      predicts *where* a specialist will pay off — the lever
                      that keeps the ensemble concise (specialise only the
                      low-salience tier).
``capability_pareto`` the (params, retained) frontier — "concise" made
                      measurable: the smallest ensemble at each capability.

The empirical anchor (bio-sae, held-out synthetic motifs): supervising a
specialist on the *strong* ESM substrate at the concept's *natural
granularity* recovered 6/6 motifs (occ-AUC 0.998) a plain SAE left at 0 %,
and routing it into the ensemble lifted the motif tier +0.105 with a +0.021
ensemble lift over every single recipe — while the salient categorical tier
barely moved. See ``docs/concise-via-routing.md``.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from saeforge.sweep_capability import _best_auc_per_feature

# The per-label best-latent symmetric AUC kernel sae-forge already ships — one
# source of truth so router numbers match the capability sweep exactly.
best_auc_per_label = _best_auc_per_feature


@runtime_checkable
class Recipe(Protocol):
    """A named specialist: anything that maps activations to a latent matrix.

    ``name`` identifies it in the router; ``encode`` returns ``(N, d_latent)``.
    A raw-host recipe is just ``encode = identity``; a supervised specialist is
    a trained encoder; a polygram-tier recipe slices a compressed basis.
    """

    name: str

    def encode(self, X: np.ndarray) -> np.ndarray: ...


def recipe_auc_matrix(
    recipe_latents: Sequence[np.ndarray],
    Y: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], np.ndarray] = best_auc_per_label,
) -> np.ndarray:
    """Stack per-recipe per-label best-latent AUC into an ``(R, V)`` matrix.

    ``recipe_latents[r]`` is recipe ``r``'s ``(N, d_r)`` latent feed on the
    *same* rows that ``Y`` (``(N, V)``) labels. Recipes may differ in width.
    Labels a recipe cannot score (no positives/negatives) come back ``NaN``;
    :func:`ensemble_route` is NaN-safe.
    """
    rows = [np.asarray(scorer(np.asarray(Z, dtype=np.float64), Y), dtype=np.float64)
            for Z in recipe_latents]
    return np.vstack(rows)


def ensemble_route(
    recipe_auc,
    recipe_names: Optional[Sequence[str]] = None,
    host: int = 0,
    eps: float = 1e-9,
) -> dict:
    """Per-label router over a recipe × label AUC matrix (the ISF mechanism).

    ``recipe_auc`` is ``(R, V)``. Implements ``R[v] = argmax_m AUC[m, v]``:
    every label is routed to the recipe that discriminates it best, and the
    ensemble takes that best AUC. NaN-safe — a label scorable by at least one
    recipe is kept; a label no recipe can score is dropped (and counted).

    Returns the per-label ``router`` (+ ``router_names``), per-recipe and
    ensemble mean AUC, the **ensemble lift** over the best *single* recipe (the
    H-ISF headline: diversity only helps if the routed ensemble beats every
    individual recipe), ``retained`` vs the host recipe, the fraction of labels
    where the ensemble strictly beats host, and the router composition.
    """
    A = np.asarray(recipe_auc, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"recipe_auc must be 2-D (R, V), got shape {A.shape}")
    R, V = A.shape
    names = (list(recipe_names) if recipe_names is not None
             else [f"recipe_{i}" for i in range(R)])
    if len(names) != R:
        raise ValueError(f"recipe_names ({len(names)}) != n_recipes ({R})")

    scorable = ~np.isnan(A).all(axis=0)            # at least one recipe scores it
    n_dropped = int((~scorable).sum())
    A = A[:, scorable]
    if A.shape[1] == 0:
        raise ValueError("no scorable labels (every label is NaN for every recipe)")

    A_for_argmax = np.where(np.isnan(A), -np.inf, A)
    router = A_for_argmax.argmax(axis=0)           # (V',)
    ensemble_best = np.nanmax(A, axis=0)           # (V',)
    per_recipe_mauc = np.nanmean(A, axis=1)        # (R,)
    ensemble_mauc = float(np.mean(ensemble_best))
    best_single = float(np.nanmax(per_recipe_mauc))
    host_auc = A[host]
    host_mauc = float(np.nanmean(host_auc))
    # frac-beats-host only over labels the host itself can score.
    host_scorable = ~np.isnan(host_auc)
    beats = (ensemble_best[host_scorable] > host_auc[host_scorable] + eps)
    return {
        "router": router.tolist(),
        "router_names": [names[i] for i in router],
        "ensemble_best": ensemble_best.tolist(),
        "per_recipe_mauc": {names[i]: float(per_recipe_mauc[i]) for i in range(R)},
        "ensemble_mauc": ensemble_mauc,
        "best_single_recipe": names[int(np.nanargmax(per_recipe_mauc))],
        "ensemble_lift": ensemble_mauc - best_single,
        "host": names[host],
        "host_mauc": host_mauc,
        "retained": ensemble_mauc / host_mauc if host_mauc > 0 else float("nan"),
        "frac_beats_host": float(beats.mean()) if beats.size else 0.0,
        "router_composition": {names[i]: int((router == i).sum()) for i in range(R)},
        "n_labels_scored": int(A.shape[1]),
        "n_labels_dropped": n_dropped,
    }


def salience_headroom(host_auc) -> np.ndarray:
    """``1 − host_auc`` per label — the cheap predictor of specialist payoff.

    The lesson that keeps the ensemble *concise*: a specialist only pays off
    where the host substrate doesn't already surface the concept. High headroom
    (low host AUC) → specialise; near-zero headroom → a plain readout suffices.
    Empirically the routed ensemble lift tracks this almost exactly (bio-sae:
    motif tier host 0.893 → +0.105 lift; categorical host 0.951 → +0.015).
    """
    a = np.asarray(host_auc, dtype=np.float64)
    return np.clip(1.0 - a, 0.0, 1.0)


def capability_pareto(points: Sequence[tuple]) -> list[tuple]:
    """Pareto frontier of ``(params, retained)`` — concision made measurable.

    Given ``[(params, retained), …]`` for candidate ensembles, return the
    non-dominated set sorted by params ascending: the smallest ensemble that
    achieves each capability level (lower params better, higher retained
    better). This is the artifact "a concise interpretability model" should be
    reported as.
    """
    pts = sorted(((float(p), float(r)) for p, r in points), key=lambda t: (t[0], -t[1]))
    frontier: list[tuple] = []
    best_r = -np.inf
    for params, retained in pts:
        if retained > best_r:
            frontier.append((params, retained))
            best_r = retained
    return frontier
