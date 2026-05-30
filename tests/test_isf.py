"""Tests for ``saeforge.isf`` — recipe-agnostic per-label routing.

Pure-numpy; no host model. The router and the salience diagnostic are the
portable core of the concise-via-routing methodology (docs/concise-via-routing.md),
so these pin the H-ISF invariants: the routed ensemble beats every single
recipe on a diverse fixture, and the salience headroom predicts where that
lift lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from saeforge import (
    capability_pareto,
    ensemble_route,
    recipe_auc_matrix,
    salience_headroom,
)
from saeforge.isf import best_auc_per_label


def test_recipe_auc_matrix_shapes_and_values():
    # Two recipes, one latent each that perfectly separates one of two labels.
    n = 40
    rng = np.random.default_rng(0)
    Y = np.zeros((n, 2), dtype=np.uint8)
    Y[:20, 0] = 1
    Y[::2, 1] = 1
    z0 = np.where(Y[:, [0]] == 1, 5.0, -5.0) + 0.01 * rng.standard_normal((n, 1))
    z1 = np.where(Y[:, [1]] == 1, 5.0, -5.0) + 0.01 * rng.standard_normal((n, 1))
    A = recipe_auc_matrix([z0, z1], Y)
    assert A.shape == (2, 2)
    assert A[0, 0] > 0.99 and A[1, 1] > 0.99          # each recipe owns its label
    assert A[0, 1] < 0.7 and A[1, 0] < 0.7


def test_ensemble_route_beats_every_single_recipe():
    # esm host strong on labels 0-1; specialist strong on the hard labels 2-3.
    A = [
        [0.90, 0.95, 0.70, 0.72],   # esm (host)
        [0.60, 0.62, 0.65, 0.66],   # jepa_unsup
        [0.55, 0.58, 0.99, 0.98],   # p1_specialist
    ]
    out = ensemble_route(A, ["esm", "jepa_unsup", "p1_specialist"], host=0)
    assert out["router_names"] == ["esm", "esm", "p1_specialist", "p1_specialist"]
    assert out["ensemble_best"] == [0.90, 0.95, 0.99, 0.98]
    assert out["ensemble_lift"] > 0                  # beats best single recipe
    assert out["retained"] > 1.0                     # beats the host on average
    assert out["frac_beats_host"] == 0.5             # the 2 hard labels
    assert out["router_composition"] == {"esm": 2, "jepa_unsup": 0, "p1_specialist": 2}
    assert out["n_labels_dropped"] == 0


def test_ensemble_route_is_nan_safe():
    # label 2 unscorable by recipe 0 (NaN); label 3 unscorable by everyone.
    A = np.array([
        [0.90, 0.80, np.nan, np.nan],
        [0.60, 0.70, 0.99, np.nan],
    ])
    out = ensemble_route(A, ["host", "spec"], host=0)
    assert out["n_labels_dropped"] == 1              # label 3 dropped
    assert out["n_labels_scored"] == 3
    # label 2 routes to the only recipe that scores it
    assert out["router_names"][2] == "spec"


def test_single_recipe_has_zero_lift():
    out = ensemble_route([[0.8, 0.9, 0.7]], ["only"], host=0)
    assert out["ensemble_lift"] == 0.0
    assert out["retained"] == 1.0
    assert out["frac_beats_host"] == 0.0


def test_ensemble_route_validates():
    with pytest.raises(ValueError, match="2-D"):
        ensemble_route([0.5, 0.6])
    with pytest.raises(ValueError, match="recipe_names"):
        ensemble_route([[0.5, 0.6]], ["a", "b"])
    with pytest.raises(ValueError, match="no scorable"):
        ensemble_route([[np.nan, np.nan]], ["x"])


def test_salience_headroom_predicts_specialisation_need():
    host_auc = np.array([0.99, 0.95, 0.70, 0.55])
    h = salience_headroom(host_auc)
    assert np.allclose(h, [0.01, 0.05, 0.30, 0.45])
    # the lowest-salience label has the most headroom (most to gain)
    assert h.argmax() == 3


def test_capability_pareto_drops_dominated_points():
    pts = [(1000, 0.90), (2000, 0.88), (3000, 0.95), (1500, 0.92), (5000, 0.95)]
    front = capability_pareto(pts)
    # (2000,0.88) dominated by (1000,0.90); (5000,0.95) dominated by (3000,0.95)
    assert front == [(1000.0, 0.90), (1500.0, 0.92), (3000.0, 0.95)]


def test_best_auc_per_label_matches_sweep_kernel():
    # best_auc_per_label is the exact sae-forge capability kernel.
    n = 30
    Y = np.zeros((n, 1), dtype=np.uint8)
    Y[:15, 0] = 1
    z = np.where(Y == 1, 1.0, 0.0) + 0.0
    auc = best_auc_per_label(z, Y)
    assert auc.shape == (1,)
    assert auc[0] > 0.99
