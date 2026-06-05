"""Router parity: the torch-batched router == polygram's per-vector route.

``PolygramHeuristicRouter.route`` must be a strict vectorisation of
``ExpertDictionary.route`` — same summed-activation scoring, same stable
descending order — not a re-implementation with different tie-breaking.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("polygram")

import torch  # noqa: E402

from saeforge._moe.routers import PolygramHeuristicRouter  # noqa: E402


def _expert_dictionary(W_dec: np.ndarray, coherence_threshold: float = 0.0,
                       max_features_per_expert: int | None = 16):
    from polygram import Dictionary, Feature, HEA_Rung2, cluster_experts

    n = W_dec.shape[0]
    n_qubits = max(1, int(np.ceil(np.log2(max(2, n)))))
    feats = [Feature(name=f"f_{i}", cluster="c0", beta=0.0) for i in range(n)]
    dictionary = Dictionary(
        name="t",
        features=feats,
        hierarchy={"c0": [f"f_{i}" for i in range(n)]},
        encoding=HEA_Rung2(depth=1, n_qubits=n_qubits),
    )
    return cluster_experts(
        dictionary,
        decoder_vectors=W_dec,
        method="cosine",
        coherence_threshold=coherence_threshold,
        max_features_per_expert=max_features_per_expert,
    )


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_torch_router_matches_polygram_per_vector(top_k):
    rng = np.random.default_rng(0)
    n_features, d_model = 96, 64
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float64)
    ed = _expert_dictionary(W_dec)

    feature_to_expert = np.asarray(ed._feature_to_expert, dtype=np.int64)
    router = PolygramHeuristicRouter(feature_to_expert, ed.n_experts)

    # Positive, well-separated activations → exact tie-breaking matches.
    activations = rng.random((32, n_features)).astype(np.float64)
    torch_route = router.route(torch.from_numpy(activations), top_k=top_k).numpy()
    polygram_route = np.array(
        [ed.route(activations[i], top_k) for i in range(activations.shape[0])],
        dtype=np.int64,
    )
    assert np.array_equal(torch_route, polygram_route)


def test_router_preserves_leading_batch_dims():
    rng = np.random.default_rng(1)
    n_features, d_model = 64, 48
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float64)
    ed = _expert_dictionary(W_dec)
    feature_to_expert = np.asarray(ed._feature_to_expert, dtype=np.int64)
    router = PolygramHeuristicRouter(feature_to_expert, ed.n_experts)

    acts = torch.rand(4, 8, n_features, dtype=torch.float64)
    routed = router.route(acts, top_k=2)
    assert routed.shape == (4, 8, 2)
    assert routed.dtype == torch.int64

    # Matches the flattened per-vector reference.
    flat = acts.reshape(-1, n_features).numpy()
    ref = np.array([ed.route(flat[i], 2) for i in range(flat.shape[0])], dtype=np.int64)
    assert np.array_equal(routed.reshape(-1, 2).numpy(), ref)


def test_router_rejects_out_of_range_top_k():
    rng = np.random.default_rng(2)
    W_dec = rng.standard_normal((32, 24)).astype(np.float64)
    ed = _expert_dictionary(W_dec, max_features_per_expert=8)
    router = PolygramHeuristicRouter(
        np.asarray(ed._feature_to_expert, dtype=np.int64), ed.n_experts
    )
    with pytest.raises(ValueError, match="top_k"):
        router.route(torch.rand(4, 32, dtype=torch.float64), top_k=ed.n_experts + 1)
