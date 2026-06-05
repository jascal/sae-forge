"""Tests for ``saeforge.forge_to_moe`` / ``ForgedMoE`` (sae-moe-forge v1).

Covers the spec scenarios and acceptance bands in
``openspec/changes/add-sae-moe-forge/specs/sae-moe-forge/spec.md``:
mechanical bands A/B/D (universal), faithfulness Band C-strict on a
clusterable basis, the routing/shape contract, and the clean-error
surface for deferred expert/router types.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("polygram")

import torch  # noqa: E402

from saeforge import ForgedMoE, ForgedMoEConfig, forge_to_moe  # noqa: E402
from saeforge.basis import FeatureBasis  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _clustered_W_dec(d_model: int, n_clusters: int, per: int, seed: int = 0) -> np.ndarray:
    """``(n_clusters*per, d_model)`` decoder with deliberate cosine clusters.

    Each cluster's rows point near a shared orthonormal centroid (tiny
    additive noise → intra-cluster cosine ≈ 0.99); centroids across
    clusters are orthogonal. Mirrors the prototype's synthetic fixture.
    """
    rng = np.random.default_rng(seed)
    centroids, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    centroids = centroids[:n_clusters]
    rows = []
    for c in range(n_clusters):
        for _ in range(per):
            row = centroids[c] + rng.standard_normal(d_model) * 0.01
            rows.append(row / (np.linalg.norm(row) + 1e-10))
    return np.stack(rows).astype(np.float64)


def _basis(W_dec: np.ndarray, *, checkpoint_path: str | None = None) -> FeatureBasis:
    n = W_dec.shape[0]
    norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        polygram_checkpoint_path=checkpoint_path,
    )


def _expert_dictionary(W_dec: np.ndarray, *, coherence_threshold: float = 0.5,
                       max_features_per_expert: int | None = None):
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


@pytest.fixture
def clusterable_768():
    """4 clusters of 32 over a 768-d residual — the Band-C-strict fixture."""
    W = _clustered_W_dec(d_model=768, n_clusters=4, per=32, seed=0)
    return _basis(W), _expert_dictionary(W)


@pytest.fixture
def clusterable_8experts():
    """8 clusters of 16 over a 64-d residual — the routing-shape fixture."""
    W = _clustered_W_dec(d_model=64, n_clusters=8, per=16, seed=1)
    return _basis(W), _expert_dictionary(W)


# ---------------------------------------------------------------------------
# forge_to_moe entry point
# ---------------------------------------------------------------------------


def test_explicit_expert_dictionary_sets_config(clusterable_768):
    basis, ed = clusterable_768
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    assert moe.config.n_experts == ed.n_experts
    assert moe.config.k_experts == 2
    assert moe.config.expert_type == "sub_dictionary"
    assert moe.config.router_type == "polygram_heuristic"
    assert moe.config.n_features == basis.n_features
    assert moe.config.d_model == basis.d_model


def test_auto_cluster_from_checkpoint(tmp_path):
    from safetensors.numpy import save_file

    W = _clustered_W_dec(d_model=768, n_clusters=4, per=32, seed=2).astype(np.float32)
    ckpt = tmp_path / "sae.safetensors"
    save_file({"W_dec": W}, str(ckpt))

    basis = FeatureBasis.from_polygram_checkpoint(ckpt)
    assert basis.polygram_checkpoint_path == str(ckpt)

    # No expert_dictionary kwarg → polygram clustering happens internally.
    moe = forge_to_moe(basis, k_experts=2, coherence_threshold=0.5)
    assert moe.config.n_experts >= 2
    assert moe.config.source_basis_checkpoint == str(ckpt)

    host = torch.randn(2, 16, 768)
    assert moe(host).shape == (2, 16, 768)


def test_no_trainable_parameters(clusterable_768):
    basis, ed = clusterable_768
    moe = forge_to_moe(basis, expert_dictionary=ed)
    assert list(moe.parameters()) == []


# ---------------------------------------------------------------------------
# Mechanical bands
# ---------------------------------------------------------------------------


def test_band_a_k_equals_n_experts_collapses_to_flat(clusterable_768):
    """Band A: k = n_experts reproduces the flat SAE within 1e-5 MSE/coord."""
    basis, ed = clusterable_768
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=ed.n_experts)
    host = torch.randn(256, basis.d_model)
    features = moe.encode(host)
    flat = features @ moe.experts.W_dec
    routed = moe(host)
    mse = float(((routed - flat) ** 2).mean())
    assert mse <= 1e-5


def test_band_b_sparsity_gain(clusterable_8experts):
    """Band B: counted decode-cost ratio sits in [2/E - 0.05, 2/E + 0.05]."""
    basis, ed = clusterable_8experts
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    E = moe.config.n_experts
    host = torch.randn(4, 32, basis.d_model)
    features = moe.encode(host)
    top_k = moe.router.route(features, 2)
    moe_cost = moe.experts.effective_decode_cost(top_k)
    flat_cost = host.shape[0] * host.shape[1] * basis.n_features
    ratio = moe_cost / flat_cost
    assert (2 / E - 0.05) <= ratio <= (2 / E + 0.05)


def test_band_d_config_round_trip(clusterable_768):
    """Band D: config.to_dict() → from_dict() reconstructs an equal config."""
    basis, ed = clusterable_768
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    rt = ForgedMoEConfig.from_dict(moe.config.to_dict())
    assert rt == moe.config


def test_band_d_save_load_byte_identical_forward(clusterable_768, tmp_path):
    """Band D extension: a reloaded module reproduces the same reconstruction."""
    basis, ed = clusterable_768
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    host = torch.randn(4, 16, basis.d_model)
    out = moe(host)

    moe.save_pretrained(tmp_path / "forged")
    reloaded = ForgedMoE.load_pretrained(tmp_path / "forged")
    assert reloaded.config == moe.config
    assert torch.equal(reloaded(host), out)


def test_save_load_round_trip_from_real_checkpoint(tmp_path):
    """End-to-end: forge from an on-disk checkpoint, save, reload, forward-match.

    Exercises the auto-cluster path AND the persistence round-trip on a
    real (small) basis checkpoint — load → forward must reproduce the
    pre-save reconstruction byte-for-byte, with dtypes preserved.
    """
    from safetensors.numpy import save_file

    W = _clustered_W_dec(d_model=768, n_clusters=4, per=32, seed=5).astype(np.float32)
    ckpt = tmp_path / "sae.safetensors"
    save_file({"W_dec": W}, str(ckpt))

    basis = FeatureBasis.from_polygram_checkpoint(ckpt)
    moe = forge_to_moe(basis, k_experts=2, coherence_threshold=0.5)
    host = torch.randn(2, 16, 768)
    out = moe(host)

    moe.save_pretrained(tmp_path / "forged")
    reloaded = ForgedMoE.load_pretrained(tmp_path / "forged")

    assert reloaded.config == moe.config
    assert reloaded.config.source_basis_checkpoint == str(ckpt)
    assert reloaded.encoder_weight.dtype == moe.encoder_weight.dtype
    assert reloaded.experts.feature_to_expert.dtype == torch.int64
    assert torch.equal(reloaded(host), out)


# ---------------------------------------------------------------------------
# Faithfulness (basis-split)
# ---------------------------------------------------------------------------


def test_band_c_strict_clusterable(clusterable_768):
    """Band C-strict: clusterable basis → routed-vs-flat <= 0.5 * flat-vs-host."""
    basis, ed = clusterable_768
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    assert not moe.coherence_diagnostic.low_coherence
    report = moe.faithfulness_report(torch.randn(4, 64, basis.d_model))
    assert report.ratio <= 0.5


def test_coherence_diagnostic_flags_isotropic_basis():
    """A near-isotropic basis is flagged low-coherence (Band-C advisory only)."""
    rng = np.random.default_rng(7)
    W = rng.standard_normal((96, 768)).astype(np.float64)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    basis = _basis(W)
    ed = _expert_dictionary(W, coherence_threshold=0.0, max_features_per_expert=16)
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    assert moe.coherence_diagnostic.low_coherence
    # The advisory report still populates a finite ratio for inspection.
    report = moe.faithfulness_report(torch.randn(4, 32, 768))
    assert np.isfinite(report.ratio)


# ---------------------------------------------------------------------------
# Routing / shape contract
# ---------------------------------------------------------------------------


def test_routed_reconstruction_shape_and_distinct_experts(clusterable_8experts):
    basis, ed = clusterable_8experts
    assert ed.n_experts == 8
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    host = torch.randn(4, 32, 64)
    assert moe(host).shape == (4, 32, 64)
    routes = moe.route(host)
    assert routes.shape == (4, 32, 2)
    flat = routes.reshape(-1, 2)
    for row in flat:
        ids = row.tolist()
        assert len(set(ids)) == 2
        assert all(0 <= i < 8 for i in ids)


def test_feature_partition_complete_and_disjoint(clusterable_8experts):
    basis, ed = clusterable_8experts
    moe = forge_to_moe(basis, expert_dictionary=ed)
    ids = [set(t.tolist()) for t in moe.experts.expert_feature_ids]
    union: set[int] = set().union(*ids)
    assert union == set(range(basis.n_features))
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            assert ids[a].isdisjoint(ids[b])


def test_expert_load_tracks_then_resets(clusterable_8experts):
    basis, ed = clusterable_8experts
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
    host = torch.randn(4, 32, 64)

    moe(host, track_load=True)
    load = moe.expert_load()
    assert load is not None
    assert load.shape == (moe.config.n_experts,)
    assert load.sum().item() == pytest.approx(1.0)

    moe(host)  # plain forward clears the tracked load
    assert moe.expert_load() is None


# ---------------------------------------------------------------------------
# Clean-error surface
# ---------------------------------------------------------------------------


def test_tiny_mlp_expert_type_raises(clusterable_768):
    basis, ed = clusterable_768
    with pytest.raises(NotImplementedError, match="add-moe-tiny-mlp-experts"):
        forge_to_moe(basis, expert_dictionary=ed, expert_type="tiny_mlp")


def test_residual_block_expert_type_raises(clusterable_768):
    basis, ed = clusterable_768
    with pytest.raises(NotImplementedError, match="add-moe-residual-block-experts"):
        forge_to_moe(basis, expert_dictionary=ed, expert_type="residual_block")


def test_linear_router_raises(clusterable_768):
    basis, ed = clusterable_768
    with pytest.raises(NotImplementedError, match="add-moe-trained-router"):
        forge_to_moe(basis, expert_dictionary=ed, router_type="linear")


def test_mlp_router_raises(clusterable_768):
    basis, ed = clusterable_768
    with pytest.raises(NotImplementedError, match="add-moe-trained-router"):
        forge_to_moe(basis, expert_dictionary=ed, router_type="mlp")


def test_invalid_k_experts_raises(clusterable_768):
    basis, ed = clusterable_768
    with pytest.raises(ValueError, match=r"\[1,"):
        forge_to_moe(basis, expert_dictionary=ed, k_experts=ed.n_experts + 5)


def test_mismatched_expert_dictionary_n_features_raises(clusterable_8experts):
    basis, ed = clusterable_8experts
    other = _basis(_clustered_W_dec(64, 2, 8, seed=3))  # 16 features, not 128
    with pytest.raises(ValueError, match="n_features"):
        forge_to_moe(other, expert_dictionary=ed)


def test_missing_checkpoint_path_raises(clusterable_768):
    basis, _ed = clusterable_768
    assert basis.polygram_checkpoint_path is None
    with pytest.raises(ValueError, match="add-moe-explicit-cluster-construction"):
        forge_to_moe(basis)


# ---------------------------------------------------------------------------
# FeatureBasis.polygram_checkpoint_path round-trip (tasks §3)
# ---------------------------------------------------------------------------


def test_feature_basis_dict_round_trip_preserves_checkpoint_path():
    W = _clustered_W_dec(64, 2, 8, seed=4)
    basis = _basis(W, checkpoint_path="/some/where/sae.safetensors")
    rt = FeatureBasis.from_dict(basis.to_dict())
    assert rt.polygram_checkpoint_path == "/some/where/sae.safetensors"
    assert np.array_equal(rt.W_dec, basis.W_dec)
    assert np.array_equal(rt.kept_ids, basis.kept_ids)
