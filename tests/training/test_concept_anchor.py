"""Unit tests for `saeforge.training.concept_anchor`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from saeforge.basis import FeatureBasis
from saeforge.training.concept_anchor import (
    LABEL_SOURCE_REGISTRY,
    LabelSource,
    PolygramClusterLabelSource,
    register_label_source,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_polygram_clusters():
    assert "polygram-clusters" in LABEL_SOURCE_REGISTRY
    assert LABEL_SOURCE_REGISTRY["polygram-clusters"] is PolygramClusterLabelSource


def test_register_label_source_decorator(tmp_name="_test_temp_source"):
    @register_label_source(tmp_name)
    class _TempSource:
        def prepare(self, model, iterator):
            return 1

        def labels_for_batch(self, batch, hidden_states):
            return torch.zeros(1, 1, 1)

    assert tmp_name in LABEL_SOURCE_REGISTRY
    assert LABEL_SOURCE_REGISTRY[tmp_name] is _TempSource
    # Clean up so subsequent tests aren't polluted
    LABEL_SOURCE_REGISTRY.pop(tmp_name, None)


def test_register_label_source_rejects_duplicate():
    @register_label_source("_dup_test_1")
    class _A:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_label_source("_dup_test_1")
        class _B:
            pass

    LABEL_SOURCE_REGISTRY.pop("_dup_test_1", None)


# ---------------------------------------------------------------------------
# PolygramClusterLabelSource
# ---------------------------------------------------------------------------


def _make_fake_basis(n_kept: int, d_model: int, *, n_clusters: int | None = 4) -> FeatureBasis:
    """Build a small FeatureBasis with metadata for the concept-anchor tests."""
    rng = np.random.default_rng(0)
    w_dec = rng.standard_normal((n_kept, d_model)).astype(np.float32)
    return FeatureBasis(
        kept_ids=np.arange(n_kept),
        W_dec=w_dec,
        merged_norms=np.linalg.norm(w_dec, axis=1),
        original_norms=np.linalg.norm(w_dec, axis=1),
        scale_compression_ratio=1.0,
        metadata={"n_clusters": n_clusters} if n_clusters is not None else {},
    )


class _ToyModel(nn.Module):
    """Tiny module whose forward returns `last_hidden_state`-shaped output
    via the `output_hidden_states=True` contract."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.config = type("Cfg", (), {"hidden_size": d_model})()

    def forward(self, input_ids, output_hidden_states: bool = False):
        h = self.embed(input_ids)
        logits = self.proj(h)
        if output_hidden_states:
            return type("Out", (), {"logits": logits, "hidden_states": (h,), "last_hidden_state": h})()
        return logits


def _make_iterator(n_batches: int, B: int, T: int, vocab: int):
    """Cheap input-id iterator."""
    for _ in range(n_batches):
        yield torch.randint(0, vocab, (B, T))


def test_polygram_cluster_source_prepare_returns_n_concepts():
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    it = _make_iterator(4, B=2, T=8, vocab=10)
    n_concepts = src.prepare(model, it)
    assert n_concepts == 4


def test_polygram_cluster_source_rejects_trivial_n_clusters():
    basis = _make_fake_basis(n_kept=4, d_model=16, n_clusters=1)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    with pytest.raises(ValueError, match="needs >= 2 clusters"):
        src.prepare(model, _make_iterator(4, 2, 8, 10))


def test_polygram_cluster_source_rejects_missing_n_clusters():
    basis = _make_fake_basis(n_kept=4, d_model=16, n_clusters=None)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    with pytest.raises(ValueError, match="n_clusters="):
        src.prepare(model, _make_iterator(4, 2, 8, 10))


def test_labels_for_batch_shape_and_binarity():
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    src.prepare(model, _make_iterator(4, 2, 8, 10))

    hidden = torch.randn(2, 8, 16)
    labels = src.labels_for_batch(batch=torch.zeros(2, 8, dtype=torch.long),
                                  hidden_states=hidden)
    assert labels.shape == (2, 8, 4)
    assert labels.dtype == torch.float32
    # All values are 0.0 or 1.0
    assert set(labels.unique().tolist()) <= {0.0, 1.0}


def test_labels_for_batch_accepts_basis_space_residual():
    """When the forward mode is `native_in_basis` the residual stream
    is already in n_features space (one dim per kept feature) rather
    than d_model. The backend SHALL detect this by trailing-dim shape
    and skip the pseudoinverse projection."""
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    src.prepare(model, _make_iterator(4, 2, 8, 10))

    # Hidden states in BASIS space, shape (B, T, n_features=8).
    hidden_basis = torch.randn(2, 8, 8)
    labels = src.labels_for_batch(
        batch=torch.zeros(2, 8, dtype=torch.long),
        hidden_states=hidden_basis,
    )
    assert labels.shape == (2, 8, 4)
    assert set(labels.unique().tolist()) <= {0.0, 1.0}


def test_labels_for_batch_rejects_unrecognised_trailing_dim():
    """If hidden_states.shape[-1] matches neither d_model nor n_features
    the backend SHALL raise a clear error instead of producing a
    confusing matmul shape error."""
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    src.prepare(model, _make_iterator(4, 2, 8, 10))

    # Wrong trailing dim — neither 8 nor 16.
    bogus = torch.randn(2, 8, 32)
    with pytest.raises(RuntimeError, match="hidden_states.shape\\[-1\\]"):
        src.labels_for_batch(
            batch=torch.zeros(2, 8, dtype=torch.long),
            hidden_states=bogus,
        )


def test_labels_for_batch_requires_hidden_states():
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    src.prepare(model, _make_iterator(4, 2, 8, 10))
    with pytest.raises(ValueError, match="requires `hidden_states`"):
        src.labels_for_batch(batch=torch.zeros(2, 8, dtype=torch.long),
                             hidden_states=None)


def test_labels_for_batch_before_prepare_raises():
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis)
    with pytest.raises(RuntimeError, match="call prepare"):
        src.labels_for_batch(batch=torch.zeros(1, 4, dtype=torch.long),
                             hidden_states=torch.zeros(1, 4, 16))


def test_polygram_cluster_source_satisfies_protocol():
    """Static check: PolygramClusterLabelSource is a LabelSource."""
    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis)
    assert isinstance(src, LabelSource)


def test_prepare_warns_when_cluster_assignments_missing():
    """When the basis metadata lacks ``cluster_assignments``, the
    backend falls back to a round-robin partition and emits a
    `UserWarning` so users can tell the supervision is operating on a
    coarse mapping."""
    import warnings

    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    # Confirm there is no cluster_assignments key in our test basis
    assert "cluster_assignments" not in basis.metadata
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        src.prepare(model, _make_iterator(4, 2, 8, 10))
    msgs = [str(w.message) for w in captured]
    assert any("round-robin" in m for m in msgs), (
        f"expected a round-robin warning, got: {msgs}"
    )


def test_prepare_no_warning_when_cluster_assignments_present():
    """When metadata supplies ``cluster_assignments``, no warning fires."""
    import warnings

    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    # Inject explicit cluster assignments — feature i goes to cluster i//2
    basis.metadata["cluster_assignments"] = [i // 2 for i in range(8)]
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=2)
    model = _ToyModel(vocab_size=10, d_model=16)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        src.prepare(model, _make_iterator(4, 2, 8, 10))
    round_robin_msgs = [
        str(w.message) for w in captured if "round-robin" in str(w.message)
    ]
    assert not round_robin_msgs, (
        f"expected no round-robin warning, got: {round_robin_msgs}"
    )


def test_prepare_warns_when_iterator_short():
    """When the iterator has fewer batches than ``calibration_batches``,
    the backend warns that the training loop will see 0 batches."""
    import warnings

    basis = _make_fake_basis(n_kept=8, d_model=16, n_clusters=4)
    src = PolygramClusterLabelSource(polygram_basis=basis, calibration_batches=10)
    model = _ToyModel(vocab_size=10, d_model=16)
    # Iterator only yields 2 batches but calibration wants 10
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        src.prepare(model, _make_iterator(2, 2, 8, 10))
    short_msgs = [
        str(w.message) for w in captured
        if "consumed only" in str(w.message)
    ]
    assert len(short_msgs) == 1, (
        f"expected exactly one iterator-short warning, got: "
        f"{[str(w.message) for w in captured]}"
    )
