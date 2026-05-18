"""Unit tests for :class:`saeforge.eval.GroundTruthTarget`.

Covers:

- Protocol conformance (``@runtime_checkable`` ``FaithfulnessTarget``).
- Class attributes (``name``, ``better_when``).
- Constructor validation (``labels`` shape, ``pool``, ``scorer``).
- ``score(...)`` error paths: missing ctx key, shape mismatch.
- ``score(...)`` happy path: identity fixture across all three
  ``pool`` modes.
- Default ``hidden_extractor`` duck typing: ``.transformer`` →
  ``.model`` → ``RuntimeError`` with ``hidden_extractor=`` mention.
- AUC parity with ``sklearn.metrics.roc_auc_score`` (continuous +
  tie-heavy fixtures). The ties case is the regression test for
  Decision 2 — it fails if anyone reverts to ordinal ranks.
- Degenerate-label handling (all-zero or all-one column → 0.5).
- ``host`` is never consulted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("torch")
import torch  # noqa: E402

from saeforge.eval import FaithfulnessTarget, GroundTruthTarget  # noqa: E402
from saeforge.eval.targets.gt_alignment import (  # noqa: E402
    _default_hidden_extractor,
    _pairwise_auc,
)


# --------------------------------------------------------------------------
# Class-attribute / protocol smokes
# --------------------------------------------------------------------------


def test_gt_alignment_target_satisfies_protocol() -> None:
    target = GroundTruthTarget(labels=np.eye(4))
    assert isinstance(target, FaithfulnessTarget)


def test_class_attributes() -> None:
    assert GroundTruthTarget.name == "gt_alignment"
    assert GroundTruthTarget.better_when == "higher"


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------


def test_constructor_rejects_1d_labels() -> None:
    with pytest.raises(ValueError, match=r"2D array"):
        GroundTruthTarget(labels=np.array([0, 1, 0, 1]))


def test_constructor_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match=r"N>=1 and M>=1"):
        GroundTruthTarget(labels=np.zeros((0, 3)))
    with pytest.raises(ValueError, match=r"N>=1 and M>=1"):
        GroundTruthTarget(labels=np.zeros((4, 0)))


def test_constructor_rejects_unsupported_pool() -> None:
    with pytest.raises(ValueError, match=r"pool='invalid'"):
        GroundTruthTarget(labels=np.eye(4), pool="invalid")  # type: ignore[arg-type]


def test_constructor_rejects_unsupported_scorer() -> None:
    with pytest.raises(ValueError, match=r"scorer='pearson'"):
        GroundTruthTarget(labels=np.eye(4), scorer="pearson")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# score(...) error paths
# --------------------------------------------------------------------------


class _ExplodingForged:
    """A `forged` whose attribute access raises — used to assert that
    error paths short-circuit before the extractor is consulted."""

    @property
    def torch_module(self) -> Any:  # pragma: no cover — must never run
        raise AssertionError(
            "torch_module accessed before ctx validation completed"
        )


def test_missing_eval_input_ids_raises_before_extractor() -> None:
    target = GroundTruthTarget(labels=np.eye(4))
    with pytest.raises(KeyError, match=r"_eval_input_ids"):
        target.score(forged=_ExplodingForged(), host=None, ctx={})


def test_none_eval_input_ids_raises_before_extractor() -> None:
    target = GroundTruthTarget(labels=np.eye(4))
    with pytest.raises(KeyError, match=r"_eval_input_ids"):
        target.score(
            forged=_ExplodingForged(),
            host=None,
            ctx={"_eval_input_ids": None},
        )


def test_shape_mismatch_raises_value_error() -> None:
    target = GroundTruthTarget(labels=np.eye(4))  # (4, 4)
    input_ids = torch.zeros((5, 8), dtype=torch.long)  # batch=5
    with pytest.raises(ValueError, match=r"4.*5|5.*4"):
        target.score(
            forged=_ExplodingForged(),
            host=None,
            ctx={"_eval_input_ids": input_ids},
        )


# --------------------------------------------------------------------------
# Happy-path scoring with explicit hidden_extractor
# --------------------------------------------------------------------------


def _identity_extractor_factory(hidden: torch.Tensor):
    """Returns an extractor that ignores ``forged`` and returns ``hidden``."""

    def _extract(forged: Any, input_ids: Any) -> torch.Tensor:
        return hidden

    return _extract


def test_identity_extractor_scores_near_one_with_mean_pool() -> None:
    """A 3D hidden tensor whose mean across seq equals the label matrix
    plus noise columns should AUC ~ 1.0."""
    rng = np.random.default_rng(0)
    n, m = 16, 3
    hidden_size = 8

    label_block = rng.integers(0, 2, size=(n, m)).astype(np.float32)
    # If a column happens to be all-zero, force one positive so the
    # AUC is well-defined for the assertion below.
    for j in range(m):
        if label_block[:, j].sum() == 0:
            label_block[0, j] = 1.0
        if label_block[:, j].sum() == n:
            label_block[0, j] = 0.0
    noise = rng.standard_normal((n, hidden_size - m)).astype(np.float32) * 0.01

    pooled = np.concatenate([label_block, noise], axis=1)  # (n, hidden_size)
    # Build a (n, seq=5, hidden_size) tensor whose mean across seq equals pooled.
    seq = 5
    hidden_3d = torch.tensor(pooled[:, None, :]).repeat(1, seq, 1)

    target = GroundTruthTarget(
        labels=label_block,
        hidden_extractor=_identity_extractor_factory(hidden_3d),
    )
    input_ids = torch.zeros((n, seq), dtype=torch.long)
    score, perp = target.score(
        forged=_ExplodingForged(),
        host=None,
        ctx={"_eval_input_ids": input_ids},
    )
    assert score > 0.95
    assert perp == pytest.approx(max(0.0, 1.0 - score))


def test_identity_extractor_with_pool_last() -> None:
    """With ``pool='last'``, only the final seq position is consulted."""
    rng = np.random.default_rng(1)
    n, m, hidden_size, seq = 12, 2, 6, 4
    labels = rng.integers(0, 2, size=(n, m)).astype(np.float32)
    for j in range(m):
        labels[0, j] = 1.0
        labels[1, j] = 0.0

    last_pos = np.concatenate(
        [labels, rng.standard_normal((n, hidden_size - m)).astype(np.float32) * 0.01],
        axis=1,
    )
    # Garbage everywhere except the last seq position.
    hidden = torch.randn(n, seq, hidden_size)
    hidden[:, -1, :] = torch.tensor(last_pos)

    target = GroundTruthTarget(
        labels=labels,
        pool="last",
        hidden_extractor=_identity_extractor_factory(hidden),
    )
    score, _ = target.score(
        forged=_ExplodingForged(),
        host=None,
        ctx={"_eval_input_ids": torch.zeros((n, seq), dtype=torch.long)},
    )
    assert score > 0.95


def test_identity_extractor_with_pool_max() -> None:
    """With ``pool='max'``, max across seq recovers the signal."""
    rng = np.random.default_rng(2)
    n, m, hidden_size, seq = 12, 2, 6, 4
    labels = rng.integers(0, 2, size=(n, m)).astype(np.float32)
    for j in range(m):
        labels[0, j] = 1.0
        labels[1, j] = 0.0

    signal = np.concatenate(
        [labels, rng.standard_normal((n, hidden_size - m)).astype(np.float32) * 0.01],
        axis=1,
    )
    # Put the signal at exactly one (random) seq position per row;
    # everything else is strongly negative so max picks the signal.
    hidden = -10.0 * torch.ones(n, seq, hidden_size)
    positions = rng.integers(0, seq, size=n)
    for i in range(n):
        hidden[i, positions[i], :] = torch.tensor(signal[i])

    target = GroundTruthTarget(
        labels=labels,
        pool="max",
        hidden_extractor=_identity_extractor_factory(hidden),
    )
    score, _ = target.score(
        forged=_ExplodingForged(),
        host=None,
        ctx={"_eval_input_ids": torch.zeros((n, seq), dtype=torch.long)},
    )
    assert score > 0.95


def test_extractor_may_pre_pool_to_2d() -> None:
    """If the user-supplied extractor returns a 2D tensor, the target
    skips pooling entirely."""
    rng = np.random.default_rng(3)
    n, m = 8, 2
    labels = rng.integers(0, 2, size=(n, m)).astype(np.float32)
    labels[0, :] = 1.0
    labels[1, :] = 0.0

    pooled = torch.tensor(
        np.concatenate([labels, rng.standard_normal((n, 4)) * 0.01], axis=1),
        dtype=torch.float32,
    )
    target = GroundTruthTarget(
        labels=labels,
        hidden_extractor=_identity_extractor_factory(pooled),
    )
    score, _ = target.score(
        forged=_ExplodingForged(),
        host=None,
        ctx={"_eval_input_ids": torch.zeros((n, 4), dtype=torch.long)},
    )
    assert score > 0.95


def test_host_is_never_consulted() -> None:
    """Passing a host whose ``forward`` raises must still produce a
    valid ``(score, perp)`` tuple — the target ignores host."""
    rng = np.random.default_rng(4)
    n, m = 8, 2
    labels = rng.integers(0, 2, size=(n, m)).astype(np.float32)
    labels[0, :] = 1.0
    labels[1, :] = 0.0
    pooled = torch.tensor(labels, dtype=torch.float32)

    class _BoomHost:
        def forward(self, *_a, **_kw):  # pragma: no cover — must never run
            raise AssertionError("host.forward was called")

        def __call__(self, *a, **kw):  # pragma: no cover — must never run
            return self.forward(*a, **kw)

    target = GroundTruthTarget(
        labels=labels,
        hidden_extractor=_identity_extractor_factory(pooled),
    )
    score, perp = target.score(
        forged=_ExplodingForged(),
        host=_BoomHost(),
        ctx={"_eval_input_ids": torch.zeros((n, 4), dtype=torch.long)},
    )
    assert isinstance(score, float)
    assert isinstance(perp, float)


# --------------------------------------------------------------------------
# Default extractor: duck-typing across .transformer and .model
# --------------------------------------------------------------------------


class _FakeTransformerModule:
    """Forged-shape stub exposing ``.torch_module.transformer(input_ids)``."""

    def __init__(self, hidden: torch.Tensor) -> None:
        class _Inner:
            def transformer(self, _input_ids):
                return hidden

        self.torch_module = _Inner()


class _FakeLlamaModule:
    """Forged-shape stub exposing ``.torch_module.model(input_ids)``."""

    def __init__(self, hidden: torch.Tensor) -> None:
        class _Inner:
            def model(self, _input_ids):
                return hidden

        self.torch_module = _Inner()


class _FakeBareModule:
    """Forged-shape stub with neither ``.transformer`` nor ``.model``."""

    def __init__(self) -> None:
        class _Inner:
            pass

        self.torch_module = _Inner()


def test_default_extractor_picks_up_transformer_shape() -> None:
    hidden = torch.randn(4, 3, 8)
    forged = _FakeTransformerModule(hidden)
    out = _default_hidden_extractor(forged, torch.zeros(4, 3, dtype=torch.long))
    assert torch.equal(out, hidden.detach().cpu())


def test_default_extractor_falls_back_to_model_shape() -> None:
    hidden = torch.randn(4, 3, 8)
    forged = _FakeLlamaModule(hidden)
    out = _default_hidden_extractor(forged, torch.zeros(4, 3, dtype=torch.long))
    assert torch.equal(out, hidden.detach().cpu())


def test_default_extractor_raises_on_exotic_module() -> None:
    forged = _FakeBareModule()
    with pytest.raises(RuntimeError, match=r"hidden_extractor"):
        _default_hidden_extractor(forged, torch.zeros(4, 3, dtype=torch.long))


def test_default_extractor_error_names_both_attribute_attempts() -> None:
    forged = _FakeBareModule()
    with pytest.raises(RuntimeError) as exc:
        _default_hidden_extractor(forged, torch.zeros(4, 3, dtype=torch.long))
    msg = str(exc.value)
    assert ".transformer" in msg
    assert ".model" in msg
    assert type(forged.torch_module).__name__ in msg


# --------------------------------------------------------------------------
# AUC parity vs sklearn
# --------------------------------------------------------------------------


def test_pairwise_auc_parity_on_continuous_fixture() -> None:
    """On a 32×4 scores / 32×3 labels fixture with continuous scores,
    ``_pairwise_auc`` matches sklearn within fp noise."""
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    n, f, m = 32, 4, 3
    scores = rng.standard_normal((n, f))
    labels = rng.integers(0, 2, size=(n, m))
    # Force every column to have both classes.
    for j in range(m):
        labels[0, j] = 1
        labels[1, j] = 0

    auc = _pairwise_auc(scores, labels)
    expected = np.empty((f, m))
    for fi in range(f):
        for mj in range(m):
            expected[fi, mj] = sklearn.roc_auc_score(labels[:, mj], scores[:, fi])
    np.testing.assert_allclose(auc, expected, atol=1e-12)


def test_pairwise_auc_parity_on_tied_fixture() -> None:
    """Regression test for Decision 2 — ``rankdata(method='average')``
    matches sklearn on tie-heavy fixtures. Ordinal ranks (the
    pre-Decision-2 implementation) would silently drift here.
    """
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    n, f, m = 32, 4, 3
    # Round to 1 decimal so many rows share scores → ties.
    scores = np.round(rng.standard_normal((n, f)), 1)
    labels = rng.integers(0, 2, size=(n, m))
    for j in range(m):
        labels[0, j] = 1
        labels[1, j] = 0

    auc = _pairwise_auc(scores, labels)
    expected = np.empty((f, m))
    for fi in range(f):
        for mj in range(m):
            expected[fi, mj] = sklearn.roc_auc_score(labels[:, mj], scores[:, fi])
    np.testing.assert_allclose(auc, expected, atol=1e-12)


def test_pairwise_auc_degenerate_label_column_is_chance() -> None:
    """All-zero (or all-one) label columns can't be scored; return 0.5
    (chance) silently."""
    rng = np.random.default_rng(2)
    scores = rng.standard_normal((16, 4))
    labels = np.zeros((16, 3))
    labels[:, 0] = 1.0  # all-positive: degenerate
    labels[:, 1] = 0.0  # all-negative: degenerate
    labels[0, 2] = 1.0  # well-defined

    auc = _pairwise_auc(scores, labels)
    np.testing.assert_array_equal(auc[:, 0], 0.5)
    np.testing.assert_array_equal(auc[:, 1], 0.5)
    assert not np.all(auc[:, 2] == 0.5)  # the well-defined column is data-driven
