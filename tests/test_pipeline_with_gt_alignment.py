"""End-to-end: ``ForgePipeline(faithfulness=GroundTruthTarget(...))``
threads GT-alignment through the FSM ctx, bypasses the KL family
default, and records ``"gt_alignment"`` as the active target name.

This is the integration counterpart to
``tests/test_gt_alignment_target.py`` (the unit-level coverage). The
fixture is a 3-cluster mixture-of-gaussians with one-hot cluster
labels; a custom ``hidden_extractor`` returns the cluster signature
directly so the AUC saturates near 1.0 regardless of what the
synthetic forged model's residual stream happens to look like. The
test is about pipeline plumbing — the unit tests cover the AUC and
default extractor in isolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("torch")
import torch  # noqa: E402

from saeforge import ForgePipeline, SubspaceProjector  # noqa: E402
from saeforge.eval import GroundTruthTarget  # noqa: E402


def _mixture_of_gaussians_labels(n_per_cluster: int = 4, n_clusters: int = 3):
    """Returns ``(labels, signal)`` where ``labels`` is ``(N, n_clusters)``
    one-hot and ``signal`` is ``(N, hidden_size>=n_clusters)`` such that
    feature ``c`` AUC against label column ``c`` is ~1.0."""
    rng = np.random.default_rng(0)
    n = n_per_cluster * n_clusters
    cluster_ids = np.repeat(np.arange(n_clusters), n_per_cluster)
    labels = np.eye(n_clusters, dtype=np.float32)[cluster_ids]  # (n, n_clusters)
    # Hidden signal: the label one-hot plus low-amplitude noise in the
    # remaining hidden_size - n_clusters columns. Pooled across seq,
    # this gives feature c near-perfectly aligned with label c.
    pad = rng.standard_normal((n, 5)).astype(np.float32) * 0.01
    signal = np.concatenate([labels, pad], axis=1)
    return labels, signal


def test_gt_alignment_threads_through_fsm_path(
    tiny_gpt2, tiny_synthetic_basis, tmp_path, monkeypatch
) -> None:
    """The FSM synthetic path with ``faithfulness=GroundTruthTarget``
    MUST score via the target and MUST NOT consult ``_kl_from_input_ids``.
    """

    def _boom(*args, **kwargs):  # pragma: no cover — only hit on regression
        raise AssertionError(
            "_kl_from_input_ids was called even though faithfulness="
            "GroundTruthTarget(...) was set on the pipeline"
        )

    monkeypatch.setattr("saeforge.forge._kl_from_input_ids", _boom)

    labels, signal = _mixture_of_gaussians_labels()
    n = labels.shape[0]

    # The extractor ignores the forged model and returns a 2D tensor;
    # the target's pool step is a no-op for 2D inputs.
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    calls = {"n": 0}

    def _extractor(forged: Any, input_ids: Any) -> torch.Tensor:
        calls["n"] += 1
        return signal_tensor

    target = GroundTruthTarget(labels=labels, hidden_extractor=_extractor)
    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        faithfulness=target,
        orchestrator="fsm",
    )
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (n, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "gt-fsm", eval_input_ids=input_ids
    )

    assert result.faithfulness_target_name == "gt_alignment"
    assert result.faithfulness >= 0.7
    # The target was actually consulted (i.e. the FSM didn't short-
    # circuit before scoring).
    assert calls["n"] >= 1


def test_gt_alignment_threads_through_imperative_path(
    tiny_gpt2, tiny_synthetic_basis, tmp_path, monkeypatch
) -> None:
    """The imperative synthetic path with ``faithfulness=GroundTruthTarget``
    also routes around KL.
    """

    def _boom(*args, **kwargs):  # pragma: no cover — only hit on regression
        raise AssertionError(
            "_kl_from_input_ids was called on the imperative path even "
            "though faithfulness=GroundTruthTarget(...) was set"
        )

    monkeypatch.setattr("saeforge.forge._kl_from_input_ids", _boom)

    labels, signal = _mixture_of_gaussians_labels()
    n = labels.shape[0]
    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    def _extractor(forged: Any, input_ids: Any) -> torch.Tensor:
        return signal_tensor

    target = GroundTruthTarget(labels=labels, hidden_extractor=_extractor)
    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        faithfulness=target,
    )
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (n, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "gt-imperative", eval_input_ids=input_ids
    )

    assert result.faithfulness_target_name == "gt_alignment"
    assert result.faithfulness >= 0.7
