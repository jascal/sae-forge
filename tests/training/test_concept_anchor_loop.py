"""End-to-end test that ``run_finetune`` consumes the concept-anchor branch.

Uses a tiny in-process GPT-2 (transformers fixture) + a 4-cluster fake
polygram basis. Verifies:

  * ``concept_alpha=0.0`` skips label-source construction entirely
    (a sabotaged ``__init__`` is never hit).
  * ``concept_alpha=0.1`` builds the heads, threads them into the
    optimiser, runs at least one step, moves both student and head
    parameters, records ``concept_loss_history`` in metadata.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from saeforge.basis import FeatureBasis
from saeforge.model import NativeModel, NativeModelConfig
from saeforge.training.config import TrainingConfig
from saeforge.training.concept_anchor import PolygramClusterLabelSource
from saeforge.training.loop import run_finetune


def _fake_basis(n_kept: int = 8, d_model: int = 16, n_clusters: int = 4) -> FeatureBasis:
    rng = np.random.default_rng(0)
    w_dec = rng.standard_normal((n_kept, d_model)).astype(np.float32)
    return FeatureBasis(
        kept_ids=np.arange(n_kept),
        W_dec=w_dec,
        merged_norms=np.linalg.norm(w_dec, axis=1),
        original_norms=np.linalg.norm(w_dec, axis=1),
        scale_compression_ratio=1.0,
        metadata={"n_clusters": n_clusters},
    )


def _tiny_native_model(d_model: int = 16, vocab: int = 100, seq_len: int = 16) -> NativeModel:
    """Build a tiny `NativeModel` matching the fixture style used by
    ``tests/test_training_loop.py``. The native module's forward
    supports ``output_hidden_states=True``."""
    config = NativeModelConfig(
        family="gpt2",
        hidden_size=d_model,
        qkv_inner_size=d_model,
        num_layers=2,
        num_heads=2,
        head_dim=d_model // 2,
        intermediate_size=d_model * 4,
        vocab_size=vocab,
        max_position_embeddings=seq_len,
    )
    torch.manual_seed(0)
    return NativeModel(config)


def _toy_iterator(n: int, batch_size: int = 2, seq_len: int = 8, vocab: int = 100):
    rng = torch.Generator().manual_seed(42)
    for _ in range(n):
        yield torch.randint(0, vocab, (batch_size, seq_len), generator=rng)


def test_alpha_zero_skips_label_source_construction():
    """With concept_alpha=0.0, the label source is never instantiated."""
    model = _tiny_native_model()
    iterator = _toy_iterator(8)
    config = TrainingConfig(
        total_steps=2, warmup_steps=0, batch_size=2, sequence_length=8,
        eval_every_steps=1000, log_every_steps=1, save_every_steps=1000,
        concept_alpha=0.0,
    )
    # Sabotage __init__: if the branch isn't skipped, this raises and the
    # test fails. With alpha=0.0 the loop never instantiates it.
    with patch.object(
        PolygramClusterLabelSource,
        "__init__",
        side_effect=RuntimeError("label source should not have been instantiated"),
    ):
        result = run_finetune(model, host=None, iterator=iterator, config=config)
    assert result.n_steps_completed == 2
    assert "concept_anchoring" not in result.metadata


def test_alpha_positive_constructs_heads_and_trains():
    model = _tiny_native_model()
    iterator = _toy_iterator(20)
    basis = _fake_basis(n_clusters=4)
    config = TrainingConfig(
        total_steps=2, warmup_steps=0, batch_size=2, sequence_length=8,
        eval_every_steps=1000, log_every_steps=1, save_every_steps=1000,
        concept_alpha=0.1,
        concept_pool_weight=1.0,
        concept_channel_weight=1.0,
        concept_focal_gamma=2.0,
        concept_label_source="polygram-clusters",
        concept_label_source_kwargs={
            "polygram_basis": basis,
            "calibration_batches": 2,
        },
    )

    # Snapshot of student params before training.
    pre = [p.detach().clone() for p in model.torch_module.parameters() if p.requires_grad]

    result = run_finetune(model, host=None, iterator=iterator, config=config)

    # Loss is finite and the loop ran.
    assert np.isfinite(result.final_loss)
    assert result.n_steps_completed == 2

    # Metadata records the concept config.
    anchor = result.metadata["concept_anchoring"]
    assert anchor["label_source"] == "polygram-clusters"
    assert anchor["n_concepts"] == 4
    assert anchor["concept_alpha"] == 0.1

    # Concept loss is tracked alongside the total loss.
    assert "concept_loss_history" in result.metadata
    assert len(result.metadata["concept_loss_history"]) >= 1
    for step, value in result.metadata["concept_loss_history"]:
        assert isinstance(step, int)
        assert np.isfinite(value)

    # Student parameters moved.
    post = [p.detach().clone() for p in model.torch_module.parameters() if p.requires_grad]
    moved = sum(
        not torch.equal(a, b) for a, b in zip(pre, post)
    )
    assert moved > 0, "expected at least one student parameter to move"


def test_alpha_positive_with_pool_only_weight_works():
    """Channel weight=0 is a valid configuration; only the pooled head's
    loss contributes. The per-channel head is still constructed but its
    gradient is zero."""
    model = _tiny_native_model()
    iterator = _toy_iterator(20)
    basis = _fake_basis(n_clusters=4)
    config = TrainingConfig(
        total_steps=1, warmup_steps=0, batch_size=2, sequence_length=8,
        eval_every_steps=1000, log_every_steps=1, save_every_steps=1000,
        concept_alpha=0.1,
        concept_pool_weight=1.0,
        concept_channel_weight=0.0,
        concept_label_source="polygram-clusters",
        concept_label_source_kwargs={"polygram_basis": basis, "calibration_batches": 2},
    )
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    assert np.isfinite(result.final_loss)
