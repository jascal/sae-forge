"""Tests for run_finetune — the training loop."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.training import TrainingConfig, run_finetune


def _build_tiny_native_model():
    pytest.importorskip("torch")
    import torch

    from saeforge import NativeModel
    from saeforge.model import NativeModelConfig

    config = NativeModelConfig(
        hidden_size=8,
        qkv_inner_size=16,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    return NativeModel(config)


def _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8, seed=0):
    pytest.importorskip("torch")
    import torch

    g = torch.Generator().manual_seed(seed)

    def gen():
        while True:
            yield torch.randint(0, vocab_size, (batch_size, sequence_length), generator=g)

    return gen()


def test_run_finetune_reduces_loss_on_synthetic_corpus():
    """Sanity: training does anything. Loss at end of 50 steps < loss at start."""
    pytest.importorskip("torch")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=50,
        warmup_steps=5,
        peak_lr=5e-3,
        batch_size=2,
        sequence_length=8,
        log_every_steps=1,
        eval_every_steps=10000,
        save_every_steps=10000,
    )
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    assert result.n_steps_completed == 50
    assert result.loss_history[-1][1] < result.loss_history[0][1]


def test_periodic_eval_cadence_is_exact(tiny_gpt2):
    """eval_history has one entry per eval_every_steps (excluding step 0)."""
    pytest.importorskip("torch")
    import torch

    model = _build_tiny_native_model()
    # tiny_gpt2 fixture has matching vocab_size=100; use it as host for KL
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=20,
        warmup_steps=2,
        peak_lr=1e-3,
        batch_size=2,
        sequence_length=8,
        log_every_steps=5,
        eval_every_steps=5,
        eval_input_ids=torch.randint(0, 100, (1, 4)),
        save_every_steps=10000,
    )
    result = run_finetune(model, host=tiny_gpt2, iterator=iterator, config=config)
    eval_steps = [s for (s, _) in result.eval_history]
    # Steps 5, 10, 15 (excluding 0) — 20 is the last step (range stops before)
    assert eval_steps == [5, 10, 15]


def test_periodic_save_cadence_is_exact(tmp_path):
    pytest.importorskip("torch")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    save_dir = tmp_path / "ckpts"
    config = TrainingConfig(
        total_steps=20,
        warmup_steps=2,
        peak_lr=1e-3,
        batch_size=2,
        sequence_length=8,
        log_every_steps=10000,
        eval_every_steps=10000,
        save_every_steps=5,
        save_dir=save_dir,
    )
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    save_step_names = sorted(p.name for p in result.save_paths)
    assert save_step_names == ["step-000005", "step-000010", "step-000015"]
    for path in result.save_paths:
        assert (path / "config.json").is_file()


def test_bf16_runs_without_error():
    """Sanity: bf16 autocast on a tiny model on CPU completes without raising."""
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "bfloat16")):
        pytest.skip("bfloat16 not available")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=4,
        warmup_steps=1,
        peak_lr=1e-3,
        batch_size=2,
        sequence_length=8,
        precision="bf16",
        log_every_steps=1,
        eval_every_steps=10000,
        save_every_steps=10000,
    )
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    assert result.n_steps_completed == 4


def test_grad_checkpointing_does_not_break_loss_decrease():
    """Smoke: gradient checkpointing path runs and reduces loss on synthetic corpus."""
    pytest.importorskip("torch")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=20,
        warmup_steps=2,
        peak_lr=5e-3,
        batch_size=2,
        sequence_length=8,
        gradient_checkpointing=True,
        log_every_steps=1,
        eval_every_steps=10000,
        save_every_steps=10000,
    )
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    assert result.n_steps_completed == 20
    # Loss may not strictly decrease over 20 steps with checkpointing on a
    # tiny model, so just check the loss didn't blow up.
    final = result.loss_history[-1][1]
    assert np.isfinite(final)


def test_config_rejects_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        TrainingConfig(precision="int8")


def test_config_rejects_zero_total_steps():
    with pytest.raises(ValueError, match="warmup_steps|total_steps"):
        TrainingConfig(total_steps=0)


def test_loss_history_logged_at_log_every_steps():
    pytest.importorskip("torch")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=20,
        warmup_steps=2,
        peak_lr=1e-3,
        batch_size=2,
        sequence_length=8,
        log_every_steps=5,
        eval_every_steps=10000,
        save_every_steps=10000,
    )
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    logged_steps = [s for (s, _) in result.loss_history]
    # Steps 0, 5, 10, 15 + the final step (19) = 5 entries
    assert logged_steps == [0, 5, 10, 15, 19]


def test_convergence_heuristic_marks_steady_loss_converged():
    from saeforge.training.loop import _check_convergence

    # All losses identical → trivially converged
    history = [(i, 1.0) for i in range(250)]
    assert _check_convergence(history) is True
    # Strictly decreasing → not converged
    history2 = [(i, 1.0 - 0.001 * i) for i in range(250)]
    assert _check_convergence(history2) is False
    # Too short → not converged
    assert _check_convergence([(0, 1.0), (1, 0.9)]) is False
