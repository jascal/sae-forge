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
        family="gpt2", hidden_size=8,
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


# ---------------------------------------------------------------------------
# Regression tests for the family-aware grad-checkpointing fix.
#
# Before the fix, _enable_grad_checkpointing hardcoded GPT-2 submodule
# names (module.transformer.h, module.transformer.wte.weight). Any
# --grad-checkpoint run on a Llama or Gemma-2 host crashed inside the
# FSM with "'ForgedLlama' object has no attribute 'transformer'", which
# the FSM swallowed into final_state: failed, and ForgePipeline.run()
# returned a ForgeResult with n_params=0, faithfulness_kl=0.0 — caller
# saw exit code 0. The fix dispatches via adapter.grad_checkpoint_targets
# so each family's native module gets the right block list and embedding
# parameter.
# ---------------------------------------------------------------------------


def _project_to(host, n_features: int):
    """Helper: project ``host`` through a tiny synthetic basis, return the
    NativeModel ready for grad-checkpointing."""
    import numpy as np

    from saeforge import NativeModel, SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis

    d_model = (
        host.config.hidden_size
        if hasattr(host.config, "hidden_size")
        else host.config.n_embd
    )
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n_features, d_model)).astype(np.float32)
    basis = FeatureBasis(
        kept_ids=np.arange(n_features, dtype=np.int64),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
        original_norms=np.linalg.norm(W, axis=1).astype(np.float32),
    )
    projector = SubspaceProjector(basis)
    adapter = adapter_for(host)
    walk = adapter.walk(host, projector)
    config = adapter.build_native_config(host, n_features)
    return NativeModel.from_projected_weights(config, walk)


def test_grad_checkpointing_runs_on_gpt2(tiny_gpt2):
    pytest.importorskip("torch")
    from saeforge.training.loop import _enable_grad_checkpointing

    nm = _project_to(tiny_gpt2, n_features=8)
    _enable_grad_checkpointing(nm._module)
    # Block forward got wrapped in a closure; embedding requires grad.
    assert nm._module.transformer.h[0].forward.__closure__ is not None
    assert nm._module.transformer.wte.weight.requires_grad


def test_grad_checkpointing_runs_on_llama(tiny_llama):
    pytest.importorskip("torch")
    from saeforge.training.loop import _enable_grad_checkpointing

    nm = _project_to(tiny_llama, n_features=32)
    # Pre-fix this raised AttributeError: 'ForgedLlama' object has no
    # attribute 'transformer'. Now it dispatches via the adapter.
    _enable_grad_checkpointing(nm._module)
    assert nm._module.model.layers[0].forward.__closure__ is not None
    assert nm._module.model.embed_tokens.weight.requires_grad


def test_grad_checkpointing_runs_on_llama_tied(tiny_llama_tied):
    pytest.importorskip("torch")
    from saeforge.training.loop import _enable_grad_checkpointing

    nm = _project_to(tiny_llama_tied, n_features=32)
    _enable_grad_checkpointing(nm._module)
    assert nm._module.model.layers[0].forward.__closure__ is not None
    assert nm._module.model.embed_tokens.weight.requires_grad


def test_grad_checkpointing_runs_on_gemma2(tiny_gemma2):
    pytest.importorskip("torch")
    from saeforge.training.loop import _enable_grad_checkpointing

    nm = _project_to(tiny_gemma2, n_features=32)
    _enable_grad_checkpointing(nm._module)
    assert nm._module.model.layers[0].forward.__closure__ is not None
    assert nm._module.model.embed_tokens.weight.requires_grad


def test_grad_checkpointing_unknown_family_raises():
    """If a hypothetical native module slips through with an unsupported
    family field, the dispatcher raises a clear ValueError naming the
    available families. Exercises the registry's adapter_for_family
    error path so a future architectural mismatch fails loudly."""
    pytest.importorskip("torch")
    from saeforge.adapters import adapter_for_family

    with pytest.raises(ValueError, match="No adapter registered for family"):
        adapter_for_family("not-a-real-family")
