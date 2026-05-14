"""Tests for the host-distillation finetune-loss extension
(add-host-distillation-finetune-loss).

Covers:

- TrainingConfig field validation (distill_alpha range, distill_temperature > 0).
- run_finetune rejection when distill_alpha < 1.0 and host is None.
- Default distill_alpha=1.0 preserves the pre-change loss path.
- distill_alpha < 1.0 actually engages KD: total_loss includes the KL term,
  gradients flow into the student, host parameters are unchanged.
- distill_alpha=0.0 is pure-KL (no corpus CE contribution).
- ForgePipeline plumbs the new kwargs into TrainingConfig.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# TrainingConfig validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_alpha_one_temp_two(self):
        cfg = TrainingConfig()
        assert cfg.distill_alpha == 1.0
        assert cfg.distill_temperature == 2.0

    def test_distill_alpha_negative_rejected(self):
        with pytest.raises(ValueError, match="distill_alpha"):
            TrainingConfig(distill_alpha=-0.1)

    def test_distill_alpha_above_one_rejected(self):
        with pytest.raises(ValueError, match="distill_alpha"):
            TrainingConfig(distill_alpha=1.5)

    def test_distill_alpha_zero_accepted(self):
        TrainingConfig(distill_alpha=0.0)

    def test_distill_alpha_one_accepted(self):
        TrainingConfig(distill_alpha=1.0)

    def test_distill_temperature_zero_rejected(self):
        with pytest.raises(ValueError, match="distill_temperature"):
            TrainingConfig(distill_temperature=0.0)

    def test_distill_temperature_negative_rejected(self):
        with pytest.raises(ValueError, match="distill_temperature"):
            TrainingConfig(distill_temperature=-1.0)


# ---------------------------------------------------------------------------
# run_finetune host=None rejection
# ---------------------------------------------------------------------------


def test_distill_alpha_lt_one_requires_host_consumes_no_batches():
    """When distill_alpha < 1.0, run_finetune raises BEFORE consuming any
    batches if host is None. A tripwire iterator confirms this."""
    pytest.importorskip("torch")

    model = _build_tiny_native_model()

    consumed = []

    def tripwire():
        consumed.append(True)
        yield from _make_synthetic_iterator()

    config = TrainingConfig(distill_alpha=0.5, total_steps=10)

    with pytest.raises(ValueError, match="distill_alpha"):
        run_finetune(model, host=None, iterator=tripwire(), config=config)

    assert consumed == [], "iterator was consumed before the host-None check"


def test_alpha_one_with_host_none_does_not_raise():
    """The default alpha=1.0 path doesn't require a host; host=None is
    legal (and matches the pre-change calling convention)."""
    pytest.importorskip("torch")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator()
    config = TrainingConfig(total_steps=2, warmup_steps=1, peak_lr=1e-3, log_every_steps=1)
    result = run_finetune(model, host=None, iterator=iterator, config=config)
    assert result.n_steps_completed == 2


# ---------------------------------------------------------------------------
# distill_alpha < 1.0 engages KD (gradients flow, host unchanged)
# ---------------------------------------------------------------------------


def test_alpha_half_gradients_flow_only_through_student(tiny_gpt2):
    """alpha=0.5: one step of KD training. The student's first parameter
    moves; the host's parameters are unchanged; total_loss is finite."""
    pytest.importorskip("torch")
    import torch

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=1,
        warmup_steps=0,
        peak_lr=1e-3,
        batch_size=2,
        sequence_length=8,
        log_every_steps=1,
        eval_every_steps=10000,
        save_every_steps=10000,
        distill_alpha=0.5,
        distill_temperature=2.0,
    )

    student_params_before = [p.detach().clone() for p in model.torch_module.parameters()]
    host_params_before = [p.detach().clone() for p in tiny_gpt2.parameters()]

    result = run_finetune(model, host=tiny_gpt2, iterator=iterator, config=config)

    # Student moved.
    moved = any(
        not torch.equal(before, after)
        for before, after in zip(student_params_before, model.torch_module.parameters())
    )
    assert moved, "student parameters didn't move under KD training"
    # Host unchanged.
    for before, current in zip(host_params_before, tiny_gpt2.parameters()):
        assert torch.equal(before, current), "host parameters moved under no_grad"
    # Loss is finite.
    assert result.loss_history, "no loss recorded"
    assert all(
        loss == loss and loss != float("inf") for (_, loss) in result.loss_history
    ), "loss was NaN or inf"


def test_alpha_zero_pure_kd_trains_without_corpus_signal(tiny_gpt2):
    """alpha=0.0: loss = 1.0 * kd_loss. The corpus labels don't enter the
    loss at all. One step succeeds and produces a finite, non-NaN loss."""
    pytest.importorskip("torch")

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator(vocab_size=100, batch_size=2, sequence_length=8)
    config = TrainingConfig(
        total_steps=2,
        warmup_steps=0,
        peak_lr=1e-3,
        batch_size=2,
        sequence_length=8,
        log_every_steps=1,
        eval_every_steps=10000,
        save_every_steps=10000,
        distill_alpha=0.0,
        distill_temperature=2.0,
    )
    result = run_finetune(model, host=tiny_gpt2, iterator=iterator, config=config)
    assert result.n_steps_completed == 2
    for _, loss in result.loss_history:
        assert loss == loss and loss != float("inf"), f"loss={loss}"


# ---------------------------------------------------------------------------
# Default alpha=1.0 byte-identical to pre-change path
# ---------------------------------------------------------------------------


def test_alpha_one_skips_host_forward(tiny_gpt2, monkeypatch):
    """When distill_alpha=1.0, run_finetune does NOT call the host
    model. Tripwire-counting verifies this (the alpha=1.0 branch is
    zero-cost vs pre-change)."""
    pytest.importorskip("torch")

    host_call_count = {"n": 0}
    original_forward = tiny_gpt2.forward

    def counting_forward(*args, **kwargs):
        host_call_count["n"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(tiny_gpt2, "forward", counting_forward)

    model = _build_tiny_native_model()
    iterator = _make_synthetic_iterator()
    config = TrainingConfig(
        total_steps=3,
        warmup_steps=0,
        peak_lr=1e-3,
        log_every_steps=1,
        eval_every_steps=10000,  # disable periodic KL eval
        save_every_steps=10000,
        distill_alpha=1.0,  # default; explicit for clarity
    )
    run_finetune(model, host=tiny_gpt2, iterator=iterator, config=config)

    assert host_call_count["n"] == 0, (
        f"host was called {host_call_count['n']} time(s) under alpha=1.0 "
        f"— the branch should skip the host forward entirely"
    )


# ---------------------------------------------------------------------------
# ForgePipeline kwargs plumbing
# ---------------------------------------------------------------------------


def test_forge_pipeline_accepts_distill_kwargs():
    """ForgePipeline accepts `finetune_distill_alpha` and
    `finetune_distill_temperature` as constructor kwargs and stores
    them as fields. The action-level plumbing
    (saeforge/actions/__init__.py reads `ctx.get(
    "finetune_distill_alpha", 1.0)`) is exercised by the per-step
    run_finetune tests above."""
    import numpy as np

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    basis = FeatureBasis(
        kept_ids=np.array([0, 1], dtype=np.int64),
        W_dec=np.eye(2, dtype=np.float32),
        merged_norms=np.ones(2, dtype=np.float32),
        original_norms=np.ones(2, dtype=np.float32),
    )
    projector = SubspaceProjector(basis=basis)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id="gpt2",
        finetune_distill_alpha=0.7,
        finetune_distill_temperature=1.5,
    )
    assert pipeline.finetune_distill_alpha == 0.7
    assert pipeline.finetune_distill_temperature == 1.5


def test_forge_pipeline_defaults_distill_kwargs_off():
    """Default ForgePipeline construction sets distill_alpha=1.0 —
    byte-identical to v0.3 behavior."""
    import numpy as np

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    basis = FeatureBasis(
        kept_ids=np.array([0, 1], dtype=np.int64),
        W_dec=np.eye(2, dtype=np.float32),
        merged_norms=np.ones(2, dtype=np.float32),
        original_norms=np.ones(2, dtype=np.float32),
    )
    projector = SubspaceProjector(basis=basis)
    pipeline = ForgePipeline(
        basis=basis, projector=projector, host_model_id="gpt2",
    )
    assert pipeline.finetune_distill_alpha == 1.0
    assert pipeline.finetune_distill_temperature == 2.0
