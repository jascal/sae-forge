"""Tests for the adaptive-regrow controller and composed action.

Three groups:

1. ``TestController`` — unit tests for ``RegrowController.next_count``
   (deterministic, bounds, monotone-in-gap, damping-effect, cold-start).
2. ``TestComposedAction`` — unit tests for ``adapt_and_regrow`` covering
   the disabled / cold-start / enabled paths.
3. ``TestSyntheticGrowth`` — multi-cycle integration tests that drive
   the composed action with synthetic compression results and assert
   the growth profile properties pinned in the capability spec.

Plus determinism + byte-equivalence-when-disabled gates.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from saeforge.basis import RegrowController


# ----------------------------------------------------------------------
# 1. Controller unit tests
# ----------------------------------------------------------------------


class TestController:
    """Controller equation: bounds, determinism, damping, target-reached."""

    def test_identical_inputs_return_identical_outputs(self):
        a = RegrowController.next_count(50, 100, 5, 32, 0.5)
        b = RegrowController.next_count(50, 100, 5, 32, 0.5)
        assert a == b
        assert 5 <= a <= 32

    def test_target_reached_returns_regrow_count(self):
        """When the basis already meets/exceeds the target, return the base."""
        assert RegrowController.next_count(150, 100, 5, 32, 0.5) == 5
        assert RegrowController.next_count(100, 100, 5, 32, 0.5) == 5

    def test_large_gap_bounded_by_regrow_max(self):
        """An enormous gap with no damping still caps at regrow_max."""
        assert RegrowController.next_count(0, 1000, 5, 32, 1.0) == 32

    def test_monotone_non_decreasing_in_gap(self):
        """All else equal, larger gap → not-smaller effective count (until capped)."""
        values = [
            RegrowController.next_count(kept, 200, 5, 64, 0.5)
            for kept in (180, 150, 120, 80, 40, 0)
        ]
        # Each successive value must be >= the previous (gap grows).
        for prev, curr in zip(values, values[1:]):
            assert curr >= prev, values

    def test_damping_factor_effect(self):
        """Higher damping → larger effective count for the same gap (until capped)."""
        # gap=100, no cap → returns round(100 * damping) clamped to [5, 1000]
        assert (
            RegrowController.next_count(0, 100, 5, 1000, 0.25)
            < RegrowController.next_count(0, 100, 5, 1000, 0.5)
            < RegrowController.next_count(0, 100, 5, 1000, 0.75)
            <= RegrowController.next_count(0, 100, 5, 1000, 1.0)
        )

    def test_damping_zero_returns_regrow_count(self):
        """Zero damping is a no-growth controller — clamps to the base floor."""
        assert RegrowController.next_count(0, 1000, 5, 64, 0.0) == 5

    def test_bounds_invariant_holds_for_arbitrary_inputs(self):
        """Across a small grid of inputs, ``regrow_count <= v <= regrow_max``."""
        for kept in (0, 10, 50, 100):
            for target in (0, 50, 200, 1000):
                for base in (0, 5, 16):
                    for cap in (max(base, 1), 32, 128):
                        if cap < base:
                            continue
                        for damp in (0.0, 0.25, 0.5, 0.75, 1.0):
                            v = RegrowController.next_count(kept, target, base, cap, damp)
                            assert base <= v <= cap, (kept, target, base, cap, damp, v)


# ----------------------------------------------------------------------
# 2. Composed-action unit tests
# ----------------------------------------------------------------------


def _base_ctx_for_regrowth(*, tmp_dir, compressed_path: str | None = None) -> dict:
    """Minimal ctx that lets ``perform_regrowth`` short-circuit into pass-through.

    No ``compression_report_path`` → action takes the pass-through branch,
    which doesn't need polygram or torch. Lets the composed-action tests
    run hermetically.
    """
    return {
        "regrow_count": 5,
        "compressed_sae_path": compressed_path or str(tmp_dir / "compressed.safetensors"),
        "inner_refine_idx": 0,
        "output_dir": str(tmp_dir),
        "transitions_log": [],
    }


class TestComposedAction:
    """``adapt_and_regrow`` paths: disabled, cold-start, enabled."""

    def test_disabled_toggle_short_circuits_to_perform_regrowth(self, tmp_path):
        """``adaptive_regrow=False`` MUST NOT invoke the controller or write effective_regrow_count."""
        from saeforge.actions import adapt_and_regrow

        ctx = _base_ctx_for_regrowth(tmp_dir=tmp_path)
        ctx.update(
            adaptive_regrow=False,
            regrow_max=32,
            n_features_target=128,
            regrow_damping=0.5,
            current_feature_count=80,
        )

        with patch.object(RegrowController, "next_count") as mock_next:
            delta = adapt_and_regrow(ctx, None)
            mock_next.assert_not_called()

        assert "effective_regrow_count" not in ctx
        # Pass-through perform_regrowth returns these two keys.
        assert delta == {
            "regrown_sae_path": ctx["compressed_sae_path"],
            "inner_refine_idx": 1,
        }
        # Exactly one log entry (the inner perform_regrowth) and it's
        # the pass-through mode — same shape as v0.2.
        assert len(ctx["transitions_log"]) == 1
        assert ctx["transitions_log"][0]["action"] == "perform_regrowth"

    def test_cold_start_short_circuits_to_perform_regrowth(self, tmp_path):
        """First cycle (current_feature_count=0) must skip the controller."""
        from saeforge.actions import adapt_and_regrow

        ctx = _base_ctx_for_regrowth(tmp_dir=tmp_path)
        ctx.update(
            adaptive_regrow=True,
            regrow_max=32,
            n_features_target=128,
            regrow_damping=0.5,
            current_feature_count=0,  # cold start
        )

        with patch.object(RegrowController, "next_count") as mock_next:
            adapt_and_regrow(ctx, None)
            mock_next.assert_not_called()

        assert "effective_regrow_count" not in ctx
        actions = [e["action"] for e in ctx["transitions_log"]]
        assert actions == ["perform_regrowth"]

    def test_enabled_warm_cycle_invokes_controller_and_logs_both_actions(self, tmp_path):
        """Enabled + warm: controller runs, effective_regrow_count set, two log entries."""
        from saeforge.actions import adapt_and_regrow

        ctx = _base_ctx_for_regrowth(tmp_dir=tmp_path)
        ctx.update(
            adaptive_regrow=True,
            regrow_max=32,
            n_features_target=128,
            regrow_damping=0.5,
            current_feature_count=80,
        )

        delta = adapt_and_regrow(ctx, None)

        # Controller: gap = 128 - 80 = 48, damped = round(48 * 0.5) = 24,
        # clamped to [5, 32] → 24.
        assert ctx["effective_regrow_count"] == 24
        assert delta["effective_regrow_count"] == 24

        # Two log entries in this exact order.
        actions = [e["action"] for e in ctx["transitions_log"]]
        assert actions == ["adapt_regrow_count", "perform_regrowth"]

        adapt_entry = ctx["transitions_log"][0]
        assert adapt_entry["value"] == 24
        assert adapt_entry["gap"] == 48
        assert adapt_entry["target"] == 128

    def test_enabled_overshoot_returns_regrow_count_no_log_pollution(self, tmp_path):
        """When current_feature_count >= target, controller returns regrow_count."""
        from saeforge.actions import adapt_and_regrow

        ctx = _base_ctx_for_regrowth(tmp_dir=tmp_path)
        ctx.update(
            adaptive_regrow=True,
            regrow_max=32,
            n_features_target=64,
            regrow_damping=0.5,
            current_feature_count=100,  # already past target
        )

        adapt_and_regrow(ctx, None)

        # Controller fires (warm cycle) but returns regrow_count.
        assert ctx["effective_regrow_count"] == 5
        actions = [e["action"] for e in ctx["transitions_log"]]
        assert actions == ["adapt_regrow_count", "perform_regrowth"]


# ----------------------------------------------------------------------
# 3. Synthetic multi-cycle integration tests
# ----------------------------------------------------------------------


def _run_n_cycles(
    n_cycles: int,
    *,
    initial_kept: int,
    target: int,
    base: int,
    cap: int,
    damping: float,
    feature_regrowth_fn=None,
) -> list[dict]:
    """Drive ``adapt_and_regrow`` for N cycles with synthetic state.

    Returns a list of per-cycle dicts:
    ``{"kept_before": int, "effective": int, "kept_after": int}``.

    Between cycles we simulate compression by setting
    ``current_feature_count = kept_after`` (so the next cycle's
    controller sees the previous cycle's regrown size). The
    ``feature_regrowth_fn`` argument lets tests control the basis-size
    delta per cycle (default: ``kept_after = kept_before + effective``,
    i.e. every regrown slot survives the next compression).
    """
    from saeforge.actions import adapt_and_regrow

    if feature_regrowth_fn is None:
        feature_regrowth_fn = lambda before, eff: before + eff  # noqa: E731

    ctx: dict = {
        "regrow_count": base,
        "compressed_sae_path": "/tmp/cycles.safetensors",
        "inner_refine_idx": 0,
        "output_dir": "/tmp",
        "transitions_log": [],
        "adaptive_regrow": True,
        "regrow_max": cap,
        "n_features_target": target,
        "regrow_damping": damping,
        "current_feature_count": initial_kept,
    }
    history: list[dict] = []
    for _ in range(n_cycles):
        kept_before = ctx["current_feature_count"]
        # Reset effective_regrow_count between cycles so the controller
        # genuinely recomputes (mirrors the production flow where
        # compress_with_polygram doesn't carry stale ctx).
        ctx.pop("effective_regrow_count", None)
        adapt_and_regrow(ctx, None)
        effective = ctx.get("effective_regrow_count", base)
        kept_after = feature_regrowth_fn(kept_before, effective)
        history.append(
            {"kept_before": kept_before, "effective": effective, "kept_after": kept_after}
        )
        ctx["current_feature_count"] = kept_after
    return history


def test_adaptive_regrow_grows_smoothly_toward_target():
    """100 → 300 with cap=64, damping=0.5, 6 cycles — concrete scenario from §8.1."""
    history = _run_n_cycles(
        n_cycles=6,
        initial_kept=100,
        target=300,
        base=5,
        cap=64,
        damping=0.5,
    )
    # (a) every effective in [5, 64].
    for h in history:
        assert 5 <= h["effective"] <= 64, h

    # (b) sequence is monotone non-increasing as the gap closes.
    effectives = [h["effective"] for h in history]
    for prev, curr in zip(effectives, effectives[1:]):
        assert curr <= prev, effectives

    # (c) final kept is in [260, 300] — close to target, not exceeding.
    final = history[-1]["kept_after"]
    assert 260 <= final <= 300, history


def test_adaptive_regrow_respects_regrow_max():
    """Huge gap (kept=0, target=10000) must still be capped at regrow_max each cycle."""
    history = _run_n_cycles(
        n_cycles=4,
        initial_kept=10,
        target=10000,
        base=5,
        cap=32,
        damping=1.0,  # no damping → still capped
    )
    for h in history:
        assert h["effective"] <= 32, h


def test_adaptive_regrow_falls_back_to_regrow_count_when_target_reached():
    """If kept >= target from cycle 1, controller always returns regrow_count."""
    history = _run_n_cycles(
        n_cycles=4,
        initial_kept=500,
        target=300,
        base=5,
        cap=64,
        damping=0.5,
        # The basis doesn't grow further — feature_regrowth_fn keeps it
        # at the same size so subsequent cycles also see overshoot.
        feature_regrowth_fn=lambda before, eff: before,
    )
    for h in history:
        assert h["effective"] == 5, h


# ----------------------------------------------------------------------
# 4. Byte-equivalence + determinism gates
# ----------------------------------------------------------------------


def test_byte_equivalent_when_adaptive_regrow_disabled(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """Setting the adaptive knobs without the master toggle MUST be a no-op for forged weights."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import hashlib

    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    torch.manual_seed(0)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    # v0.2 minimal config.
    baseline = ForgePipeline(
        basis=tiny_synthetic_basis, projector=projector, orchestrator="fsm"
    )
    baseline_result = baseline.run_synthetic(
        tiny_gpt2, tmp_path / "baseline", eval_input_ids=eval_input_ids
    )

    # adaptive_regrow=False but with the other three knobs set —
    # validation must NOT raise and the resulting weights MUST match.
    with_inert_knobs = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        adaptive_regrow=False,
        regrow_max=64,
        n_features_target=128,
        regrow_damping=0.7,
    )
    inert_result = with_inert_knobs.run_synthetic(
        tiny_gpt2, tmp_path / "inert", eval_input_ids=eval_input_ids
    )

    def _sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    baseline_weights = tmp_path / "baseline" / "forged" / "model.safetensors"
    inert_weights = tmp_path / "inert" / "forged" / "model.safetensors"
    assert _sha(baseline_weights) == _sha(inert_weights)
    assert baseline_result.n_params == inert_result.n_params


def test_two_runs_same_seed_byte_identical_under_adaptive_regrow(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """Determinism: two adaptive runs with identical config produce identical artifacts."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import hashlib

    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    pytest.importorskip("polygram")
    from polygram import RegrowConfig

    projector = SubspaceProjector(tiny_synthetic_basis)
    torch.manual_seed(0)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    # ``regrow_count > 0`` requires an explicit RegrowConfig at the
    # ForgePipeline construction boundary even when the FSM path will
    # never actually invoke polygram (no validation_report → pass-through
    # compression → pass-through regrow). The determinism gate is over
    # the controller + FSM dispatch, not over polygram itself.
    regrow = RegrowConfig(model_name="gpt2", layer=0)

    def _build():
        return ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            orchestrator="fsm",
            regrow=regrow,
            adaptive_regrow=True,
            regrow_count=5,
            regrow_max=32,
            n_features_target=128,
            regrow_damping=0.5,
        )

    r1 = _build().run_synthetic(tiny_gpt2, tmp_path / "run1", eval_input_ids=eval_input_ids)
    r2 = _build().run_synthetic(tiny_gpt2, tmp_path / "run2", eval_input_ids=eval_input_ids)

    def _sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    w1 = tmp_path / "run1" / "forged" / "model.safetensors"
    w2 = tmp_path / "run2" / "forged" / "model.safetensors"
    assert _sha(w1) == _sha(w2)

    actions1 = [e["action"] for e in r1.extras["transitions_log"]]
    actions2 = [e["action"] for e in r2.extras["transitions_log"]]
    assert actions1 == actions2
