"""End-to-end FSM tests for the v0.2 continual-learning extensions.

Stub-driven: bypasses the heavy compress/regrow/project actions by
monkey-patching ``saeforge.actions.ACTION_TABLE`` with trivial
handlers, so we can exercise the FSM topology itself without waiting
for torch to do real work. Uses the hierarchical orchestrator
(``run_machine``) — the same path the real forge takes — so the
transcript reflects the composed hierarchy's action ordering.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest


def _build_stub_table(transcript: list[str], advance_predicate=None):
    """Stub ACTION_TABLE that records action names and mimics counter writes.

    ``load_and_scan`` is a composed action that runs both
    ``load_sae_and_corpus`` and ``scan_activations`` under the real
    implementation. The stub appends both inner names to preserve the
    v0.2 transcript shape under the hierarchy.
    """

    def make(name):
        def action(c, _payload=None):
            transcript.append(name)
            if name == "compress_with_polygram" and c.get("regrow_count", 0) == 0:
                return {"inner_refine_idx": c.get("inner_refine_idx", 0) + 1}
            if name == "perform_regrowth":
                return {"inner_refine_idx": c.get("inner_refine_idx", 0) + 1}
            if name == "evaluate_faithfulness" and advance_predicate is not None:
                return {"advance_stream": advance_predicate(c)}
            if name == "advance_to_next_task":
                return {
                    "task_idx": c.get("task_idx", 0) + 1,
                    "inner_refine_idx": 0,
                    "tokens_seen_in_task": 0,
                    "current_iter": 0,
                    "advance_stream": False,
                }
            if name == "rotate_for_next_iter":
                return {
                    "current_iter": c.get("current_iter", 0) + 1,
                    "inner_refine_idx": 0,
                }
            return {}
        return action

    def load_and_scan_stub(c, _payload=None):
        # Mirror the real composed helper: record both inner action names.
        transcript.append("load_sae_and_corpus")
        transcript.append("scan_activations")
        return {}

    def adapt_and_regrow_stub(c, _payload=None):
        # adaptive-regrow: the FSM's `compressed → regrown` transition
        # now dispatches `adapt_and_regrow`, which short-circuits to
        # `perform_regrowth` when adaptive_regrow=False (the v0.2
        # byte-equivalent path). When adaptive_regrow=True, the
        # controller logs one extra `adapt_regrow_count` entry BEFORE
        # the perform_regrowth log entry. The stub mirrors that
        # transcript-shape contract.
        if c.get("adaptive_regrow"):
            transcript.append("adapt_regrow_count")
        transcript.append("perform_regrowth")
        return {"inner_refine_idx": c.get("inner_refine_idx", 0) + 1}

    table = {
        name: make(name)
        for name in (
            "load_sae_and_corpus",
            "scan_activations",
            "compress_with_polygram",
            "perform_regrowth",
            "project_to_subspace",
            "fine_tune_model",
            "evaluate_faithfulness",
            "advance_to_next_task",
            "rotate_for_next_iter",
            "save_final_model",
            "log_error",
        )
    }
    table["load_and_scan"] = load_and_scan_stub
    table["adapt_and_regrow"] = adapt_and_regrow_stub
    return table


@contextmanager
def _stubbed_action_table(transcript: list[str], advance_predicate=None):
    """Swap ACTION_TABLE for the duration of the with-block."""
    from saeforge import actions as actions_mod

    saved = dict(actions_mod.ACTION_TABLE)
    actions_mod.ACTION_TABLE.clear()
    actions_mod.ACTION_TABLE.update(_build_stub_table(transcript, advance_predicate))
    try:
        yield
    finally:
        actions_mod.ACTION_TABLE.clear()
        actions_mod.ACTION_TABLE.update(saved)


def _drive(ctx):
    """Run the hierarchical FSM with stubbed actions; return ``(transcript, final_state)``."""
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import run_machine

    transcript: list[str] = []
    with _stubbed_action_table(transcript):
        run_machine(ctx)
    final = "done" if not ctx.get("error_message") else "failed"
    return transcript, final


def _base_ctx(**overrides):
    """Default ctx with v0.1-equivalent values; override with kwargs."""
    ctx = {
        "sae_checkpoint": "/tmp/x",
        "host_model_id": "gpt2",
        "output_dir": "/tmp/o",
        "iterations": 1,
        "regrow_count": 0,
        "current_iter": 0,
        "n_tasks": 1,
        "task_idx": 0,
        "task_trigger": "labeled",
        "tokens_seen_in_task": 0,
        "loss_delta_threshold": 0.0,
        "recent_eval_losses": [],
        "advance_stream": False,
        "should_continue": False,
        "inner_refine_passes": 1,
        "inner_refine_idx": 0,
        "protect_top_k": 0,
        "protect_score": "mean_act",
        "protected_features": [],
        "feature_usage": [],
        "activation_buffer_size": 4096,
        "replay_ratio": 0.0,
        "replay_policy": "reservoir",
        "replay_buffer_size": 0,
    }
    ctx.update(overrides)
    return ctx


# ---------- Default-knob equivalence (the hard contract) ----------


def test_default_knobs_v01_topology_with_one_extra_hop():
    """Under v0.2 defaults the trace = v0.1 sequence + one activations_scanned hop."""
    transcript, final = _drive(_base_ctx())
    assert final == "done"
    expected = [
        "load_sae_and_corpus",
        "scan_activations",
        "compress_with_polygram",
        "project_to_subspace",
        "fine_tune_model",
        "evaluate_faithfulness",
        "save_final_model",
    ]
    assert transcript == expected


# ---------- Basis loop ----------


def test_basis_loop_inner_refine_passes_three_with_regrow():
    """3 compress passes, 3 regrow passes, exit from regrown to projected."""
    transcript, final = _drive(_base_ctx(regrow_count=1, inner_refine_passes=3))
    assert final == "done"
    n_compress = transcript.count("compress_with_polygram")
    n_regrow = transcript.count("perform_regrowth")
    # 3 compress + 3 regrow expected; the first compress comes from the
    # activations_scanned -> compressed transition.
    assert n_compress == 3, transcript
    assert n_regrow == 3, transcript


def test_basis_loop_zero_regrow_no_self_loop():
    """With regrow_count=0 we never enter `regrown` regardless of inner_refine_passes."""
    transcript, final = _drive(_base_ctx(regrow_count=0, inner_refine_passes=5))
    assert final == "done"
    assert transcript.count("perform_regrowth") == 0
    # compress fires once per pass when regrow=0; idx increments to 5 then exits.
    assert transcript.count("compress_with_polygram") == 5


def test_basis_loop_passes_one_matches_v01_with_regrow():
    """inner_refine_passes=1 + regrow_count>0 = v0.1 behavior: one compress, one regrow, then project."""
    transcript, final = _drive(_base_ctx(regrow_count=1, inner_refine_passes=1))
    assert final == "done"
    assert transcript.count("compress_with_polygram") == 1
    assert transcript.count("perform_regrowth") == 1


# ---------- Stream loop ----------


def _drive_with_eval_advance(ctx, advance_predicate):
    """Drive the FSM with a custom ``evaluate_faithfulness`` advance predicate."""
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import run_machine

    transcript: list[str] = []
    with _stubbed_action_table(transcript, advance_predicate=advance_predicate):
        run_machine(ctx)
    final = "done" if not ctx.get("error_message") else "failed"
    return transcript, final


def test_stream_loop_labeled_three_tasks():
    """n_tasks=3 + labeled trigger: two stream advances, one final save."""
    ctx = _base_ctx(n_tasks=3, task_trigger="labeled")
    transcript, final = _drive_with_eval_advance(
        ctx,
        lambda c: c.get("task_idx", 0) + 1 < c.get("n_tasks", 1),
    )
    assert final == "done"
    assert transcript.count("advance_to_next_task") == 2
    assert transcript.count("save_final_model") == 1
    # Three eval cycles total
    assert transcript.count("evaluate_faithfulness") == 3
    assert ctx["task_idx"] == 2


def test_stream_advance_dominates_should_continue():
    """With both advance_stream=true and should_continue=true, stream wins."""
    ctx = _base_ctx(n_tasks=2, iterations=3, task_trigger="labeled")

    def advance(c):
        # Bound should_continue by iterations so the refine loop has a
        # natural exit. The point of the test is the *first* eval where
        # both flags are true: stream_advance must win that race.
        c["should_continue"] = c.get("current_iter", 0) + 1 < c.get("iterations", 1)
        return c.get("task_idx", 0) + 1 < c.get("n_tasks", 1)

    transcript, final = _drive_with_eval_advance(ctx, advance)
    assert final == "done"
    # On the first eval (task 0, current_iter 0), both predicates are true.
    # Stream wins → advance_to_next_task fires, no rotate yet. On task 1
    # the stream is exhausted so the refine loop runs current_iter -> iterations.
    assert transcript.count("advance_to_next_task") == 1
    assert transcript.count("save_final_model") == 1


# ---------- Refine loop preservation ----------


def test_refine_loop_unchanged_under_n_tasks_one():
    """v0.1 refine loop semantics survive: 3-iter forge with n_tasks=1."""
    ctx = _base_ctx(n_tasks=1, iterations=3)

    def advance(c):
        # advance_stream stays false; should_continue flips per iteration.
        # With n_tasks=1, advance_stream is always false. should_continue follows
        # current_iter < iterations.
        c["should_continue"] = c.get("current_iter", 0) + 1 < c.get("iterations", 1)
        return False  # never advance stream

    transcript, final = _drive_with_eval_advance(ctx, advance)
    assert final == "done"
    # The refine loop fires `rotate_for_next_iter` for each refine pass.
    assert transcript.count("rotate_for_next_iter") == 2  # 3 iterations -> 2 rotations
    assert transcript.count("save_final_model") == 1
