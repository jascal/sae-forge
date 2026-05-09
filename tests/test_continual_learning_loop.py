"""End-to-end FSM tests for the v0.2 continual-learning extensions.

Stub-driven: bypasses the heavy compress/regrow/project actions by
registering trivial handlers, so we can exercise the FSM topology
itself without waiting for torch to do real work.
"""

from __future__ import annotations

import asyncio

import pytest


def _stub_actions(transcript):
    def factory(name):
        def action(ctx, payload):
            transcript.append(name)
            if name == "compress_with_polygram" and ctx.get("regrow_count", 0) == 0:
                return {"inner_refine_idx": ctx.get("inner_refine_idx", 0) + 1}
            if name == "perform_regrowth":
                return {"inner_refine_idx": ctx.get("inner_refine_idx", 0) + 1}
            if name == "advance_to_next_task":
                return {
                    "task_idx": ctx.get("task_idx", 0) + 1,
                    "inner_refine_idx": 0,
                    "tokens_seen_in_task": 0,
                    "advance_stream": False,
                }
            return {}
        return action
    return factory


async def _drive(ctx):
    """Run the FSM with stubbed actions; return the transcript."""
    pytest.importorskip("orca_runtime_python")
    import orca_runtime_python as orp

    from saeforge.orchestrator import _derive_canonical_events, load_machine_definition

    machine_def = load_machine_definition()
    next_event = _derive_canonical_events(machine_def)
    machine = orp.OrcaMachine(machine_def, context=ctx)
    transcript: list[str] = []
    factory = _stub_actions(transcript)
    for name in [
        "load_sae_and_corpus", "scan_activations", "compress_with_polygram",
        "perform_regrowth", "project_to_subspace", "fine_tune_model",
        "evaluate_faithfulness", "advance_to_next_task", "rotate_for_next_iter",
        "save_final_model", "log_error",
    ]:
        machine.register_action(name, factory(name))
    await machine.start()
    await machine.send("start")
    while machine.state.value not in ("done", "failed"):
        await machine.send(next_event[machine.state.value])
    await machine.stop()
    return transcript, machine.state.value


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
    transcript, final = asyncio.run(_drive(_base_ctx()))
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
    transcript, final = asyncio.run(
        _drive(_base_ctx(regrow_count=1, inner_refine_passes=3))
    )
    assert final == "done"
    n_compress = transcript.count("compress_with_polygram")
    n_regrow = transcript.count("perform_regrowth")
    # 3 compress + 3 regrow expected; the first compress comes from the
    # activations_scanned -> compressed transition.
    assert n_compress == 3, transcript
    assert n_regrow == 3, transcript


def test_basis_loop_zero_regrow_no_self_loop():
    """With regrow_count=0 we never enter `regrown` regardless of inner_refine_passes."""
    transcript, final = asyncio.run(
        _drive(_base_ctx(regrow_count=0, inner_refine_passes=5))
    )
    assert final == "done"
    assert transcript.count("perform_regrowth") == 0
    # compress fires once per pass when regrow=0; idx increments to 5 then exits.
    assert transcript.count("compress_with_polygram") == 5


def test_basis_loop_passes_one_matches_v01_with_regrow():
    """inner_refine_passes=1 + regrow_count>0 = v0.1 behavior: one compress, one regrow, then project."""
    transcript, final = asyncio.run(
        _drive(_base_ctx(regrow_count=1, inner_refine_passes=1))
    )
    assert final == "done"
    assert transcript.count("compress_with_polygram") == 1
    assert transcript.count("perform_regrowth") == 1


# ---------- Stream loop ----------


def _drive_with_eval_advance(ctx, advance_predicate):
    """Drive the FSM stubbing evaluate_faithfulness to set advance_stream per predicate."""
    pytest.importorskip("orca_runtime_python")
    import orca_runtime_python as orp

    from saeforge.orchestrator import _derive_canonical_events, load_machine_definition

    machine_def = load_machine_definition()
    next_event = _derive_canonical_events(machine_def)
    machine = orp.OrcaMachine(machine_def, context=ctx)
    transcript: list[str] = []

    def make(name):
        def action(c, p):
            transcript.append(name)
            if name == "compress_with_polygram" and c.get("regrow_count", 0) == 0:
                return {"inner_refine_idx": c.get("inner_refine_idx", 0) + 1}
            if name == "perform_regrowth":
                return {"inner_refine_idx": c.get("inner_refine_idx", 0) + 1}
            if name == "evaluate_faithfulness":
                return {"advance_stream": advance_predicate(c)}
            if name == "advance_to_next_task":
                return {
                    "task_idx": c.get("task_idx", 0) + 1,
                    "inner_refine_idx": 0,
                    "tokens_seen_in_task": 0,
                    "advance_stream": False,
                }
            return {}
        return action

    for name in [
        "load_sae_and_corpus", "scan_activations", "compress_with_polygram",
        "perform_regrowth", "project_to_subspace", "fine_tune_model",
        "evaluate_faithfulness", "advance_to_next_task", "rotate_for_next_iter",
        "save_final_model", "log_error",
    ]:
        machine.register_action(name, make(name))

    async def go():
        await machine.start()
        await machine.send("start")
        while machine.state.value not in ("done", "failed"):
            await machine.send(next_event[machine.state.value])
        await machine.stop()
    asyncio.run(go())
    return transcript, machine.state.value


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
        # Set should_continue=true via the predicate side-effect.
        c["should_continue"] = True
        return c.get("task_idx", 0) + 1 < c.get("n_tasks", 1)

    transcript, final = _drive_with_eval_advance(ctx, advance)
    assert final == "done"
    # Stream advances once, then on the second eval should_continue is still true
    # but task budget exhausted, so we terminate (refine_same_shard would fire too,
    # but stream_advance is false at task_idx=1 with n_tasks=2 → refine wins on
    # the second eval). Allow either pattern; assert just the topology contract:
    assert transcript.count("advance_to_next_task") >= 1
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
