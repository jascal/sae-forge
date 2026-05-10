"""§6.5 / §7.5 — multi-shard hierarchy integration tests.

These tests drive the FSM with ``n_tasks > 1`` via the same
``run_synthetic`` entry point real callers use. The hierarchy MUST
re-spawn ``RefineMachine`` (and through it ``BasisMachine``) on
each shard advance — verifying that compound-state re-entry works
correctly across multiple cycles, which single-shard byte-equivalence
cannot catch.
"""

from __future__ import annotations

import pytest


def test_fsm_with_n_tasks_two_runs_advance_to_next_task_once(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """``n_tasks = 2`` with the labeled trigger SHALL fire one advance + one save."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        orchestrator="fsm",
        n_tasks=2,
        task_trigger="labeled",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "multi-shard", eval_input_ids=eval_input_ids
    )

    log = result.extras["transitions_log"]
    actions = [e["action"] for e in log]
    # advance_to_next_task fires once (between the two shards).
    assert actions.count("advance_to_next_task") == 1, actions
    # save_final_model fires exactly once at termination.
    assert actions.count("save_final_model") == 1, actions
    # The full forge pipeline runs twice — two compresses, two projects,
    # two evaluates. ``load_and_scan`` runs at the start of each shard
    # (the entering state of each fresh RefineMachine spawn), expanding
    # to two ``load_sae_and_corpus`` + two ``scan_activations`` entries.
    assert actions.count("load_sae_and_corpus") == 2, actions
    assert actions.count("scan_activations") == 2, actions
    assert actions.count("compress_with_polygram") == 2, actions
    assert actions.count("project_to_subspace") == 2, actions
    assert actions.count("evaluate_faithfulness") == 2, actions

    # Final state and ctx invariants.
    assert result.extras["final_state"] == "done"


def test_fsm_n_tasks_two_machine_path_evolves_correctly(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """``transitions_log[*].machine_path`` SHALL accurately reflect the active sub-machine."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        orchestrator="fsm",
        n_tasks=2,
        task_trigger="labeled",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2, tmp_path / "machine-path", eval_input_ids=eval_input_ids
    )

    log = result.extras["transitions_log"]
    paths_by_action = {}
    for entry in log:
        paths_by_action.setdefault(entry["action"], set()).add(entry["machine_path"])

    # Basis-loop actions live in stream/refine/basis.
    assert paths_by_action["compress_with_polygram"] == {"stream/refine/basis"}
    assert paths_by_action["project_to_subspace"] == {"stream/refine/basis"}
    assert paths_by_action["fine_tune_model"] == {"stream/refine/basis"}
    # Evaluate runs at the refine level.
    assert paths_by_action["evaluate_faithfulness"] == {"stream/refine"}
    # advance_to_next_task and save_final_model fire from StreamMachine.
    assert paths_by_action["advance_to_next_task"] == {"stream"}
    assert paths_by_action["save_final_model"] == {"stream"}
    # load_sae_and_corpus / scan_activations run inside load_and_scan,
    # which is the transition action on RefineMachine.entering → refining.
    # The orchestrator updates _machine_path on compound-state entry, so
    # by the time _log fires for these inner actions we are already in
    # stream/refine.
    assert paths_by_action["load_sae_and_corpus"] == {"stream/refine"}
    assert paths_by_action["scan_activations"] == {"stream/refine"}
