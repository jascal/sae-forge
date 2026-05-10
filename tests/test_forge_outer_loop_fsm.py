"""Tests for the v0.1 forge-outer-loop FSM orchestrator."""

from __future__ import annotations

import hashlib

import pytest


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hierarchy_has_three_machines_with_v02_state_distribution():
    """The hierarchy distributes the v0.2 ten states across three sub-machines.

    The flat ten states map as follows:

    - StreamMachine: ``init`` (initial), ``streaming`` (compound),
      ``next_shard``, ``done`` (final), ``failed`` (final)
    - RefineMachine: ``entering`` (collapses v0.2 ``loaded`` +
      ``activations_scanned``), ``refining`` (compound), ``evaluating``
      (renamed from v0.2 ``evaluated``), ``exiting`` (final), ``failed``
    - BasisMachine: ``starting`` (initial), ``compressed``, ``regrown``,
      ``projected``, ``finetuned``, ``done`` (final), ``failed``
    """
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import load_machine_hierarchy

    defs = load_machine_hierarchy()
    by_name = {d.name: d for d in defs}
    assert set(by_name) == {"StreamMachine", "RefineMachine", "BasisMachine"}
    assert {s.name for s in by_name["StreamMachine"].states} == {
        "init", "streaming", "next_shard", "done", "failed",
    }
    assert {s.name for s in by_name["RefineMachine"].states} == {
        "entering", "refining", "evaluating", "exiting", "failed",
    }
    assert {s.name for s in by_name["BasisMachine"].states} == {
        "starting", "compressed", "regrown", "projected", "finetuned",
        "done", "failed",
    }


def test_compound_states_invoke_their_children():
    """``StreamMachine.streaming`` invokes ``RefineMachine``; ``RefineMachine.refining`` invokes ``BasisMachine``."""
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import load_machine_hierarchy

    by_name = {d.name: d for d in load_machine_hierarchy()}
    streaming = next(s for s in by_name["StreamMachine"].states if s.name == "streaming")
    assert streaming.invoke is not None
    assert streaming.invoke.machine == "RefineMachine"
    assert streaming.invoke.on_done == "refine_done"

    refining = next(s for s in by_name["RefineMachine"].states if s.name == "refining")
    assert refining.invoke is not None
    assert refining.invoke.machine == "BasisMachine"
    assert refining.invoke.on_done == "basis_done"

    assert all(s.invoke is None for s in by_name["BasisMachine"].states)


def test_hierarchy_has_required_guards_per_machine():
    """Guards are scoped to the sub-machine that owns the state they protect.

    Basis-loop guards live in ``BasisMachine``; refine-loop guards live in
    ``RefineMachine``; stream-loop guards live in ``StreamMachine``. The
    v0.2 ``refine_same_shard`` is renamed ``refine_continue`` because the
    "same shard" semantic is internalized inside ``RefineMachine``.
    """
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import load_machine_hierarchy

    by_name = {d.name: d for d in load_machine_hierarchy()}
    assert {"should_regrow", "no_regrow_more_passes", "no_regrow_done",
            "basis_loop_continue", "basis_loop_done"} <= set(by_name["BasisMachine"].guards)
    assert {"refine_continue", "refine_exit"} <= set(by_name["RefineMachine"].guards)
    assert {"stream_advance", "terminate_run"} <= set(by_name["StreamMachine"].guards)


def test_fsm_run_synthetic_end_to_end(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "fsm-out", eval_input_ids=eval_input_ids)
    assert result.n_params > 0
    assert result.faithfulness_kl is not None
    assert result.faithfulness_kl >= 0.0
    assert (tmp_path / "fsm-out" / "forged" / "config.json").is_file()
    assert (tmp_path / "fsm-out" / "forged" / "model.safetensors").is_file()
    assert result.extras["final_state"] == "done"


def test_fsm_transitions_log_has_full_sequence(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "fsm-log", eval_input_ids=eval_input_ids)

    actions_in_order = [entry["action"] for entry in result.extras["transitions_log"]]
    expected = [
        "load_sae_and_corpus",
        "scan_activations",  # v0.2 inserts a no-op pass-through here under defaults
        "compress_with_polygram",
        "project_to_subspace",
        "fine_tune_model",
        "evaluate_faithfulness",
        "save_final_model",
    ]
    assert actions_in_order == expected, actions_in_order


def test_imperative_and_fsm_byte_equivalent(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """v0.1 -> v0.2 migration safety net: byte-identical forged weights."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)

    torch.manual_seed(0)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    imp = ForgePipeline(basis=tiny_synthetic_basis, projector=projector, orchestrator="imperative")
    imp_result = imp.run_synthetic(tiny_gpt2, tmp_path / "imp", eval_input_ids=eval_input_ids)

    fsm = ForgePipeline(basis=tiny_synthetic_basis, projector=projector, orchestrator="fsm")
    fsm_result = fsm.run_synthetic(tiny_gpt2, tmp_path / "fsm", eval_input_ids=eval_input_ids)

    imp_weights = tmp_path / "imp" / "forged" / "model.safetensors"
    fsm_weights = tmp_path / "fsm" / "forged" / "model.safetensors"

    assert _file_sha256(imp_weights) == _file_sha256(fsm_weights)
    assert imp_result.n_params == fsm_result.n_params


def test_fsm_quantum_aware_topology_unchanged(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """Setting quantum_aware=True must not change the state set or transitions."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector
    from saeforge.orchestrator import load_machine_definition

    states_classical = {s.name for s in load_machine_definition().states}
    transitions_classical = {
        (t.source, t.event, t.guard, t.target, t.action) for t in load_machine_definition().transitions
    }

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        quantum_aware=True,
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "qa", eval_input_ids=eval_input_ids)
    assert result.extras["final_state"] == "done"

    states_after = {s.name for s in load_machine_definition().states}
    transitions_after = {
        (t.source, t.event, t.guard, t.target, t.action) for t in load_machine_definition().transitions
    }
    assert states_classical == states_after
    assert transitions_classical == transitions_after


def test_fsm_orca_extra_missing_raises_actionable_import_error(monkeypatch):
    """When orca_runtime_python is not importable, run_machine raises a clear error."""
    import sys

    saved = sys.modules.pop("orca_runtime_python", None)
    monkeypatch.setitem(sys.modules, "orca_runtime_python", None)
    try:
        from saeforge.orchestrator import run_machine

        with pytest.raises(ImportError, match=r"\[orca\]"):
            run_machine({})
    finally:
        if saved is not None:
            sys.modules["orca_runtime_python"] = saved
        else:
            sys.modules.pop("orca_runtime_python", None)
