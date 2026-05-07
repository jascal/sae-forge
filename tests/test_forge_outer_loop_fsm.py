"""Tests for the v0.1 forge-outer-loop FSM orchestrator."""

from __future__ import annotations

import hashlib

import pytest


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_machine_loads_and_has_nine_states():
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import load_machine_definition

    m = load_machine_definition()
    state_names = {s.name for s in m.states}
    assert state_names == {
        "init",
        "loaded",
        "compressed",
        "regrown",
        "projected",
        "finetuned",
        "evaluated",
        "done",
        "failed",
    }
    initial = [s for s in m.states if s.is_initial]
    assert [s.name for s in initial] == ["init"]
    finals = sorted(s.name for s in m.states if s.is_final)
    assert finals == ["done", "failed"]


def test_machine_has_required_guards():
    pytest.importorskip("orca_runtime_python")
    from saeforge.orchestrator import load_machine_definition

    m = load_machine_definition()
    assert "should_regrow" in m.guards
    assert "no_regrow" in m.guards
    assert "should_continue_loop" in m.guards


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
