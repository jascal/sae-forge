"""§9 visualizer tests for the auto-generated Mermaid diagram."""

from __future__ import annotations

import pytest

orca_runtime = pytest.importorskip("orca_runtime_python")


def test_emitted_mermaid_is_state_diagram_v2():
    from saeforge.machines.visualize import to_mermaid
    from saeforge.orchestrator import load_machine_hierarchy

    out = to_mermaid(load_machine_hierarchy())
    assert out.startswith("stateDiagram-v2")
    assert out.endswith("\n")


def test_emitted_mermaid_contains_every_state_in_the_hierarchy():
    """All 16 state names across the three machines SHALL appear in the diagram."""
    from saeforge.machines.visualize import to_mermaid
    from saeforge.orchestrator import load_machine_hierarchy

    defs = load_machine_hierarchy()
    out = to_mermaid(defs)
    expected_names = {s.name for d in defs for s in d.states}
    # Final states render as `[*]` rather than by name. Collect non-final
    # state names explicitly.
    non_final = {s.name for d in defs for s in d.states if not s.is_final}
    for name in non_final:
        assert name in out, f"state {name!r} missing from emitted Mermaid"


def test_emitted_mermaid_nests_compound_states_as_subgraphs():
    """``StreamMachine.streaming`` SHALL contain ``RefineMachine``; same for refining → BasisMachine."""
    from saeforge.machines.visualize import to_mermaid
    from saeforge.orchestrator import load_machine_hierarchy

    out = to_mermaid(load_machine_hierarchy())
    assert 'state "streaming" as streaming {' in out
    assert 'state "refining" as refining {' in out
    # The basis-loop initial state must appear inside the refining block,
    # which is in turn inside the streaming block.
    streaming_open = out.index('state "streaming" as streaming {')
    refining_open = out.index('state "refining" as refining {')
    starting_index = out.index("starting")
    assert streaming_open < refining_open < starting_index


def test_emitted_mermaid_labels_guards_and_actions():
    """Transition labels SHALL include guard names and action names where present."""
    from saeforge.machines.visualize import to_mermaid
    from saeforge.orchestrator import load_machine_hierarchy

    out = to_mermaid(load_machine_hierarchy())
    # Spot-check several known guard/action labels.
    assert "[should_regrow]" in out
    assert "[basis_loop_continue]" in out
    assert "[stream_advance]" in out
    assert "/ compress_with_polygram" in out
    assert "/ load_and_scan" in out
    assert "/ save_final_model" in out
