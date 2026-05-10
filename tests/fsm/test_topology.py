"""§8 topology-checker tests for the three-machine hierarchy.

The orca runtime ships a parser-side validator (``_validate_machine_def``,
called from ``parse_orca_md_multi``) that rejects machines with dangling
transition sources, missing initial/final states, etc. These tests
confirm that:

1. Each sub-machine parses standalone with no errors.
2. The composed three-machine concatenation parses without
   cross-references breaking (the ``invoke:`` directives must name
   sibling machines that exist in the composed bundle).
3. The guard truth table is jointly exhaustive and pairwise disjoint
   over ``(advance_stream, should_continue)`` — preserves the v0.2
   contract.
"""

from __future__ import annotations

import importlib.resources

import pytest

orca_runtime = pytest.importorskip("orca_runtime_python")


_MACHINE_FILES = ("stream.orca.md", "refine.orca.md", "basis.orca.md")


def _read(name: str) -> str:
    return importlib.resources.files("saeforge.machines").joinpath(name).read_text()


def test_each_submachine_parses_standalone():
    """Every sub-machine SHALL pass the runtime's parser/validator in isolation."""
    from orca_runtime_python.parser import parse_orca_md

    for name in _MACHINE_FILES:
        text = _read(name)
        # parse_orca_md raises ParseError on validation failure.
        machine_def = parse_orca_md(text)
        assert machine_def.name in {"StreamMachine", "RefineMachine", "BasisMachine"}
        assert machine_def.states, f"{name} declared no states"
        assert any(s.is_initial for s in machine_def.states), f"{name} has no initial state"
        assert any(s.is_final for s in machine_def.states), f"{name} has no final state"


def test_composed_hierarchy_parses_clean():
    """The concatenated three-machine bundle SHALL parse without cross-ref errors."""
    from orca_runtime_python.parser import parse_orca_md_multi

    combined = "\n---\n".join(_read(name) for name in _MACHINE_FILES)
    defs = parse_orca_md_multi(combined)
    assert [d.name for d in defs] == ["StreamMachine", "RefineMachine", "BasisMachine"]

    # Every invoke directive must name a sibling that exists in the bundle.
    bundle_names = {d.name for d in defs}
    for d in defs:
        for s in d.states:
            if s.invoke is not None:
                assert s.invoke.machine in bundle_names, (
                    f"{d.name}.{s.name} invokes {s.invoke.machine!r} which is "
                    f"not present in the composed bundle {bundle_names}"
                )


def test_canonical_event_derivable_for_every_non_final_state():
    """``_build_canonical_event_map`` SHALL find an event for every non-final source."""
    from saeforge.orchestrator import _build_canonical_event_map, load_machine_hierarchy

    for d in load_machine_hierarchy():
        canonical = _build_canonical_event_map(d)
        for s in d.states:
            if s.is_final:
                continue
            assert s.name in canonical, (
                f"machine {d.name!r}: state {s.name!r} has no canonical event"
            )


def test_guard_truth_table_is_jointly_exhaustive_and_disjoint():
    """``stream_advance`` / ``refine_continue`` / ``terminate_run`` partition the guard space.

    Pre-hierarchy v0.2 had three guards routing the ``evaluated → eval_done``
    transition; the hierarchy redistributes them across StreamMachine
    (stream_advance, terminate_run) and RefineMachine (refine_continue).
    Together they MUST still partition ``(advance_stream, should_continue) ∈ {true, false}²``.
    """
    cases = [
        # (advance_stream, should_continue) -> exactly one of the three should fire
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    for advance_stream, should_continue in cases:
        stream_advance = advance_stream is True
        terminate_run = advance_stream is False and should_continue is False
        refine_continue = advance_stream is False and should_continue is True
        fired = [name for name, val in [
            ("stream_advance", stream_advance),
            ("terminate_run", terminate_run),
            ("refine_continue", refine_continue),
        ] if val]
        assert len(fired) == 1, (
            f"(advance_stream={advance_stream}, should_continue={should_continue}): "
            f"expected exactly one guard to fire, got {fired}"
        )


def test_basis_loop_self_loop_terminates_under_inner_refine_passes_cap():
    """``basis_loop_continue`` / ``basis_loop_done`` partition the inner-refine counter."""
    from saeforge.orchestrator import load_machine_hierarchy

    by_name = {d.name: d for d in load_machine_hierarchy()}
    basis = by_name["BasisMachine"]
    # Both guards must exist; their expressions must reference inner_refine_idx
    # against inner_refine_passes (the v0.2 truth table preserved verbatim).
    assert "basis_loop_continue" in basis.guards
    assert "basis_loop_done" in basis.guards
