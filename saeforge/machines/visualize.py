"""Mermaid emitter for the three-machine forge hierarchy.

Auto-generates a ``stateDiagram-v2`` block from the parsed hierarchy
returned by ``saeforge.orchestrator.load_machine_hierarchy``. The
emitter is the canonical source of truth for the diagram embedded
in ``docs/advanced-fsm-options.md`` — a CI test
(``tests/fsm/test_diagram_drift.py``) asserts the committed diagram
matches the live emit on every run, so drift can't land.

The output is plain Mermaid text (no library dependency); rendering
is done client-side by GitHub, MkDocs, or whatever consumer reads
the doc.
"""

from __future__ import annotations

from typing import Iterable


def to_mermaid(hierarchy_defs: list) -> str:
    """Render a composed three-machine hierarchy as a single Mermaid block.

    ``hierarchy_defs`` is the list returned by ``load_machine_hierarchy()``
    (outermost first). Sub-machines are nested under their parent's
    compound state via Mermaid ``state X { ... }`` subgraphs, which is
    the v2 syntax for hierarchical state diagrams.
    """
    by_name = {d.name: d for d in hierarchy_defs}
    if not hierarchy_defs:
        return "stateDiagram-v2\n  [*] --> [*]\n"

    root = hierarchy_defs[0]
    lines: list[str] = ["stateDiagram-v2"]
    lines.extend(_render_machine(root, by_name, depth=1))
    return "\n".join(lines) + "\n"


def _render_machine(machine_def, by_name: dict, *, depth: int) -> list[str]:
    """Render a single machine + recurse into invoked children."""
    indent = "  " * depth
    out: list[str] = []
    initial = next((s for s in machine_def.states if s.is_initial), None)
    if initial is not None:
        out.append(f"{indent}[*] --> {initial.name}")

    # Build a transitions-by-source index so we can group rendering.
    by_source: dict[str, list] = {}
    for t in machine_def.transitions:
        by_source.setdefault(t.source, []).append(t)

    # Render every state. Compound (invoke) states get a nested subgraph.
    for s in machine_def.states:
        if s.invoke is not None and s.invoke.machine in by_name:
            child_def = by_name[s.invoke.machine]
            out.append(f'{indent}state "{s.name}" as {s.name} {{')
            out.extend(_render_machine(child_def, by_name, depth=depth + 1))
            out.append(f"{indent}}}")
        # Final states get explicit terminator markers in transitions below.

    # Render transitions. Targets that are final states render as `[*]`.
    final_names = {s.name for s in machine_def.states if s.is_final}
    for source, transitions in by_source.items():
        for t in transitions:
            if t.event == "error":
                # error transitions are rendered explicitly so reviewers
                # can see the failure paths in the diagram.
                pass
            target = "[*]" if t.target in final_names else t.target
            label = _format_transition_label(t)
            out.append(f"{indent}{source} --> {target} : {label}")

    return out


def _format_transition_label(transition) -> str:
    """``event [guard] / action`` — minimal, machine-readable, not pretty-printed."""
    parts = [transition.event]
    guard = getattr(transition, "guard", None)
    if guard:
        parts.append(f"[{guard}]")
    action = getattr(transition, "action", None)
    if action:
        parts.append(f"/ {action}")
    return " ".join(parts)


def _iter_states(defs: Iterable) -> Iterable:
    for d in defs:
        for s in d.states:
            yield s
