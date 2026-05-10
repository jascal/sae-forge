"""§9.4 — committed Mermaid diagram MUST match the live emit.

The diagram embedded in ``docs/advanced-fsm-options.md`` is the
auto-generated output of ``saeforge.machines.visualize.to_mermaid``.
This test extracts the committed block and asserts it equals the
live emit byte-for-byte. If it fails, regenerate the diagram by
running ``sae-forge inspect --fsm-diagram`` and pasting the output
into the doc between the BEGIN/END markers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

orca_runtime = pytest.importorskip("orca_runtime_python")


_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "advanced-fsm-options.md"
_BLOCK_RE = re.compile(
    r"<!-- BEGIN AUTO-GENERATED FSM DIAGRAM -->\n```mermaid\n(.*?)```\n<!-- END AUTO-GENERATED FSM DIAGRAM -->",
    re.DOTALL,
)


def _extract_committed_diagram() -> str:
    text = _DOC_PATH.read_text()
    match = _BLOCK_RE.search(text)
    assert match is not None, (
        f"could not find auto-generated FSM diagram block in {_DOC_PATH}; "
        "it should be wrapped in BEGIN/END HTML comments around a "
        "```mermaid``` fenced block"
    )
    return match.group(1)


def test_committed_diagram_matches_live_emit():
    """The committed Mermaid block SHALL byte-equal ``to_mermaid(load_machine_hierarchy())``."""
    from saeforge.machines.visualize import to_mermaid
    from saeforge.orchestrator import load_machine_hierarchy

    committed = _extract_committed_diagram()
    live = to_mermaid(load_machine_hierarchy())
    if committed != live:
        pytest.fail(
            "Committed FSM diagram has drifted from saeforge/machines/{stream,refine,basis}.orca.md.\n"
            "Regenerate by running:\n\n"
            "    sae-forge inspect --fsm-diagram\n\n"
            "Paste the output between the BEGIN/END markers in "
            f"{_DOC_PATH.relative_to(Path.cwd())}.\n\n"
            "Live diff (first 500 chars):\n"
            f"committed:\n{committed[:500]}\n\nlive:\n{live[:500]}"
        )
