"""FSM orchestrator — wraps OrcaMachine to drive the full forge pipeline.

The orchestrator is intentionally thin: it parses the .orca.md, derives
the canonical post-action event for each non-final state from the
transition table, and drives the runtime until a final state is
reached. All control-flow decisions (basis-loop, refine-loop, stream-
loop) live in the machine's guard expressions; this file does not
duplicate them.
"""

from __future__ import annotations

import asyncio
import importlib.resources

from saeforge.utils.lazy import require_extra


_FINAL_STATES = {"done", "failed"}


def load_machine_definition():
    """Parse the canonical sae_forge.orca.md from package resources."""
    runtime = require_extra("orca_runtime_python", "orca")
    text = importlib.resources.files("saeforge.machines").joinpath("sae_forge.orca.md").read_text()
    return runtime.parse_orca_md(text)


def _derive_canonical_events(machine_def) -> dict[str, str]:
    """Build the state -> next-event map from the parsed transition table.

    For each non-final source state we pick its single non-error
    outgoing event as the canonical post-action event. If a state has
    multiple distinct non-error events (none today) the first wins —
    callers can override by editing the .orca.md.
    """
    canonical: dict[str, str] = {}
    for transition in machine_def.transitions:
        if transition.event == "error":
            continue
        canonical.setdefault(transition.source, transition.event)
    return canonical


def run_machine(initial_context: dict) -> dict:
    """Drive the SaeForge FSM synchronously and return the final context.

    ``initial_context`` is mutated in place AND returned.
    """
    runtime = require_extra("orca_runtime_python", "orca")
    asyncio.run(_run_async(runtime, initial_context))
    return initial_context


async def _run_async(runtime, ctx: dict) -> None:
    from saeforge.actions import ACTION_TABLE

    machine_def = load_machine_definition()
    next_event_for_state = _derive_canonical_events(machine_def)
    machine = runtime.OrcaMachine(machine_def, context=ctx)

    for name, handler in ACTION_TABLE.items():
        machine.register_action(name, handler)

    await machine.start()

    await _step(machine, "start")
    while machine.state.value not in _FINAL_STATES:
        current = machine.state.value
        event = next_event_for_state.get(current)
        if event is None:
            raise RuntimeError(
                f"no canonical event derivable for state {current!r}; the "
                ".orca.md transition table must declare at least one "
                "non-error outgoing transition for every non-final state"
            )
        await _step(machine, event)

    await machine.stop()


async def _step(machine, event_name: str) -> None:
    """Send an event; convert action exceptions into the FSM ``error`` event."""
    try:
        await machine.send(event_name)
    except Exception as e:
        machine.context["error_message"] = f"{type(e).__name__}: {e}"
        await machine.send("error", payload={"error": str(e)})
