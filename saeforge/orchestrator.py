"""FSM orchestrator — wraps OrcaMachine to drive the full forge pipeline."""

from __future__ import annotations

import asyncio
import importlib.resources

from saeforge.utils.lazy import require_extra


# State -> event-to-fire-after-its-action mapping. Each non-final state has one
# canonical "done" event whose name encodes which stage just completed.
_NEXT_EVENT_FOR_STATE = {
    "loaded": "load_done",
    "compressed": "compress_done",
    "regrown": "regrowth_done",
    "projected": "projection_done",
    "finetuned": "finetune_done",
    "evaluated": "eval_done",
}

_FINAL_STATES = {"done", "failed"}


def load_machine_definition():
    """Parse the canonical sae_forge.orca.md from package resources."""
    runtime = require_extra("orca_runtime_python", "orca")
    text = importlib.resources.files("saeforge.machines").joinpath("sae_forge.orca.md").read_text()
    return runtime.parse_orca_md(text)


def run_machine(initial_context: dict) -> dict:
    """Drive the SaeForge FSM synchronously and return the final context.

    ``initial_context`` is mutated in place AND returned. The caller can read
    final state values from the returned dict.
    """
    runtime = require_extra("orca_runtime_python", "orca")
    asyncio.run(_run_async(runtime, initial_context))
    return initial_context


async def _run_async(runtime, ctx: dict) -> None:
    from saeforge.actions import ACTION_TABLE

    machine_def = load_machine_definition()
    machine = runtime.OrcaMachine(machine_def, context=ctx)

    for name, handler in ACTION_TABLE.items():
        machine.register_action(name, handler)

    await machine.start()

    # Drive transitions: start, then fire the canonical event for each state we land in.
    await _step(machine, "start")
    while machine.state.value not in _FINAL_STATES:
        current = machine.state.value
        event = _NEXT_EVENT_FOR_STATE.get(current)
        if event is None:
            raise RuntimeError(f"no canonical event registered for state {current!r}")
        await _step(machine, event)

    await machine.stop()


async def _step(machine, event_name: str) -> None:
    """Send an event; convert action exceptions into the FSM ``error`` event."""
    try:
        await machine.send(event_name)
    except Exception as e:
        machine.context["error_message"] = f"{type(e).__name__}: {e}"
        await machine.send("error", payload={"error": str(e)})
