"""FSM orchestrator — composes three orca sub-machines into one forge run.

The hierarchical-fsm capability replaces the v0.2 flat ten-state machine
with three composed orca sub-machines:

- ``StreamMachine`` (outermost) — shard / task management.
- ``RefineMachine`` (middle) — per-shard convergence (load+scan, basis
  loop, evaluate, refine-loop arbitration).
- ``BasisMachine`` (innermost) — compress/regrow/project/finetune.

Composition uses ``orca_runtime_python``'s native ``- invoke:`` directive
and ``parse_orca_md_multi`` loader. The orchestrator wires every spawned
child machine to share its parent's context dict (so the hierarchy looks
like a single ctx-shared FSM at runtime, matching the v0.2 mental model)
and register the same ``ACTION_TABLE`` (so actions resolve regardless of
which machine emitted the event).

The driver loop walks down the ``_active_invoke`` chain to find the
current leaf machine, sends its canonical event, and repeats until the
top-level StreamMachine reaches a final state. The runtime auto-bubbles
``[final]`` of a child to its parent's ``on_done`` event, so the leaf
naturally moves up the hierarchy as sub-machines complete.
"""

from __future__ import annotations

import asyncio
import importlib.resources

from saeforge.utils.lazy import require_extra


_FINAL_STATES = {"done", "failed"}
_MACHINE_FILES = ("stream.orca.md", "refine.orca.md", "basis.orca.md")
_PATH_SUFFIX = {"StreamMachine": "stream", "RefineMachine": "refine", "BasisMachine": "basis"}


def load_machine_hierarchy():
    """Parse the three forge sub-machines and return ``[stream, refine, basis]``.

    Returns a list of ``MachineDef`` objects in nesting order (outermost
    first). The orchestrator uses index 0 as the root machine and indices
    1+ as siblings registered on the root for invoke resolution.
    """
    require_extra("orca_runtime_python", "orca")
    from orca_runtime_python.parser import parse_orca_md_multi

    texts = [
        importlib.resources.files("saeforge.machines").joinpath(name).read_text()
        for name in _MACHINE_FILES
    ]
    return parse_orca_md_multi("\n---\n".join(texts))


def load_machine_definition():
    """Backward-compatible alias returning the root (StreamMachine) def.

    Kept for callers that introspect the machine for state-set sanity
    checks. New callers should prefer ``load_machine_hierarchy()`` to see
    the full hierarchy.
    """
    return load_machine_hierarchy()[0]


def _build_canonical_event_map(machine_def) -> dict[str, str]:
    """For each non-final source state, pick the first non-error outgoing event."""
    canonical: dict[str, str] = {}
    for transition in machine_def.transitions:
        if transition.event == "error":
            continue
        canonical.setdefault(transition.source, transition.event)
    return canonical


def _make_shared_ctx_machine_class(runtime, action_table: dict, sibling_defs: dict):
    """Build a runtime-OrcaMachine subclass with shared ctx + action propagation.

    The base ``OrcaMachine.start_child_machine`` creates each child with a
    fresh context dict (``dict(child_def.context)``) and an empty action
    handler table. The forge hierarchy needs the opposite: every machine
    in the tree shares one ctx dict (so writes in one sub-machine are
    visible to all) and every machine sees the full ACTION_TABLE.
    """

    OrcaMachine = runtime.OrcaMachine

    class SharedCtxOrcaMachine(OrcaMachine):
        async def start_child_machine(self, state_name, invoke_def):
            if invoke_def.machine not in sibling_defs:
                return
            child_def = sibling_defs[invoke_def.machine]

            # Shared ctx: pass the parent's dict object directly (NOT a copy).
            # The runtime's default would call ``dict(child_def.context)``;
            # we override with the live shared dict.
            child = SharedCtxOrcaMachine(
                definition=child_def,
                event_bus=self.event_bus,
                context=self.context,
            )
            # Children can spawn their own children too (BasisMachine has none
            # today, but the registration is harmless and future-proof).
            child.register_machines(sibling_defs)
            for name, handler in action_table.items():
                child.register_action(name, handler)

            # _machine_path: push the child's suffix on entry; the closure
            # below pops it back when the child finalizes.
            previous_path = self.context.get("_machine_path", "stream")
            suffix = _PATH_SUFFIX.get(invoke_def.machine, invoke_def.machine.lower())
            self.context["_machine_path"] = (
                f"{previous_path}/{suffix}" if previous_path else suffix
            )

            self._child_machines[state_name] = child
            self._active_invoke = state_name

            async def on_transition_handler(_old, new):
                if new.is_compound():
                    return
                child_state = new.leaf()
                child_state_def = child._find_state_def(child_state)
                if child_state_def and child_state_def.is_final:
                    # Distinguish success vs failure: the runtime fires
                    # on_done for any final state, but a child reaching
                    # ``failed`` is a propagation error — bubble up via
                    # the parent's ``error`` event so the parent enters
                    # its own ``failed`` state and runs ``log_error``.
                    payload = {
                        "child": invoke_def.machine,
                        "final_state": child_state,
                        "context": child.context,
                        "error": self.context.get("error_message", ""),
                    }
                    await child.stop()
                    self._child_machines.pop(state_name, None)
                    if self._active_invoke == state_name:
                        self._active_invoke = None
                    self.context["_machine_path"] = previous_path
                    if child_state == "failed":
                        await self.send("error", payload)
                    elif invoke_def.on_done:
                        await self.send(invoke_def.on_done, payload)

            child.on_transition = on_transition_handler
            await child.start()

    return SharedCtxOrcaMachine


def _find_active_leaf(machine):
    """Walk down the ``_active_invoke`` chain to the deepest live child."""
    current = machine
    while True:
        active = current._active_invoke
        if not active:
            return current
        child = current._child_machines.get(active)
        if child is None:
            return current
        current = child


def run_machine(initial_context: dict) -> dict:
    """Drive the hierarchical forge FSM synchronously and return final ctx.

    ``initial_context`` is mutated in place AND returned.
    """
    runtime = require_extra("orca_runtime_python", "orca")
    asyncio.run(_run_async(runtime, initial_context))
    return initial_context


async def _run_async(runtime, ctx: dict) -> None:
    from saeforge.actions import ACTION_TABLE

    defs = load_machine_hierarchy()
    root_def = defs[0]
    sibling_defs = {d.name: d for d in defs}  # include root so children can re-invoke
    canonical_per_machine = {d.name: _build_canonical_event_map(d) for d in defs}

    # Initialize _machine_path before machine spawn so the first _log entry
    # records the right path.
    ctx.setdefault("_machine_path", _PATH_SUFFIX[root_def.name])

    SharedCtxMachine = _make_shared_ctx_machine_class(runtime, ACTION_TABLE, sibling_defs)
    machine = SharedCtxMachine(root_def, context=ctx)
    machine.register_machines(sibling_defs)
    for name, handler in ACTION_TABLE.items():
        machine.register_action(name, handler)

    await machine.start()
    await _step(machine, "start")

    while machine.state.value not in _FINAL_STATES:
        leaf = _find_active_leaf(machine)
        leaf_state = leaf.state.leaf() if leaf.state.is_compound() else leaf.state.value
        if leaf_state in _FINAL_STATES and leaf is machine:
            break
        canonical = canonical_per_machine.get(leaf.definition.name, {})
        event = canonical.get(leaf_state)
        if event is None:
            raise RuntimeError(
                f"no canonical event derivable for state {leaf_state!r} in "
                f"machine {leaf.definition.name!r}; the .orca.md transition "
                "table must declare at least one non-error outgoing transition "
                "for every non-final state"
            )
        await _step(leaf, event)

    await machine.stop()


async def _step(machine, event_name: str) -> None:
    """Send an event; convert action exceptions into the FSM ``error`` event."""
    try:
        await machine.send(event_name)
    except Exception as e:
        machine.context["error_message"] = f"{type(e).__name__}: {e}"
        await machine.send("error", payload={"error": str(e)})
