"""Runtime probe for orca-runtime-python compound-state support.

This test is the **gate** for the hierarchical-fsm change: it confirms
that `parse_orca_md_multi`, `> invoke: ChildMachine`, and the parent's
`on_done` event wiring all work together end-to-end. If this test
fails, the hierarchical-fsm refactor is parked pending a runtime fix
(per `openspec/changes/hierarchical-fsm/tasks.md` §1.1–1.2).

It also empirically pins the context-scoping behavior that the design
hinges on: whether parent and child share a context dict, or each
machine owns its own. The answer drives how `load_machine_hierarchy`
wires the three forge sub-machines.
"""

from __future__ import annotations

import asyncio

import pytest

orca_runtime = pytest.importorskip("orca_runtime_python")


_PARENT = """
# machine Outer

## context

| Field | Type | Default |
|-------|------|---------|
| outer_count | int | 0 |
| inner_done_seen | bool | false |

## state init [initial]
## state running

- invoke: Inner
- on_done: -> child_done

## state done [final]

## transitions

| Source  | Event       | Guard | Target  | Action          |
|---------|-------------|-------|---------|-----------------|
| init    | start       |       | running |                 |
| running | child_done  |       | done    | record_finished |

## actions

| Name            | Signature       |
|-----------------|-----------------|
| record_finished | (ctx) -> Context |
"""

_CHILD = """
# machine Inner

## context

| Field | Type | Default |
|-------|------|---------|
| inner_count | int | 0 |

## state working [initial]
## state finished [final]

## transitions

| Source  | Event  | Guard | Target   | Action      |
|---------|--------|-------|----------|-------------|
| working | finish |       | finished | bump_inner  |

## actions

| Name       | Signature       |
|------------|-----------------|
| bump_inner | (ctx) -> Context |
"""


def _run() -> tuple[dict, dict]:
    """Execute the Outer{invoke: Inner} fixture; return (outer_ctx, inner_ctx)."""
    parent_def, child_def = orca_runtime.parser.parse_orca_md_multi(
        _PARENT + "\n---\n" + _CHILD
    )
    assert parent_def.name == "Outer"
    assert child_def.name == "Inner"

    captured_inner_ctx: dict = {}

    def record_finished(ctx, payload):
        # The on_done event delivers the child's final context as payload.context.
        if payload and "context" in payload:
            captured_inner_ctx.update(payload["context"])
        return {"inner_done_seen": True, "outer_count": ctx["outer_count"] + 1}

    def bump_inner(ctx, _payload):
        return {"inner_count": ctx["inner_count"] + 1}

    parent = orca_runtime.OrcaMachine(parent_def, context=dict(parent_def.context))
    parent.register_machines({"Inner": child_def})
    parent.register_action("record_finished", record_finished)

    async def go():
        await parent.start()
        await parent.send("start")
        await asyncio.sleep(0)  # yield for child startup
        child = parent._child_machines.get("running")
        assert child is not None, "child machine should be live during running state"
        # Action handlers are per-machine; the parent's handler table is
        # not visible to the spawned child. The orchestrator must register
        # actions on every child machine — pinned by this probe.
        child.register_action("bump_inner", bump_inner)
        await child.send("finish")
        # Give the on_done handler time to fire and parent transition to run
        await asyncio.sleep(0)
        await parent.stop()
        return parent.context, child.context

    outer_ctx, inner_ctx = asyncio.run(go())
    inner_ctx.update(captured_inner_ctx)  # keep the captured copy too
    return outer_ctx, inner_ctx


def test_compound_state_invokes_child():
    """Outer's `running` state spawns Inner; Inner's [final] fires Outer's on_done."""
    outer_ctx, inner_ctx = _run()
    assert outer_ctx["inner_done_seen"] is True, (
        "parent's on_done handler did not fire — runtime probe failed"
    )
    assert outer_ctx["outer_count"] == 1
    assert inner_ctx["inner_count"] == 1


def test_parent_and_child_have_separate_contexts():
    """Empirically pin: parent and child each own their own ctx dict.

    This is load-bearing for the hierarchical-fsm design. If contexts
    were shared, we would not need explicit input/on_done payload
    wiring. They are not shared (verified here), so the forge
    orchestrator must merge child ctx back via on_done payloads.
    """
    outer_ctx, inner_ctx = _run()
    assert "inner_count" not in outer_ctx or outer_ctx.get("inner_count") == 0, (
        "outer should not see inner_count unless we merged it; got "
        f"outer_ctx={outer_ctx}"
    )
    assert "outer_count" not in inner_ctx, (
        f"inner should not see outer_count; got inner_ctx={inner_ctx}"
    )
