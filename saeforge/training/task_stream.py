"""Task stream abstractions for continual-learning forge runs.

Three concrete streams cover the three ``task_trigger`` modes:

- ``LabeledTaskStream``: a finite list of explicit shards (paths or
  in-memory iterables). ``next()`` advances down the list.
- ``TokenBudgetTaskStream``: a single underlying source, chunked by
  fine-tune tokens processed. The orchestrator decides *when* to call
  ``next()`` based on the ``advance_stream`` ctx flag; the stream
  itself just hands out the next chunk.
- ``LossDriftTaskStream``: a single source, advanced when a held-out
  probe loss climbs (the trigger logic lives in
  ``evaluate_task_advance``; the stream is a thin wrapper over the
  source that produces "next chunk" each time ``next()`` is called).

A process-local registry maps ``task_iterator_id`` strings to live
``TaskStream`` instances so the FSM context (which must be JSON-ish)
can reference a stream by handle without serializing it.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator


class TaskStream(ABC):
    """Yields one tokenized iterator per task. ``None`` signals exhaustion."""

    @abstractmethod
    def next(self) -> Iterable[Any] | None:
        """Return the next task's iterator, or None if the stream is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of tasks if known; otherwise -1 for open streams."""
        raise NotImplementedError


class LabeledTaskStream(TaskStream):
    """Finite list of (already-tokenized iterators or corpus paths)."""

    def __init__(self, tasks: list[Any]) -> None:
        if not tasks:
            raise ValueError("LabeledTaskStream requires at least one task")
        self._tasks = list(tasks)
        self._idx = 0

    def next(self) -> Iterable[Any] | None:
        if self._idx >= len(self._tasks):
            return None
        task = self._tasks[self._idx]
        self._idx += 1
        return task

    def __len__(self) -> int:
        return len(self._tasks)


class TokenBudgetTaskStream(TaskStream):
    """Wraps a single iterable, hands out chunks of approximately ``tokens_per_task``.

    The chunking is approximate — we don't slice mid-batch. A "task" is
    "the next batch sequence until cumulative-tokens >= tokens_per_task."
    The actual count happens in ``fine_tune_model`` via
    ``ctx['tokens_seen_in_task']``; this stream just produces a fresh
    sub-iterator from the same underlying source on each ``next()``.

    Termination: the underlying source must be finite or the caller must
    bound by ``n_tasks``. An infinite source + no n_tasks cap will run
    forever (the orchestrator's transition budget catches this).
    """

    def __init__(self, source: Iterable[Any], tokens_per_task: int) -> None:
        if tokens_per_task <= 0:
            raise ValueError(
                f"tokens_per_task must be > 0, got {tokens_per_task}"
            )
        self._source_iter: Iterator[Any] = iter(source)
        self._tokens_per_task = tokens_per_task
        self._exhausted = False

    def next(self) -> Iterable[Any] | None:
        if self._exhausted:
            return None
        # Yield a generator that pulls from the shared source iterator.
        # When the source raises StopIteration mid-task, mark exhausted.
        return self._budget_chunk()

    def _budget_chunk(self) -> Iterator[Any]:
        # The fine-tune action is responsible for stopping based on
        # ctx['tokens_seen_in_task']. This generator just exposes the
        # underlying source until it runs out.
        for batch in self._source_iter:
            yield batch
        self._exhausted = True

    def __len__(self) -> int:
        return -1  # unknown until exhausted


class LossDriftTaskStream(TaskStream):
    """Wraps a single iterable, hands out a chunk per ``next()`` call.

    Behaviorally identical to ``TokenBudgetTaskStream`` from the FSM's
    perspective — the difference is in *which* ctx flag the
    ``evaluate_task_advance`` action checks. We keep them as separate
    classes for clarity and to leave room for trigger-specific behavior
    later (e.g. snapshot the source position when drift is detected).
    """

    def __init__(self, source: Iterable[Any]) -> None:
        self._source_iter: Iterator[Any] = iter(source)
        self._exhausted = False

    def next(self) -> Iterable[Any] | None:
        if self._exhausted:
            return None
        return self._chunk()

    def _chunk(self) -> Iterator[Any]:
        for batch in self._source_iter:
            yield batch
        self._exhausted = True

    def __len__(self) -> int:
        return -1


# --- Process-local registry ---------------------------------------------------

_REGISTRY: dict[str, TaskStream] = {}


def register(stream: TaskStream) -> str:
    """Register a stream, return a process-local handle.

    The handle is a UUID4 string; callers store it in ctx and look up
    the live stream via ``get(handle)`` in actions.
    """
    handle = uuid.uuid4().hex
    _REGISTRY[handle] = stream
    return handle


def get(handle: str) -> TaskStream:
    """Retrieve a stream by handle. Raises KeyError if unknown."""
    if handle not in _REGISTRY:
        raise KeyError(
            f"task stream handle {handle!r} not registered. The handle "
            "must be created in the same Python process that runs the FSM."
        )
    return _REGISTRY[handle]


def deregister(handle: str) -> None:
    """Remove a handle from the registry (call from ForgePipeline teardown)."""
    _REGISTRY.pop(handle, None)
