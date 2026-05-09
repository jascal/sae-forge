"""Replay buffer + mixed iterator for continual-learning fine-tune.

Three policies:
- ``reservoir``: classic reservoir sampling, uniform across all history.
- ``recent_window``: FIFO ring of the last N sequences.
- ``per_task``: stratified across ``task_idx``; requires task labels on add.

The buffer stores tokenized sequences (whatever the corpus iterator
yields — typically a tensor or array of input_ids). It does not assume
torch; ``MixedIterator`` round-robins between a primary iterator and a
sampler from the buffer at the configured ratio.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Iterable, Iterator


_VALID_POLICIES = ("reservoir", "recent_window", "per_task")


class ReplayBuffer:
    """Bounded buffer of past training sequences with three sampling policies.

    The buffer is sequence-level — one entry is whatever the upstream
    iterator yields per step (typically a batch tensor). Token-level
    reservoir is tracked as a future refinement (tasks.md §12.4).
    """

    def __init__(
        self,
        size: int,
        policy: str = "reservoir",
        seed: int | None = None,
    ) -> None:
        if size < 0:
            raise ValueError(f"ReplayBuffer size must be >= 0, got {size}")
        if policy not in _VALID_POLICIES:
            raise ValueError(
                f"unknown replay policy {policy!r}; must be one of {_VALID_POLICIES}"
            )
        self.size = size
        self.policy = policy
        self._rng = random.Random(seed)
        # Reservoir state
        self._items: list[Any] = []
        self._n_seen = 0
        # Recent-window state
        self._window: deque[Any] = deque(maxlen=size if size > 0 else None)
        # Per-task state: dict[task_id, list[seq]] with per-task capacity
        self._per_task: dict[int, list[Any]] = {}
        self._per_task_cap: int | None = None

    def configure_per_task(self, n_tasks: int) -> None:
        """Set the per-task capacity. Required before ``add`` for ``per_task``.

        Splits ``size`` evenly across ``n_tasks``. Floor-divides; any
        remainder is unused (the alternative — distributing it
        unevenly — would bias the first few tasks).
        """
        if self.policy != "per_task":
            return
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        self._per_task_cap = self.size // n_tasks

    def add(self, sequence: Any, task_id: int | None = None) -> None:
        """Insert one sequence into the buffer per the configured policy."""
        if self.size == 0:
            return

        if self.policy == "reservoir":
            self._add_reservoir(sequence)
        elif self.policy == "recent_window":
            self._window.append(sequence)
        else:  # per_task
            if task_id is None:
                raise ValueError(
                    "per_task replay requires task_id on add; got None"
                )
            if self._per_task_cap is None:
                raise RuntimeError(
                    "per_task replay buffer used before configure_per_task() was called"
                )
            slot = self._per_task.setdefault(task_id, [])
            if len(slot) < self._per_task_cap:
                slot.append(sequence)
            else:
                # Reservoir sampling within the per-task slot keeps representation
                # uniform across the task's data without unbounded growth.
                idx = self._rng.randrange(self._per_task_cap)
                slot[idx] = sequence

    def _add_reservoir(self, sequence: Any) -> None:
        self._n_seen += 1
        if len(self._items) < self.size:
            self._items.append(sequence)
        else:
            # Replace at random index `i` with probability size / n_seen.
            j = self._rng.randrange(self._n_seen)
            if j < self.size:
                self._items[j] = sequence

    def __len__(self) -> int:
        if self.policy == "reservoir":
            return len(self._items)
        if self.policy == "recent_window":
            return len(self._window)
        return sum(len(slot) for slot in self._per_task.values())

    def is_empty(self) -> bool:
        return len(self) == 0

    def sample(self, n: int) -> list[Any]:
        """Draw n items from the buffer with replacement.

        With-replacement keeps `MixedIterator` simple and is fine for
        replay (we reuse old data deliberately). For unbiased reservoir
        coverage the buffer itself already maintains uniformity; the
        sampler just draws from it.
        """
        if self.is_empty():
            return []
        if self.policy == "reservoir":
            pool: list[Any] = self._items
        elif self.policy == "recent_window":
            pool = list(self._window)
        else:
            pool = [seq for slot in self._per_task.values() for seq in slot]
        return [self._rng.choice(pool) for _ in range(n)]


class MixedIterator:
    """Round-robin iterator yielding from primary or replay per replay_ratio.

    The ratio is enforced as a deterministic schedule: every ``cycle``
    yields, exactly ``round(cycle * replay_ratio)`` come from the
    replay buffer. Cycle is hardcoded to 100 — fine-grained enough for
    any practical ratio and avoids floating-point drift across long runs.

    When the primary iterator is exhausted, the mixed iterator stops —
    replay tokens are *padding*, not a substitute for the primary
    stream. When the replay buffer is empty (e.g. on the very first
    task), every step yields from primary regardless of ratio.
    """

    _CYCLE = 100

    def __init__(
        self,
        primary: Iterable[Any],
        replay: ReplayBuffer,
        replay_ratio: float,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError(
                f"replay_ratio must be in [0, 1]; got {replay_ratio}"
            )
        self._primary_iter: Iterator[Any] = iter(primary)
        self._replay = replay
        self._replay_ratio = replay_ratio
        self._rng = random.Random(seed)
        self._n_replay_per_cycle = round(self._CYCLE * replay_ratio)
        self._cycle_step = 0
        self._replay_indices = self._build_replay_indices()
        # Counters surfaced for tests / debugging.
        self.n_primary_yielded = 0
        self.n_replay_yielded = 0

    def _build_replay_indices(self) -> set[int]:
        if self._n_replay_per_cycle == 0:
            return set()
        # Spread replay positions uniformly across the cycle so the
        # primary stream is not back-loaded after a burst of replays.
        idxs = set()
        for k in range(self._n_replay_per_cycle):
            idxs.add(int(k * self._CYCLE / self._n_replay_per_cycle))
        return idxs

    def __iter__(self) -> "MixedIterator":
        return self

    def __next__(self) -> Any:
        use_replay = (
            self._cycle_step in self._replay_indices and not self._replay.is_empty()
        )
        self._cycle_step = (self._cycle_step + 1) % self._CYCLE
        if use_replay:
            sample = self._replay.sample(1)[0]
            self.n_replay_yielded += 1
            return sample
        # Pull primary first; on StopIteration the counter does not advance.
        item = next(self._primary_iter)
        self.n_primary_yielded += 1
        return item
