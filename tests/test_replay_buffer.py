"""Tests for saeforge.training.replay — ReplayBuffer + MixedIterator."""

from __future__ import annotations

import pytest

from saeforge.training import MixedIterator, ReplayBuffer


def test_buffer_zero_size_is_inert():
    buf = ReplayBuffer(size=0)
    buf.add("seq1")
    assert len(buf) == 0
    assert buf.is_empty()
    assert buf.sample(5) == []


def test_reservoir_caps_at_size():
    buf = ReplayBuffer(size=4, policy="reservoir", seed=0)
    for i in range(20):
        buf.add(f"seq{i}")
    assert len(buf) == 4


def test_recent_window_is_fifo():
    buf = ReplayBuffer(size=3, policy="recent_window")
    for i in range(5):
        buf.add(f"seq{i}")
    samples = buf.sample(20)
    held = set(samples)
    assert held == {"seq2", "seq3", "seq4"}


def test_per_task_requires_configure():
    buf = ReplayBuffer(size=6, policy="per_task")
    with pytest.raises(RuntimeError, match="configure_per_task"):
        buf.add("seq1", task_id=0)


def test_per_task_stratifies_capacity():
    buf = ReplayBuffer(size=6, policy="per_task")
    buf.configure_per_task(n_tasks=3)
    for tid in range(3):
        for i in range(10):
            buf.add(f"task{tid}_seq{i}", task_id=tid)
    # 6 / 3 = 2 slots per task
    assert len(buf) == 6
    samples = buf.sample(60)
    counts = {tid: 0 for tid in range(3)}
    for s in samples:
        for tid in range(3):
            if s.startswith(f"task{tid}_"):
                counts[tid] += 1
                break
    # Each task should be represented (sample-with-replacement, so just check >0).
    assert all(c > 0 for c in counts.values()), counts


def test_per_task_requires_task_id():
    buf = ReplayBuffer(size=4, policy="per_task")
    buf.configure_per_task(n_tasks=2)
    with pytest.raises(ValueError, match="task_id"):
        buf.add("seq")


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown replay policy"):
        ReplayBuffer(size=10, policy="not_a_policy")


def test_negative_size_raises():
    with pytest.raises(ValueError, match=">= 0"):
        ReplayBuffer(size=-1)


def test_mixed_iterator_replay_ratio_zero_yields_only_primary():
    primary = iter(["p0", "p1", "p2", "p3", "p4"])
    buf = ReplayBuffer(size=4, policy="recent_window")
    buf.add("R0")
    mix = MixedIterator(primary, buf, replay_ratio=0.0)
    out = list(mix)
    assert out == ["p0", "p1", "p2", "p3", "p4"]
    assert mix.n_primary_yielded == 5
    assert mix.n_replay_yielded == 0


def test_mixed_iterator_with_replay_emits_both_streams():
    """At replay_ratio=0.5 over 100 calls, exactly 50 come from each source.

    The fine-tune loop, not the iterator, decides total iterations — so we
    explicitly bound the test to 100 calls rather than calling list().
    """
    primary = iter([f"p{i}" for i in range(200)])
    buf = ReplayBuffer(size=4, policy="recent_window")
    for i in range(4):
        buf.add(f"R{i}")
    mix = MixedIterator(primary, buf, replay_ratio=0.5)
    out = [next(mix) for _ in range(100)]
    primary_count = sum(1 for x in out if x.startswith("p"))
    replay_count = sum(1 for x in out if x.startswith("R"))
    assert primary_count == 50, primary_count
    assert replay_count == 50, replay_count
    assert mix.n_primary_yielded == 50
    assert mix.n_replay_yielded == 50


def test_mixed_iterator_empty_buffer_falls_back_to_primary():
    """When the buffer is empty (e.g. task 0), every step yields from primary."""
    primary = iter([f"p{i}" for i in range(10)])
    buf = ReplayBuffer(size=4)
    mix = MixedIterator(primary, buf, replay_ratio=0.5)
    out = list(mix)
    assert out == [f"p{i}" for i in range(10)]
    assert mix.n_replay_yielded == 0


def test_mixed_iterator_invalid_ratio_raises():
    with pytest.raises(ValueError, match="replay_ratio"):
        MixedIterator(iter([]), ReplayBuffer(size=1), replay_ratio=1.5)
