"""Tests for saeforge.training.task_stream — three TaskStream variants + registry."""

from __future__ import annotations

import pytest

from saeforge.training import (
    LabeledTaskStream,
    LossDriftTaskStream,
    TokenBudgetTaskStream,
)
from saeforge.training import task_stream as ts_module


def test_labeled_task_stream_yields_in_order():
    s = LabeledTaskStream(["task_a", "task_b", "task_c"])
    assert s.next() == "task_a"
    assert s.next() == "task_b"
    assert s.next() == "task_c"
    assert s.next() is None
    assert len(s) == 3


def test_labeled_task_stream_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        LabeledTaskStream([])


def test_token_budget_task_stream_chunks_share_source():
    """Each next() returns a generator that pulls from the shared underlying iterator."""
    source = iter(range(10))
    s = TokenBudgetTaskStream(source, tokens_per_task=4)
    chunk1 = s.next()
    # Pull 4 items from chunk1
    pulled = [next(chunk1), next(chunk1), next(chunk1), next(chunk1)]
    assert pulled == [0, 1, 2, 3]
    chunk2 = s.next()
    assert next(chunk2) == 4  # continues from shared iterator


def test_token_budget_rejects_invalid_chunk():
    with pytest.raises(ValueError, match="tokens_per_task"):
        TokenBudgetTaskStream(iter([]), tokens_per_task=0)


def test_loss_drift_task_stream_basic_behavior():
    s = LossDriftTaskStream(iter(["a", "b", "c"]))
    chunk = s.next()
    assert next(chunk) == "a"
    assert next(chunk) == "b"


def test_registry_round_trip():
    s = LabeledTaskStream(["t0", "t1"])
    handle = ts_module.register(s)
    assert isinstance(handle, str)
    assert ts_module.get(handle) is s
    ts_module.deregister(handle)
    with pytest.raises(KeyError):
        ts_module.get(handle)


def test_registry_unknown_handle_raises_keyerror():
    with pytest.raises(KeyError, match="not registered"):
        ts_module.get("not-a-real-handle")
