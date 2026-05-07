"""Tests for the cosine-with-warmup LR schedule."""

from __future__ import annotations

import math

import pytest

from saeforge.training import cosine_with_warmup


def test_warmup_starts_small():
    lr = cosine_with_warmup(0, total_steps=100, warmup_steps=10, peak_lr=1e-3)
    assert lr == pytest.approx(1e-4, rel=1e-6)


def test_warmup_peaks_at_warmup_step():
    lr = cosine_with_warmup(10, total_steps=100, warmup_steps=10, peak_lr=1e-3)
    assert lr == pytest.approx(1e-3, rel=1e-6)


def test_floors_at_min_lr_ratio():
    lr = cosine_with_warmup(
        100, total_steps=100, warmup_steps=10, peak_lr=1e-3, min_lr_ratio=0.1
    )
    assert lr == pytest.approx(1e-4, rel=1e-6)


def test_clamps_past_total_steps():
    lr = cosine_with_warmup(
        500, total_steps=100, warmup_steps=10, peak_lr=1e-3, min_lr_ratio=0.1
    )
    assert lr == pytest.approx(1e-4, rel=1e-6)
    assert math.isfinite(lr)


def test_monotone_non_increasing_after_warmup():
    last = float("inf")
    for step in range(10, 200):
        lr = cosine_with_warmup(step, total_steps=100, warmup_steps=10, peak_lr=1e-3)
        assert lr <= last + 1e-12
        last = lr


def test_monotone_non_decreasing_during_warmup():
    last = -float("inf")
    for step in range(0, 11):
        lr = cosine_with_warmup(step, total_steps=100, warmup_steps=10, peak_lr=1e-3)
        assert lr >= last - 1e-12
        last = lr


def test_zero_warmup_returns_peak_at_step_zero():
    lr = cosine_with_warmup(0, total_steps=100, warmup_steps=0, peak_lr=1e-3)
    assert lr == pytest.approx(1e-3, rel=1e-6)
