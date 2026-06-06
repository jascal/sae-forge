"""Tests for saeforge.circuit_heads — behavioral writer-head detection (task 1.4).

The detector runs a real forward pass and reads attention. To test the SCORING + RANKING
deterministically (independent of a random model's weights) we drive it with a fake host whose
attention pattern we control: one head is a perfect Δ=1 (previous-token) mover, the rest are
diagonal. A separate smoke test confirms it runs on the real tiny GPT-2 fixture and returns
well-formed triples.
"""

from __future__ import annotations

import pytest

from saeforge import circuit_heads


class _Cfg:
    model_type = "gpt2"

    def __init__(self, n_layer, n_head):
        self.n_layer = n_layer
        self.n_head = n_head


class _AttnOut:
    def __init__(self, attentions):
        self.attentions = attentions


class _FakeTransformer:
    """Returns controlled attention: head ``mover`` is a perfect prev-token (offset -1) head."""

    def __init__(self, n_layer, n_head, mover, dup=None):
        self.n_layer, self.n_head, self.mover, self.dup = n_layer, n_head, mover, dup

    def __call__(self, input_ids, output_attentions=True):
        import torch

        Lc = input_ids.shape[1]
        atts = []
        for L in range(self.n_layer):
            a = torch.zeros(1, self.n_head, Lc, Lc)
            for h in range(self.n_head):  # default: attend to self
                a[0, h] = torch.eye(Lc)
            if self.mover is not None and L == self.mover[0]:
                h = self.mover[1]
                a[0, h].zero_()
                a[0, h, 0, 0] = 1.0
                for i in range(1, Lc):
                    a[0, h, i, i - 1] = 1.0  # previous token
            if self.dup is not None and L == self.dup[0]:
                h = self.dup[1]
                a[0, h].zero_()
                # query at pos 2 attends to pos 0 (the earlier same token, see corpus below)
                a[0, h, 2, 0] = 1.0
            atts.append(a)
        return _AttnOut(atts)


class _FakeHost:
    def __init__(self, n_layer, n_head, mover, dup=None):
        self.config = _Cfg(n_layer, n_head)
        self.transformer = _FakeTransformer(n_layer, n_head, mover, dup)


def test_prev_token_head_detected_first():
    pytest.importorskip("torch")
    host = _FakeHost(n_layer=2, n_head=4, mover=(1, 2))
    corpus = list(range(20))
    heads = circuit_heads.prev_token_heads(host, corpus, top_k=4, ctx=16)
    assert heads, "expected at least the planted prev-token head"
    L, h, score = heads[0]
    assert (L, h) == (1, 2)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_min_attention_threshold_filters_weak_heads():
    pytest.importorskip("torch")
    host = _FakeHost(n_layer=2, n_head=4, mover=(1, 2))
    # only the planted head has Δ=1 mass ~1.0; the diagonal heads score 0 → filtered out
    heads = circuit_heads.prev_token_heads(host, list(range(20)), top_k=4, ctx=16, min_attention=0.5)
    assert [(L, h) for (L, h, _s) in heads] == [(1, 2)]


def test_duplicate_token_head_detected():
    pytest.importorskip("torch")
    host = _FakeHost(n_layer=1, n_head=3, mover=None, dup=(0, 1))
    # token at index 2 repeats index 0 → a duplicate-token query position exists
    corpus = [5, 7, 5, 9, 11, 13, 5, 17]
    heads = circuit_heads.duplicate_token_heads(host, corpus, top_k=3, ctx=16, min_attention=0.0)
    assert heads
    assert (heads[0][0], heads[0][1]) == (0, 1)


def test_identify_dispatch_and_unknown_preset():
    pytest.importorskip("torch")
    host = _FakeHost(n_layer=2, n_head=4, mover=(0, 3))
    got = circuit_heads.identify(host, list(range(20)), "prev-token", ctx=16)
    assert (got[0][0], got[0][1]) == (0, 3)
    with pytest.raises(ValueError, match="prev-token"):
        circuit_heads.identify(host, list(range(20)), "no-such-preset")


def test_non_gpt2_host_raises():
    pytest.importorskip("torch")

    class _LlamaCfg:
        model_type = "llama"
        n_layer = 2
        n_head = 4

    class _Host:
        config = _LlamaCfg()
        transformer = None

    with pytest.raises(NotImplementedError, match="gpt2"):
        circuit_heads.prev_token_heads(_Host(), list(range(10)))


def test_prev_token_heads_wellformed_on_real_model(tiny_gpt2):
    """Smoke: runs on the real fixture and returns ≤ top_k sorted (L, h, score) triples in range."""
    heads = circuit_heads.prev_token_heads(tiny_gpt2, list(range(40)), top_k=3, ctx=16, min_attention=0.0)
    assert len(heads) <= 3
    nL, H = tiny_gpt2.config.n_layer, tiny_gpt2.config.n_head
    scores = [s for (_L, _h, s) in heads]
    assert scores == sorted(scores, reverse=True)
    for (L, h, s) in heads:
        assert 0 <= L < nL and 0 <= h < H
        assert 0.0 <= s <= 1.0
