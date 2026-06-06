"""Tests for the writer-output U_C — extract_writer_subspace + writer-output dispatch (task 2.3)."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.composition_subspace import (
    extract_composition_subspace,
    extract_writer_subspace,
)


def _head_ov(host, L, h):
    """The host head's OV map W_V^h W_O^h as (d, d) — the matrix whose row space U_C must keep."""
    cfg = host.config
    d, H = cfg.n_embd, cfg.n_head
    hd = d // H
    block = host.transformer.h[L]
    Wc = block.attn.c_attn.weight.detach().numpy().astype(np.float64)
    Wo = block.attn.c_proj.weight.detach().numpy().astype(np.float64)
    sl = slice(h * hd, (h + 1) * hd)
    return Wc[:, 2 * d:3 * d][:, sl] @ Wo[sl, :]


def test_writer_U_orthonormal_and_rank_bounded(tiny_gpt2):
    cs = extract_writer_subspace(tiny_gpt2, writer_heads=[(0, 1), (1, 2)], rank=4)
    d, r = cs.U.shape
    assert d == tiny_gpt2.config.n_embd
    assert r <= 4
    assert np.linalg.norm(cs.U.T @ cs.U - np.eye(r)) < 1e-5
    assert cs.metadata["mode"] == "writer-output"


def test_writer_output_preserved_invariant(tiny_gpt2):
    """Each writer head's OV output projected onto span(U_C) reproduces it (the preserve contract)."""
    writers = [(0, 1), (1, 2)]
    hd = tiny_gpt2.config.n_embd // tiny_gpt2.config.n_head
    cs = extract_writer_subspace(tiny_gpt2, writer_heads=writers, rank=2 * hd)  # captures both row spaces
    P = cs.U @ cs.U.T
    for (L, h) in writers:
        ov = _head_ov(tiny_gpt2, L, h)
        assert np.linalg.norm(ov @ P - ov) / (np.linalg.norm(ov) + 1e-12) < 1e-6


def test_writer_rank_cap_respected(tiny_gpt2):
    cs = extract_writer_subspace(tiny_gpt2, writer_heads=[(0, 0)], rank=2)
    assert cs.U.shape[1] == 2
    assert cs.source_heads == [[0, 0]]


def test_writer_source_heads_recorded(tiny_gpt2):
    cs = extract_writer_subspace(tiny_gpt2, writer_heads=[(0, 1), (1, 3)], rank=3)
    assert cs.source_heads == [[0, 1], [1, 3]]
    assert cs.metadata["writer_heads"] == [[0, 1], [1, 3]]


def test_writer_output_dispatch_replicates_per_layer(tiny_gpt2):
    """mode='writer-output' returns the SAME writer subspace at each requested layer."""
    subs = extract_composition_subspace(
        tiny_gpt2, layers=[0, 1], heads=[(0, 1), (1, 2)], rank=4, mode="writer-output"
    )
    assert set(subs) == {0, 1}
    assert np.allclose(subs[0].U, subs[1].U)
    assert subs[0].metadata["mode"] == "writer-output"


def test_writer_output_mode_rejects_non_tuple_heads(tiny_gpt2):
    with pytest.raises(ValueError, match="writer-output"):
        extract_composition_subspace(tiny_gpt2, layers=[0], heads="all", mode="writer-output")
    with pytest.raises(ValueError, match="writer-output"):
        extract_composition_subspace(tiny_gpt2, layers=[0], heads=[0, 1], mode="writer-output")


def test_extract_composition_rejects_unknown_mode(tiny_gpt2):
    with pytest.raises(ValueError, match="mode must be"):
        extract_composition_subspace(tiny_gpt2, layers=[0], heads=[(0, 1)], mode="bogus")


def test_writer_subspace_non_gpt2_raises():
    class _Cfg:
        model_type = "llama"

    class _Host:
        config = _Cfg()

    with pytest.raises(NotImplementedError, match="gpt2"):
        extract_writer_subspace(_Host(), writer_heads=[(0, 0)], rank=2)
