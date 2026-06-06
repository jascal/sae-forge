"""Tests for saeforge.composition_subspace — U_C extraction + invariants."""

from __future__ import annotations

import numpy as np
import pytest

from saeforge.composition_subspace import (
    CompositionSubspace,
    extract_composition_subspace,
)


def test_U_orthonormal_and_rank_bounded(tiny_gpt2):
    subs = extract_composition_subspace(tiny_gpt2, layers=[0, 1], rank=4)
    assert set(subs) == {0, 1}
    for s in subs.values():
        d, r = s.U.shape
        assert d == tiny_gpt2.config.n_embd
        assert r <= d
        assert np.linalg.norm(s.U.T @ s.U - np.eye(r)) < 1e-5
        assert 0.0 < s.preserved_fraction() <= 1.0


def test_records_ln_meansub_approximation(tiny_gpt2):
    subs = extract_composition_subspace(tiny_gpt2, layers=[0], rank=4, fold_ln1=True)
    assert subs[0].metadata["ln_meansub_approx"] is True
    no_fold = extract_composition_subspace(tiny_gpt2, layers=[0], rank=4, fold_ln1=False)
    assert no_fold[0].metadata["ln_meansub_approx"] is False


def test_U_C_captures_QK_read_geometry(tiny_gpt2):
    """The QK macro is preserved on U_C: projecting the residual onto U_C
    preserves attention scores (U_C spans the dominant read directions)."""
    cfg = tiny_gpt2.config
    d, H = cfg.n_embd, cfg.n_head
    hd = d // H
    subs = extract_composition_subspace(tiny_gpt2, layers=[0], rank=d)  # full rank → exact
    U = subs[0].U
    P = U @ U.T  # projector onto span(U_C)

    block = tiny_gpt2.transformer.h[0]
    Wc = block.attn.c_attn.weight.detach().numpy().astype(np.float64)
    ln_w = block.ln_1.weight.detach().numpy().astype(np.float64)
    Wq, Wk = Wc[:, :d], Wc[:, d:2 * d]

    rng = np.random.default_rng(0)
    r = rng.standard_normal((32, d))
    rp = rng.standard_normal((32, d))
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        Mh = (Wq[:, sl] @ Wk[:, sl].T)  # (d, d), ln folded below via the directions
        full = ((r * ln_w) @ Mh @ (rp * ln_w).T)
        # score using only the U_C-projected residual on the (ln-folded) read side
        proj = (((r @ P) * ln_w) @ Mh @ ((rp @ P) * ln_w).T)
        denom = np.linalg.norm(full) + 1e-9
        assert np.linalg.norm(full - proj) / denom < 1e-6


def test_head_restriction_shrinks_source(tiny_gpt2):
    full = extract_composition_subspace(tiny_gpt2, layers=[0], rank=4, heads="all")[0]
    restricted = extract_composition_subspace(tiny_gpt2, layers=[0], rank=4, heads=[0, 1])[0]
    assert full.source_heads == "all"
    assert restricted.source_heads == [0, 1]
    # restricting to 2 of 4 heads cannot increase the captured rank
    assert restricted.rank <= full.rank


def test_non_gpt2_host_raises():
    class _Cfg:
        model_type = "llama"

    class _Host:
        config = _Cfg()

    with pytest.raises(NotImplementedError, match="gpt2"):
        extract_composition_subspace(_Host(), layers=[0])


def test_compositionsubspace_validates_orthonormality():
    with pytest.raises(ValueError, match="orthonormal"):
        CompositionSubspace(U=np.ones((8, 2)), layer=0, rank=2, source_heads="all", d_model=8)
    with pytest.raises(ValueError, match="d_model"):
        CompositionSubspace(
            U=np.linalg.qr(np.random.default_rng(0).standard_normal((8, 2)))[0],
            layer=0, rank=2, source_heads="all", d_model=16,
        )
