"""Tests for add-causal-host-capability-sweep: SAELens SAE loading + causal mid-layer extraction.

No real model downloads — the causal-extraction tests use a tiny fake forged module + a monkeypatched
tokenizer, so they run in CI without network. ESM-path equivalence is covered by the existing
test_sweep_pareto_capability.py suite (which stays green).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from saeforge.sweep_capability import (  # noqa: E402
    _extract_forged_activations,
    _load_encoding_state,
)


# --------------------------------------------------------------------------- SAELens loading


def test_load_encoding_state_accepts_saelens_and_reference(tmp_path: Path):
    """`W_dec` (SAELens, (F,d)) and `decoder.weight` (reference, (d,F)) for the same dictionary yield the
    same (F, d) W_dec_full and identical row norms."""
    rng = np.random.default_rng(0)
    F, d = 12, 5
    W_fd = rng.standard_normal((F, d)).astype(np.float32)  # (n_features, d_model)

    ref_path = tmp_path / "ref.pt"
    torch.save({"decoder.weight": torch.from_numpy(W_fd.T.copy())}, ref_path)  # (d, F)
    sl_path = tmp_path / "saelens.pt"
    torch.save({"W_dec": torch.from_numpy(W_fd.copy())}, sl_path)  # (F, d)

    ref = _load_encoding_state("ref", ref_path)
    sl = _load_encoding_state("sl", sl_path)
    assert ref.W_dec_full.shape == (F, d)
    assert sl.W_dec_full.shape == (F, d)
    np.testing.assert_allclose(ref.W_dec_full, sl.W_dec_full, rtol=1e-6)
    np.testing.assert_allclose(ref.row_norms, sl.row_norms, rtol=1e-6)


def test_load_encoding_state_rejects_unknown_keys(tmp_path: Path):
    p = tmp_path / "bad.pt"
    torch.save({"not_a_decoder": torch.zeros(3, 3)}, p)
    with pytest.raises(ValueError, match="neither 'decoder.weight'"):
        _load_encoding_state("bad", p)


# --------------------------------------------------------------------------- causal forged extraction


class _Block(torch.nn.Module):
    def forward(self, x):  # identity body — the pre-hook captures this block's INPUT
        return x


class _Tx(torch.nn.Module):
    def __init__(self, n_layers: int, N: int):
        super().__init__()
        self.wte = torch.nn.Embedding(64, N)
        self.h = torch.nn.ModuleList([_Block() for _ in range(n_layers)])

    def forward(self, ids):
        x = self.wte(ids)
        for b in self.h:
            x = b(x)
        return x


class _ForgedGPT2(torch.nn.Module):
    def __init__(self, N: int, n_layers: int, family: str = "gpt2"):
        super().__init__()
        self.config = SimpleNamespace(family=family)
        self.transformer = _Tx(n_layers, N)
        self.lm_head = torch.nn.Linear(N, 64)

    def forward(self, ids):
        return self.lm_head(self.transformer(ids))


class _FakeEnc(dict):
    def to(self, _device):
        return self


class _FakeTok:
    def __call__(self, text, **_kw):
        n = max(3, min(len(str(text).split()), 6))
        return _FakeEnc(input_ids=torch.arange(1, n + 1).unsqueeze(0))


@pytest.fixture
def _patch_tokenizer(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTok())


def test_forged_causal_hook_captures_basis_resid(_patch_tokenizer):
    """The forward-pre-hook on transformer.h[layer] captures the (tokens, N) basis-space resid_pre; the
    forged module still returns logits, so without the hook there'd be no hidden state to read."""
    N, n_layers = 7, 4
    forged = _ForgedGPT2(N, n_layers)
    host = SimpleNamespace(config=SimpleNamespace(_name_or_path="gpt2"))
    seqs = ["alpha beta gamma delta", "one two three"]
    out = _extract_forged_activations(
        forged, host, seqs, device="cpu", aggregator="pool_then_encode",
        max_seq_len=128, feed="residue", host_layer=2,
    )
    # residue feed → one row per token, concatenated across sequences; width is the basis width N
    # (basis space, NOT d_model — the forged module runs native_in_basis).
    assert out.shape == (4 + 3, N)  # token counts from _FakeTok
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_forged_causal_pooled_feed(_patch_tokenizer):
    forged = _ForgedGPT2(7, 4)
    host = SimpleNamespace(config=SimpleNamespace(_name_or_path="gpt2"))
    out = _extract_forged_activations(
        forged, host, ["alpha beta gamma delta", "one two three"], device="cpu",
        aggregator="pool_then_encode", max_seq_len=128, feed="pooled", host_layer=2,
    )
    assert out.shape == (2, 7)  # one pooled row per sequence


def test_forged_causal_unsupported_family_raises(_patch_tokenizer):
    forged = _ForgedGPT2(7, 4, family="llama")
    host = SimpleNamespace(config=SimpleNamespace(_name_or_path="gpt2"))
    with pytest.raises(NotImplementedError, match="llama"):
        _extract_forged_activations(
            forged, host, ["alpha beta"], device="cpu", aggregator="pool_then_encode",
            max_seq_len=128, feed="residue", host_layer=2,
        )


def test_forged_causal_missing_tokenizer_id_raises(_patch_tokenizer):
    """The causal path requires the host tokenizer id and must NOT silently fall back to the ESM vocab."""
    forged = _ForgedGPT2(7, 4)
    host = SimpleNamespace(config=SimpleNamespace(_name_or_path=None))
    with pytest.raises(ValueError, match="host tokenizer id"):
        _extract_forged_activations(
            forged, host, ["alpha beta"], device="cpu", aggregator="pool_then_encode",
            max_seq_len=128, feed="residue", host_layer=2,
        )
