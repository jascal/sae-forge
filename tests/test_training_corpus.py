"""Tests for corpus iterator — local files, pre-tokenized passthrough."""

from __future__ import annotations

import json
import sys

import pytest


@pytest.fixture
def tiny_tokenizer():
    """A minimal tokenizer that maps chars to small int ids — enough for shape tests."""
    pytest.importorskip("torch")

    class TinyTok:
        pad_token = "[PAD]"
        eos_token = "[EOS]"

        def __call__(self, text, return_tensors=None, padding=None, truncation=None,
                     max_length=None):
            import torch
            ids = [ord(c) % 100 for c in text][: max_length or 16]
            if max_length:
                ids = ids + [0] * (max_length - len(ids))
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    return TinyTok()


def test_local_txt_iterator_yields_correct_shape(tmp_path, tiny_tokenizer):
    pytest.importorskip("torch")
    import torch

    from saeforge.training import build_iterator

    txt = tmp_path / "corpus.txt"
    txt.write_text("\n".join(f"document {i}" for i in range(10)))
    it = build_iterator(txt, tiny_tokenizer, batch_size=2, sequence_length=8)
    batches = []
    for _ in range(3):
        b = next(it)
        batches.append(b)
        assert isinstance(b, torch.Tensor)
        assert b.dtype == torch.long
        assert b.shape == (2, 8)


def test_local_jsonl_iterator(tmp_path, tiny_tokenizer):
    pytest.importorskip("torch")
    from saeforge.training import build_iterator

    jl = tmp_path / "corpus.jsonl"
    with jl.open("w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"document {i}"}) + "\n")
    it = build_iterator(jl, tiny_tokenizer, batch_size=2, sequence_length=8)
    b = next(it)
    assert b.shape == (2, 8)


def test_local_directory_iterator(tmp_path, tiny_tokenizer):
    pytest.importorskip("torch")
    from saeforge.training import build_iterator

    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    (tmp_path / "b.jsonl").write_text(json.dumps({"text": "gamma"}) + "\n")
    it = build_iterator(tmp_path, tiny_tokenizer, batch_size=2, sequence_length=8)
    b = next(it)
    assert b.shape == (2, 8)


def test_pretokenized_iterable_passthrough(tiny_tokenizer):
    pytest.importorskip("torch")
    import torch

    from saeforge.training import build_iterator

    pre = [torch.zeros((2, 8), dtype=torch.long) for _ in range(3)]
    it = build_iterator(pre, tiny_tokenizer, batch_size=2, sequence_length=8)
    out = list(it)
    assert len(out) == 3
    assert out[0].shape == (2, 8)


def test_local_path_does_not_import_datasets(tmp_path, tiny_tokenizer, monkeypatch):
    """Local-corpus iterator must not lazy-import `datasets`."""
    pytest.importorskip("torch")

    from saeforge.training import build_iterator

    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    txt = tmp_path / "corpus.txt"
    txt.write_text("a\nb\nc\nd\n")
    it = build_iterator(txt, tiny_tokenizer, batch_size=2, sequence_length=4)
    next(it)
    assert "datasets" not in sys.modules


def test_missing_local_path_raises(tmp_path, tiny_tokenizer):
    """Path that does not exist falls through to HF dataset loader, which raises
    if `datasets` isn't installed or the name doesn't match a real dataset.
    For pure local-only behaviour, use an existing path.
    """
    pytest.importorskip("torch")
    from saeforge.training import build_iterator

    bogus = tmp_path / "no-such-thing.txt"
    # When `datasets` is not installed, this should raise ImportError mentioning
    # the [recipe] extra. Otherwise it would try to load as HF dataset and fail there.
    with pytest.raises((ImportError, FileNotFoundError, Exception)):
        it = build_iterator(bogus, tiny_tokenizer, batch_size=2, sequence_length=4)
        next(it)


def test_unknown_source_type_raises(tiny_tokenizer):
    pytest.importorskip("torch")
    from saeforge.training import build_iterator

    with pytest.raises(TypeError, match="source"):
        build_iterator(42, tiny_tokenizer, batch_size=2, sequence_length=4)
