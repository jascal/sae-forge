"""Corpus iterator — local files first, HF datasets second, pre-tokenized passthrough.

Local-corpus paths trigger zero network calls (verified by the
test_local_only test). HF dataset names lazy-import `datasets`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_LOCAL_TEXT_SUFFIXES = (".txt", ".jsonl")


def build_iterator(
    source: Any,
    tokenizer: Any,
    batch_size: int,
    sequence_length: int,
) -> Iterator:
    """Construct a token-batch iterator from any of:

    1. Local ``.txt`` file (one document per line)
    2. Local ``.jsonl`` file (each line a JSON object with a ``"text"`` field)
    3. Local directory of ``.txt`` / ``.jsonl`` files (read recursively)
    4. HuggingFace dataset name (e.g. ``"HuggingFaceFW/fineweb-edu"``;
       lazy-imports ``datasets``)
    5. Pre-tokenized iterable yielding ``(batch_size, sequence_length)`` tensors

    The iterator yields int64 token-id tensors of shape
    ``(batch_size, sequence_length)``. Local-source paths cause no network
    activity beyond what the tokenizer itself may have already cached.
    """
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.exists():
            return _build_local_iterator(source_path, tokenizer, batch_size, sequence_length)
        # String that isn't a local path → assume HF dataset name
        return _build_hf_iterator(str(source), tokenizer, batch_size, sequence_length)
    if hasattr(source, "__iter__"):
        return iter(source)
    raise TypeError(
        f"build_iterator: source must be str/Path or iterable; got {type(source).__name__}"
    )


def _iter_local_files(source_path: Path) -> Iterator[Path]:
    if source_path.is_dir():
        for child in sorted(source_path.rglob("*")):
            if child.is_file() and child.suffix in _LOCAL_TEXT_SUFFIXES:
                yield child
    elif source_path.is_file():
        yield source_path
    else:
        raise FileNotFoundError(f"corpus source not found: {source_path}")


def _iter_documents(source_path: Path) -> Iterator[str]:
    for path in _iter_local_files(source_path):
        with path.open("r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    text = payload.get("text") if isinstance(payload, dict) else None
                    if text:
                        yield text
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def _build_local_iterator(
    source_path: Path,
    tokenizer: Any,
    batch_size: int,
    sequence_length: int,
) -> Iterator:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def gen():
        while True:
            batch_ids: list = []
            for text in _iter_documents(source_path):
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=sequence_length,
                )
                batch_ids.append(enc["input_ids"][0])
                if len(batch_ids) == batch_size:
                    yield torch.stack(batch_ids, dim=0).long()
                    batch_ids = []
            if not batch_ids:
                # corpus exhausted with nothing pending — restart
                continue
            # flush any partial trailing batch by repeating the last element to fill
            while len(batch_ids) < batch_size:
                batch_ids.append(batch_ids[-1])
            yield torch.stack(batch_ids, dim=0).long()

    return gen()


def _build_hf_iterator(
    name: str,
    tokenizer: Any,
    batch_size: int,
    sequence_length: int,
) -> Iterator:
    import torch

    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ImportError(
            "build_iterator with a HuggingFace dataset name needs the [recipe] extra; "
            "install it with `pip install sae-forge[recipe]`, "
            "or pass a local file path instead."
        ) from e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = hf_datasets.load_dataset(name, streaming=True, split="train")

    def gen():
        while True:
            batch_ids: list = []
            for example in ds:
                text = example.get("text") if isinstance(example, dict) else None
                if not text:
                    continue
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=sequence_length,
                )
                batch_ids.append(enc["input_ids"][0])
                if len(batch_ids) == batch_size:
                    yield torch.stack(batch_ids, dim=0).long()
                    batch_ids = []

    return gen()


def take(iterator: Iterable, n: int) -> Iterator:
    """Yield the first ``n`` items of ``iterator``."""
    for i, item in enumerate(iterator):
        if i >= n:
            break
        yield item
