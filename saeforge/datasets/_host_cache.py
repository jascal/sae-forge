"""Host-extraction cache for capability sweeps.

Capability sweeps re-extract host activations on the same protein
subset for every cell in the (encoding × width × scale_boost) cube.
The host's outputs are invariant across cells, so caching them after
the first cell turns N cells of redundant host forward passes into
1 forward pass + (N-1) cache hits.

Cache key: ``(host_model_id, sequences_hash, aggregator, max_seq_len)``.

  - ``host_model_id`` — the HF id (or local path) the host was loaded from.
  - ``sequences_hash`` — SHA-256 of the newline-joined sequence list
    (deterministic across runs over the same dataset).
  - ``aggregator`` — string label or callable's ``__name__``. Same
    sequence list with different aggregator → different cached
    output, so they must not share a key.
  - ``max_seq_len`` — truncation length applied during extraction.

On-disk format: one ``.safetensors`` file per cache key under
``cache_dir``. Filename: ``host_<sha256[:16]>.safetensors``. A
companion ``host_<sha256[:16]>.meta.json`` carries the four cache-
key components so a stale-key mismatch surfaces with a clear error.

Invalidation: cache keys are content-addressed; changing any key
component yields a new on-disk file. Stale entries (key components
match but file is corrupt / readable but wrong shape) raise a clear
``RuntimeError`` instead of silently using the bad payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HostCacheKey:
    """Identifies a host-extraction cache entry."""

    host_model_id: str
    sequences_hash: str
    aggregator: str
    max_seq_len: int
    feed: str = "pooled"

    @classmethod
    def from_inputs(
        cls,
        host_model_id: str,
        sequences: list[str],
        aggregator: "str | Any",
        max_seq_len: int,
        feed: str = "pooled",
    ) -> "HostCacheKey":
        """Build a key from the raw inputs. Hashes ``sequences`` via
        SHA-256 of the newline-joined list (deterministic; surfaces
        any ordering or content drift).

        ``feed`` distinguishes pooled vs residue extraction — the same
        sequences under different feeds produce different cached
        tensors and MUST NOT share a key.
        """
        if isinstance(aggregator, str):
            agg_str = aggregator
        elif callable(aggregator):
            agg_str = getattr(aggregator, "__name__", repr(aggregator))
        else:
            raise TypeError(
                f"HostCacheKey: aggregator must be a string or callable; "
                f"got {type(aggregator).__name__!r}"
            )
        if feed not in ("pooled", "residue"):
            raise ValueError(
                f"HostCacheKey: feed must be 'pooled' or 'residue'; "
                f"got {feed!r}"
            )
        h = hashlib.sha256()
        # Length prefix per sequence so two different lists that
        # concatenate to the same bytes don't hash equal.
        for s in sequences:
            h.update(f"{len(s)}:".encode("utf-8"))
            h.update(s.encode("utf-8"))
            h.update(b"\n")
        return cls(
            host_model_id=str(host_model_id),
            sequences_hash=h.hexdigest(),
            aggregator=agg_str,
            max_seq_len=int(max_seq_len),
            feed=feed,
        )

    def digest(self) -> str:
        """Short content-address (first 16 hex chars) used in filenames."""
        h = hashlib.sha256()
        h.update(self.host_model_id.encode("utf-8"))
        h.update(b"|")
        h.update(self.sequences_hash.encode("utf-8"))
        h.update(b"|")
        h.update(self.aggregator.encode("utf-8"))
        h.update(b"|")
        h.update(str(self.max_seq_len).encode("utf-8"))
        h.update(b"|")
        h.update(self.feed.encode("utf-8"))
        return h.hexdigest()[:16]

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "host_model_id": self.host_model_id,
            "sequences_hash": self.sequences_hash,
            "aggregator": self.aggregator,
            "max_seq_len": self.max_seq_len,
            "feed": self.feed,
        }


class HostExtractionCache:
    """File-backed cache for host activations across sweep cells.

    Usage::

        cache = HostExtractionCache(cache_dir, enabled=True)
        key = HostCacheKey.from_inputs(host_id, sequences, agg, max_seq_len)
        if cache.has(key):
            host_X = cache.load(key)
        else:
            host_X = _extract_host(...)         # caller's extractor
            cache.save(key, host_X)
    """

    def __init__(self, cache_dir: "str | Path", *, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = bool(enabled)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: HostCacheKey) -> tuple[Path, Path]:
        stem = f"host_{key.digest()}"
        return (
            self.cache_dir / f"{stem}.safetensors",
            self.cache_dir / f"{stem}.meta.json",
        )

    def has(self, key: HostCacheKey) -> bool:
        if not self.enabled:
            return False
        st_path, meta_path = self._paths(key)
        return st_path.exists() and meta_path.exists()

    def load(self, key: HostCacheKey) -> Any:
        """Returns the cached tensor. Raises ``RuntimeError`` on any
        cache-key mismatch (defensive against hash collisions)."""
        from safetensors.torch import load_file

        st_path, meta_path = self._paths(key)
        meta_on_disk = json.loads(meta_path.read_text())
        if meta_on_disk != key.to_meta_dict():
            raise RuntimeError(
                f"HostExtractionCache: meta mismatch at {meta_path}. "
                f"Expected {key.to_meta_dict()!r}, got {meta_on_disk!r}. "
                f"Cache may be corrupt; delete and re-run."
            )
        tensors = load_file(str(st_path))
        if "host_activations" not in tensors:
            raise RuntimeError(
                f"HostExtractionCache: missing 'host_activations' key in "
                f"{st_path}; cache file corrupt — delete and re-run."
            )
        return tensors["host_activations"]

    def save(self, key: HostCacheKey, tensor: Any) -> None:
        if not self.enabled:
            return
        from safetensors.torch import save_file

        st_path, meta_path = self._paths(key)
        meta_path.write_text(json.dumps(key.to_meta_dict(), indent=2))
        save_file({"host_activations": tensor.contiguous()}, str(st_path))
