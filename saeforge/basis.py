"""FeatureBasis — load a Polygram-compressed SAE checkpoint into a feature basis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FeatureBasis:
    """Surviving feature basis extracted from a Polygram-compressed SAE.

    Pure-numpy. The torch extra is not required to construct or inspect a
    basis — only to feed one to ``SubspaceProjector`` against a real host
    model.
    """

    kept_ids: np.ndarray
    W_dec: np.ndarray
    merged_norms: np.ndarray
    original_norms: np.ndarray
    scale_compression_ratio: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.W_dec.ndim != 2:
            raise ValueError(f"W_dec must be 2-D (n_kept, d_model); got shape {self.W_dec.shape}")
        n_kept = self.W_dec.shape[0]
        for name, arr in (
            ("kept_ids", self.kept_ids),
            ("merged_norms", self.merged_norms),
            ("original_norms", self.original_norms),
        ):
            if arr.shape[0] != n_kept:
                raise ValueError(
                    f"{name} length {arr.shape[0]} does not match W_dec rows {n_kept}"
                )
        self._pinv_cache: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return int(self.W_dec.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_dec.shape[1])

    def pseudoinverse(self) -> np.ndarray:
        """Return the cached Moore-Penrose pseudoinverse of ``W_dec``.

        Shape: ``(d_model, n_features)``. Right-multiplying a row in the
        host residual stream by this matrix encodes it into the basis.
        """
        if self._pinv_cache is None:
            self._pinv_cache = np.linalg.pinv(self.W_dec)
        return self._pinv_cache

    @classmethod
    def from_polygram_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        report_path: str | Path | None = None,
    ) -> FeatureBasis:
        """Load a Polygram-compressed checkpoint + companion compression report.

        ``checkpoint_path`` is the ``.safetensors`` file produced by
        ``polygram compress`` / ``polygram compress-epoch``. ``report_path``
        defaults to the same stem with a ``_compression_report.json`` suffix
        (Polygram's convention) — pass it explicitly if your filenames diverge.
        """
        raise NotImplementedError(
            "FeatureBasis.from_polygram_checkpoint is the change-2 deliverable; "
            "see openspec/changes/feature-basis/proposal.md."
        )

    def to_summary(self) -> dict:
        """Return a JSON-serializable summary for ``sae-forge inspect``."""
        return {
            "n_features": self.n_features,
            "d_model": self.d_model,
            "scale_compression_ratio": float(self.scale_compression_ratio),
            "merged_norm_mean": float(self.merged_norms.mean()) if self.n_features else 0.0,
            "merged_norm_std": float(self.merged_norms.std()) if self.n_features else 0.0,
            "original_norm_mean": float(self.original_norms.mean()) if self.n_features else 0.0,
            "kept_id_count": int(self.kept_ids.shape[0]),
        }

    def save_summary(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_summary(), indent=2))
