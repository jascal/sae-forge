"""FeatureBasis — load a Polygram-compressed SAE checkpoint into a feature basis.

Also hosts ``RegrowController``: a deterministic pure-Python controller
that computes the per-cycle ``effective_regrow_count`` for the
adaptive-regrow path in ``BasisMachine``. The controller is colocated
with ``FeatureBasis`` because both operate on the *basis-shape* concept
(number of kept features) — keeping them in one file matches the
existing single-file convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_CANDIDATE_W_DEC_KEYS = ("W_dec", "decoder.weight", "dec")
_ZERO_ROW_THRESHOLD = 1e-12
_CANDIDATE_REPORT_SUFFIXES = (
    "_compression_report.json",
    ".compression_report.json",
    "_report.json",
)


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
        ``polygram compress`` / ``polygram compress-epoch``. The companion
        report is auto-located by trying suffixes (``_compression_report.json``,
        ``.compression_report.json``, ``_report.json``) on the checkpoint's
        stem; pass ``report_path`` explicitly when filenames diverge.
        """
        checkpoint_path = Path(checkpoint_path)
        if report_path is None:
            report_path = _locate_report(checkpoint_path)
        else:
            report_path = Path(report_path)

        report = _load_report(report_path) if report_path is not None else {}
        W_dec_full = _load_w_dec(checkpoint_path)

        # kept_ids = rows with nonzero norm. Polygram's zero strategy literally
        # zeros non-representative rows; merge strategy zeros them too (only
        # the representative gets rescaled). Detecting from W_dec is uniform
        # across Compressor and EpochCompressor outputs — the latter's
        # EpochReport doesn't carry per-cluster zeroed lists.
        n_total = W_dec_full.shape[0]
        row_norms_full = np.linalg.norm(W_dec_full, axis=1)
        kept_mask = row_norms_full > _ZERO_ROW_THRESHOLD
        # Belt-and-braces: also exclude any rows the report explicitly lists
        # as zeroed, in case a future strategy keeps them at small but
        # nonzero norm.
        zeroed_ids = _collect_zeroed_ids(report)
        if zeroed_ids:
            kept_mask[list(zeroed_ids)] = False
        kept_ids = np.flatnonzero(kept_mask).astype(np.int64)

        W_dec = np.ascontiguousarray(W_dec_full[kept_ids])
        row_norms = np.linalg.norm(W_dec, axis=1)
        merged_lookup = _collect_merged_norm_by_rep(report)
        merged_norms = np.array(
            [
                merged_lookup[int(fid)] if int(fid) in merged_lookup else row_norms[i]
                for i, fid in enumerate(kept_ids)
            ],
            dtype=np.float64,
        )

        return cls(
            kept_ids=kept_ids,
            W_dec=W_dec.astype(np.float64, copy=False),
            merged_norms=merged_norms,
            original_norms=row_norms.astype(np.float64, copy=False),
            scale_compression_ratio=float(report.get("scale_compression_ratio", 1.0)),
            metadata={
                "checkpoint_path": str(checkpoint_path),
                "report_path": str(report_path) if report_path is not None else None,
                "source_checkpoint": report.get("source_checkpoint"),
                "strategy": report.get("strategy"),
                "n_total_features": int(n_total),
                "n_features_kept": int(report.get("n_features_kept", kept_ids.shape[0])),
                "n_clusters": int(report.get("n_clusters", 0)),
            },
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


class RegrowController:
    """Deterministic, pure-function controller for adaptive regrow.

    Computes the per-cycle ``effective_regrow_count`` from a single
    polygram-side signal (``n_features_kept``) and the configured
    ``n_features_target`` / ``regrow_count`` / ``regrow_max`` /
    ``regrow_damping`` knobs. See ``openspec/changes/adaptive-regrow``
    for the rationale and the rejected alternatives (PID, ML-based).

    The controller is stateless: no instance state, no class state, no
    IO, no RNG. Two calls with the same arguments return the same int.
    """

    @staticmethod
    def next_count(
        n_features_kept: int,
        n_features_target: int,
        regrow_count: int,
        regrow_max: int,
        regrow_damping: float,
    ) -> int:
        """Return the per-cycle ``effective_regrow_count`` bounded by
        ``[regrow_count, regrow_max]``.

        Equation (linear, damped, bounded; integer-valued):

            gap     = max(0, n_features_target - n_features_kept)
            damped  = int(round(gap * regrow_damping))
            return    max(regrow_count, min(damped, regrow_max))

        The ``int(round(...))`` coercion is load-bearing: ``regrow_count``
        is a feature count (integer), so the controller's return type is
        ``int`` even when the inputs include ``regrow_damping`` as a
        float. Tiebreak follows Python's banker's rounding (e.g.,
        ``round(0.5) == 0``); in practice ``gap * regrow_damping`` is
        rarely a half-integer, so this is determinism-preserving rather
        than load-bearing semantically.

        When the basis already exceeds the target (``gap == 0``) the
        controller returns ``regrow_count`` — the v0.2 fallback. No
        growth pressure beyond the configured base. See
        ``tests/fsm/test_adaptive_regrow.py::TestController::test_target_reached_returns_regrow_count``.

        **First-cycle handling** (before any compression has run): the
        composed action ``adapt_and_regrow`` short-circuits to
        ``perform_regrowth`` when the controller has no signal yet
        (``current_feature_count`` is still 0 at the start of the
        basis loop). The first regrow uses ``regrow_count`` verbatim;
        the controller only kicks in once a compression pass has
        written ``current_feature_count`` to ctx. See
        ``tests/fsm/test_adaptive_regrow.py::TestComposedAction::test_cold_start_short_circuits_to_perform_regrowth``.
        """
        gap = max(0, int(n_features_target) - int(n_features_kept))
        damped = int(round(gap * float(regrow_damping)))
        return max(int(regrow_count), min(damped, int(regrow_max)))


def _locate_report(checkpoint_path: Path) -> Path | None:
    stem = checkpoint_path.with_suffix("")
    for suffix in _CANDIDATE_REPORT_SUFFIXES:
        candidate = Path(str(stem) + suffix)
        if candidate.is_file():
            return candidate
    return None


def _load_report(report_path: Path) -> dict:
    if not report_path.is_file():
        raise FileNotFoundError(f"compression report not found: {report_path}")
    return json.loads(report_path.read_text())


def _load_w_dec(checkpoint_path: Path) -> np.ndarray:
    from safetensors import safe_open

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    with safe_open(str(checkpoint_path), framework="numpy") as f:
        keys = list(f.keys())
        chosen: str | None = None
        for candidate in _CANDIDATE_W_DEC_KEYS:
            if candidate in keys:
                chosen = candidate
                break
        if chosen is None:
            raise KeyError(
                f"no decoder weight tensor in {checkpoint_path}; "
                f"tried {_CANDIDATE_W_DEC_KEYS}, found {keys}"
            )
        tensor = f.get_tensor(chosen)
        if chosen == "decoder.weight" and tensor.shape[0] != tensor.shape[1]:
            tensor = tensor.T
    return np.asarray(tensor)


def _collect_zeroed_ids(report: dict) -> set[int]:
    zeroed: set[int] = set()
    for cluster in report.get("clusters", []) or []:
        for fid in cluster.get("zeroed", []) or []:
            zeroed.add(int(fid))
    return zeroed


def _collect_merged_norm_by_rep(report: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for cluster in report.get("clusters", []) or []:
        rep = cluster.get("representative")
        merged = cluster.get("merged_norm")
        if rep is not None and merged is not None:
            out[int(rep)] = float(merged)
    return out
