"""Unit tests for `saeforge.forge_quality` — diagnostics module."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from saeforge.forge_quality import (
    QualityThresholds,
    QualityTier,
    advise_sweep_quality,
    basis_rank_from_safetensors,
    classify_quality,
    compute_basis_rank,
    resolve_host_d_model,
)


# ---------------------------------------------------------------------------
# compute_basis_rank
# ---------------------------------------------------------------------------


class TestComputeBasisRank:
    def test_full_rank_random_matrix(self):
        W = np.random.default_rng(0).standard_normal((8, 64))
        assert compute_basis_rank(W) == 8

    def test_linearly_dependent_rows(self):
        W = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        # Fourth row is 2× the first → rank 3, not 4.
        assert compute_basis_rank(W) == 3

    def test_zero_rows_raises(self):
        W = np.zeros((0, 16), dtype=np.float64)
        with pytest.raises(ValueError, match="0 rows"):
            compute_basis_rank(W)

    def test_all_zero_rows_returns_zero(self):
        """Edge case: rows present but all zero (catastrophic SAE state).
        Polygram's pipeline shouldn't normally produce this — at least one
        representative per cluster is retained — but the rank computation
        handles it gracefully and returns 0 (no span).
        """
        W = np.zeros((4, 16), dtype=np.float64)
        assert compute_basis_rank(W) == 0

    def test_single_nonzero_row_returns_one(self):
        """Edge case: minimum non-degenerate basis."""
        W = np.zeros((1, 16), dtype=np.float64)
        W[0, 5] = 1.0
        assert compute_basis_rank(W) == 1

    def test_single_zero_row_returns_zero(self):
        """Edge case: 1-row all-zero matrix."""
        W = np.zeros((1, 16), dtype=np.float64)
        assert compute_basis_rank(W) == 0

    def test_basis_rank_from_safetensors_all_zero_returns_zero(self, tmp_path):
        """Documented contract: when every row in W_dec is zero,
        basis_rank_from_safetensors returns 0 directly without calling
        compute_basis_rank (which would raise on the empty post-filter array).
        """
        from safetensors.numpy import save_file

        W = np.zeros((4, 16), dtype=np.float32)
        path = tmp_path / "all_zero.safetensors"
        save_file({"W_dec": W}, str(path))
        assert basis_rank_from_safetensors(path) == 0


# ---------------------------------------------------------------------------
# classify_quality
# ---------------------------------------------------------------------------


class TestClassifyQuality:
    @pytest.mark.parametrize(
        "rank,d_model,expected",
        [
            (768, 768, QualityTier.SATURATED),     # ratio = 1.0
            (1000, 768, QualityTier.SATURATED),    # ratio > 1
            (500, 768, QualityTier.GOOD),          # ratio ≈ 0.65
            (384, 768, QualityTier.GOOD),          # ratio = 0.5 (boundary, ≥)
            (200, 768, QualityTier.UNDERSIZED),    # ratio ≈ 0.26
            (48, 768, QualityTier.UNDERSIZED),     # ratio = 0.0625 (boundary, ≥)
            (40, 768, QualityTier.DEGENERATE),     # ratio < 0.0625
            (1, 768, QualityTier.DEGENERATE),
            (0, 768, QualityTier.DEGENERATE),
        ],
    )
    def test_default_thresholds(self, rank, d_model, expected):
        _, tier = classify_quality(rank, d_model)
        assert tier == expected

    def test_custom_thresholds_shift_boundaries(self):
        # 1.5 good / 0.25 undersized.
        thresholds = QualityThresholds(saturated=2.0, good=1.0, undersized=0.25)
        # ratio = 1.5 → "good" under defaults (would be saturated at 1.0).
        _, tier = classify_quality(1500, 1000, thresholds)
        assert tier == QualityTier.GOOD

    def test_rejects_zero_d_model(self):
        with pytest.raises(ValueError, match="host_d_model"):
            classify_quality(8, 0)


# ---------------------------------------------------------------------------
# QualityThresholds validation
# ---------------------------------------------------------------------------


class TestQualityThresholds:
    def test_defaults(self):
        t = QualityThresholds()
        assert t.saturated == 1.0
        assert t.good == 0.5
        assert t.undersized == 0.0625

    def test_rejects_inverted_ordering(self):
        with pytest.raises(ValueError, match="saturated > good > undersized"):
            QualityThresholds(saturated=0.5, good=1.0, undersized=0.25)

    def test_rejects_negative_undersized(self):
        with pytest.raises(ValueError, match="saturated > good > undersized"):
            QualityThresholds(saturated=1.0, good=0.5, undersized=-0.1)


# ---------------------------------------------------------------------------
# resolve_host_d_model
# ---------------------------------------------------------------------------


class TestResolveHostDModel:
    def test_returns_none_when_transformers_missing(self, monkeypatch):
        """If transformers isn't installed, resolve returns (None, None) and
        logs to stderr (not raise).
        """
        import builtins

        real_import = builtins.__import__

        def _no_transformers(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_transformers)
        d_model, model_type = resolve_host_d_model("gpt2")
        assert d_model is None
        assert model_type is None

    def test_returns_none_when_autoconfig_fails(self, monkeypatch):
        """A failure deeper than ImportError (network error, gated model)
        also returns (None, None) without raising.
        """
        pytest.importorskip("transformers")
        import transformers

        def _boom(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _boom)
        d_model, model_type = resolve_host_d_model("gpt2")
        assert d_model is None
        # model_type also None because we never got the config.
        assert model_type is None

    def test_gpt2_resolves_to_768(self):
        """Real gpt2 host resolves to 768. Gated on network availability; skips
        cleanly if transformers/HF cache miss.
        """
        pytest.importorskip("transformers")
        try:
            d_model, model_type = resolve_host_d_model("gpt2")
        except Exception:
            pytest.skip("transformers / HF cache unavailable")
        if d_model is None:
            pytest.skip("network/cache miss for gpt2 config")
        assert d_model == 768
        assert model_type == "gpt2"


# ---------------------------------------------------------------------------
# advise_sweep_quality
# ---------------------------------------------------------------------------


class TestAdviseSweepQuality:
    def test_silent_on_good_setup(self, tmp_path):
        """All encodings' smallest K is good or saturated → returns None."""
        enc_path = tmp_path / "mps"
        enc_path.mkdir()
        manifest = {
            8: mock.Mock(n_features_kept=500),
            16: mock.Mock(n_features_kept=600),
        }
        # Smallest K=8 has rank 500 against d_model=768 → ratio 0.65 (good).
        advisory = advise_sweep_quality(
            encodings=[("mps", enc_path)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 500,
        )
        assert advisory is None

    def test_warns_on_degenerate(self, tmp_path):
        """Smallest K is rank 1 against d_model 768 → advisory fires."""
        enc_path = tmp_path / "mps" / "pareto"
        enc_path.mkdir(parents=True)
        # Need a file at the expected location for the path check.
        (enc_path / "k_8.safetensors").write_text("")
        manifest = {
            8: mock.Mock(n_features_kept=1),
            16: mock.Mock(n_features_kept=2),
        }
        advisory = advise_sweep_quality(
            encodings=[("mps", enc_path.parent)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 1,
        )
        assert advisory is not None
        assert "mps" in advisory
        assert "K=8" in advisory
        assert "degenerate" in advisory.lower()
        # No K meets the good threshold (384) → routes back to polygram.
        assert "polygram" in advisory.lower()
        # Fixed-wording clarification sentence present.
        assert "exploratory low-rank" in advisory

    def test_suggests_k_floor_from_manifest(self, tmp_path):
        """When a K in the manifest meets the good threshold, the advisory
        names it as a suggested floor.
        """
        enc_path = tmp_path / "mps" / "pareto"
        enc_path.mkdir(parents=True)
        (enc_path / "k_8.safetensors").write_text("")
        manifest = {
            8: mock.Mock(n_features_kept=4),
            16: mock.Mock(n_features_kept=8),
            32: mock.Mock(n_features_kept=500),  # ≥ 384 (= 768 * 0.5)
            64: mock.Mock(n_features_kept=700),
        }
        advisory = advise_sweep_quality(
            encodings=[("mps", enc_path.parent)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 4,
        )
        assert advisory is not None
        # K=32 is the smallest with n_features_kept ≥ 384.
        assert "K=32" in advisory

    def test_caveat_on_non_residual_stream_model_type(self, tmp_path):
        """When the host's model_type isn't in the residual-stream allowlist,
        the advisory prepends a caveat sentence.
        """
        enc_path = tmp_path / "wenc" / "pareto"
        enc_path.mkdir(parents=True)
        (enc_path / "k_8.safetensors").write_text("")
        manifest = {8: mock.Mock(n_features_kept=1)}
        advisory = advise_sweep_quality(
            encodings=[("wenc", enc_path.parent)],
            host_d_model=512,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 1,
            model_type="whisper",
        )
        assert advisory is not None
        assert "standard transformer architecture" in advisory
        assert "whisper" in advisory


# ---------------------------------------------------------------------------
# basis_rank_from_safetensors
# ---------------------------------------------------------------------------


class TestBasisRankFromSafetensors:
    def test_counts_surviving_features(self, tmp_path):
        from safetensors.numpy import save_file

        W = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
        # Zero out two rows to mimic polygram compression.
        W[3] = 0
        W[5] = 0
        path = tmp_path / "sae.safetensors"
        save_file({"W_dec": W}, str(path))
        # Surviving rows are 6 random vectors → rank 6.
        assert basis_rank_from_safetensors(path) == 6

    def test_missing_w_dec_raises(self, tmp_path):
        from safetensors.numpy import save_file

        path = tmp_path / "no_wdec.safetensors"
        save_file({"W_enc": np.zeros((8, 16), dtype=np.float32)}, str(path))
        with pytest.raises(KeyError, match="W_dec"):
            basis_rank_from_safetensors(path)


# ---------------------------------------------------------------------------
# advise_sweep_quality — polygram saturation note
# (add-polygram-cluster-diagnostics)
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402


def _make_encoding_with_report(
    tmp_path,
    *,
    label: str,
    largest_k: int,
    n_clusters: int,
    additional_ks: list[int] | None = None,
):
    """Materialise a fake encoding directory with a polygram-style report
    at the largest-K SAE. Returns ``(enc_dir, manifest_dict)``.
    """
    enc_dir = tmp_path / label
    pareto = enc_dir / "pareto"
    pareto.mkdir(parents=True)
    ks = sorted(set([largest_k, *(additional_ks or [])]))
    for k in ks:
        (pareto / f"k_{k}.safetensors").write_text("")
    # Drop the compression report next to the largest-K checkpoint.
    report = pareto / f"k_{largest_k}_compression_report.json"
    report.write_text(_json.dumps({"n_clusters": n_clusters, "n_zeroed": 0}))
    manifest = {k: mock.Mock(n_features_kept=k) for k in ks}
    return enc_dir, manifest


class TestAdvisePolygramSaturation:
    def test_saturation_note_appended_when_clusters_equal_capacity(
        self, tmp_path
    ):
        """rung5 sweep whose largest-K SAE has n_clusters=128 → saturation note."""
        enc_dir, manifest = _make_encoding_with_report(
            tmp_path,
            label="rung5",
            largest_k=128,
            n_clusters=128,
        )
        # Rank-tier path: smallest K=128 reports rank low enough to flag.
        advisory = advise_sweep_quality(
            encodings=[("rung5", enc_dir)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 1,
        )
        assert advisory is not None
        assert (
            "polygram_n_clusters (128) equals encoding capacity (128) "
            "— the encoding may be saturated" in advisory
        )
        assert "HEA_Rung2(n_qubits=8)" in advisory

    def test_saturation_note_alone_when_no_rank_tier_warranted(self, tmp_path):
        """Rank-tier check is silent (good basis) but cluster saturation fires
        → return a single-line advisory containing only the saturation note.
        """
        enc_dir, manifest = _make_encoding_with_report(
            tmp_path,
            label="rung5",
            largest_k=128,
            n_clusters=128,
        )
        advisory = advise_sweep_quality(
            encodings=[("rung5", enc_dir)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 500,  # ratio 0.65 → good
        )
        assert advisory is not None
        # No rank-tier scaffolding present.
        assert "forge-quality advisory" not in advisory
        # Just the saturation note.
        assert "may be saturated" in advisory
        assert "HEA_Rung2(n_qubits=8)" in advisory

    def test_no_saturation_when_clusters_below_capacity(self, tmp_path):
        """rung5 largest-K reports 6 clusters → no saturation note."""
        enc_dir, manifest = _make_encoding_with_report(
            tmp_path,
            label="rung5",
            largest_k=128,
            n_clusters=6,
        )
        advisory = advise_sweep_quality(
            encodings=[("rung5", enc_dir)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 500,  # good
        )
        # No rank-tier advisory + no saturation → None.
        assert advisory is None

    def test_no_saturation_when_capacity_unknown(self, tmp_path):
        """Unknown encoding label parses to None capacity → no saturation note."""
        enc_dir, manifest = _make_encoding_with_report(
            tmp_path,
            label="bogus_rung",
            largest_k=64,
            n_clusters=64,
        )
        advisory = advise_sweep_quality(
            encodings=[("bogus_rung", enc_dir)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 500,
        )
        # Capacity unknown → no saturation. No rank-tier issue either.
        assert advisory is None

    def test_no_saturation_when_report_missing(self, tmp_path):
        """No compression report on disk → saturation check silently skips."""
        enc_dir = tmp_path / "rung5"
        (enc_dir / "pareto").mkdir(parents=True)
        (enc_dir / "pareto" / "k_128.safetensors").write_text("")
        manifest = {128: mock.Mock(n_features_kept=128)}
        advisory = advise_sweep_quality(
            encodings=[("rung5", enc_dir)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 500,
        )
        assert advisory is None

    def test_rung4_suggests_rung5(self, tmp_path):
        enc_dir, manifest = _make_encoding_with_report(
            tmp_path,
            label="rung4",
            largest_k=32,
            n_clusters=32,
        )
        advisory = advise_sweep_quality(
            encodings=[("rung4", enc_dir)],
            host_d_model=768,
            thresholds=QualityThresholds(),
            manifest_loader=lambda p: manifest,
            basis_rank_loader=lambda p: 500,
        )
        assert advisory is not None
        assert "Rung5" in advisory


class TestQualityFloorIgnoresPolygramSaturation:
    """``quality_floor`` reacts only to ``quality_ratio``, never to polygram
    fields. Cluster saturation is descriptive, not a gate.
    """

    def test_via_sweep_runs_with_floor_when_only_saturated(
        self, tmp_path, synthetic_compressed_sae
    ):
        """Sweep with `quality_floor` succeeds when the only issue is
        cluster saturation (not rank-ratio).
        """
        import contextlib
        import json as _j
        import shutil
        from dataclasses import dataclass, field
        from pathlib import Path

        from saeforge.sweep import sweep_pareto

        # Inline minimal stub pipeline + pareto dir builder (avoid cross-
        # test-module imports, which depend on pytest rootdir/pythonpath).
        @dataclass
        class _StubResult:
            output_dir: Path
            faithfulness: float | None = 0.5
            faithfulness_target_name: str | None = "kl"
            extras: dict = field(default_factory=lambda: {"perplexity": 1.0, "final_loss": 1.0})

        @dataclass
        class _StubPipeline:
            basis: object = None
            projector: object = None
            _call_count: int = 0

            def run(self, output_dir, **kwargs):
                self._call_count += 1
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                return _StubResult(output_dir=Path(output_dir))

        @contextlib.contextmanager
        def _noop_swap(pipeline, sae_checkpoint):
            yield

        import saeforge.sweep as _sweep_mod
        original_swap = _sweep_mod._basis_swap
        _sweep_mod._basis_swap = _noop_swap
        try:
            enc_dir = tmp_path / "rung5"
            (enc_dir / "pareto").mkdir(parents=True)
            shutil.copy(
                synthetic_compressed_sae["checkpoint"],
                enc_dir / "pareto" / "k_2.safetensors",
            )
            per_k = enc_dir / "pareto" / "k_2.safetensors"
            report_path = per_k.with_name(per_k.stem + "_compression_report.json")
            report_path.write_text(_j.dumps({"n_clusters": 128, "n_zeroed": 0}))

            pipeline = _StubPipeline()
            # The synthetic SAE has 6 surviving features. Override d_model
            # to 8 so quality_ratio = 6/8 = 0.75 → good tier. Floor 0.5
            # must pass even though cluster saturation fires.
            sweep_pareto(
                pipeline,
                encodings=[("rung5", enc_dir)],
                output_dir=tmp_path / "out",
                quality_floor=0.5,
                host_d_model_override=8,
            )
            assert pipeline._call_count == 1
        finally:
            _sweep_mod._basis_swap = original_swap
