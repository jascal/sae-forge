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
