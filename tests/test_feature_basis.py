"""Tests for FeatureBasis.from_polygram_checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest

from saeforge import FeatureBasis


def test_from_polygram_checkpoint_kept_ids(synthetic_compressed_sae):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    assert basis.kept_ids.tolist() == synthetic_compressed_sae["expected_kept_ids"]
    assert basis.n_features == 6
    assert basis.d_model == 16


def test_from_polygram_checkpoint_merged_norm_picked_up(synthetic_compressed_sae):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    rep_2_idx = list(basis.kept_ids).index(2)
    assert basis.merged_norms[rep_2_idx] == pytest.approx(1.5, rel=1e-6)


def test_from_polygram_checkpoint_no_merge_falls_back_to_row_norm(synthetic_compressed_sae):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    rep_3_idx = list(basis.kept_ids).index(3)
    expected = float(np.linalg.norm(synthetic_compressed_sae["W_dec_full"][3]))
    assert basis.merged_norms[rep_3_idx] == pytest.approx(expected, rel=1e-6)


def test_from_polygram_checkpoint_scale_compression_ratio(synthetic_compressed_sae):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    assert basis.scale_compression_ratio == pytest.approx(0.92)


def test_from_polygram_checkpoint_metadata(synthetic_compressed_sae):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    assert basis.metadata["strategy"] == "merge"
    assert basis.metadata["n_total_features"] == 8
    assert basis.metadata["n_features_kept"] == 6
    assert basis.metadata["n_clusters"] == 2


def test_from_polygram_checkpoint_explicit_report_path(synthetic_compressed_sae, tmp_path):
    renamed = tmp_path / "elsewhere.json"
    renamed.write_text(synthetic_compressed_sae["report_path"].read_text())
    basis = FeatureBasis.from_polygram_checkpoint(
        synthetic_compressed_sae["checkpoint"], report_path=renamed
    )
    assert basis.n_features == 6


def test_from_polygram_checkpoint_missing_report_detects_zero_rows(
    synthetic_compressed_sae,
):
    """Without a report, zero-row detection still finds kept features.

    The synthetic fixture zeros rows 5 and 7. The loader detects this
    directly from W_dec norms even when no compression report is on
    disk. This is what makes the loader compatible with EpochCompressor
    output (whose EpochReport doesn't carry per-cluster zeroed lists).
    """
    synthetic_compressed_sae["report_path"].unlink()
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    assert basis.kept_ids.tolist() == [0, 1, 2, 3, 4, 6]
    assert basis.scale_compression_ratio == 1.0
    assert basis.metadata["n_features_kept"] == 6


def test_from_polygram_checkpoint_uncompressed_sae(tmp_path):
    """A fully-nonzero W_dec (no compression applied) keeps every row."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(2)
    W_dec = rng.standard_normal((8, 16)).astype(np.float32)
    ckpt = tmp_path / "raw.safetensors"
    save_file({"W_dec": W_dec}, str(ckpt))
    basis = FeatureBasis.from_polygram_checkpoint(ckpt)
    assert basis.n_features == 8
    assert basis.scale_compression_ratio == 1.0


def test_from_polygram_checkpoint_explicit_missing_report_raises(
    synthetic_compressed_sae, tmp_path
):
    with pytest.raises(FileNotFoundError, match="compression report"):
        FeatureBasis.from_polygram_checkpoint(
            synthetic_compressed_sae["checkpoint"],
            report_path=tmp_path / "no-such-report.json",
        )


def test_from_polygram_checkpoint_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        FeatureBasis.from_polygram_checkpoint(tmp_path / "missing.safetensors")


def test_from_polygram_checkpoint_unknown_decoder_key_raises(tmp_path):
    from safetensors.numpy import save_file

    bad = tmp_path / "no_w_dec.safetensors"
    save_file({"some_other_key": np.zeros((2, 2), dtype=np.float32)}, str(bad))
    with pytest.raises(KeyError, match="decoder weight"):
        FeatureBasis.from_polygram_checkpoint(bad)


def test_from_polygram_checkpoint_decoder_weight_transpose_path(tmp_path):
    """When the key is `decoder.weight` and matrix is non-square, loader transposes."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(1)
    out_in = rng.standard_normal((16, 8)).astype(np.float32)  # (d_model, n_features) HF convention
    ckpt = tmp_path / "transposed.safetensors"
    save_file({"decoder.weight": out_in}, str(ckpt))
    minimal = {
        "schema_version": 1,
        "source_checkpoint": "x",
        "source_checkpoint_sha256": "x",
        "output_checkpoint": str(ckpt),
        "output_checkpoint_sha256": "x",
        "validation_report_dictionary_name": "x",
        "validation_report_schema_version": 1,
        "strategy": "zero",
        "feature_ids": [],
        "clusters": [],
        "n_features_zeroed": 0,
        "n_features_kept": 8,
        "n_clusters": 0,
        "scale_compression_ratio": 1.0,
    }
    report = ckpt.with_name("transposed_compression_report.json")
    report.write_text(json.dumps(minimal))
    basis = FeatureBasis.from_polygram_checkpoint(ckpt)
    assert basis.W_dec.shape == (8, 16)


def test_pseudoinverse_is_cached(synthetic_compressed_sae):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    a = basis.pseudoinverse()
    b = basis.pseudoinverse()
    assert a is b
    assert a.shape == (basis.d_model, basis.n_features)


def test_to_summary_round_trip(synthetic_compressed_sae, tmp_path):
    basis = FeatureBasis.from_polygram_checkpoint(synthetic_compressed_sae["checkpoint"])
    out = tmp_path / "summary.json"
    basis.save_summary(out)
    parsed = json.loads(out.read_text())
    assert parsed["n_features"] == 6
    assert parsed["d_model"] == 16
    assert parsed["scale_compression_ratio"] == pytest.approx(0.92)
