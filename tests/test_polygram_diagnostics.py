"""Unit tests for ``saeforge.polygram_diagnostics``."""

from __future__ import annotations

import json
import logging
import math


from saeforge.polygram_diagnostics import (
    compute_redundancy_ratio,
    format_saturation_note,
    load_polygram_report,
    resolve_encoding_capacity,
)


# ---------------------------------------------------------------------------
# load_polygram_report
# ---------------------------------------------------------------------------


class TestLoadPolygramReport:
    def test_finds_suffix_variants(self, tmp_path):
        ckpt = tmp_path / "sae.compressed.safetensors"
        ckpt.write_bytes(b"")
        # Use the canonical suffix that polygram emits.
        report_path = tmp_path / "sae.compressed_compression_report.json"
        payload = {"n_clusters": 6, "n_zeroed": 88, "strategy": "merge"}
        report_path.write_text(json.dumps(payload))
        report = load_polygram_report(ckpt)
        assert report is not None
        assert report["n_clusters"] == 6
        assert report["n_zeroed"] == 88
        assert report["strategy"] == "merge"

    def test_finds_dot_compression_report_suffix(self, tmp_path):
        ckpt = tmp_path / "sae.safetensors"
        ckpt.write_bytes(b"")
        report_path = tmp_path / "sae.compression_report.json"
        report_path.write_text(json.dumps({"n_clusters": 3}))
        report = load_polygram_report(ckpt)
        assert report is not None
        assert report["n_clusters"] == 3

    def test_finds_report_json_suffix(self, tmp_path):
        ckpt = tmp_path / "sae.safetensors"
        ckpt.write_bytes(b"")
        report_path = tmp_path / "sae_report.json"
        report_path.write_text(json.dumps({"n_clusters": 7}))
        report = load_polygram_report(ckpt)
        assert report is not None
        assert report["n_clusters"] == 7

    def test_returns_none_on_missing(self, tmp_path, caplog):
        ckpt = tmp_path / "missing.safetensors"
        with caplog.at_level(logging.INFO, logger="saeforge.polygram_diagnostics"):
            result = load_polygram_report(ckpt)
        assert result is None
        # INFO log line emitted.
        assert any("no compression report" in r.message for r in caplog.records)

    def test_returns_none_on_malformed_json(self, tmp_path, caplog):
        ckpt = tmp_path / "sae.safetensors"
        ckpt.write_bytes(b"")
        bad_report = tmp_path / "sae_compression_report.json"
        bad_report.write_text("{")
        with caplog.at_level(logging.INFO, logger="saeforge.polygram_diagnostics"):
            result = load_polygram_report(ckpt)
        assert result is None
        assert any("failed to load report" in r.message for r in caplog.records)

    def test_handles_none_path(self):
        assert load_polygram_report(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_redundancy_ratio
# ---------------------------------------------------------------------------


class TestComputeRedundancyRatio:
    def test_basic_supervised(self):
        # Phase 6.2 supervised SAE at Rung5 cap=128: 6 clusters / 88 zeroed.
        r = compute_redundancy_ratio(6, 88)
        assert r is not None
        assert math.isclose(r, 88 / 94, rel_tol=1e-12)

    def test_basic_unsupervised(self):
        # Phase 1.6 unsupervised SAE: 7 clusters / 62 zeroed.
        r = compute_redundancy_ratio(7, 62)
        assert r is not None
        assert math.isclose(r, 62 / 69, rel_tol=1e-12)

    def test_zero_zeroed_returns_zero(self):
        # All clusters, no zeroed → ratio is 0.0 (no redundancy).
        r = compute_redundancy_ratio(6, 0)
        assert r == 0.0

    def test_none_n_clusters_returns_none(self):
        assert compute_redundancy_ratio(None, 5) is None

    def test_none_n_zeroed_returns_none(self):
        assert compute_redundancy_ratio(6, None) is None

    def test_both_none_returns_none(self):
        assert compute_redundancy_ratio(None, None) is None

    def test_both_zero_returns_none(self):
        assert compute_redundancy_ratio(0, 0) is None

    def test_negative_returns_none(self):
        assert compute_redundancy_ratio(-1, 5) is None
        assert compute_redundancy_ratio(5, -1) is None


# ---------------------------------------------------------------------------
# resolve_encoding_capacity
# ---------------------------------------------------------------------------


class TestResolveEncodingCapacity:
    def test_known_rungs_lowercase(self):
        assert resolve_encoding_capacity("rung3") == 16
        assert resolve_encoding_capacity("rung4") == 32
        assert resolve_encoding_capacity("rung5") == 128

    def test_known_rungs_titlecase(self):
        assert resolve_encoding_capacity("Rung3") == 16
        assert resolve_encoding_capacity("Rung4") == 32
        assert resolve_encoding_capacity("Rung5") == 128

    def test_known_rungs_with_whitespace(self):
        assert resolve_encoding_capacity("  rung5  ") == 128

    def test_parametric_hea_rung2_equals(self):
        assert resolve_encoding_capacity("hea_rung2(n_qubits=6)") == 64
        assert resolve_encoding_capacity("HEA_Rung2(n_qubits=8)") == 256

    def test_parametric_hea_rung2_colon(self):
        # The encoding spec parser used by other sweep flags accepts
        # ``:`` as the LABEL:VALUE separator; tolerate both forms here.
        assert resolve_encoding_capacity("hea_rung2(n_qubits:5)") == 32

    def test_parametric_hea_rung2_whitespace(self):
        assert resolve_encoding_capacity("HEA_Rung2( n_qubits = 4 )") == 16

    def test_unknown_returns_none(self):
        assert resolve_encoding_capacity("bogus") is None
        assert resolve_encoding_capacity("rung99") is None
        assert resolve_encoding_capacity("") is None
        assert resolve_encoding_capacity("mps") is None

    def test_malformed_parametric_returns_none(self):
        assert resolve_encoding_capacity("hea_rung2()") is None
        assert resolve_encoding_capacity("hea_rung2(n=6)") is None


# ---------------------------------------------------------------------------
# format_saturation_note
# ---------------------------------------------------------------------------


class TestFormatSaturationNote:
    def test_includes_all_args(self):
        note = format_saturation_note(128, 128, "HEA_Rung2(n_qubits=8)")
        assert "128" in note
        assert "HEA_Rung2(n_qubits=8)" in note
        # Spec-frozen phrasing.
        assert "may be saturated" in note
        assert "additional concepts" in note
