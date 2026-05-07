"""CLI integration tests for the polygram tuning flags introduced by
forge-polygram-tuning-passthrough.

Covers tasks.md §8.4: --coverage-target, --regrow-layer, and the
--regrow-count > 0 without --regrow-layer error path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("polygram")

from saeforge.cli import _build_parser


class TestParserAcceptsTuningFlags:
    def test_coverage_target_recorded(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
                "--coverage-target", "0.6",
            ]
        )
        assert args.coverage_target == 0.6

    def test_max_compress_iterations_recorded(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
                "--max-compress-iterations", "3",
            ]
        )
        assert args.max_compress_iterations == 3

    def test_regrow_layer_recorded(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
                "--regrow-count", "2",
                "--regrow-layer", "4",
            ]
        )
        assert args.regrow_layer == 4
        assert args.regrow_count == 2

    def test_regrow_strategy_default_none(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
            ]
        )
        assert args.regrow_strategy is None
        # All tuning flags omitted → None defaults; pipeline falls back
        # to polygram's own defaults.
        assert args.coverage_target is None
        assert args.cosine_threshold is None
        assert args.max_compress_iterations is None
        assert args.regrow_count == 0
        assert args.regrow_layer is None


class TestRegrowLayerRequiredWhenRegrowCountSet:
    def test_regrow_count_without_regrow_layer_exits_2(self, tmp_path, capsys):
        # Need a real-looking checkpoint path for FeatureBasis to fail
        # later (we expect the CLI to bail before reaching it).
        ckpt = tmp_path / "fake.safetensors"
        ckpt.write_bytes(b"")

        from saeforge.cli import main

        rc = main(
            [
                "forge",
                str(ckpt),
                "--host-model", "gpt2",
                "--output-dir", str(tmp_path / "out"),
                "--regrow-count", "2",
                # NB: no --regrow-layer — this is what we're testing.
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "--regrow-count" in captured.err
        assert "--regrow-layer" in captured.err
