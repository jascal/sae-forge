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


class TestParserAcceptsAdaptiveRegrowFlags:
    def test_adaptive_regrow_flag_defaults_to_false(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
            ]
        )
        assert args.adaptive_regrow is False
        assert args.regrow_max == 0
        assert args.n_features_target == 0
        assert args.regrow_damping == 0.5

    def test_all_adaptive_flags_recorded(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
                "--regrow-count", "5",
                "--regrow-layer", "8",
                "--adaptive-regrow",
                "--regrow-max", "64",
                "--n-features-target", "300",
                "--regrow-damping", "0.75",
            ]
        )
        assert args.adaptive_regrow is True
        assert args.regrow_max == 64
        assert args.n_features_target == 300
        assert args.regrow_damping == 0.75


class TestAdaptiveRegrowMutuallyRequired:
    def test_adaptive_regrow_without_regrow_max_exits_2(self, tmp_path, capsys):
        ckpt = tmp_path / "fake.safetensors"
        ckpt.write_bytes(b"")
        from saeforge.cli import main

        rc = main(
            [
                "forge",
                str(ckpt),
                "--host-model", "gpt2",
                "--output-dir", str(tmp_path / "out"),
                "--adaptive-regrow",
                "--n-features-target", "300",
                # NB: no --regrow-max.
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "--adaptive-regrow" in captured.err
        assert "--regrow-max" in captured.err

    def test_adaptive_regrow_without_n_features_target_exits_2(self, tmp_path, capsys):
        ckpt = tmp_path / "fake.safetensors"
        ckpt.write_bytes(b"")
        from saeforge.cli import main

        rc = main(
            [
                "forge",
                str(ckpt),
                "--host-model", "gpt2",
                "--output-dir", str(tmp_path / "out"),
                "--adaptive-regrow",
                "--regrow-max", "64",
                # NB: no --n-features-target.
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "--n-features-target" in captured.err


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


class TestInspectFsmDiagram:
    def test_fsm_diagram_emits_state_diagram_v2(self, capsys):
        pytest.importorskip("orca_runtime_python")
        from saeforge.cli import main

        rc = main(["inspect", "--fsm-diagram"])
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out.startswith("stateDiagram-v2")
        # The three sub-machines' compound states must be in the output.
        assert 'state "streaming"' in captured.out
        assert 'state "refining"' in captured.out
        assert captured.err == ""

    def test_inspect_with_no_args_errors_actionably(self, capsys):
        from saeforge.cli import main

        rc = main(["inspect"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "checkpoint" in captured.err and "--fsm-diagram" in captured.err
