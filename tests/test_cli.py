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


class TestAudioFeaturesPath:
    """§8 — forge-whisper-encoder CLI: --audio-features-path flag.

    The flag is the audio-side analog of --eval-prompts: it selects
    the cosine_faithfulness eval signal for a Whisper-encoder forge.
    argparse-level mutual exclusion with --eval-prompts; runtime
    torch.load + pass-through to ForgePipeline.eval_audio_features.
    """

    def test_audio_features_path_parses(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "openai/whisper-tiny",
                "--output-dir", "/tmp/out",
                "--audio-features-path", "/tmp/mel.pt",
            ]
        )
        assert args.audio_features_path == "/tmp/mel.pt"
        # eval_prompts default is None when audio path is selected.
        assert args.eval_prompts is None

    def test_default_audio_features_path_is_none(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "forge",
                "ckpt.safetensors",
                "--host-model", "gpt2",
                "--output-dir", "/tmp/out",
            ]
        )
        assert args.audio_features_path is None

    def test_mutual_exclusion_with_eval_prompts(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(
                [
                    "forge",
                    "ckpt.safetensors",
                    "--host-model", "openai/whisper-tiny",
                    "--output-dir", "/tmp/out",
                    "--eval-prompts", "prompts.jsonl",
                    "--audio-features-path", "/tmp/mel.pt",
                ]
            )
        # argparse exits with 2 on usage errors.
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "not allowed" in err or "mutually exclusive" in err

    def test_audio_features_path_passes_through_to_pipeline(
        self, tmp_path, monkeypatch
    ):
        """End-to-end CLI wiring: --audio-features-path → torch.load →
        ForgePipeline.eval_audio_features. Mocks the pipeline run so
        the test stays hermetic on the Intel Mac (no HF download)."""
        pytest.importorskip("torch")
        import torch

        from saeforge.cli import main

        # Write a real torch tensor and a real polygram-shape SAE so
        # FeatureBasis.from_polygram_checkpoint accepts it.
        import json
        import numpy as np
        from safetensors.numpy import save_file

        mel_path = tmp_path / "mel.pt"
        mel = torch.zeros(1, 80, 3000)
        torch.save(mel, mel_path)

        ckpt = tmp_path / "sae.compressed.safetensors"
        W = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        save_file({"W_dec": W}, str(ckpt))
        (tmp_path / "sae.compressed_compression_report.json").write_text(
            json.dumps({"schema_version": 1, "clusters": []})
        )

        # Capture the ForgePipeline kwargs without actually running.
        captured = {}

        class _StubPipeline:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, output_dir):
                from saeforge.forge import ForgeResult

                return ForgeResult(
                    model=None,
                    output_dir=output_dir,
                    n_params=0,
                    faithfulness_kl=None,
                    extras={},
                )

        monkeypatch.setattr("saeforge.ForgePipeline", _StubPipeline)

        rc = main(
            [
                "forge",
                str(ckpt),
                "--host-model", "openai/whisper-tiny",
                "--output-dir", str(tmp_path / "out"),
                "--audio-features-path", str(mel_path),
            ]
        )
        assert rc == 0
        # The loaded tensor should be the one we saved (same shape).
        assert captured["eval_audio_features"] is not None
        assert tuple(captured["eval_audio_features"].shape) == (1, 80, 3000)


class TestEvalPromptsWiring:
    """--eval-prompts JSONL-of-strings is now wired into
    ForgePipeline.eval_prompts. Full JSONL-schema support (objects
    with role / completion / metadata) is tech-debt — tracked in
    a separate backlog issue.
    """

    def test_eval_prompts_parses_jsonl_strings(self, tmp_path, monkeypatch):
        import json
        import numpy as np
        from safetensors.numpy import save_file

        from saeforge.cli import main

        # JSONL-of-strings file: each line a json-encoded prompt.
        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text(
            "\n".join(json.dumps(p) for p in [
                "The quick brown fox",
                "In a hole in the ground",
            ])
            + "\n"
        )

        ckpt = tmp_path / "sae.compressed.safetensors"
        W = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        save_file({"W_dec": W}, str(ckpt))
        (tmp_path / "sae.compressed_compression_report.json").write_text(
            json.dumps({"schema_version": 1, "clusters": []})
        )

        captured = {}

        class _StubPipeline:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, output_dir):
                from saeforge.forge import ForgeResult

                return ForgeResult(
                    model=None,
                    output_dir=output_dir,
                    n_params=0,
                    faithfulness_kl=None,
                    extras={},
                )

        monkeypatch.setattr("saeforge.ForgePipeline", _StubPipeline)

        rc = main(
            [
                "forge",
                str(ckpt),
                "--host-model", "gpt2",
                "--output-dir", str(tmp_path / "out"),
                "--eval-prompts", str(prompts_path),
            ]
        )
        assert rc == 0
        assert captured["eval_prompts"] == [
            "The quick brown fox",
            "In a hole in the ground",
        ]

    def test_eval_prompts_defaults_empty_list_when_unset(
        self, tmp_path, monkeypatch
    ):
        import json
        import numpy as np
        from safetensors.numpy import save_file

        from saeforge.cli import main

        ckpt = tmp_path / "sae.compressed.safetensors"
        W = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        save_file({"W_dec": W}, str(ckpt))
        (tmp_path / "sae.compressed_compression_report.json").write_text(
            json.dumps({"schema_version": 1, "clusters": []})
        )

        captured = {}

        class _StubPipeline:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, output_dir):
                from saeforge.forge import ForgeResult

                return ForgeResult(
                    model=None,
                    output_dir=output_dir,
                    n_params=0,
                    faithfulness_kl=None,
                    extras={},
                )

        monkeypatch.setattr("saeforge.ForgePipeline", _StubPipeline)
        rc = main(
            [
                "forge",
                str(ckpt),
                "--host-model", "gpt2",
                "--output-dir", str(tmp_path / "out"),
            ]
        )
        assert rc == 0
        assert captured["eval_prompts"] == []


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
