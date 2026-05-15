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


class TestEvalPromptsParser:
    """Closes #26 — _parse_eval_prompts handles three input shapes
    (dict-shorthand, JSON string, raw line) in a single pass and
    raises actionable ValueError on unsupported shapes.
    """

    def _parse(self, path):
        from saeforge.cli import _parse_eval_prompts

        return _parse_eval_prompts(path)

    def test_dict_shorthand(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            '{"prompt": "Hello"}\n{"prompt": "World"}\n'
        )
        assert self._parse(path) == ["Hello", "World"]

    def test_json_string(self, tmp_path):
        """Each line a bare JSON string — the existing string-only
        format the v0.4 wiring fix shipped."""
        path = tmp_path / "prompts.jsonl"
        path.write_text('"Hello"\n"World"\n')
        assert self._parse(path) == ["Hello", "World"]

    def test_plain_text(self, tmp_path):
        """Each non-empty line a raw prompt — non-JSON content is
        accepted as-is."""
        path = tmp_path / "prompts.txt"
        path.write_text("Hello\nWorld\n")
        assert self._parse(path) == ["Hello", "World"]

    def test_mixed_shapes_one_file(self, tmp_path):
        """First-shape-wins per line — a single file may interleave
        the three forms."""
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            '{"prompt": "Hello"}\n'
            '"World"\n'
            "Raw line here\n"
        )
        assert self._parse(path) == ["Hello", "World", "Raw line here"]

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "prompts.txt"
        path.write_text("\nHello\n\n\nWorld\n\n")
        assert self._parse(path) == ["Hello", "World"]

    def test_dict_without_prompt_key_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"not_prompt": "Hello"}\n')
        with pytest.raises(ValueError, match="'prompt'"):
            self._parse(path)

    def test_dict_with_non_string_prompt_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prompt": 42}\n')
        with pytest.raises(ValueError, match="must be a string"):
            self._parse(path)

    def test_invalid_json_type_raises(self, tmp_path):
        """JSON numbers, booleans, lists are unsupported shapes."""
        path = tmp_path / "bad.jsonl"
        path.write_text("[1, 2, 3]\n")
        with pytest.raises(ValueError, match="entries must be"):
            self._parse(path)


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
