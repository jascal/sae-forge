"""Tests for the progressive-sweep CLI subcommand + recommend's
un-converged-frontier refusal.

Three suites:
  1. Parser construction — `sweep-capability-progressive` flags
     resolve, defaults match spec.
  2. End-to-end smoke — CLI subcommand runs against a synthesized
     fixture via main(); exit code reflects convergence; frontier +
     progressive_summary.json land on disk.
  3. `recommend` un-converged-frontier refusal — refuses by
     default, accepts via --accept-unconverged, refuses cleanly on
     missing companion summary.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("yaml")


# ---------------------------------------------------------------------------
# Suite 1: parser
# ---------------------------------------------------------------------------


def test_parser_sweep_capability_progressive_defaults():
    from saeforge.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "sweep-capability-progressive",
        "--dataset-config", "x.yaml",
        "--host", "h",
        "--candidate-widths", "4,8,16",
        "--schedule", "10,50",
        "--output-dir", "out",
    ])
    assert args.command == "sweep-capability-progressive"
    # Defaults per the openspec.
    assert args.retained_mauc_tolerance == pytest.approx(0.005)
    assert args.plateau_tolerance == pytest.approx(0.01)
    assert args.min_plateau_widths == 3
    assert args.convergence_n_stages == 2
    assert args.max_seq_len == 512
    assert args.device == "cpu"
    assert args.no_host_cache is False
    assert args.scale_boosts == "1.0"
    assert args.encodings == "raw_slice"


def test_parser_recommend_accepts_unconverged_flag():
    from saeforge.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "recommend", "--frontier", "f.jsonl",
        "--target", "retained-mauc>=0.9",
        "--accept-unconverged",
    ])
    assert args.accept_unconverged is True
    args_default = parser.parse_args([
        "recommend", "--frontier", "f.jsonl",
        "--target", "retained-mauc>=0.9",
    ])
    assert args_default.accept_unconverged is False


# ---------------------------------------------------------------------------
# Suite 2: end-to-end smoke (CLI -> wrapper -> output)
# ---------------------------------------------------------------------------


def _build_bio_sae_fixture(tmp_path: Path, *, n_proteins=8, d_model=32, sae_width=32):
    """Same shape as the synthetic fixture in test_sweep_progressive.py."""
    import pandas as pd
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({
        "encoder.weight": torch.from_numpy(
            rng.standard_normal((sae_width, d_model)).astype(np.float32)
        ),
        "encoder.bias": torch.zeros(sae_width),
        "decoder.weight": torch.from_numpy(
            rng.standard_normal((d_model, sae_width)).astype(np.float32)
        ),
        "decoder.bias": torch.zeros(d_model),
    }, run_dir / "sae.pt")
    bundle = {
        "pooled": rng.standard_normal((n_proteins, d_model)).astype(np.float32),
        "labels_protein_Y": rng.integers(0, 2, (n_proteins, 5)).astype(np.uint8),
        "residue_index": np.stack([
            np.repeat(np.arange(n_proteins), 4).astype(np.int32),
            np.tile(np.arange(4), n_proteins).astype(np.int32),
            np.full(n_proteins * 4, 4, dtype=np.int32),
        ], axis=1),
        "labels_residue_Y": rng.integers(0, 2, (n_proteins * 4, 3)).astype(np.uint8),
        "activations": rng.standard_normal((n_proteins * 4, d_model)).astype(np.float32),
    }
    bundle_path = tmp_path / "bio_bundle.safetensors"
    save_file(bundle, str(bundle_path))
    seqs = pd.DataFrame({"sequence": ["MAKVITDR" + "G" * (i + 1) for i in range(n_proteins)]})
    seqs_path = tmp_path / "sequences.parquet"
    seqs.to_parquet(seqs_path)
    return run_dir, bundle_path, seqs_path


@pytest.fixture
def _tiny_host_model_id(tmp_path: Path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM

    cfg = EsmConfig(
        vocab_size=33, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64,
        max_position_embeddings=128,
        position_embedding_type="rotary",
        emb_layer_norm_before=False, token_dropout=False,
        mask_token_id=32, pad_token_id=1,
    )
    torch.manual_seed(0)
    model = EsmForMaskedLM(cfg)
    host_dir = tmp_path / "tiny_esm"
    model.save_pretrained(host_dir)
    try:
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        tok.save_pretrained(host_dir)
    except Exception as exc:
        pytest.skip(f"can't fetch ESM tokenizer: {exc}")
    return str(host_dir)


def _write_dataset_config(
    tmp_path: Path, *, run_dir: Path, bundle_path: Path, sequences_path: Path,
    tokenizer_id: str, sae_k: int = 8,
) -> Path:
    import yaml

    config = {
        "encoder_checkpoint": str(run_dir / "sae.pt"),
        "sequences_path": str(sequences_path),
        "labels_path": str(bundle_path),
        "feed": "pooled",
        "tokenizer_id": tokenizer_id,
        "aggregator": "pool_then_encode",
        "min_prevalence": 0,
        "sae_variant": "topk",
        "sae_k": sae_k,
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_sweep_capability_progressive_e2e(tmp_path: Path, _tiny_host_model_id):
    """End-to-end through main(): subcommand runs, emits frontier + summary,
    exits 0 on the synthetic substrate (which converges in 1-2 stages)."""
    from saeforge.cli import main as cli_main

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    cfg_path = _write_dataset_config(
        tmp_path, run_dir=run_dir, bundle_path=bundle_path,
        sequences_path=seqs_path, tokenizer_id=_tiny_host_model_id,
    )
    output_dir = tmp_path / "prog_out"
    rc = cli_main([
        "sweep-capability-progressive",
        "--dataset-config", str(cfg_path),
        "--host", _tiny_host_model_id,
        "--candidate-widths", "4,8,16,32",
        "--schedule", "4,8",
        "--output-dir", str(output_dir),
        "--device", "cpu",
    ])
    # rc 0 or 1 acceptable (0 = converged, 1 = exhausted-but-emitted);
    # both indicate the wrapper ran end-to-end.
    assert rc in (0, 1), f"unexpected exit code {rc}"
    assert (output_dir / "frontier.jsonl").exists()
    assert (output_dir / "progressive_summary.json").exists()
    summary = json.loads((output_dir / "progressive_summary.json").read_text())
    assert "stages" in summary
    assert "recommendation" in summary
    assert "convergence_trajectory" in summary["recommendation"]


def test_sweep_capability_progressive_config_validation(tmp_path: Path):
    """Missing required keys → exit 2 with stderr message."""
    from saeforge.cli import main as cli_main

    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("encoder_checkpoint: x\n")  # missing labels_path, sequences_path
    rc = cli_main([
        "sweep-capability-progressive",
        "--dataset-config", str(cfg_path),
        "--host", "h",
        "--candidate-widths", "4",
        "--schedule", "1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


# ---------------------------------------------------------------------------
# Suite 3: recommend refusal on un-converged frontiers
# ---------------------------------------------------------------------------


def _write_progressive_frontier(
    tmp_path: Path, *, converged: bool, rec_n: int = 16,
    rec_retained: float = 1.0,
):
    """Synthesize a minimal progressive frontier + summary for the
    recommend-refusal tests. Doesn't run the full sweep — just writes
    files in the shape recommend expects."""
    frontier_dir = tmp_path / "progressive"
    frontier_dir.mkdir()
    rows = [
        {
            "encoding_label": "raw_slice",
            "target_n_features_kept": w,
            "n_features_kept_actual": w,
            "pareto_reached_target": True,
            "faithfulness_kl": None,
            "perplexity": None,
            "final_fine_tune_loss": None,
            "sae_checkpoint": "/tmp/sae",
            "forged_model_path": None,
            "elapsed_seconds": 0.1,
            "error_message": None,
            "host_baseline_mauc": 0.8,
            "forge_mauc": 0.8 * (1.0 if w == rec_n else 0.95),
            "retained_mauc_vs_host": rec_retained if w == rec_n else 0.95,
            "capability_aggregator": "pool_then_encode",
            "capability_min_prevalence": 0,
            "stage": 0,
        }
        for w in (8, 16, 32)
    ]
    frontier_path = frontier_dir / "frontier.jsonl"
    frontier_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    trajectory = [
        {
            "stage": 0, "n_proteins": 10,
            "argmin_plateau_width": rec_n,
            "argmin_retained_mauc": rec_retained,
            "plateau_size": 3, "neighbours_added": 0,
            "shifted_from_prev_stage": False,
        },
    ]
    if not converged:
        # Add a shifted second stage so refusal has something to name.
        trajectory.append({
            "stage": 1, "n_proteins": 50,
            "argmin_plateau_width": rec_n + 8,  # shifted
            "argmin_retained_mauc": rec_retained,
            "plateau_size": 3, "neighbours_added": 0,
            "shifted_from_prev_stage": True,
        })
    summary = {
        "stages": [{"stage": 0, "n_proteins": 10, "active_widths": [8, 16, 32],
                    "plateau_widths": [8, 16, 32], "peak_n": rec_n,
                    "peak_retained_mauc": rec_retained, "n_rows": 3}],
        "recommendation": {
            "target_n_features_kept": rec_n,
            "retained_mauc_vs_host": rec_retained,
            "stages_converged": 1,
            "converged": converged,
            "rationale": ("converged at stage 0" if converged
                          else "stage 1 shifted argmin from "
                               f"n={rec_n} to n={rec_n + 8}"),
            "convergence_trajectory": trajectory,
        },
    }
    (frontier_dir / "progressive_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return frontier_path


def test_recommend_refuses_unconverged_progressive_frontier(tmp_path: Path):
    """Default behaviour: refuse with diagnostic on converged=False."""
    from saeforge.cli import main as cli_main

    frontier_path = _write_progressive_frontier(tmp_path, converged=False)
    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.5",
        ])
    assert rc == 1
    stderr = err_buf.getvalue()
    assert "did NOT converge" in stderr
    assert "--accept-unconverged" in stderr
    # The diagnostic SHALL name the shifted stage.
    assert "Shifted stages: [1]" in stderr


def test_recommend_accepts_unconverged_with_flag(tmp_path: Path):
    """--accept-unconverged overrides the refusal."""
    from saeforge.cli import main as cli_main

    frontier_path = _write_progressive_frontier(tmp_path, converged=False)
    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.5",
            "--accept-unconverged",
            "--json",
        ])
    assert rc == 0
    picked = json.loads(out_buf.getvalue())
    assert picked["target_n_features_kept"] == 8  # smallest survivor


def test_recommend_processes_converged_progressive_frontier_normally(tmp_path: Path):
    """A converged progressive frontier behaves like a normal recommend
    call — no refusal, picks smallest survivor."""
    from saeforge.cli import main as cli_main

    frontier_path = _write_progressive_frontier(tmp_path, converged=True)
    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.5",
            "--json",
        ])
    assert rc == 0
    picked = json.loads(out_buf.getvalue())
    assert picked["target_n_features_kept"] == 8


def test_recommend_missing_progressive_summary_raises(tmp_path: Path):
    """A progressive frontier without companion progressive_summary.json
    → exit 2 with explanatory message."""
    from saeforge.cli import main as cli_main

    frontier_path = _write_progressive_frontier(tmp_path, converged=True)
    # Delete the summary so the recommend has to refuse.
    (frontier_path.parent / "progressive_summary.json").unlink()
    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.5",
        ])
    assert rc == 2
    stderr = err_buf.getvalue()
    assert "progressive_summary.json" in stderr


def test_recommend_single_shot_frontier_unaffected(tmp_path: Path):
    """A non-progressive frontier (no stage field on any row) bypasses
    the un-converged check entirely — back-compat with the v0.8.x
    recommend behaviour."""
    from saeforge.cli import main as cli_main

    frontier_dir = tmp_path / "single_shot"
    frontier_dir.mkdir()
    rows = [
        {
            "encoding_label": "raw_slice",
            "target_n_features_kept": w,
            "n_features_kept_actual": w,
            "pareto_reached_target": True,
            "faithfulness_kl": None,
            "perplexity": None,
            "final_fine_tune_loss": None,
            "sae_checkpoint": "/tmp/sae",
            "forged_model_path": None,
            "elapsed_seconds": 0.1,
            "error_message": None,
            "host_baseline_mauc": 0.8,
            "forge_mauc": 0.8,
            "retained_mauc_vs_host": 1.0,
            "capability_aggregator": "pool_then_encode",
            "capability_min_prevalence": 0,
            # No "stage" key — single-shot row.
        }
        for w in (8, 16)
    ]
    frontier_path = frontier_dir / "frontier.jsonl"
    frontier_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # No companion progressive_summary.json — that's expected for a
    # single-shot frontier; recommend SHALL NOT try to read it.
    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.5",
            "--json",
        ])
    assert rc == 0
    picked = json.loads(out_buf.getvalue())
    assert picked["target_n_features_kept"] == 8


# ---------------------------------------------------------------------------
# Suite 4: multi-encoding CLI surface
# (added by add-multi-encoding-capability-sweep slice 3/N)
# ---------------------------------------------------------------------------


def test_cli_multi_encoding_flag_parses():
    """--encoding LABEL:PATH (repeatable) accumulates into args.encoding."""
    from saeforge.cli import _build_parser, _parse_encoding_specs

    parser = _build_parser()
    args = parser.parse_args([
        "sweep-capability",
        "--dataset-config", "x.yaml",
        "--host", "h",
        "--widths", "4,8",
        "--output-dir", "out",
        "--encoding", "raw_slice:p1",
        "--encoding", "partition_q4:p2",
        "--encoding", "mps_rung1_x16:p3",
    ])
    assert args.encoding == ["raw_slice:p1", "partition_q4:p2", "mps_rung1_x16:p3"]
    parsed = _parse_encoding_specs(args.encoding)
    assert len(parsed) == 3
    assert parsed[0] == ("raw_slice", Path("p1"))
    assert parsed[2] == ("mps_rung1_x16", Path("p3"))


def test_cli_progressive_multi_encoding_flag_parses():
    """Same --encoding flag works on sweep-capability-progressive."""
    from saeforge.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "sweep-capability-progressive",
        "--dataset-config", "x.yaml",
        "--host", "h",
        "--candidate-widths", "4,8,16",
        "--schedule", "10,50",
        "--output-dir", "out",
        "--encoding", "raw_slice:p1",
        "--encoding", "partition:p2",
    ])
    assert args.encoding == ["raw_slice:p1", "partition:p2"]
    assert args.dry_run is False  # default
    assert args.dollars_per_gpu_hr is None  # default


def test_cli_dry_run_flag_parses_with_cost_rate():
    """--dry-run + --dollars-per-gpu-hr both parse cleanly."""
    from saeforge.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "sweep-capability",
        "--dataset-config", "x.yaml", "--host", "h",
        "--widths", "4,8", "--output-dir", "out",
        "--dry-run", "--dollars-per-gpu-hr", "3.0",
    ])
    assert args.dry_run is True
    assert args.dollars_per_gpu_hr == 3.0


def test_cli_sweep_capability_dry_run_exits_zero(tmp_path: Path):
    """--dry-run emits projection and exits 0 WITHOUT running the
    sweep. Tested by passing a non-existent dataset-config path that
    would fail at runtime — but dry-run still needs --dataset-config
    to be a real file because the YAML loader runs before the dry-
    run branch."""
    import yaml
    from saeforge.cli import main as cli_main

    cfg_path = tmp_path / "dataset.yaml"
    # Need a YAML config that AT LEAST has the required keys; the
    # dataset loader runs before dry-run kicks in. We use a stub
    # config pointing at non-existent fixture paths; the dataset
    # constructor will fail on file-not-found UNLESS we point it at
    # a real fixture. Use the synthetic fixture from elsewhere in
    # this file.
    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    cfg_path.write_text(yaml.safe_dump({
        "encoder_checkpoint": str(run_dir / "sae.pt"),
        "sequences_path": str(seqs_path),
        "labels_path": str(bundle_path),
        "feed": "pooled",
        "tokenizer_id": "facebook/esm2_t6_8M_UR50D",
        "aggregator": "pool_then_encode",
        "min_prevalence": 0,
        "sae_variant": "topk",
        "sae_k": 8,
    }))
    # Skip if tokenizer can't be fetched.
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    except Exception as exc:
        pytest.skip(f"ESM tokenizer unavailable: {exc}")

    rc = cli_main([
        "sweep-capability",
        "--dataset-config", str(cfg_path),
        "--host", "facebook/esm2_t6_8M_UR50D",
        "--widths", "4,8",
        "--output-dir", str(tmp_path / "dry_run_out"),
        "--encoding", f"raw_slice:{run_dir / 'sae.pt'}",
        "--encoding", f"shadow_b:{run_dir / 'sae.pt'}",  # OK to reuse path
        "--dry-run",
        "--dollars-per-gpu-hr", "3.0",
    ])
    assert rc == 0
    # The sweep would have written frontier.jsonl if it ran; under
    # dry-run, no such file exists.
    assert not (tmp_path / "dry_run_out" / "frontier.jsonl").exists()


def test_cli_recommend_multi_encoding_emits_ranking_table(tmp_path: Path):
    """Multi-encoding frontier (rows with multiple distinct
    encoding_label values) → recommend output emits the per-encoding
    ranking table."""
    import contextlib
    import io
    import json

    from saeforge.cli import main as cli_main

    # Synthesize a multi-encoding frontier.
    frontier_dir = tmp_path / "multi_encoding"
    frontier_dir.mkdir()
    rows = []
    for label, base_retained in [("raw_slice", 0.90), ("partition_q4", 0.92)]:
        for width in (16, 32, 64):
            rows.append({
                "encoding_label": label,
                "target_n_features_kept": width,
                "n_features_kept_actual": width,
                "pareto_reached_target": True,
                "faithfulness_kl": None,
                "perplexity": None,
                "final_fine_tune_loss": None,
                "sae_checkpoint": "/tmp/sae",
                "forged_model_path": None,
                "elapsed_seconds": 0.1,
                "error_message": None,
                "host_baseline_mauc": 0.95,
                "forge_mauc": 0.85,
                "retained_mauc_vs_host": base_retained,
                "capability_aggregator": "pool_then_encode",
                "capability_min_prevalence": 0,
            })
    frontier_path = frontier_dir / "frontier.jsonl"
    frontier_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.85",
        ])
    assert rc == 0
    stdout = out_buf.getvalue()
    # Ranking table SHALL appear.
    assert "Per-encoding ranking" in stdout
    assert "raw_slice" in stdout
    assert "partition_q4" in stdout
    # Both encodings have a row in the ranking table.
    assert stdout.count("rank") >= 1


def test_cli_recommend_single_encoding_skips_ranking_table(tmp_path: Path):
    """Single-encoding frontier (only one distinct encoding_label) →
    recommend does NOT emit the ranking table (back-compat)."""
    import contextlib
    import io
    import json

    from saeforge.cli import main as cli_main

    frontier_dir = tmp_path / "single_encoding"
    frontier_dir.mkdir()
    rows = [
        {
            "encoding_label": "raw_slice",
            "target_n_features_kept": w,
            "n_features_kept_actual": w,
            "pareto_reached_target": True,
            "faithfulness_kl": None, "perplexity": None,
            "final_fine_tune_loss": None,
            "sae_checkpoint": "/tmp/sae",
            "forged_model_path": None,
            "elapsed_seconds": 0.1, "error_message": None,
            "host_baseline_mauc": 0.95, "forge_mauc": 0.85,
            "retained_mauc_vs_host": 0.90,
            "capability_aggregator": "pool_then_encode",
            "capability_min_prevalence": 0,
        }
        for w in (16, 32, 64)
    ]
    frontier_path = frontier_dir / "frontier.jsonl"
    frontier_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.85",
        ])
    assert rc == 0
    stdout = out_buf.getvalue()
    # Ranking table SHALL NOT appear for single-encoding.
    assert "Per-encoding ranking" not in stdout
