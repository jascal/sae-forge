"""Real-action tests for the FSM compress / regrow / fine-tune stages.

These exercise the gating-on-input behaviour: when the right ctx fields
are present, the actions actually run polygram / torch training; when
absent, they remain pass-throughs (covered in test_forge_outer_loop_fsm.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_uncompressed_sae(path: Path, n_features: int, d_model: int) -> None:
    """Build an 8-feature, 16-d uncompressed SAE checkpoint for compression tests."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    save_file(
        {
            "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32),
            "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
            "b_enc": np.zeros((n_features,), dtype=np.float32),
            "b_dec": np.zeros((d_model,), dtype=np.float32),
        },
        str(path),
    )


def test_compress_action_runs_polygram_when_validation_report_provided(
    tmp_path, synthetic_validation_report, tiny_gpt2
):
    pytest.importorskip("polygram")
    pytest.importorskip("orca_runtime_python")

    from saeforge.actions import compress_with_polygram

    sae_path = tmp_path / "uncompressed.safetensors"
    _write_uncompressed_sae(sae_path, n_features=8, d_model=16)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    ctx = {
        "current_sae_path": str(sae_path),
        "output_dir": str(output_dir),
        "validation_report_path": str(synthetic_validation_report["path"]),
        "compression_strategy": "merge",
        "rep_selection": "n_fires",
        "transitions_log": [],
    }
    delta = compress_with_polygram(ctx, None)
    ctx.update(delta)

    assert Path(ctx["compressed_sae_path"]).is_file()
    assert Path(ctx["compression_report_path"]).is_file()
    assert ctx["current_feature_count"] > 0
    assert ctx["current_feature_count"] < 8  # something got compressed
    last_log = ctx["transitions_log"][-1]
    assert last_log["mode"] == "polygram"


def test_compress_action_passes_through_without_validation_report(tmp_path):
    from saeforge.actions import compress_with_polygram

    ctx = {
        "current_sae_path": "/some/sae.safetensors",
        "output_dir": str(tmp_path),
        "validation_report_path": None,
        "transitions_log": [],
    }
    delta = compress_with_polygram(ctx, None)
    assert delta["compressed_sae_path"] == "/some/sae.safetensors"
    assert ctx["transitions_log"][-1]["mode"] == "passthrough"


def test_fine_tune_action_reduces_loss(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")

    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        finetune_steps=4,
        finetune_lr=1e-2,
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    finetune_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))
    result = pipeline.run_synthetic(
        tiny_gpt2,
        tmp_path / "ft",
        eval_input_ids=eval_input_ids,
        finetune_input_ids=finetune_input_ids,
    )
    losses = None
    for entry in result.extras["transitions_log"]:
        if entry.get("action") == "fine_tune_model" and entry.get("mode") == "trained":
            losses = (entry["loss_first"], entry["loss_last"])
            break
    assert losses is not None, "fine_tune_model should have run in trained mode"
    assert losses[1] <= losses[0] + 1e-3, f"loss should not regress: {losses}"


def test_fine_tune_action_passes_through_without_input_ids(tmp_path, tiny_gpt2, tiny_synthetic_basis):
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")

    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "noft", eval_input_ids=eval_input_ids)

    finetune_entry = next(
        e for e in result.extras["transitions_log"] if e.get("action") == "fine_tune_model"
    )
    assert finetune_entry["mode"] == "passthrough"


def test_full_compress_then_forge_via_fsm(
    tmp_path, synthetic_validation_report, tiny_gpt2
):
    """End-to-end: uncompressed SAE → FSM compresses → projects → forges → eval."""
    pytest.importorskip("polygram")
    pytest.importorskip("orca_runtime_python")
    pytest.importorskip("torch")

    import torch

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    sae_path = tmp_path / "uncompressed.safetensors"
    _write_uncompressed_sae(sae_path, n_features=8, d_model=16)

    # Provide an "any" basis to satisfy ForgePipeline's __init__ — the FSM
    # will reload from disk via the compress action, not use this basis.
    rng = np.random.default_rng(1)
    placeholder_W = rng.standard_normal((8, 16)).astype(np.float64)
    placeholder = FeatureBasis(
        kept_ids=np.arange(8),
        W_dec=placeholder_W,
        merged_norms=np.linalg.norm(placeholder_W, axis=1),
        original_norms=np.linalg.norm(placeholder_W, axis=1),
    )
    projector = SubspaceProjector(placeholder)
    pipeline = ForgePipeline(
        basis=placeholder,
        projector=projector,
        orchestrator="fsm",
        validation_report_path=str(synthetic_validation_report["path"]),
        compression_strategy="merge",
        rep_selection="n_fires",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        tiny_gpt2,
        tmp_path / "full",
        eval_input_ids=eval_input_ids,
        sae_checkpoint=sae_path,
    )

    summary = json.loads((tmp_path / "full" / "forge_result.json").read_text())
    assert summary["compress_mode"] == "polygram"
    assert summary["n_features"] is not None and summary["n_features"] < 8
    assert (tmp_path / "full" / "compressed.safetensors").is_file()
    assert (tmp_path / "full" / "compressed_compression_report.json").is_file()
    assert (tmp_path / "full" / "forged" / "model.safetensors").is_file()
    assert result.faithfulness_kl is not None
