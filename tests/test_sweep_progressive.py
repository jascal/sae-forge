"""Tests for sweep_pareto_capability_progressive + helper functions.

Three suites:
  1. Pure helpers (no forge needed): plateau identification, neighbour
     expansion, convergence detector.
  2. ParetoFrontierRow.stage field round-trip + back-compat with v0.8.x
     frontier files (no stage field).
  3. End-to-end progressive sweep against the synthetic ESM fixture
     from test_sweep_pareto_capability.py — convergence detection
     fires cleanly on a well-behaved fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Suite 1: pure helper functions
# ---------------------------------------------------------------------------


def _make_row(width: int, retained: float, stage: int | None = None):
    """Build a minimal ParetoFrontierRow with capability fields
    populated and the rest at sensible defaults."""
    from saeforge.sweep import ParetoFrontierRow

    return ParetoFrontierRow(
        encoding_label="x",
        target_n_features_kept=width,
        n_features_kept_actual=width,
        pareto_reached_target=True,
        faithfulness_kl=None,
        perplexity=None,
        final_fine_tune_loss=None,
        sae_checkpoint="/tmp/sae",
        forged_model_path=None,
        elapsed_seconds=0.1,
        error_message=None,
        host_baseline_mauc=0.8,
        forge_mauc=0.8 * retained,
        retained_mauc_vs_host=retained,
        capability_aggregator="pool_then_encode",
        capability_min_prevalence=0,
        stage=stage,
    )


def test_plateau_identification_within_tolerance():
    """Widths within plateau_tolerance of peak form the plateau."""
    from saeforge.sweep_capability_progressive import _identify_plateau

    rows = [
        _make_row(8, 0.85),
        _make_row(16, 1.00),  # peak
        _make_row(32, 0.995),  # within 0.01 of peak
        _make_row(64, 0.95),   # outside 0.01 of peak
    ]
    plateau, peak_retained, peak_n = _identify_plateau(
        rows, plateau_tolerance=0.01, min_plateau_widths=2,
    )
    assert plateau == (16, 32)
    assert peak_retained == pytest.approx(1.0)
    assert peak_n == 16


def test_plateau_min_floor_widens_tolerance():
    """min_plateau_widths floor forces widening when tolerance is too
    tight."""
    from saeforge.sweep_capability_progressive import _identify_plateau

    rows = [
        _make_row(8, 0.80),
        _make_row(16, 1.00),  # peak
        _make_row(32, 0.90),  # outside 0.01 plateau
        _make_row(64, 0.95),
    ]
    plateau, _, _ = _identify_plateau(
        rows, plateau_tolerance=0.01, min_plateau_widths=3,
    )
    # Should widen to top-3: (16, 64, 32) sorted → (16, 32, 64)
    assert plateau == (16, 32, 64)


def test_plateau_handles_empty_success_set():
    """All-failed stage returns empty plateau, sentinel peak_n."""
    from saeforge.sweep import ParetoFrontierRow
    from saeforge.sweep_capability_progressive import _identify_plateau

    failed = ParetoFrontierRow(
        encoding_label="x", target_n_features_kept=8,
        n_features_kept_actual=None, pareto_reached_target=None,
        faithfulness_kl=None, perplexity=None,
        final_fine_tune_loss=None, sae_checkpoint="/tmp/sae",
        forged_model_path=None, elapsed_seconds=0.0,
        error_message="forge crashed",
    )
    plateau, peak_retained, peak_n = _identify_plateau(
        [failed], plateau_tolerance=0.01, min_plateau_widths=1,
    )
    assert plateau == ()
    assert np.isnan(peak_retained)
    assert peak_n == -1


def test_neighbour_expansion_picks_immediate_candidates():
    """Plateau {16, 32, 64} on candidates {4, 8, 16, 32, 64, 128, 256}
    → next-stage active {8, 16, 32, 64, 128}."""
    from saeforge.sweep_capability_progressive import _expand_neighbours

    candidates = [4, 8, 16, 32, 64, 128, 256]
    actives = _expand_neighbours([16, 32, 64], candidates)
    assert actives == (8, 16, 32, 64, 128)


def test_neighbour_expansion_handles_edges():
    """Plateau-member at the start / end of candidates has no
    out-of-range neighbour."""
    from saeforge.sweep_capability_progressive import _expand_neighbours

    candidates = [4, 8, 16, 32]
    # Plateau at the leftmost candidate: no left neighbour.
    assert _expand_neighbours([4], candidates) == (4, 8)
    # Plateau at the rightmost candidate: no right neighbour.
    assert _expand_neighbours([32], candidates) == (16, 32)


def test_neighbour_expansion_dedups():
    """Adjacent plateau members share neighbours; result is sorted
    unique."""
    from saeforge.sweep_capability_progressive import _expand_neighbours

    candidates = [4, 8, 16, 32, 64]
    actives = _expand_neighbours([16, 32], candidates)
    # 16's neighbours: 8, 32. 32's neighbours: 16, 64.
    # Union: {8, 16, 32, 64}.
    assert actives == (8, 16, 32, 64)


def test_convergence_detector_fires_on_stable_run():
    """K-in-a-row stable stages → converged."""
    from saeforge.sweep_capability_progressive import (
        ConvergenceTrajectoryEntry,
        _detect_convergence,
    )

    traj = [
        ConvergenceTrajectoryEntry(0, 10, 16, 0.95, 3, 2, False),
        ConvergenceTrajectoryEntry(1, 50, 16, 0.96, 3, 2, False),
        ConvergenceTrajectoryEntry(2, 200, 16, 0.96, 3, 2, False),
    ]
    converged, stages = _detect_convergence(
        traj, convergence_n_stages=2, retained_mauc_tolerance=0.01,
    )
    assert converged is True
    # Trailing stable run: stages 1 and 2 both non-shifted, AUC stable.
    assert stages >= 2


def test_convergence_detector_rejects_shift():
    """argmin shifts between consecutive stages → NOT converged."""
    from saeforge.sweep_capability_progressive import (
        ConvergenceTrajectoryEntry,
        _detect_convergence,
    )

    traj = [
        ConvergenceTrajectoryEntry(0, 10, 16, 0.95, 3, 2, False),
        ConvergenceTrajectoryEntry(1, 50, 32, 0.96, 3, 2, True),  # shifted
        ConvergenceTrajectoryEntry(2, 200, 32, 0.96, 3, 2, False),
    ]
    converged, stages = _detect_convergence(
        traj, convergence_n_stages=2, retained_mauc_tolerance=0.01,
    )
    # Trailing two stages are 1 and 2; stage 1 shifted. Not converged.
    assert converged is False


def test_convergence_detector_rejects_auc_drift():
    """Same width but retained_mauc drift beyond tolerance → NOT
    converged."""
    from saeforge.sweep_capability_progressive import (
        ConvergenceTrajectoryEntry,
        _detect_convergence,
    )

    traj = [
        ConvergenceTrajectoryEntry(0, 10, 16, 0.95, 3, 2, False),
        ConvergenceTrajectoryEntry(1, 50, 16, 0.97, 3, 2, False),  # +0.02
    ]
    converged, _ = _detect_convergence(
        traj, convergence_n_stages=2, retained_mauc_tolerance=0.005,
    )
    assert converged is False


def test_convergence_detector_too_few_stages():
    """Trajectory shorter than convergence_n_stages → never converged."""
    from saeforge.sweep_capability_progressive import (
        ConvergenceTrajectoryEntry,
        _detect_convergence,
    )

    traj = [
        ConvergenceTrajectoryEntry(0, 10, 16, 0.95, 3, 2, False),
    ]
    converged, stages = _detect_convergence(
        traj, convergence_n_stages=2, retained_mauc_tolerance=0.01,
    )
    assert converged is False
    assert stages == 1


# ---------------------------------------------------------------------------
# Suite 2: ParetoFrontierRow.stage round-trip
# ---------------------------------------------------------------------------


def test_row_stage_field_round_trip():
    """Populated stage survives to_json_dict / from_json_dict."""
    from saeforge.sweep import ParetoFrontierRow

    row = _make_row(16, 0.95, stage=2)
    d = row.to_json_dict()
    assert d["stage"] == 2
    back = ParetoFrontierRow.from_json_dict(d)
    assert back.stage == 2


def test_row_stage_omitted_when_none():
    """Default stage=None is omitted from JSON — v0.8.x back-compat."""
    from saeforge.sweep import ParetoFrontierRow

    row = ParetoFrontierRow(
        encoding_label="x", target_n_features_kept=1,
        n_features_kept_actual=1, pareto_reached_target=True,
        faithfulness_kl=0.5, perplexity=1.0,
        final_fine_tune_loss=None, sae_checkpoint="/tmp",
        forged_model_path=None, elapsed_seconds=0.1,
        error_message=None,
    )
    d = row.to_json_dict()
    assert "stage" not in d


def test_row_stage_validation_rejects_negative():
    from saeforge.sweep import ParetoFrontierRow

    with pytest.raises(ValueError, match="stage"):
        ParetoFrontierRow(
            encoding_label="x", target_n_features_kept=1,
            n_features_kept_actual=1, pareto_reached_target=True,
            faithfulness_kl=0.5, perplexity=1.0,
            final_fine_tune_loss=None, sae_checkpoint="/tmp",
            forged_model_path=None, elapsed_seconds=0.1,
            error_message=None,
            stage=-1,
        )


def test_v08x_row_loads_with_stage_none():
    """A pre-change v0.8.x payload (no stage key) loads with stage=None."""
    from saeforge.sweep import ParetoFrontierRow

    payload = {
        "encoding_label": "Rung5", "target_n_features_kept": 16,
        "n_features_kept_actual": 16, "pareto_reached_target": True,
        "faithfulness_kl": 0.5, "perplexity": 1.6,
        "final_fine_tune_loss": None, "sae_checkpoint": "/tmp/sae",
        "forged_model_path": None, "elapsed_seconds": 4.2,
        "error_message": None,
    }
    row = ParetoFrontierRow.from_json_dict(payload)
    assert row.stage is None


# ---------------------------------------------------------------------------
# Suite 3: end-to-end progressive sweep (smoke)
# ---------------------------------------------------------------------------


@pytest.fixture
def _tiny_host_model_id(tmp_path: Path):
    """Tiny ESM host saved to a temp dir; reused across progressive
    stages."""
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


def _build_bio_sae_fixture(tmp_path: Path, *, n_proteins=8, d_model=32, sae_width=32):
    """Same fixture builder as test_sweep_pareto_capability.py.

    Skips cleanly when pandas isn't installed — only the suite-3
    end-to-end tests call this helper; the pure-helper + row-schema
    tests in this module don't need pandas and continue to run.
    """
    pytest.importorskip("pandas")
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


def test_progressive_sweep_end_to_end(tmp_path: Path, _tiny_host_model_id):
    """End-to-end: 2-stage schedule on a synthetic fixture, asserts
    history is populated + recommendation emitted."""
    from saeforge import sweep_pareto_capability_progressive
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=8, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    history = sweep_pareto_capability_progressive(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        candidate_widths=[4, 8, 16, 32],
        n_proteins_schedule=[4, 8],
        output_dir=tmp_path / "progressive_out",
        device="cpu",
    )
    assert len(history.stages) >= 1
    assert history.recommendation.target_n_features_kept >= 0
    # The frontier.jsonl SHALL exist and carry stage tags.
    frontier = tmp_path / "progressive_out" / "frontier.jsonl"
    assert frontier.exists()
    import json
    lines = frontier.read_text().splitlines()
    assert lines, "frontier.jsonl is empty"
    parsed = [json.loads(line) for line in lines]
    assert all("stage" in entry for entry in parsed), (
        "every progressive frontier row SHALL carry the stage field"
    )

    # progressive_summary.json SHALL be on disk with the expected shape.
    summary_path = tmp_path / "progressive_out" / "progressive_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "stages" in summary and "recommendation" in summary
    assert "convergence_trajectory" in summary["recommendation"]


def test_progressive_single_element_schedule_is_single_shot(
    tmp_path: Path, _tiny_host_model_id,
):
    """Single-element n_proteins_schedule degenerates to single-shot
    with converged=True by definition."""
    from saeforge import sweep_pareto_capability_progressive
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=8, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    history = sweep_pareto_capability_progressive(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        candidate_widths=[4, 8, 16, 32],
        n_proteins_schedule=[8],  # single element
        output_dir=tmp_path / "progressive_single",
        device="cpu",
    )
    assert len(history.stages) == 1
    assert history.recommendation.converged is True, (
        "single-element schedule SHALL declare converged=True by "
        "definition"
    )


def test_progressive_validates_inputs(tmp_path: Path):
    """Schedule must be monotone + non-empty; widths non-empty; schedule
    can't exceed dataset size."""
    from saeforge import sweep_pareto_capability_progressive
    from saeforge.datasets import CapabilityDataset

    # Minimal dataset; the validation we're testing fires before
    # anything actually runs.
    ds = CapabilityDataset(
        sequences=["MAKVITDR"] * 3,
        labels=np.zeros((3, 4), dtype=np.uint8),
        encoder=lambda x: x[:, :4],
        tokenizer_id="any",
        feed="pooled",
    )
    with pytest.raises(ValueError, match="non-empty"):
        sweep_pareto_capability_progressive(
            sae_checkpoint="/tmp/sae",
            host_model_id="any",
            dataset=ds,
            candidate_widths=[],
            n_proteins_schedule=[3],
            output_dir=tmp_path / "out",
        )
    with pytest.raises(ValueError, match="monotone"):
        sweep_pareto_capability_progressive(
            sae_checkpoint="/tmp/sae",
            host_model_id="any",
            dataset=ds,
            candidate_widths=[4, 8],
            n_proteins_schedule=[3, 2],  # non-monotone
            output_dir=tmp_path / "out",
        )
    with pytest.raises(ValueError, match="exceeds the dataset's"):
        sweep_pareto_capability_progressive(
            sae_checkpoint="/tmp/sae",
            host_model_id="any",
            dataset=ds,
            candidate_widths=[4],
            n_proteins_schedule=[10],  # > 3 sequences
            output_dir=tmp_path / "out",
        )
