"""§6.4 acceptance test: ``load_and_scan`` preserves v0.2 transitions_log shape.

The hierarchical-fsm refactor collapses the v0.2 ``loaded`` +
``activations_scanned`` two-state pair into a single
``RefineMachine.entering`` state with a composed ``load_and_scan``
action. The composed helper MUST log both inner action names in order
so ``transitions_log`` consumers see the same sequence as v0.2.

This test pins that contract under both gating modes:

- ``protect_top_k == 0`` (v0.2 default): scan_activations is a true
  pass-through; both inner names still appear in the log; protected
  set stays empty.
- ``protect_top_k > 0``: scan_activations runs the protected-set
  selection; both inner names appear; ``protected_features`` is
  non-empty after.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _seed_sae_checkpoint(tmp_path: Path) -> Path:
    """Build a tiny synthetic SAE checkpoint that ``load_sae_and_corpus`` accepts.

    The action only checks that ``ctx["sae_checkpoint"]`` is an existing
    file; it does NOT parse the contents. We just need a real path on
    disk pointing to *something*.
    """
    sae_path = tmp_path / "tiny.safetensors"
    sae_path.write_bytes(b"\x00")  # any nonzero file
    return sae_path


def _seed_polygram_basis(tmp_path: Path) -> Path:
    """Build a minimal polygram-compressed checkpoint that ``FeatureBasis.from_polygram_checkpoint`` loads.

    Reuses the same shape as ``tests/conftest.py::synthetic_compressed_sae``
    but inlined here so the test does not depend on that fixture's
    cleanup model.
    """
    from safetensors.numpy import save_file
    import json

    rng = np.random.default_rng(0)
    n_total, d_model = 4, 8
    W_dec = rng.standard_normal((n_total, d_model)).astype(np.float32)
    W_dec[3] = 0.0  # one zeroed row to mimic compression bookkeeping
    sae_path = tmp_path / "compressed.safetensors"
    save_file(
        {
            "decoder.weight": W_dec,
            "encoder.weight": W_dec.T,
            "encoder.bias": np.zeros(n_total, dtype=np.float32),
        },
        str(sae_path),
    )
    report_path = sae_path.with_suffix(".compression_report.json")
    report = {
        "schema_version": "0.1",
        "kept": [0, 1, 2],
        "dropped": [3],
        "clusters": [
            {"representative": 3, "members": [3], "strategy": "zero"},
        ],
        "merged_norms": {},
        "metrics": {},
    }
    report_path.write_text(json.dumps(report))
    return sae_path


def test_load_and_scan_passthrough_when_protect_top_k_zero(tmp_path):
    """Default knobs: both inner action names log; protected set stays empty."""
    pytest.importorskip("orca_runtime_python")
    from saeforge.actions import load_and_scan

    sae = _seed_sae_checkpoint(tmp_path)
    ctx = {
        "sae_checkpoint": str(sae),
        "protect_top_k": 0,
        "_machine_path": "stream/refine",
        "transitions_log": [],
    }

    delta = load_and_scan(ctx, None)

    actions = [entry["action"] for entry in ctx["transitions_log"]]
    assert actions == ["load_sae_and_corpus", "scan_activations"]
    # Pass-through delta from scan_activations under protect_top_k=0
    # is empty; load_sae_and_corpus writes current_sae_path.
    assert delta == {"current_sae_path": str(sae)}
    assert ctx.get("protected_features", []) == []
    # machine_path is recorded on every log entry.
    assert all(e["machine_path"] == "stream/refine" for e in ctx["transitions_log"])


def test_load_and_scan_runs_protected_set_when_top_k_positive(tmp_path):
    """``protect_top_k > 0``: scan_activations populates ``protected_features``."""
    pytest.importorskip("orca_runtime_python")
    pytest.importorskip("torch")
    from saeforge.actions import load_and_scan

    sae = _seed_polygram_basis(tmp_path)
    ctx = {
        "sae_checkpoint": str(sae),
        "protect_top_k": 2,
        "protect_score": "mean_act",
        "_machine_path": "stream/refine",
        "transitions_log": [],
    }

    load_and_scan(ctx, None)

    actions = [entry["action"] for entry in ctx["transitions_log"]]
    assert actions == ["load_sae_and_corpus", "scan_activations"]
    assert len(ctx.get("protected_features", [])) == 2
    # feature_usage gets populated by the v0.2 scorer.
    assert len(ctx.get("feature_usage", [])) > 0
