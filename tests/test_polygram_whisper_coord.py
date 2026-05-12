"""Polygram coordination smoke (§11.1 of forge-whisper-encoder).

Verifies that a polygram-format compressed checkpoint at Whisper-
encoder dimensions loads cleanly through
``FeatureBasis.from_polygram_checkpoint`` and feeds the
``WhisperEncoderAdapter`` end-to-end. The loader is architecture-
agnostic by design — it reads ``W_dec`` + the companion report and
emits a ``FeatureBasis`` regardless of host architecture — but the
five-SAE polygram panel that motivated v0.2's ``uniform-sphere``
profile includes Whisper-tiny enc.b2 (6,144 × 384) and Whisper-large-v1
enc.b16 (20,480 × 1,280) at sizes the pre-v0.2 GPT-2-small calibration
never saw. This test pins the loader on a synthetic checkpoint
shaped like the smaller of those two SAEs, then runs it through the
adapter against the ``tiny_synthetic_whisper`` host fixture.

We do not download a real polygram-compressed Whisper SAE here:

- The bf16 dtype path that Llama-Scope inspect surfaced is fixed in
  polygram 0.2.0 already; modern TopK SAEs (Whisper included) hit
  exactly that code path, so a synthetic fp32 checkpoint exercises
  the file-format contract without depending on a 20MB+ download.
- The Intel Mac dev hardware is the personal-project compute floor;
  shipping real-checkpoint validation in CI would couple this
  repo's tests to HF availability + network.
- A future real-checkpoint integration smoke is tracked as a
  follow-up under §11 (a manual one-shot, not a CI test).
"""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def whisper_shaped_polygram_checkpoint(tmp_path):
    """Synthesize a polygram-format compressed SAE at Whisper-encoder dims.

    Shape: 256 features over a 64-d residual stream. Matches the
    ``tiny_synthetic_whisper`` fixture's ``d_model=64`` so the
    forge pipeline can chain through the adapter without a basis-
    shape mismatch. The 256-feature width gives ~4x over-completeness
    — well into the regime where ``uniform-sphere`` is the
    polygram-recommended profile (see polygram's
    ``docs/research/sae-geometry-regimes.md``).

    Cluster {2, 5} collapses onto representative 2 with merged_norm
    1.5; row 5 is zeroed. Cluster {7, 11} zeroes onto representative
    7 with no merged_norm. Two zeroed rows lets the loader exercise
    its kept-id detection on a Whisper-shape basis.
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(42)
    n_total = 256
    d_model = 64
    W_dec = rng.standard_normal((n_total, d_model)).astype(np.float32)
    W_dec[5] = 0.0
    W_dec[11] = 0.0
    rep_norm = np.linalg.norm(W_dec[2])
    if rep_norm > 0:
        W_dec[2] *= 1.5 / rep_norm

    checkpoint = tmp_path / "whisper_sae.compressed.safetensors"
    save_file({"W_dec": W_dec}, str(checkpoint))

    report = {
        "schema_version": 1,
        "source_checkpoint": "whisper_sae.safetensors",
        "source_checkpoint_sha256": "deadbeef",
        "output_checkpoint": str(checkpoint),
        "output_checkpoint_sha256": "feedface",
        "validation_report_dictionary_name": "synthetic_whisper",
        "validation_report_schema_version": 1,
        "strategy": "merge",
        "feature_ids": [2, 5, 7, 11],
        "clusters": [
            {
                "cluster_id": 0,
                "members": [2, 5],
                "representative": 2,
                "zeroed": [5],
                "cluster_norm_mean": 1.4,
                "cluster_norm_std": 0.1,
                "merged_norm": 1.5,
            },
            {
                "cluster_id": 1,
                "members": [7, 11],
                "representative": 7,
                "zeroed": [11],
                "cluster_norm_mean": 1.0,
                "cluster_norm_std": 0.05,
                "merged_norm": None,
            },
        ],
        "n_features_zeroed": 2,
        "n_features_kept": n_total - 2,
        "n_clusters": 2,
        "scale_compression_ratio": 0.92,
        # Polygram 0.2.0 stamps the geometric profile name on the
        # report; sae-forge does not consume the profile (it's a
        # polygram-side compression knob) but we record it here so
        # the round-trip metadata reflects the recommended setting.
        "profile": "uniform-sphere",
    }
    report_path = tmp_path / "whisper_sae.compressed_compression_report.json"
    report_path.write_text(json.dumps(report))

    return {
        "checkpoint": checkpoint,
        "report_path": report_path,
        "report": report,
        "n_total": n_total,
        "d_model": d_model,
    }


# ---------------------------------------------------------------------------
# Loader → FeatureBasis on Whisper-shape dims
# ---------------------------------------------------------------------------


class TestLoader:
    def test_from_polygram_checkpoint_loads_whisper_shape(
        self, whisper_shaped_polygram_checkpoint
    ):
        from saeforge.basis import FeatureBasis

        c = whisper_shaped_polygram_checkpoint
        basis = FeatureBasis.from_polygram_checkpoint(c["checkpoint"])

        # Expected kept count: n_total minus the 2 zeroed rows.
        assert basis.n_features == c["n_total"] - 2
        assert basis.d_model == c["d_model"]
        # The basis preserves the kept ids in their original positions.
        zeroed = {5, 11}
        expected_kept_ids = np.array(
            [i for i in range(c["n_total"]) if i not in zeroed],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(basis.kept_ids, expected_kept_ids)

    def test_loader_records_profile_via_metadata_passthrough(
        self, whisper_shaped_polygram_checkpoint
    ):
        """Polygram 0.2.0 stamps ``profile`` on the report. sae-forge
        does not gate any behavior on the profile (it's a polygram-side
        knob), but the loader's ``metadata`` dict should not strip it —
        downstream consumers (logging, audit) can read it back from
        the on-disk report directly."""
        from saeforge.basis import FeatureBasis

        c = whisper_shaped_polygram_checkpoint
        basis = FeatureBasis.from_polygram_checkpoint(c["checkpoint"])
        # The basis itself doesn't carry the profile — sae-forge has no
        # contract with it. But the report path should survive so
        # consumers can inspect it.
        assert basis.metadata["report_path"] == str(c["report_path"])
        # And the on-disk report still names the recommended profile.
        report = json.loads(open(c["report_path"]).read())
        assert report["profile"] == "uniform-sphere"


# ---------------------------------------------------------------------------
# End-to-end: polygram checkpoint → FeatureBasis → forged Whisper encoder
# ---------------------------------------------------------------------------


class TestEndToEndForge:
    def test_polygram_basis_drives_whisper_forge(
        self, whisper_shaped_polygram_checkpoint, tiny_synthetic_whisper
    ):
        """A FeatureBasis loaded from a polygram-format Whisper-shape
        checkpoint feeds the WhisperEncoderAdapter walk + forge
        pipeline without shape mismatches."""
        import torch

        from saeforge.adapters import adapter_for
        from saeforge.audio_data import synthetic_mel_features
        from saeforge.basis import FeatureBasis
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        basis = FeatureBasis.from_polygram_checkpoint(
            whisper_shaped_polygram_checkpoint["checkpoint"]
        )
        # auto scale_boost for over-complete bases (n_features=254 > d_model=64).
        projector = SubspaceProjector(basis, scale_boost="auto")

        adapter = adapter_for(tiny_synthetic_whisper)
        assert adapter.family == "whisper_encoder"
        walk = adapter.walk(tiny_synthetic_whisper, projector)
        config = adapter.build_native_config(
            tiny_synthetic_whisper, basis.n_features
        )
        forged = NativeModel.from_projected_weights(config, walk)

        # Forward-shape sanity through the forged module.
        mel = synthetic_mel_features(0, n_frames=200)
        with torch.no_grad():
            out = forged.torch_module(mel)
        assert out.shape == (1, 100, basis.n_features)
        assert torch.isfinite(out).all().item()

    def test_cosine_eval_runs_on_polygram_basis(
        self, whisper_shaped_polygram_checkpoint, tiny_synthetic_whisper
    ):
        """End-to-end check that cosine_faithfulness runs against a
        polygram-loaded basis without crashing. The score value isn't
        meaningful for synthetic weights; assert only the [0, 1]
        contract holds."""
        import torch

        from saeforge.adapters import adapter_for
        from saeforge.audio_data import synthetic_mel_features
        from saeforge.audio_eval import cosine_faithfulness
        from saeforge.basis import FeatureBasis
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        basis = FeatureBasis.from_polygram_checkpoint(
            whisper_shaped_polygram_checkpoint["checkpoint"]
        )
        projector = SubspaceProjector(basis, scale_boost="auto")
        adapter = adapter_for(tiny_synthetic_whisper)
        walk = adapter.walk(tiny_synthetic_whisper, projector)
        config = adapter.build_native_config(
            tiny_synthetic_whisper, basis.n_features
        )
        forged = NativeModel.from_projected_weights(config, walk)

        mel = synthetic_mel_features(0)
        with torch.no_grad():
            score = cosine_faithfulness(forged, tiny_synthetic_whisper, mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
