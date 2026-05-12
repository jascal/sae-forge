"""Tests for ``saeforge.audio_data.synthetic_mel_features`` (§7 of the
forge-whisper-encoder change).

The synthesizer is the test/example replacement for the
``[audio]``-extra-gated ``librosa`` / ``WhisperFeatureExtractor``
pipeline. These tests check the shape and dtype contract, determinism
under seeding, batch independence, and that the output actually flows
through the host Whisper encoder + the forged encoder without
shape/dtype mismatches.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Shape / dtype contract
# ---------------------------------------------------------------------------


class TestShape:
    def test_default_shape_matches_whisper_input(self):
        from saeforge.audio_data import synthetic_mel_features

        out = synthetic_mel_features(seed=0)
        assert tuple(out.shape) == (1, 80, 3000)

    def test_returns_float32_torch_tensor(self):
        import torch

        from saeforge.audio_data import synthetic_mel_features

        out = synthetic_mel_features(seed=0)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32

    def test_custom_shape_kwargs_honored(self):
        from saeforge.audio_data import synthetic_mel_features

        out = synthetic_mel_features(seed=1, batch=4, n_mels=40, n_frames=600)
        assert tuple(out.shape) == (4, 40, 600)


# ---------------------------------------------------------------------------
# Determinism + seed semantics
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_output(self):
        import torch

        from saeforge.audio_data import synthetic_mel_features

        a = synthetic_mel_features(seed=42)
        b = synthetic_mel_features(seed=42)
        assert torch.equal(a, b)

    def test_different_seed_different_noise(self):
        import torch

        from saeforge.audio_data import synthetic_mel_features

        a = synthetic_mel_features(seed=0)
        b = synthetic_mel_features(seed=1)
        # The sweep envelope is seed-independent, so they're not
        # orthogonal — but the noise is, so they're not bit-equal.
        assert not torch.equal(a, b)


class TestBatchIndependence:
    def test_batch_elements_decorrelated(self):
        from saeforge.audio_data import synthetic_mel_features

        out = synthetic_mel_features(seed=0, batch=2)
        # Per-batch sub-seeds → the additive noise differs across the
        # batch axis even though the sweep envelope is shared. Mean
        # absolute difference should be appreciable.
        diff = (out[0] - out[1]).abs().mean().item()
        assert diff > 0.01

    def test_batch_one_matches_no_batch_axis_default(self):
        """A scalar at seed S, batch=1 should produce the same first-
        element noise as batch=2 at the same seed."""
        from saeforge.audio_data import synthetic_mel_features

        b1 = synthetic_mel_features(seed=7, batch=1)
        b2 = synthetic_mel_features(seed=7, batch=2)
        # First batch element is the same — the sub-seed is drawn
        # deterministically in batch order.
        assert (b1[0] - b2[0]).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------


class TestMagnitude:
    def test_values_bounded(self):
        from saeforge.audio_data import synthetic_mel_features

        out = synthetic_mel_features(seed=0).numpy()
        # sine envelope in [-1, 1] + 0.05-std Gaussian noise; should
        # stay well within [-1.5, 1.5] across a 3000-frame draw.
        assert out.min() > -1.5
        assert out.max() < 1.5

    def test_no_nans_or_infs(self):
        from saeforge.audio_data import synthetic_mel_features

        out = synthetic_mel_features(seed=0).numpy()
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Integration: synthesizer + WhisperModel + ForgedWhisperEncoder
# ---------------------------------------------------------------------------


class TestWhisperIntegration:
    def test_feeds_host_whisper_encoder(self, tiny_synthetic_whisper):
        """Confirms §7.3: the tiny_synthetic_whisper fixture has an
        accessible ``.encoder`` that consumes synthetic mel features
        and returns a BaseModelOutput-like object."""
        import torch

        from saeforge.audio_data import synthetic_mel_features

        mel = synthetic_mel_features(seed=0)
        with torch.no_grad():
            out = tiny_synthetic_whisper.encoder(mel)
        # last_hidden_state shape: (batch, n_frames // 2, d_model)
        assert out.last_hidden_state.shape == (1, 1500, 64)

    def test_feeds_forged_whisper_encoder(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        import torch

        from saeforge.adapters import adapter_for
        from saeforge.audio_data import synthetic_mel_features
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(feature_basis_64_to_32)
        adapter = adapter_for(tiny_synthetic_whisper)
        walk = adapter.walk(tiny_synthetic_whisper, projector)
        config = adapter.build_native_config(
            tiny_synthetic_whisper, feature_basis_64_to_32.n_features
        )
        forged = NativeModel.from_projected_weights(config, walk)

        mel = synthetic_mel_features(seed=0)
        with torch.no_grad():
            out = forged.torch_module(mel)
        assert out.shape == (1, 1500, feature_basis_64_to_32.n_features)
        assert torch.isfinite(out).all().item()


# ---------------------------------------------------------------------------
# tiny_synthetic_whisper fixture sanity (§7.3)
# ---------------------------------------------------------------------------


class TestFixture:
    def test_fixture_is_whisper_model_with_encoder(
        self, tiny_synthetic_whisper
    ):
        # Plain attribute checks — no assertions about WhisperModel
        # subclass identity (transformers churns class names across
        # versions; the duck-typed contract is what matters).
        assert hasattr(tiny_synthetic_whisper, "encoder")
        assert hasattr(tiny_synthetic_whisper.encoder, "conv1")
        assert hasattr(tiny_synthetic_whisper.encoder, "conv2")
        assert hasattr(tiny_synthetic_whisper.encoder, "embed_positions")
        assert hasattr(tiny_synthetic_whisper.encoder, "layers")
        assert hasattr(tiny_synthetic_whisper.encoder, "layer_norm")
        assert len(tiny_synthetic_whisper.encoder.layers) == 2
