"""Synthetic mel-spectrogram fixtures for Whisper forge tests + examples.

The real-audio path needs a feature extractor that turns ``.wav`` /
``.flac`` files into the ``(batch, n_mels=80, n_frames=3000)``
log-mel-spectrogram tensors Whisper consumes. That pipeline lives
behind the ``[audio]`` extra and depends on ``librosa`` /
``transformers.WhisperFeatureExtractor``. For tests, examples, and
smoke runs we don't need acoustically meaningful audio — we just
need a tensor of the right shape with smooth structure and bounded
magnitude so the forge's forward pass exercises every path.

This module provides exactly that: a pure-numpy synthesizer
producing a sine-sweep + Gaussian noise pattern that mimics the
shape and magnitude profile of a real log-mel spectrogram. No
``[audio]`` extra required — only numpy + torch (already in the
hard dependency set).

Why a sine sweep specifically: real mel spectrograms of voiced
audio carry coherent structure across both axes (harmonics across
mel bins, slow envelope changes across frames). A sweep — where
each mel bin emits a sinusoid of slightly different period along
the time axis — captures that "smooth in both axes" character
without committing to any particular acoustic content.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saeforge.utils.lazy import require_extra


def synthetic_mel_features(
    seed: int,
    *,
    batch: int = 1,
    n_mels: int = 80,
    n_frames: int = 3000,
    noise_std: float = 0.05,
) -> Any:
    """Return a deterministic, sweep-shaped synthetic mel spectrogram.

    Parameters
    ----------
    seed:
        Numpy ``default_rng`` seed; same seed produces bit-identical
        output across calls.
    batch:
        Number of independent mel-spectrogram samples to stack along
        the batch axis. Each batch element gets a unique sub-seed so
        the per-element noise patterns are decorrelated.
    n_mels:
        Number of mel bins on axis 1. Default ``80`` matches every
        OpenAI Whisper variant (tiny through large-v3).
    n_frames:
        Number of time frames on axis 2. Default ``3000`` matches
        Whisper's required input length
        (``max_source_positions * conv1.stride * conv2.stride =
        1500 * 1 * 2``).
    noise_std:
        Standard deviation of the additive Gaussian noise. ``0.05``
        keeps the overall magnitude close to the sine envelope so
        the sweep structure dominates and the forge sees coherent
        input.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, n_mels, n_frames)``, dtype ``float32``,
        magnitude roughly in ``[-1.05, 1.05]`` (sine envelope ±1
        plus a few standard deviations of noise). Suitable as direct
        input to ``WhisperModel.encoder``,
        ``WhisperForConditionalGeneration``, and
        :class:`ForgedWhisperEncoder`.

    Notes
    -----
    The sweep pattern is ``sin(2π * (mel_idx + 1) / (n_mels + 1) *
    frame_idx / n_frames * sweep_periods)`` where ``sweep_periods``
    is a fixed constant (``8``) — every mel bin gets a slightly
    different period across the time axis, producing the desired
    smooth-in-both-axes character. The seed only affects the
    additive noise; the sweep envelope is deterministic given
    ``(n_mels, n_frames)``.
    """
    torch = require_extra("torch", "torch")

    sweep_periods = 8.0
    # (n_mels,): periods range smoothly from a fraction-of-sweep up
    # to ``sweep_periods``. The +1 offset avoids period=0 at mel 0.
    mel_axis = np.arange(n_mels, dtype=np.float64).reshape(n_mels, 1)
    frame_axis = np.arange(n_frames, dtype=np.float64).reshape(1, n_frames)
    period_per_mel = (mel_axis + 1.0) / (n_mels + 1.0) * sweep_periods
    sweep = np.sin(2.0 * np.pi * period_per_mel * frame_axis / n_frames)

    out = np.empty((batch, n_mels, n_frames), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for b in range(batch):
        # Per-batch sub-seed so noise across batch elements is
        # independent. Using rng.spawn would be cleaner on newer
        # numpy, but stays compatible with the venv pin via
        # explicit child Generator.
        sub = np.random.default_rng(rng.integers(0, 2**31 - 1))
        noise = sub.standard_normal((n_mels, n_frames)) * noise_std
        out[b] = (sweep + noise).astype(np.float32)

    return torch.from_numpy(out)
