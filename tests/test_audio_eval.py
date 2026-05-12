"""Tests for ``saeforge.audio_eval.cosine_faithfulness`` (§4.4 of the
forge-whisper-encoder change).

The function takes a forged Whisper encoder + host model + mel
features and returns a `[0.0, 1.0]` scalar measuring per-frame
cosine similarity between the forged output and the host's encoder
states projected through the same SAE basis. The tests here use
lightweight stubs for both ``forged`` (a torch module exposing
``basis_encode`` + a controllable ``forward``) and ``host`` (an
object with a ``.encoder(...)`` that returns
``BaseModelOutput(last_hidden_state=...)``) so we can exercise the
metric math directly. End-to-end correctness on a real Whisper
fixture is tracked alongside the audio-side smoke in
``tests/test_whisper_encoder_module.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _stub_forged(basis_encode, forward_output):
    """Build a stand-in for the ``NativeModel`` argument.

    ``forged.torch_module(audio_features)`` returns ``forward_output``;
    ``forged.torch_module.basis_encode`` is the ``(d, f)`` matrix the
    metric uses to project host states into the SAE basis.
    """
    import torch
    import torch.nn as nn

    class _StubForged(nn.Module):
        def __init__(self, basis_encode_t, forward_output_t):
            super().__init__()
            self.register_buffer("basis_encode", basis_encode_t)
            self._forward_output = forward_output_t

        def forward(self, _audio_features):
            return self._forward_output

    module = _StubForged(
        torch.as_tensor(basis_encode, dtype=torch.float32),
        torch.as_tensor(forward_output, dtype=torch.float32),
    )
    return SimpleNamespace(torch_module=module)


def _stub_host(host_states):
    """Build a stand-in for the host model.

    ``host.encoder(audio_features)`` returns a BaseModelOutput-like
    object exposing ``last_hidden_state``. The metric's host extraction
    looks for ``host.encoder`` (the ``WhisperModel`` branch).
    """
    import torch

    state_t = torch.as_tensor(host_states, dtype=torch.float32)
    encoder_out = SimpleNamespace(last_hidden_state=state_t)

    class _StubEncoder:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, _audio_features):
            return encoder_out

    class _StubHost:
        encoder = _StubEncoder()

        def to(self, _device):
            return self

        def eval(self):
            return self

    return _StubHost()


# ---------------------------------------------------------------------------
# Identical states → 1.0
# ---------------------------------------------------------------------------


class TestIdentical:
    def test_forged_equals_projected_host_returns_one(self):
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(0)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((2, 5, d)).astype(np.float32)
        projected = (
            torch.as_tensor(host_states, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()

        forged = _stub_forged(basis_encode, projected)
        host = _stub_host(host_states)
        # audio_features unused by the stubs; pass anything tensor-shaped.
        audio_features = torch.zeros(2, 80, 100)

        score = cosine_faithfulness(forged, host, audio_features)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_forged_equals_scaled_projected_host_still_one(self):
        """Cosine is scale-invariant: a constant rescaling of forged
        states doesn't change the metric."""
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(1)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((1, 3, d)).astype(np.float32)
        projected = (
            torch.as_tensor(host_states, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()

        forged = _stub_forged(basis_encode, projected * 7.5)
        host = _stub_host(host_states)
        audio_features = torch.zeros(1, 80, 6)

        score = cosine_faithfulness(forged, host, audio_features)
        assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Orthogonal / zero states → 0.0
# ---------------------------------------------------------------------------


class TestOrthogonal:
    def test_zero_forged_states_return_zero(self):
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(2)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((1, 3, d)).astype(np.float32)
        forged_states = np.zeros((1, 3, f), dtype=np.float32)

        forged = _stub_forged(basis_encode, forged_states)
        host = _stub_host(host_states)
        audio_features = torch.zeros(1, 80, 6)

        score = cosine_faithfulness(forged, host, audio_features)
        assert score == 0.0

    def test_zero_host_states_return_zero(self):
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(3)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = np.zeros((1, 3, d), dtype=np.float32)
        forged_states = rng.standard_normal((1, 3, f)).astype(np.float32)

        forged = _stub_forged(basis_encode, forged_states)
        host = _stub_host(host_states)
        audio_features = torch.zeros(1, 80, 6)

        score = cosine_faithfulness(forged, host, audio_features)
        assert score == 0.0

    def test_anti_correlated_states_clamp_to_zero(self):
        """Cosine = -1 forged vs projected host should clamp to 0 since
        the FSM min_faithfulness predicate assumes non-negative."""
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(4)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((1, 3, d)).astype(np.float32)
        projected = (
            torch.as_tensor(host_states, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()

        forged = _stub_forged(basis_encode, -projected)
        host = _stub_host(host_states)
        audio_features = torch.zeros(1, 80, 6)

        score = cosine_faithfulness(forged, host, audio_features)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Noise monotonically degrades similarity
# ---------------------------------------------------------------------------


class TestNoiseMonotonic:
    def test_increasing_noise_decreases_score(self):
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(5)
        d, f = 16, 8
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((2, 7, d)).astype(np.float32)
        projected = (
            torch.as_tensor(host_states, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()

        host = _stub_host(host_states)
        audio_features = torch.zeros(2, 80, 14)

        scores = []
        for sigma in (0.0, 0.1, 0.5, 1.5):
            noise = rng.standard_normal(projected.shape).astype(np.float32) * sigma
            forged = _stub_forged(basis_encode, projected + noise)
            scores.append(
                cosine_faithfulness(forged, host, audio_features)
            )

        # Monotone-decreasing across the noise sweep.
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
            f"Cosine score not monotonically decreasing with noise: {scores}"
        )
        # And the noiseless point is 1.0 (within fp32 noise).
        assert scores[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Batch averaging
# ---------------------------------------------------------------------------


class TestBatchAveraging:
    def test_batch_axis_averaged(self):
        """A batch of (good, bad) frames should average to roughly the
        mean of the per-example cosines."""
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(6)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)

        good_host = rng.standard_normal((1, 5, d)).astype(np.float32)
        bad_host = rng.standard_normal((1, 5, d)).astype(np.float32)
        host_stack = np.concatenate([good_host, bad_host], axis=0)

        good_projected = (
            torch.as_tensor(good_host, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()
        bad_forged = rng.standard_normal((1, 5, f)).astype(np.float32)
        forged_stack = np.concatenate([good_projected, bad_forged], axis=0)

        forged = _stub_forged(basis_encode, forged_stack)
        host = _stub_host(host_stack)
        audio_features = torch.zeros(2, 80, 10)
        score_batched = cosine_faithfulness(forged, host, audio_features)

        # Compute the single-example scores and verify the batch result
        # is the mean of the two (within fp32 noise).
        forged_good = _stub_forged(basis_encode, good_projected)
        host_good = _stub_host(good_host)
        forged_bad = _stub_forged(basis_encode, bad_forged)
        host_bad = _stub_host(bad_host)
        af1 = torch.zeros(1, 80, 10)
        score_good = cosine_faithfulness(forged_good, host_good, af1)
        score_bad = cosine_faithfulness(forged_bad, host_bad, af1)
        expected = (score_good + score_bad) / 2.0
        assert score_batched == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# fp16 / bf16 input passthrough
# ---------------------------------------------------------------------------


class TestLowPrecisionInput:
    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
    def test_low_precision_states_handled(self, dtype_name):
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        dtype = getattr(torch, dtype_name)
        rng = np.random.default_rng(7)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((1, 3, d)).astype(np.float32)
        projected = (
            torch.as_tensor(host_states, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()

        # Build the stubs and cast their forward outputs / host states
        # down to the low-precision dtype to mimic running the encoders
        # at fp16 / bf16.
        forged = _stub_forged(basis_encode, projected)
        forged.torch_module._forward_output = forged.torch_module._forward_output.to(dtype)

        host = _stub_host(host_states)
        host.encoder = host.encoder
        # Cast the host encoder output as well.
        out_t = host.encoder(None).last_hidden_state.to(dtype)
        host.encoder.__class__.__call__ = (  # type: ignore[method-assign]
            lambda self, _af, _out=SimpleNamespace(last_hidden_state=out_t): _out
        )

        audio_features = torch.zeros(1, 80, 6)
        score = cosine_faithfulness(forged, host, audio_features)
        # fp16/bf16 round-trip is lossy but should still register ≈ 1.0
        # since the matched-pair construction has high SNR.
        assert score == pytest.approx(1.0, abs=2e-2)


# ---------------------------------------------------------------------------
# End-to-end smoke through the real forged encoder
# ---------------------------------------------------------------------------


class TestPrecomputedHostStates:
    """The optional ``precomputed_host_states`` kwarg lets a caller skip
    the host encoder forward entirely. The FSM uses this for the
    pre-capture fast path; tests verify the helper accepts ``host=None``
    when precomputed states are supplied and uses them directly.
    """

    def test_precomputed_states_skip_host_forward(self):
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(8)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((1, 3, d)).astype(np.float32)
        projected = (
            torch.as_tensor(host_states, dtype=torch.float32)
            @ torch.as_tensor(basis_encode, dtype=torch.float32)
        ).numpy()

        forged = _stub_forged(basis_encode, projected)
        # host=None: the helper SHALL never look at it when
        # precomputed_host_states is provided.
        audio_features = torch.zeros(1, 80, 6)
        score = cosine_faithfulness(
            forged,
            host=None,
            audio_features=audio_features,
            precomputed_host_states=torch.as_tensor(host_states),
        )
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_precomputed_states_match_running_host(self):
        """Computing cosine via precomputed_host_states gives the same
        result as running the host inside the helper."""
        import torch

        from saeforge.audio_eval import cosine_faithfulness

        rng = np.random.default_rng(9)
        d, f = 8, 4
        basis_encode = rng.standard_normal((d, f)).astype(np.float32)
        host_states = rng.standard_normal((1, 3, d)).astype(np.float32)
        forged_states = rng.standard_normal((1, 3, f)).astype(np.float32) * 0.1

        forged = _stub_forged(basis_encode, forged_states)
        host = _stub_host(host_states)
        audio_features = torch.zeros(1, 80, 6)

        score_via_host = cosine_faithfulness(forged, host, audio_features)
        score_via_precomputed = cosine_faithfulness(
            forged,
            host=None,
            audio_features=audio_features,
            precomputed_host_states=torch.as_tensor(host_states),
        )
        assert score_via_host == pytest.approx(score_via_precomputed, abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-end smoke through the real forged encoder
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Run cosine_faithfulness through the real ``ForgedWhisperEncoder``
    + ``WhisperModel`` host. This isn't a strict numerical assertion —
    the forge approximates the host (ε from LayerNorm projection plus
    ε_conv from the frozen-copied stem) so cosine is in `[0, 1]` but
    not pinned at 1. The test only asserts the function runs end-to-
    end and returns a sensible scalar.
    """

    def test_returns_valid_scalar_on_real_forge(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        import torch

        from saeforge.adapters import adapter_for
        from saeforge.audio_eval import cosine_faithfulness
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(feature_basis_64_to_32)
        adapter = adapter_for(tiny_synthetic_whisper)
        walk = adapter.walk(tiny_synthetic_whisper, projector)
        config = adapter.build_native_config(
            tiny_synthetic_whisper, feature_basis_64_to_32.n_features
        )
        forged = NativeModel.from_projected_weights(config, walk)

        torch.manual_seed(0)
        # HF WhisperEncoder enforces input length == max_source_positions *
        # conv1.stride * conv2.stride = 1500 * 1 * 2 = 3000. The forged
        # encoder is more permissive (any length up to the pos table), so
        # the strict-length input comes from the host's check.
        audio_features = torch.randn(1, 80, 3000)
        score = cosine_faithfulness(
            forged, tiny_synthetic_whisper, audio_features
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
