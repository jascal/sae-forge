"""Tests for ForgedWhisperEncoder (§3.8 of the forge-whisper-encoder change).

Covers the forward shape contract, the conv-stem-frozen invariant (the
adapter walk's frozen-copied entries arrive bit-for-bit on the forged
module), the ``basis_encode`` d → f bridge buffer is present and
state-dict-resident, and ``NativeModel.{save,load}_pretrained`` round-
trips a forged Whisper encoder without parameter or buffer drift.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_forged(host, basis):
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    projector = SubspaceProjector(basis)
    adapter = adapter_for(host)
    walk = adapter.walk(host, projector)
    config = adapter.build_native_config(host, basis.n_features)
    return NativeModel.from_projected_weights(config, walk), walk, projector


# ---------------------------------------------------------------------------
# Forward shape contract
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_forward_returns_per_frame_encoder_states(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        import torch

        nm, _, _ = _build_forged(tiny_synthetic_whisper, feature_basis_64_to_32)
        # n_frames=200 → after conv2(stride=2): 100 output frames; well
        # under max_source_positions=1500 so positional embeddings cover.
        x = torch.randn(2, 80, 200, dtype=torch.float32)
        with torch.no_grad():
            y = nm.torch_module(x)
        assert y.shape == (2, 100, feature_basis_64_to_32.n_features)
        assert torch.isfinite(y).all().item(), (
            "Forged Whisper encoder produced non-finite values on a "
            "well-formed random input — likely a buffer init issue."
        )

    def test_forward_rejects_input_exceeding_pos_table(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        import torch

        nm, _, _ = _build_forged(tiny_synthetic_whisper, feature_basis_64_to_32)
        # max_source_positions=1500 → max input frames after conv2 stride=2
        # is 1500, so input frames > 3000 overshoot the positional table.
        too_long = torch.randn(1, 80, 4000, dtype=torch.float32)
        with pytest.raises(ValueError, match="positional embedding table"):
            with torch.no_grad():
                nm.torch_module(too_long)


# ---------------------------------------------------------------------------
# Conv-stem-frozen invariant
# ---------------------------------------------------------------------------


class TestConvStemFrozen:
    def test_conv_and_pos_embeddings_bit_for_bit(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        nm, _, _ = _build_forged(tiny_synthetic_whisper, feature_basis_64_to_32)
        host_enc = tiny_synthetic_whisper.encoder
        forged_state = nm.torch_module.state_dict()

        for key, host_tensor in [
            ("conv1.weight", host_enc.conv1.weight),
            ("conv1.bias", host_enc.conv1.bias),
            ("conv2.weight", host_enc.conv2.weight),
            ("conv2.bias", host_enc.conv2.bias),
            ("embed_positions.weight", host_enc.embed_positions.weight),
        ]:
            forged_arr = forged_state[key].cpu().float().numpy()
            host_arr = host_tensor.detach().cpu().float().numpy()
            np.testing.assert_allclose(forged_arr, host_arr, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# basis_encode buffer
# ---------------------------------------------------------------------------


class TestBasisEncodeBuffer:
    def test_buffer_is_not_a_parameter(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        nm, _, _ = _build_forged(tiny_synthetic_whisper, feature_basis_64_to_32)
        param_names = {n for n, _ in nm.torch_module.named_parameters()}
        buffer_names = {n for n, _ in nm.torch_module.named_buffers()}
        assert "basis_encode" not in param_names
        assert "basis_encode" in buffer_names

    def test_buffer_value_matches_projector_encode(
        self, tiny_synthetic_whisper, feature_basis_64_to_32
    ):
        import torch

        nm, walk, projector = _build_forged(
            tiny_synthetic_whisper, feature_basis_64_to_32
        )
        expected = (
            projector.basis.pseudoinverse() * projector.scale_boost
        ).astype(np.float64)
        actual = nm.torch_module.basis_encode.detach().cpu().float().numpy()
        # Cast both to fp32 for the comparison since the module's buffer
        # was stored at the state_dict's native dtype.
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.astype(np.float32),
            atol=1e-6,
        )
        # And it survives via the walk dict identically.
        np.testing.assert_allclose(
            walk["basis_encode"].astype(np.float32),
            expected.astype(np.float32),
            atol=1e-6,
        )
        _ = torch  # silence pyflakes — torch import enforces the [torch] extra


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_state_dict_round_trip(
        self, tiny_synthetic_whisper, feature_basis_64_to_32, tmp_path
    ):
        import torch

        from saeforge.model import NativeModel

        nm, _, _ = _build_forged(tiny_synthetic_whisper, feature_basis_64_to_32)
        save_dir = tmp_path / "forged_whisper"
        nm.save_pretrained(save_dir)

        loaded = NativeModel.load_pretrained(save_dir)

        # Every state_dict entry — params + buffers — round-trips bit-equal.
        orig_state = nm.torch_module.state_dict()
        loaded_state = loaded.torch_module.state_dict()
        assert set(orig_state) == set(loaded_state)
        for key, orig_tensor in orig_state.items():
            assert torch.equal(orig_tensor, loaded_state[key]), (
                f"state_dict entry {key!r} drifted across save/load round-trip"
            )

        # And forward output is identical.
        x = torch.randn(1, 80, 200, dtype=torch.float32)
        with torch.no_grad():
            y_orig = nm.torch_module(x)
            y_loaded = loaded.torch_module(x)
        assert torch.equal(y_orig, y_loaded)

    def test_config_round_trip_preserves_whisper_fields(
        self, tiny_synthetic_whisper, feature_basis_64_to_32, tmp_path
    ):
        from saeforge.model import NativeModel

        nm, _, _ = _build_forged(tiny_synthetic_whisper, feature_basis_64_to_32)
        save_dir = tmp_path / "forged_whisper_cfg"
        nm.save_pretrained(save_dir)
        loaded = NativeModel.load_pretrained(save_dir)

        assert loaded.config.family == "whisper_encoder"
        assert loaded.config.output_kind == "encoder_states"
        assert loaded.config.vocab_size == 0
        assert loaded.config.hidden_size == feature_basis_64_to_32.n_features
        assert loaded.config.num_layers == 2
        assert loaded.config.num_heads == 4
