"""Tests for the audio-side ForgePipeline plumbing (§6 of the
forge-whisper-encoder change).

Construction-time validation of the eval_audio_features /
eval_prompts mutual exclusion, plus an end-to-end check that the
field flows through ``_build_fsm_ctx`` into the
``_eval_audio_features`` slot the action layer reads.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Mutual exclusion: eval_audio_features XOR eval_prompts
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    def test_both_set_raises(self, tiny_synthetic_basis):
        import torch

        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        with pytest.raises(ValueError, match="mutually exclusive"):
            ForgePipeline(
                basis=tiny_synthetic_basis,
                projector=projector,
                eval_prompts=["hello world"],
                eval_audio_features=audio,
            )

    def test_only_audio_set_constructs(self, tiny_synthetic_basis):
        import torch

        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        # No raise.
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            eval_audio_features=audio,
        )
        assert pipeline.eval_audio_features is audio
        assert pipeline.eval_prompts == []

    def test_only_prompts_set_constructs(self, tiny_synthetic_basis):
        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        # No raise.
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            eval_prompts=["hello world"],
        )
        assert pipeline.eval_audio_features is None
        assert pipeline.eval_prompts == ["hello world"]

    def test_neither_set_constructs(self, tiny_synthetic_basis):
        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        # Default state, no raise.
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
        )
        assert pipeline.eval_audio_features is None
        assert pipeline.eval_prompts == []


# ---------------------------------------------------------------------------
# Issue #27 — FSM gaps: finetune + hybrid-bridge rejection for whisper_encoder
# ---------------------------------------------------------------------------


class TestWhisperEncoderFinetuneRejected:
    """v0.4 forge-whisper-encoder forges have a frozen-copied conv stem
    and no per-frame loss signal defined. Setting ``finetune_steps > 0``
    on an audio pipeline must be rejected at construction time, before
    the FSM can run an optimizer step over the conv weights."""

    def test_finetune_steps_rejected_on_whisper_encoder(
        self, tiny_synthetic_basis
    ):
        import torch

        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        with pytest.raises(ValueError, match="finetune_steps > 0"):
            ForgePipeline(
                basis=tiny_synthetic_basis,
                projector=projector,
                eval_audio_features=audio,
                finetune_steps=10,
            )

    def test_finetune_steps_allowed_on_lm_pipeline(self, tiny_synthetic_basis):
        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        # No audio features → LM-family forge → finetune_steps is fine.
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            finetune_steps=10,
        )
        assert pipeline.finetune_steps == 10
        assert pipeline.eval_audio_features is None

    def test_finetune_zero_allowed_on_whisper_encoder(
        self, tiny_synthetic_basis
    ):
        """The default ``finetune_steps=0`` is the supported path for
        audio forges — eval-only, no fine-tune."""
        import torch

        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            eval_audio_features=audio,
            finetune_steps=0,
        )
        assert pipeline.eval_audio_features is audio


class TestWhisperEncoderHybridBridgeRejected:
    """``hybrid_bridge=True`` wires three bases (embed / mid / lm_head)
    onto the LM residual stream. ForgedWhisperEncoder already carries a
    d→f bridge in the ``basis_encode`` buffer, has no ``lm_head``, and
    layering the LM hybrid path on top would either double-project or
    crash deep in the projection step. Must be rejected at construction."""

    def test_hybrid_bridge_rejected_on_whisper_encoder(
        self, tiny_synthetic_basis
    ):
        import torch

        from saeforge import ForgePipeline, SubspaceProjector
        from saeforge.basis import FeatureBasis
        import numpy as np

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        # hybrid_bridge requires basis_embed + basis_lm_head; provide
        # matching-shape stand-ins so the existing hybrid-bridge invariant
        # checks pass, and the whisper-specific rejection is the one that
        # fires.
        n = tiny_synthetic_basis.n_features
        d = tiny_synthetic_basis.d_model
        rng = np.random.default_rng(0)
        W = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(W, axis=1).astype(np.float32)
        side_basis = FeatureBasis(
            kept_ids=np.arange(n, dtype=np.int64),
            W_dec=W,
            merged_norms=norms,
            original_norms=norms,
        )
        with pytest.raises(ValueError, match="hybrid_bridge=True is not supported"):
            ForgePipeline(
                basis=tiny_synthetic_basis,
                projector=projector,
                eval_audio_features=audio,
                hybrid_bridge=True,
                basis_embed=side_basis,
                basis_lm_head=side_basis,
            )

    def test_hybrid_bridge_allowed_on_lm_pipeline(self, tiny_synthetic_basis):
        """The LM hybrid-bridge path is unchanged."""
        import numpy as np

        from saeforge import ForgePipeline, SubspaceProjector
        from saeforge.basis import FeatureBasis

        projector = SubspaceProjector(tiny_synthetic_basis)
        n = tiny_synthetic_basis.n_features
        d = tiny_synthetic_basis.d_model
        rng = np.random.default_rng(0)
        W = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(W, axis=1).astype(np.float32)
        side_basis = FeatureBasis(
            kept_ids=np.arange(n, dtype=np.int64),
            W_dec=W,
            merged_norms=norms,
            original_norms=norms,
        )
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            hybrid_bridge=True,
            basis_embed=side_basis,
            basis_lm_head=side_basis,
        )
        assert pipeline.hybrid_bridge is True
        assert pipeline.eval_audio_features is None


# ---------------------------------------------------------------------------
# eval_audio_features flows through _build_fsm_ctx to _eval_audio_features
# ---------------------------------------------------------------------------


class TestCtxWiring:
    def test_audio_features_populates_eval_audio_features_key(
        self, tiny_synthetic_basis
    ):
        import torch

        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            eval_audio_features=audio,
        )

        ctx = pipeline._build_fsm_ctx(
            sae_checkpoint="dummy.safetensors",
            output_dir="/tmp/x",
            host_model=None,
            eval_input_ids=None,
            eval_audio_features=pipeline.eval_audio_features,
            finetune_input_ids=None,
            finetune_iterator=None,
            host_model_id="<test>",
        )
        assert ctx["_eval_audio_features"] is audio
        assert ctx["_eval_input_ids"] is None

    def test_eval_encoder_states_populates_ctx_field(
        self, tiny_synthetic_basis
    ):
        import torch

        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        audio = torch.zeros(1, 80, 3000)
        precaptured = torch.zeros(1, 1500, 64)
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            eval_audio_features=audio,
            eval_encoder_states=precaptured,
        )
        ctx = pipeline._build_fsm_ctx(
            sae_checkpoint="dummy.safetensors",
            output_dir="/tmp/x",
            host_model=None,
            eval_input_ids=None,
            eval_audio_features=pipeline.eval_audio_features,
            eval_encoder_states=pipeline.eval_encoder_states,
            finetune_input_ids=None,
            finetune_iterator=None,
            host_model_id="<test>",
        )
        assert ctx["_eval_audio_features"] is audio
        assert ctx["_eval_encoder_states"] is precaptured

    def test_absent_audio_features_yields_none(self, tiny_synthetic_basis):
        from saeforge import ForgePipeline, SubspaceProjector

        projector = SubspaceProjector(tiny_synthetic_basis)
        pipeline = ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
        )
        ctx = pipeline._build_fsm_ctx(
            sae_checkpoint="dummy.safetensors",
            output_dir="/tmp/x",
            host_model=None,
            eval_input_ids=None,
            finetune_input_ids=None,
            finetune_iterator=None,
            host_model_id="<test>",
        )
        # Default kwarg path: ctx key exists with None value so the action
        # can probe it with a single .get(...) call without branching on
        # presence.
        assert ctx["_eval_audio_features"] is None
        assert ctx["_eval_encoder_states"] is None
