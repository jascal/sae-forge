"""Tests for the audio-side ForgePipeline plumbing (§6 of the
forge-whisper-encoder change).

Construction-time validation of the eval_audio_features /
eval_prompts mutual exclusion, plus an end-to-end check that the
field flows through ``_build_fsm_ctx`` into the
``_eval_audio_features`` slot the action layer reads.
"""

from __future__ import annotations

import pytest


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
