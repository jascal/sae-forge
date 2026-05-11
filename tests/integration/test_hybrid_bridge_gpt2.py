"""End-to-end hybrid-bridge forge on a tiny untied GPT-2 (T0).

Mirrors the cross-architecture defaults-validation surface documented in
``openspec/changes/hybrid-bridge-forge/design.md`` § "Cross-architecture
validation tiering". T0 = tiny_gpt2 on CPU; the same test runs at T1
(real ``gpt2`` host) via the comparison harness in
``scripts/compare_single_vs_hybrid_gpt2.py``.

This integration test must:
- Construct a hybrid-bridge pipeline against an untied tiny GPT-2.
- Project all weights, build the forged model with bridges attached.
- Confirm bridge parameters appear in the forged ``state_dict``.
- Round-trip via safetensors.
- Confirm the byte-equivalence-when-disabled scenario: a pipeline with
  ``hybrid_bridge=False`` produces the same forged weights regardless of
  whether the unused ``basis_embed`` / ``basis_lm_head`` fields are populated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _basis(*, n=8, d=16, seed=0):
    from saeforge.basis import FeatureBasis

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, d)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1),
        original_norms=np.linalg.norm(W, axis=1),
    )


@pytest.fixture
def tiny_gpt2_untied_4layer():
    """Untied 4-layer tiny GPT-2 — minimum n_layer for hybrid (>= 3)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=4,
        n_head=4,
        n_inner=32,
        tie_word_embeddings=False,
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture
def tiny_gpt2_tied_4layer():
    """Tied 4-layer tiny GPT-2 — must be refused under hybrid_bridge."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=4,
        n_head=4,
        n_inner=32,
        tie_word_embeddings=True,
    )
    return GPT2LMHeadModel(config).eval()


def _hybrid_pipeline(host, *, n_features=8):
    """Construct a ForgePipeline against an in-memory host (no .from_pretrained).

    Mirrors what ``ForgePipeline._run_real_imperative`` would do, but skipping
    the HF Hub round-trip so the test stays offline.
    """
    from saeforge.bridges import BridgeConfig
    from saeforge.forge import ForgePipeline
    from saeforge.projector import SubspaceProjector

    d = host.config.n_embd
    b_mid = _basis(n=n_features, d=d, seed=10)
    b_embed = _basis(n=n_features, d=d, seed=20)
    b_lm = _basis(n=n_features, d=d, seed=30)
    return ForgePipeline(
        basis=b_mid,
        projector=SubspaceProjector(b_mid),
        host_model_id="<offline>",
        hybrid_bridge=True,
        basis_embed=b_embed,
        basis_lm_head=b_lm,
        bridge_config=BridgeConfig(),
    )


def _build_forged_module(pipeline, host):
    """Drive the pipeline through projection + NativeModel construction, returning the forged model."""
    from saeforge.model import NativeModel, _config_from_host

    bundle = pipeline._build_hybrid_bundle(host)
    weights = pipeline.projector.project_module(host, hybrid=bundle)
    config = _config_from_host(host, pipeline.basis.n_features)
    if bundle is not None:
        config.bridges = True
        config.bridge_init = pipeline.bridge_config.init
        config.bridge_nonlin = pipeline.bridge_config.nonlin
        config.bridge_pre_layernorm = pipeline.bridge_config.pre_layernorm
    return NativeModel.from_projected_weights(config, weights)


class TestT0TinyGpt2Smoke:
    def test_hybrid_forge_constructs_with_bridges(self, tiny_gpt2_untied_4layer):
        import torch

        pipeline = _hybrid_pipeline(tiny_gpt2_untied_4layer)
        model = _build_forged_module(pipeline, tiny_gpt2_untied_4layer)
        # Bridge submodules exist on the transformer.
        bridges = model.torch_module.transformer.bridges
        assert bridges is not None
        assert "emb_mid" in bridges
        assert "mid_lm" in bridges
        # Bridge parameters appear in state_dict.
        sd = model.torch_module.state_dict()
        assert any(k.startswith("transformer.bridges.emb_mid.") for k in sd)
        assert any(k.startswith("transformer.bridges.mid_lm.") for k in sd)

    def test_forward_pass_finite(self, tiny_gpt2_untied_4layer):
        import torch

        pipeline = _hybrid_pipeline(tiny_gpt2_untied_4layer)
        model = _build_forged_module(pipeline, tiny_gpt2_untied_4layer)
        input_ids = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            logits = model.forward(input_ids)
        assert logits.shape == (2, 8, 100)
        assert torch.isfinite(logits).all()

    def test_safetensors_round_trip(self, tiny_gpt2_untied_4layer, tmp_path: Path):
        import torch

        from saeforge.model import NativeModel

        pipeline = _hybrid_pipeline(tiny_gpt2_untied_4layer)
        model = _build_forged_module(pipeline, tiny_gpt2_untied_4layer)
        model.save_pretrained(tmp_path)
        loaded = NativeModel.load_pretrained(tmp_path)

        sd_orig = model.torch_module.state_dict()
        sd_loaded = loaded.torch_module.state_dict()
        assert set(sd_orig.keys()) == set(sd_loaded.keys())
        for k in sd_orig:
            assert torch.allclose(sd_orig[k], sd_loaded[k]), f"mismatch on {k}"


class TestTiedEmbeddingRefusal:
    def test_tied_embeddings_refused_at_run_time(self, tiny_gpt2_tied_4layer):
        pipeline = _hybrid_pipeline(tiny_gpt2_tied_4layer)
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            pipeline._build_hybrid_bundle(tiny_gpt2_tied_4layer)


class TestByteEquivalenceWhenDisabled:
    """Confirm hybrid_bridge=False with stale extras is byte-identical to the v0 minimal call.

    This is the load-bearing safety net documented in the proposal: every
    existing single-basis caller stays bit-for-bit unchanged after this
    change ships.
    """

    def test_disabled_with_extras_matches_minimal(self, tiny_gpt2_untied_4layer):
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis(n=8, d=16, seed=42)

        # Pipeline with stale extras and hybrid_bridge=False
        pipe_extras = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
            hybrid_bridge=False,
            basis_embed=_basis(seed=999),
            basis_lm_head=_basis(seed=998),
        )
        # Minimal pipeline — no hybrid fields at all
        pipe_min = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
        )

        bundle_extras = pipe_extras._build_hybrid_bundle(tiny_gpt2_untied_4layer)
        bundle_min = pipe_min._build_hybrid_bundle(tiny_gpt2_untied_4layer)
        assert bundle_extras is None
        assert bundle_min is None

        w_extras = pipe_extras.projector.project_module(
            tiny_gpt2_untied_4layer, hybrid=bundle_extras
        )
        w_min = pipe_min.projector.project_module(tiny_gpt2_untied_4layer, hybrid=bundle_min)
        assert set(w_extras.keys()) == set(w_min.keys())
        for k in w_extras:
            np.testing.assert_array_equal(w_extras[k], w_min[k])

    def test_native_module_byte_identical_without_bridges_flag(
        self, tiny_gpt2_untied_4layer
    ):
        """Forging without the bridge flag MUST produce the same nn.Module shape as before this change."""
        import torch

        from saeforge.forge import ForgePipeline
        from saeforge.model import NativeModel, _config_from_host
        from saeforge.projector import SubspaceProjector

        b = _basis(n=8, d=16, seed=7)
        pipeline = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
        )
        weights = pipeline.projector.project_module(tiny_gpt2_untied_4layer)
        config = _config_from_host(tiny_gpt2_untied_4layer, b.n_features)
        model = NativeModel.from_projected_weights(config, weights)
        # Without bridges flag set, the transformer has no bridge submodule.
        assert model.torch_module.transformer.bridges is None
        # State dict has no bridge keys.
        sd = model.torch_module.state_dict()
        assert not any(".bridges." in k for k in sd)
