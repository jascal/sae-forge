"""End-to-end hybrid-bridge forge on a tiny untied Qwen2 (qkv_bias=True).

Same shape as ``tests/integration/test_hybrid_bridge_llama.py``. Confirms
the qkv_bias + bridges combination round-trips cleanly — Qwen2 emits
Q/K/V bias state-dict entries that are NOT projected, and the bridge
state-dict entries are projected only through the bridge config (not
through the basis). Both groups must survive save/load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _basis(*, n=32, d=128, seed=0):
    from saeforge.basis import FeatureBasis

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, d)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1),
        original_norms=np.linalg.norm(W, axis=1),
    )


def _hybrid_pipeline(host, *, n_features=32):
    from saeforge.bridges import BridgeConfig
    from saeforge.forge import ForgePipeline
    from saeforge.projector import SubspaceProjector

    d = host.config.hidden_size
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


class TestT0TinyQwen2Smoke:
    def test_hybrid_forge_constructs_with_bridges(self, tiny_qwen2_untied_4layer):
        pipeline = _hybrid_pipeline(tiny_qwen2_untied_4layer)
        model = _build_forged_module(pipeline, tiny_qwen2_untied_4layer)
        bridges = model.torch_module.model.bridges
        assert bridges is not None
        assert "emb_mid" in bridges
        assert "mid_lm" in bridges
        sd = model.torch_module.state_dict()
        # Bridge keys present
        assert any(k.startswith("model.bridges.emb_mid.") for k in sd)
        assert any(k.startswith("model.bridges.mid_lm.") for k in sd)
        # Qwen2 Q/K/V bias keys also present (the load-bearing check that
        # qkv_bias + bridges coexist cleanly)
        for i in range(4):
            for qkv in ("q_proj", "k_proj", "v_proj"):
                assert f"model.layers.{i}.self_attn.{qkv}.bias" in sd

    def test_forward_pass_finite(self, tiny_qwen2_untied_4layer):
        import torch

        pipeline = _hybrid_pipeline(tiny_qwen2_untied_4layer)
        model = _build_forged_module(pipeline, tiny_qwen2_untied_4layer)
        input_ids = torch.randint(0, 1024, (2, 8))
        with torch.no_grad():
            logits = model.forward(input_ids)
        assert logits.shape == (2, 8, 1024)
        assert torch.isfinite(logits).all()

    def test_safetensors_round_trip(self, tiny_qwen2_untied_4layer, tmp_path: Path):
        import torch

        from saeforge.model import NativeModel

        pipeline = _hybrid_pipeline(tiny_qwen2_untied_4layer)
        model = _build_forged_module(pipeline, tiny_qwen2_untied_4layer)
        model.save_pretrained(tmp_path)
        loaded = NativeModel.load_pretrained(tmp_path)
        sd_orig = model.torch_module.state_dict()
        sd_loaded = loaded.torch_module.state_dict()
        assert set(sd_orig.keys()) == set(sd_loaded.keys())
        for k in sd_orig:
            assert torch.allclose(sd_orig[k], sd_loaded[k]), f"mismatch on {k}"


class TestByteEquivalenceWhenDisabled:
    def test_disabled_path_has_no_bridges_in_state_dict(self, tiny_qwen2_untied_4layer):
        from saeforge.forge import ForgePipeline
        from saeforge.model import NativeModel, _config_from_host
        from saeforge.projector import SubspaceProjector

        b = _basis(n=32, d=128, seed=7)
        pipeline = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
        )
        weights = pipeline.projector.project_module(tiny_qwen2_untied_4layer)
        config = _config_from_host(tiny_qwen2_untied_4layer, b.n_features)
        model = NativeModel.from_projected_weights(config, weights)
        assert model.torch_module.model.bridges is None
        sd = model.torch_module.state_dict()
        assert not any(".bridges." in k for k in sd)
        # The Qwen2 Q/K/V biases are still present on the single-basis path
        # (this is the regression gate that the conftest 4-layer bump and the
        # bridge wiring didn't disturb the qkv_bias plumbing).
        for i in range(4):
            assert f"model.layers.{i}.self_attn.q_proj.bias" in sd
