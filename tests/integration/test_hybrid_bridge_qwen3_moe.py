"""End-to-end hybrid-bridge forge on a synthetic small Qwen3-MoE.

Satisfies the ``hybrid-bridge-llama-family`` family-coverage requirement
for Qwen3-MoE: bridges + MoE routing + Q/K-norm all coexist cleanly in
the forged state_dict and survive save/load.

Requires ``transformers >= 4.51``. Skips gracefully on older installs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers", minversion="4.51")


def _basis(*, n=32, d=128, seed=0):
    from saeforge.basis import FeatureBasis

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, d)).astype(np.float64)
    norms = np.linalg.norm(W, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n),
        W_dec=W,
        merged_norms=norms,
        original_norms=norms,
    )


def _hybrid_pipeline(host, *, n_features=32):
    from saeforge.bridges import BridgeConfig
    from saeforge.forge import ForgePipeline
    from saeforge.projector import SubspaceProjector

    d = host.config.hidden_size
    b_mid = _basis(n=n_features, d=d, seed=10)
    return ForgePipeline(
        basis=b_mid,
        projector=SubspaceProjector(b_mid),
        host_model_id="<offline>",
        hybrid_bridge=True,
        basis_embed=_basis(n=n_features, d=d, seed=20),
        basis_lm_head=_basis(n=n_features, d=d, seed=30),
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


class TestT0TinyQwen3MoESmoke:
    def test_hybrid_forge_constructs_with_bridges_and_moe_and_qknorm(
        self, tiny_qwen3_moe_untied
    ):
        pipeline = _hybrid_pipeline(tiny_qwen3_moe_untied)
        model = _build_forged_module(pipeline, tiny_qwen3_moe_untied)

        # Bridges
        bridges = model.torch_module.model.bridges
        assert bridges is not None
        assert "emb_mid" in bridges
        assert "mid_lm" in bridges

        # MoE structure on every block
        cfg = tiny_qwen3_moe_untied.config
        for i, layer in enumerate(model.torch_module.model.layers):
            assert hasattr(layer.mlp, "gate"), f"layer {i} missing mlp.gate"
            assert hasattr(layer.mlp, "experts"), f"layer {i} missing mlp.experts"
            assert len(layer.mlp.experts) == cfg.num_experts
            # qk_norm
            assert layer.self_attn.q_norm is not None
            assert layer.self_attn.k_norm is not None

        # State dict contains both bridge keys AND per-expert keys AND q_norm keys
        sd = model.torch_module.state_dict()
        assert any(k.startswith("model.bridges.emb_mid.") for k in sd)
        assert any(k.startswith("model.bridges.mid_lm.") for k in sd)
        for i in range(cfg.num_hidden_layers):
            assert f"model.layers.{i}.mlp.gate.weight" in sd
            for e in range(cfg.num_experts):
                assert f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight" in sd
            for qk in ("q_norm", "k_norm"):
                assert f"model.layers.{i}.self_attn.{qk}.weight" in sd

    def test_forward_pass_finite(self, tiny_qwen3_moe_untied):
        import torch

        pipeline = _hybrid_pipeline(tiny_qwen3_moe_untied)
        model = _build_forged_module(pipeline, tiny_qwen3_moe_untied)
        input_ids = torch.randint(0, 1024, (2, 8))
        with torch.no_grad():
            logits = model.forward(input_ids)
        assert logits.shape == (2, 8, 1024)
        assert torch.isfinite(logits).all()

    def test_safetensors_round_trip(self, tiny_qwen3_moe_untied, tmp_path: Path):
        import torch

        from saeforge.model import NativeModel

        pipeline = _hybrid_pipeline(tiny_qwen3_moe_untied)
        model = _build_forged_module(pipeline, tiny_qwen3_moe_untied)
        model.save_pretrained(tmp_path)
        loaded = NativeModel.load_pretrained(tmp_path)
        sd_orig = model.torch_module.state_dict()
        sd_loaded = loaded.torch_module.state_dict()
        assert set(sd_orig.keys()) == set(sd_loaded.keys())
        for k in sd_orig:
            assert torch.allclose(sd_orig[k], sd_loaded[k]), f"mismatch on {k}"


class TestByteEquivalenceWhenDisabled:
    def test_disabled_path_keeps_full_moe_plumbing(self, tiny_qwen3_moe_untied):
        """``hybrid_bridge=False`` produces a forged module with no bridge keys
        but full MoE/qk_norm plumbing intact (regression gate)."""
        from saeforge.forge import ForgePipeline
        from saeforge.model import NativeModel, _config_from_host
        from saeforge.projector import SubspaceProjector

        b = _basis(n=32, d=128, seed=7)
        pipeline = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
        )
        weights = pipeline.projector.project_module(tiny_qwen3_moe_untied)
        config = _config_from_host(tiny_qwen3_moe_untied, b.n_features)
        model = NativeModel.from_projected_weights(config, weights)

        # No bridges
        assert model.torch_module.model.bridges is None
        sd = model.torch_module.state_dict()
        assert not any(".bridges." in k for k in sd)

        # Full MoE + qk_norm plumbing
        cfg = tiny_qwen3_moe_untied.config
        for i in range(cfg.num_hidden_layers):
            assert f"model.layers.{i}.mlp.gate.weight" in sd
            for e in range(cfg.num_experts):
                assert f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight" in sd
            assert f"model.layers.{i}.self_attn.q_norm.weight" in sd
            assert f"model.layers.{i}.self_attn.k_norm.weight" in sd
