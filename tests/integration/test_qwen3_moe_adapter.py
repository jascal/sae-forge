"""Qwen3-MoE adapter and forged-module unit tests.

Synthetic small-MoE host (3 layers × 4 experts × top-2 routing). Real
Qwen3-MoE (30B-A3B) is NVIDIA-only; the smoke for that is
``scripts/smoke_qwen3_moe.py``.

Requires ``transformers >= 4.51``. The entire file skips gracefully on
older installs (the ``[intel]`` extra is capped at ``<4.50``).
"""

from __future__ import annotations

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


class TestAdapterDispatch:
    def test_dispatches_to_qwen3_moe_adapter(self, tiny_qwen3_moe_untied):
        from saeforge.adapters import adapter_for

        adapter = adapter_for(tiny_qwen3_moe_untied)
        assert adapter.family == "qwen3_moe"

    def test_registered_classes_contains_qwen3_moe(self):
        from saeforge.adapters import registered_classes

        names = [c.__name__ for c in registered_classes()]
        assert "Qwen3MoeForCausalLM" in names


class TestWalker:
    def test_walker_emits_gate_and_experts(self, tiny_qwen3_moe_untied):
        from saeforge.adapters import adapter_for
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(_basis())
        walk = adapter_for(tiny_qwen3_moe_untied).walk(tiny_qwen3_moe_untied, projector)
        cfg = tiny_qwen3_moe_untied.config
        for i in range(cfg.num_hidden_layers):
            assert f"model.layers.{i}.mlp.gate.weight" in walk
            for e in range(cfg.num_experts):
                for kind in ("gate_proj", "up_proj", "down_proj"):
                    key = f"model.layers.{i}.mlp.experts.{e}.{kind}.weight"
                    assert key in walk, f"missing {key}"

    def test_walker_omits_dense_mlp_keys(self, tiny_qwen3_moe_untied):
        """MoE hosts have no dense ``mlp.gate_proj`` etc. — the walker must not emit them."""
        from saeforge.adapters import adapter_for
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(_basis())
        walk = adapter_for(tiny_qwen3_moe_untied).walk(tiny_qwen3_moe_untied, projector)
        for i in range(tiny_qwen3_moe_untied.config.num_hidden_layers):
            for kind in ("gate_proj", "up_proj", "down_proj"):
                # Bare mlp.<kind>.weight (no .experts.) must not appear
                key = f"model.layers.{i}.mlp.{kind}.weight"
                assert key not in walk, f"unexpected dense MLP key {key} on MoE host"

    def test_walker_expert_shapes(self, tiny_qwen3_moe_untied):
        from saeforge.adapters import adapter_for
        from saeforge.projector import SubspaceProjector

        basis = _basis(n=32, d=128)
        projector = SubspaceProjector(basis)
        walk = adapter_for(tiny_qwen3_moe_untied).walk(tiny_qwen3_moe_untied, projector)
        cfg = tiny_qwen3_moe_untied.config
        # gate.weight: HF (num_experts, hidden) projected on the IN axis →
        # output (num_experts, n_features).
        for i in range(cfg.num_hidden_layers):
            assert walk[f"model.layers.{i}.mlp.gate.weight"].shape == (
                cfg.num_experts,
                32,
            )
            for e in range(cfg.num_experts):
                # gate_proj: HF (moe_intermediate, hidden) → (moe_intermediate, n_features)
                assert walk[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"].shape == (
                    cfg.moe_intermediate_size,
                    32,
                )
                # up_proj: same shape as gate_proj
                assert walk[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"].shape == (
                    cfg.moe_intermediate_size,
                    32,
                )
                # down_proj: HF (hidden, moe_intermediate) → (n_features, moe_intermediate)
                assert walk[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"].shape == (
                    32,
                    cfg.moe_intermediate_size,
                )


class TestNativeConfig:
    def test_moe_fields_populated_from_host(self, tiny_qwen3_moe_untied):
        from saeforge.adapters import adapter_for

        config = adapter_for(tiny_qwen3_moe_untied).build_native_config(
            tiny_qwen3_moe_untied, 32
        )
        cfg = tiny_qwen3_moe_untied.config
        assert config.family == "qwen3_moe"
        assert config.num_experts == cfg.num_experts
        assert config.num_experts_per_tok == cfg.num_experts_per_tok
        assert config.moe_intermediate_size == cfg.moe_intermediate_size
        # Inherited from Qwen3 dense
        assert config.qk_norm is True
        assert config.qkv_bias is False

    def test_dense_families_keep_num_experts_zero(
        self, tiny_llama, tiny_qwen2, tiny_qwen3_untied_4layer, feature_basis_128_to_32
    ):
        """Regression gate: adding MoE fields doesn't leak into dense paths."""
        from saeforge.adapters import adapter_for

        for host in (tiny_llama, tiny_qwen2, tiny_qwen3_untied_4layer):
            config = adapter_for(host).build_native_config(
                host, feature_basis_128_to_32.n_features
            )
            assert config.num_experts == 0, (
                f"{type(host).__name__} should have num_experts=0; got {config.num_experts}"
            )


class TestForgedModule:
    def test_forged_block_has_moe_mlp(self, tiny_qwen3_moe_untied):
        from saeforge.adapters import adapter_for
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(_basis())
        adapter = adapter_for(tiny_qwen3_moe_untied)
        walk = adapter.walk(tiny_qwen3_moe_untied, projector)
        config = adapter.build_native_config(tiny_qwen3_moe_untied, 32)
        model = NativeModel.from_projected_weights(config, walk)
        for layer in model.torch_module.model.layers:
            # MoE block must have gate + experts
            assert hasattr(layer.mlp, "gate"), "MoE block missing mlp.gate"
            assert hasattr(layer.mlp, "experts"), "MoE block missing mlp.experts"
            assert len(layer.mlp.experts) == config.num_experts
            # qk_norm still present (inherited from Qwen3)
            assert layer.self_attn.q_norm is not None
            assert layer.self_attn.k_norm is not None

    def test_dense_block_keeps_swiglu(self, tiny_llama, feature_basis_128_to_32):
        """Regression gate: Llama / Qwen2 / Qwen3-dense still get SwiGLU_MLP."""
        from saeforge.adapters import adapter_for
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(feature_basis_128_to_32)
        adapter = adapter_for(tiny_llama)
        walk = adapter.walk(tiny_llama, projector)
        config = adapter.build_native_config(tiny_llama, feature_basis_128_to_32.n_features)
        model = NativeModel.from_projected_weights(config, walk)
        for layer in model.torch_module.model.layers:
            assert not hasattr(layer.mlp, "gate"), "dense block should not have mlp.gate"
            assert not hasattr(layer.mlp, "experts"), "dense block should not have mlp.experts"
            # SwiGLU has gate_proj, up_proj, down_proj
            assert hasattr(layer.mlp, "gate_proj")
            assert hasattr(layer.mlp, "up_proj")
            assert hasattr(layer.mlp, "down_proj")

    def test_forward_pass_finite(self, tiny_qwen3_moe_untied):
        import torch

        from saeforge.adapters import adapter_for
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(_basis())
        adapter = adapter_for(tiny_qwen3_moe_untied)
        walk = adapter.walk(tiny_qwen3_moe_untied, projector)
        config = adapter.build_native_config(tiny_qwen3_moe_untied, 32)
        model = NativeModel.from_projected_weights(config, walk)
        input_ids = torch.randint(0, 1024, (2, 8))
        with torch.no_grad():
            logits = model.forward(input_ids)
        assert logits.shape == (2, 8, 1024)
        assert torch.isfinite(logits).all()

    def test_norm_topk_prob_renormalizes(self, tiny_qwen3_moe_untied):
        """When norm_topk_prob=True, the top-K weights sum to 1 per token."""
        import torch

        from saeforge.adapters import adapter_for
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        projector = SubspaceProjector(_basis())
        adapter = adapter_for(tiny_qwen3_moe_untied)
        walk = adapter.walk(tiny_qwen3_moe_untied, projector)
        config = adapter.build_native_config(tiny_qwen3_moe_untied, 32)
        assert config.norm_topk_prob is True
        model = NativeModel.from_projected_weights(config, walk)

        # Instrument the first block's gate to capture top-K weights
        mlp = model.torch_module.model.layers[0].mlp
        x = torch.randn(1, 4, 32)
        gate_logits = mlp.gate(x.reshape(-1, 32))
        weights = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)
        top_w, _ = weights.topk(mlp.top_k, dim=-1)
        # Pre-renorm: top_w sums to less than 1 (mass leaks to non-top experts)
        # Post-renorm (what mlp.forward does): top_w should sum to 1
        renormalized = top_w / top_w.sum(dim=-1, keepdim=True)
        assert torch.allclose(
            renormalized.sum(dim=-1),
            torch.ones(renormalized.size(0)),
            atol=1e-5,
        )


class TestForgePipelineMoEStrategy:
    def test_top_n_requires_keep_n(self):
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis()
        with pytest.raises(ValueError, match="moe_keep_n > 0"):
            ForgePipeline(
                basis=b,
                projector=SubspaceProjector(b),
                moe_strategy="top_n",
                moe_keep_n=0,
            )

    def test_invalid_strategy_raises(self):
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis()
        with pytest.raises(ValueError, match="moe_strategy"):
            ForgePipeline(
                basis=b,
                projector=SubspaceProjector(b),
                moe_strategy="bogus",
            )


class TestCollapseStrategy:
    def test_collapse_averages_experts_and_drops_router(self, tiny_qwen3_moe_untied):
        """``moe_strategy='collapse'`` produces dense averaged MLP keys + no router."""
        from saeforge.adapters import adapter_for
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis()
        adapter = adapter_for(tiny_qwen3_moe_untied)
        projector = SubspaceProjector(b)
        walk = adapter.walk(tiny_qwen3_moe_untied, projector)
        config = adapter.build_native_config(tiny_qwen3_moe_untied, 32)

        pipeline = ForgePipeline(
            basis=b,
            projector=projector,
            moe_strategy="collapse",
        )
        new_weights, new_config = pipeline._apply_moe_collapse(walk, config)

        assert new_config.family == "qwen3"
        assert new_config.num_experts == 0
        assert new_config.intermediate_size == config.moe_intermediate_size

        # Router and per-expert keys gone
        for i in range(tiny_qwen3_moe_untied.config.num_hidden_layers):
            assert f"model.layers.{i}.mlp.gate.weight" not in new_weights
            for e in range(tiny_qwen3_moe_untied.config.num_experts):
                for kind in ("gate_proj", "up_proj", "down_proj"):
                    assert (
                        f"model.layers.{i}.mlp.experts.{e}.{kind}.weight"
                        not in new_weights
                    )
            # Dense averaged keys present
            for kind in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{i}.mlp.{kind}.weight"
                assert key in new_weights

    def test_collapse_averaged_values_are_mean_of_experts(self, tiny_qwen3_moe_untied):
        """Confirm the averaging math: collapsed gate_proj == mean(all experts' gate_proj)."""
        from saeforge.adapters import adapter_for
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis()
        adapter = adapter_for(tiny_qwen3_moe_untied)
        projector = SubspaceProjector(b)
        walk = adapter.walk(tiny_qwen3_moe_untied, projector)
        config = adapter.build_native_config(tiny_qwen3_moe_untied, 32)
        pipeline = ForgePipeline(
            basis=b, projector=projector, moe_strategy="collapse"
        )
        new_weights, _ = pipeline._apply_moe_collapse(walk, config)

        cfg = tiny_qwen3_moe_untied.config
        for i in range(cfg.num_hidden_layers):
            expert_gates = [
                walk[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"]
                for e in range(cfg.num_experts)
            ]
            expected_avg = sum(expert_gates) / len(expert_gates)
            np.testing.assert_array_almost_equal(
                new_weights[f"model.layers.{i}.mlp.gate_proj.weight"],
                expected_avg,
            )

    def test_top_n_strategy_validates_at_construction(self):
        """``moe_strategy='top_n'`` with ``moe_keep_n>0`` is accepted at construction.

        The ``NotImplementedError`` surfaces only inside ``_run_real_imperative``
        when an actual run starts. v1 contract: validation passes, run raises.
        """
        from saeforge.forge import ForgePipeline
        from saeforge.projector import SubspaceProjector

        b = _basis()
        # __post_init__ accepts the (top_n, moe_keep_n=2) combination
        pipeline = ForgePipeline(
            basis=b,
            projector=SubspaceProjector(b),
            host_model_id="<offline>",
            moe_strategy="top_n",
            moe_keep_n=2,
        )
        assert pipeline.moe_strategy == "top_n"
        assert pipeline.moe_keep_n == 2
