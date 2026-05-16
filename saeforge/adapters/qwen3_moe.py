"""Qwen3-MoE architecture adapter.

Qwen3-MoE replaces Qwen3 dense's single SwiGLU MLP with a router
(``mlp.gate``) + N independent expert MLPs
(``mlp.experts.{i}.{gate,up,down}_proj``), where the router selects top-K
experts per token and weighted-sums their outputs.

Inherits everything else from Qwen3 dense (Q/K per-head RMSNorm, no Q/K/V
biases, SwiGLU experts, RMSNorm, GQA, RoPE). The walker's MoE emission is
host-attribute-driven (``hasattr(block.mlp, "experts")``) and lives in the
shared ``LlamaAdapter.walk`` — this adapter just stamps ``family="qwen3_moe"``
and detects the MoE-specific fields for ``NativeModelConfig``.

What's intentionally NOT replicated:

- Sliding-window attention. The native module uses standard causal attention
  everywhere; long-context drift accepted as ``ε_attn`` per
  ``docs/algorithm.md`` §5.

Requires ``transformers >= 4.51`` (Qwen3-MoE landed alongside Qwen3 dense
in 4.51). The registration at the bottom of this module is guarded with a
try/except so older installs silently skip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saeforge.adapters.qwen3 import Qwen3Adapter

if TYPE_CHECKING:  # pragma: no cover
    from saeforge.model import NativeModelConfig


class Qwen3MoEAdapter(Qwen3Adapter):
    """Adapter for HF :class:`transformers.Qwen3MoeForCausalLM`."""

    family = "qwen3_moe"

    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> "NativeModelConfig":
        from dataclasses import replace

        base = super().build_native_config(
            host, n_features, attention_width=attention_width
        )
        cfg = host.config
        return replace(
            base,
            family=self.family,
            num_experts=int(getattr(cfg, "num_experts", 0)),
            num_experts_per_tok=int(getattr(cfg, "num_experts_per_tok", 0)),
            moe_intermediate_size=int(getattr(cfg, "moe_intermediate_size", 0)),
            norm_topk_prob=bool(getattr(cfg, "norm_topk_prob", True)),
        )


try:
    from transformers import Qwen3MoeForCausalLM

    from saeforge.adapters import register_adapter

    register_adapter(Qwen3MoeForCausalLM, Qwen3MoEAdapter())
except ImportError:  # pragma: no cover — transformers < 4.51
    pass
