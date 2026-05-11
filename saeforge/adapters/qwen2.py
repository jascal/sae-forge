"""Qwen2 architecture adapter.

Qwen2 is architecturally Llama-shaped (SwiGLU MLP, RMSNorm, GQA, RoPE)
with one key difference: Q/K/V projections have biases. The Llama
adapter machinery is generalised to handle the bias when ``qkv_bias``
is set on :class:`saeforge.model.NativeModelConfig`; this adapter just
stamps ``family="qwen2"`` and inherits Llama's walker + the bias
auto-detection in :meth:`LlamaAdapter.build_native_config`.

What's intentionally NOT replicated:

- Qwen2's sliding-window attention. The native module uses standard
  causal attention everywhere; long-context drift accepted as
  ``ε_attn`` per ``docs/algorithm.md`` §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saeforge.adapters.llama import LlamaAdapter

if TYPE_CHECKING:  # pragma: no cover
    from saeforge.model import NativeModelConfig


class Qwen2Adapter(LlamaAdapter):
    """Adapter for HF :class:`transformers.Qwen2ForCausalLM`."""

    family = "qwen2"

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
        return replace(base, family=self.family)


try:
    from transformers import Qwen2ForCausalLM

    from saeforge.adapters import register_adapter

    register_adapter(Qwen2ForCausalLM, Qwen2Adapter())
except ImportError:  # pragma: no cover
    pass
