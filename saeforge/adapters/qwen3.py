"""Qwen3 dense architecture adapter.

Qwen3 dense is architecturally Llama-shaped (SwiGLU MLP, RMSNorm, GQA, RoPE)
with one structural addition vs Qwen2: per-head RMSNorm on Q and K applied
between the per-head reshape and the scaled dot-product. The weights are
head-dim aligned (shape ``(head_dim,)``), so they pass through the projector
unchanged — same pattern as Qwen2's Q/K/V biases.

The Llama walker emits ``q_norm`` / ``k_norm`` pass-through automatically
whenever the host has those submodules (host-attribute-driven, family-
agnostic). The Llama factory's :class:`LlamaSelfAttention` constructs the
forward-side RMSNorm modules when ``cfg.qk_norm`` is True.

Qwen3 dense also drops Qwen2's Q/K/V biases. The inherited ``qkv_bias``
auto-detection in :meth:`LlamaAdapter.build_native_config` picks up
``qkv_bias=False`` from the host's first block, so no Qwen3-specific bias
code is needed.

What's intentionally NOT replicated:

- Sliding-window attention. The native module uses standard causal
  attention everywhere; long-context drift accepted as ``ε_attn`` per
  ``docs/algorithm.md`` §5.

Requires ``transformers >= 4.51`` (Qwen3 support landed in 4.51). The
registration at the bottom of this module is guarded with a try/except so
older installs (the ``[intel]`` extra is capped at 4.49) silently skip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saeforge.adapters.llama import LlamaAdapter

if TYPE_CHECKING:  # pragma: no cover
    from saeforge.model import NativeModelConfig


class Qwen3Adapter(LlamaAdapter):
    """Adapter for HF :class:`transformers.Qwen3ForCausalLM`."""

    family = "qwen3"

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
        # Detect Q/K-norm presence directly on the host's first block.
        # Robust across HF config-attribute renames; the truth is the
        # actual nn.Module structure the HF loader built.
        qk_norm = (
            len(host.model.layers) > 0
            and getattr(host.model.layers[0].self_attn, "q_norm", None) is not None
        )
        return replace(base, family=self.family, qk_norm=qk_norm)


try:
    from transformers import Qwen3ForCausalLM

    from saeforge.adapters import register_adapter

    register_adapter(Qwen3ForCausalLM, Qwen3Adapter())
except ImportError:  # pragma: no cover — exercised on transformers < 4.51
    pass
