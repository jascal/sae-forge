"""Gemma-2 architecture adapter.

Gemma-2 shares the Llama-shaped attention + SwiGLU MLP layout, with
two architectural additions:

1. **Four RMSNorms per block** — Gemma-2 wraps the attention and MLP
   blocks with extra norms inside the residual:

   ```
   x = x + post_attention_layernorm(self_attn(input_layernorm(x)))
   x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
   ```

   vs. Llama's two:

   ```
   x = x + self_attn(input_layernorm(x))
   x = x + mlp(post_attention_layernorm(x))
   ```

   The two extra norms (``pre_feedforward_layernorm``,
   ``post_feedforward_layernorm``) project via the same
   ``project_residual_aligned`` helper that Llama uses.

2. **Optional logit soft-capping** — ``final_logit_softcapping`` and
   ``attn_logit_softcapping`` clamp logits / attention scores via
   ``tanh(x / cap) * cap``. Surfaced on the ``NativeModelConfig`` and
   applied at forward time by the shared
   :func:`saeforge.adapters.llama.build_llama_family_module` factory.
   The projection itself is unaffected.

What's intentionally NOT replicated in v0.2:

- Gemma-2's alternating local/global attention (sliding-window mask).
  The native module uses the standard causal mask everywhere; the
  long-context drift is accepted as ``ε_attn`` per
  ``docs/algorithm.md`` §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


from saeforge.adapters.llama import LlamaAdapter

if TYPE_CHECKING:  # pragma: no cover
    from saeforge.model import NativeModelConfig


class Gemma2Adapter(LlamaAdapter):
    """Adapter for HF :class:`transformers.Gemma2ForCausalLM`.

    Inherits the Llama walker logic and adds the two extra per-layer
    RMSNorm keys plus the soft-cap config-passthrough.
    """

    family = "gemma2"

    # Override the Llama list to add the two extra Gemma-2 norms.
    _norms_per_layer = (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    )

    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> "NativeModelConfig":
        # Build the Llama-shaped config first, then stamp the Gemma-2
        # extras (family + softcaps) on top via dataclasses.replace.
        from dataclasses import replace

        base = super().build_native_config(
            host, n_features, attention_width=attention_width
        )
        cfg = host.config
        return replace(
            base,
            family=self.family,
            final_logit_softcap=getattr(cfg, "final_logit_softcapping", None),
            attn_logit_softcap=getattr(cfg, "attn_logit_softcapping", None),
        )

    def native_module_class(self) -> type:
        # Same factory as Llama — the family field on the config branches
        # the forward pass for the four-norm Gemma-2 layout.
        from saeforge.adapters.llama import _get_forged_llama_class

        return _get_forged_llama_class()


# Register at import time. ``Gemma2ForCausalLM`` does NOT subclass
# ``LlamaForCausalLM`` so registration order doesn't matter for
# correctness — but we register Gemma-2 before Llama anyway to keep the
# more specific-first-discovered order documented in __init__.py.
try:
    from transformers import Gemma2ForCausalLM

    from saeforge.adapters import register_adapter

    register_adapter(Gemma2ForCausalLM, Gemma2Adapter())
except ImportError:  # pragma: no cover
    pass
