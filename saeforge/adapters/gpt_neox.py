"""GPT-NeoX / Pythia architecture adapter (``add-gpt-neox-adapter``).

Forges ``GPTNeoXForCausalLM`` hosts (EleutherAI Pythia 14m–2.8b, GPT-NeoX-20B, etc.). GPT-NeoX combines
features no existing adapter had together, so this module adds them:

- **Parallel residual** (``use_parallel_residual=True``): attention reads ``input_layernorm(x)`` and the MLP
  reads ``post_attention_layernorm(x)`` — both off the SAME pre-block ``x``, summed: ``x + attn + mlp``
  (not sequential like GPT-2 / Llama).
- **Partial rotary** (``rotary_pct``, e.g. 0.25): RoPE on only the first ``int(head_dim*rotary_pct)`` dims of
  each head; the rest pass through. Uses :func:`saeforge._positional.rope.apply_rotary_pos_emb_partial`.
- **LayerNorm with bias** (not RMSNorm) on every norm + a final layer norm.
- **Fused QKV** ``query_key_value`` ``(3*hidden, hidden)`` reshaped per-head ``[q|k|v]`` (HF's
  ``view(..., num_heads, 3*head_dim).chunk(3, -1)``), separate ``dense`` out-proj, GELU MLP
  (``dense_h_to_4h`` / ``dense_4h_to_h``) — all with biases — and **untied** ``embed_in`` / ``embed_out``.

The native forward is validated against a real (tiny-random) ``GPTNeoXForCausalLM`` on an identity basis: the
forged logits match the host's to float tolerance (see ``scripts/prototype_gpt_neox.py`` /
``tests/test_gpt_neox_adapter.py``). v1: ``attention_width="host"`` only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saeforge.adapters.base import ArchitectureAdapter, to_numpy
from saeforge.utils.lazy import require_extra


class GPTNeoXAdapter(ArchitectureAdapter):
    """Adapter for HF GPT-NeoX hosts (``GPTNeoXForCausalLM`` / ``GPTNeoXModel``)."""

    family = "gpt_neox"

    def walk(
        self,
        host: Any,
        projector: "Any",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        if attention_width != "host":
            raise NotImplementedError(
                f"GPTNeoXAdapter v1 supports attention_width='host' only; got {attention_width!r}. "
                f"feature_native (both-sides QKV projection) is a follow-up."
            )
        try:
            from transformers import GPTNeoXForCausalLM, GPTNeoXModel
        except ImportError as e:
            raise ImportError(
                "GPTNeoXAdapter.walk needs the [torch] extra; install with `pip install sae-forge[torch]`."
            ) from e

        if isinstance(host, GPTNeoXForCausalLM):
            core = host.gpt_neox
            embed_out_weight = host.embed_out.weight
        elif isinstance(host, GPTNeoXModel):
            core = host
            embed_out_weight = None
        else:  # pragma: no cover — registry should never dispatch here
            raise NotImplementedError(
                f"GPTNeoXAdapter.walk only handles HF GPT-NeoX hosts; got {type(host).__name__}"
            )

        out: dict[str, np.ndarray] = {}
        out["gpt_neox.embed_in.weight"] = projector.project_embed(to_numpy(core.embed_in.weight))

        for i, block in enumerate(core.layers):
            p = f"gpt_neox.layers.{i}"
            # Two LayerNorms (weight + bias), both residual-aligned.
            for norm in ("input_layernorm", "post_attention_layernorm"):
                ln = getattr(block, norm)
                out[f"{p}.{norm}.weight"] = projector.project_residual_aligned(to_numpy(ln.weight))
                out[f"{p}.{norm}.bias"] = projector.project_residual_aligned(to_numpy(ln.bias))
            # Attention: fused query_key_value reads the residual (project the d_model/in axis); its bias is
            # in head space (3*hidden) → unprojected. dense writes the residual; its bias is residual-space.
            attn = block.attention
            out[f"{p}.attention.query_key_value.weight"] = projector.project_residual_output(
                to_numpy(attn.query_key_value.weight)
            )
            if attn.query_key_value.bias is not None:
                out[f"{p}.attention.query_key_value.bias"] = to_numpy(attn.query_key_value.bias)
            out[f"{p}.attention.dense.weight"] = projector.project_residual_input(
                to_numpy(attn.dense.weight)
            )
            if attn.dense.bias is not None:
                out[f"{p}.attention.dense.bias"] = projector.project_residual_bias(to_numpy(attn.dense.bias))
            # MLP (GELU): h_to_4h reads residual (bias in MLP-hidden space → unprojected); 4h_to_h writes it.
            mlp = block.mlp
            out[f"{p}.mlp.dense_h_to_4h.weight"] = projector.project_residual_output(
                to_numpy(mlp.dense_h_to_4h.weight)
            )
            if mlp.dense_h_to_4h.bias is not None:
                out[f"{p}.mlp.dense_h_to_4h.bias"] = to_numpy(mlp.dense_h_to_4h.bias)
            out[f"{p}.mlp.dense_4h_to_h.weight"] = projector.project_residual_input(
                to_numpy(mlp.dense_4h_to_h.weight)
            )
            if mlp.dense_4h_to_h.bias is not None:
                out[f"{p}.mlp.dense_4h_to_h.bias"] = projector.project_residual_bias(
                    to_numpy(mlp.dense_4h_to_h.bias)
                )

        out["gpt_neox.final_layer_norm.weight"] = projector.project_residual_aligned(
            to_numpy(core.final_layer_norm.weight)
        )
        out["gpt_neox.final_layer_norm.bias"] = projector.project_residual_aligned(
            to_numpy(core.final_layer_norm.bias)
        )
        if embed_out_weight is not None:
            out["embed_out.weight"] = projector.project_unembed(to_numpy(embed_out_weight))
        return out

    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> "Any":
        from saeforge.model import NativeModelConfig

        if attention_width != "host":
            raise NotImplementedError(
                f"GPTNeoXAdapter v1 supports attention_width='host' only; got {attention_width!r}."
            )
        cfg = host.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        attention_bias = bool(getattr(cfg, "attention_bias", True))
        # Modern transformers (>=4.5x) migrates rotary knobs into ``cfg.rope_parameters``
        # ({'rope_theta', 'partial_rotary_factor', ...}); older configs expose the legacy top-level
        # ``rotary_emb_base`` / ``rotary_pct``. Read rope_parameters first, then fall back.
        rp = getattr(cfg, "rope_parameters", None) or {}
        partial = rp.get(
            "partial_rotary_factor",
            getattr(cfg, "partial_rotary_factor", getattr(cfg, "rotary_pct", 1.0)),
        )
        theta = rp.get(
            "rope_theta", getattr(cfg, "rope_theta", getattr(cfg, "rotary_emb_base", 10000.0))
        )
        return NativeModelConfig(
            family="gpt_neox",
            hidden_size=n_features,
            qkv_inner_size=cfg.num_attention_heads * head_dim,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            head_dim=head_dim,
            intermediate_size=cfg.intermediate_size,
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            layer_norm_epsilon=float(getattr(cfg, "layer_norm_eps", 1e-5)),
            attention_width=attention_width,
            tied_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
            qkv_bias=attention_bias,
            rope_theta=float(theta),
            partial_rotary_factor=float(partial),
        )

    def native_module_class(self) -> type:
        return _get_forged_gpt_neox_class()

    def grad_checkpoint_targets(self, module):
        return module.gpt_neox.layers, module.gpt_neox.embed_in.weight


# ---------------------------------------------------------------------------
# Native module factory.
# ---------------------------------------------------------------------------

_FORGED_GPT_NEOX_CLASS = None


def build_gpt_neox_module(config: "Any"):
    """Construct a GPT-NeoX native module. Lazy-imports torch."""
    return _get_forged_gpt_neox_class()(config)


def _get_forged_gpt_neox_class():
    global _FORGED_GPT_NEOX_CLASS
    if _FORGED_GPT_NEOX_CLASS is not None:
        return _FORGED_GPT_NEOX_CLASS

    import math

    torch = require_extra("torch", "torch")
    import torch.nn as nn
    import torch.nn.functional as F

    from saeforge._positional.rope import (
        apply_rotary_pos_emb_partial,
        compute_rope_cache,
    )

    class GPTNeoXAttention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.num_heads = cfg.num_heads
            self.head_dim = cfg.head_dim
            inner = cfg.qkv_inner_size  # num_heads * head_dim
            bias = bool(getattr(cfg, "qkv_bias", True))
            self.query_key_value = nn.Linear(cfg.hidden_size, 3 * inner, bias=bias)
            self.dense = nn.Linear(inner, cfg.hidden_size, bias=bias)
            self.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
            self.rotary_ndims = int(self.head_dim * float(getattr(cfg, "partial_rotary_factor", 1.0)))

        def forward(self, x):
            prefix = x.shape[:-1]  # (..., seq)
            # Fused QKV → per-head [q|k|v]: HF does view(*input, num_heads, 3*head_dim).transpose, chunk(3,-1).
            qkv = self.query_key_value(x).view(*prefix, self.num_heads, 3 * self.head_dim).transpose(-3, -2)
            q, k, v = qkv.chunk(3, dim=-1)  # each (..., num_heads, seq, head_dim)
            # Partial rotary on the first rotary_ndims dims; cache built for rotary_ndims.
            if self.rotary_ndims > 0:
                cos, sin = compute_rope_cache(
                    q.shape[-2], self.rotary_ndims, theta=self.rope_theta, device=q.device, dtype=q.dtype,
                )
                q, k = apply_rotary_pos_emb_partial(q, k, cos, sin, self.rotary_ndims)
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            seq = scores.size(-1)
            causal = torch.triu(
                torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = (attn @ v).transpose(-3, -2).contiguous()
            out = out.view(*out.shape[:-2], self.num_heads * self.head_dim)
            return self.dense(out)

    class GPTNeoXMLP(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.dense_h_to_4h = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
            self.dense_4h_to_h = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

        def forward(self, x):
            return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))  # HF "gelu" = exact GELU

    class GPTNeoXLayer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            eps = cfg.layer_norm_epsilon
            self.input_layernorm = nn.LayerNorm(cfg.hidden_size, eps=eps)
            self.post_attention_layernorm = nn.LayerNorm(cfg.hidden_size, eps=eps)
            self.attention = GPTNeoXAttention(cfg)
            self.mlp = GPTNeoXMLP(cfg)

        def forward(self, x):
            # Parallel residual: attn and mlp BOTH read the pre-block x (through their own norms), summed.
            attn_out = self.attention(self.input_layernorm(x))
            mlp_out = self.mlp(self.post_attention_layernorm(x))
            return x + attn_out + mlp_out

    class GPTNeoXModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embed_in = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList([GPTNeoXLayer(cfg) for _ in range(cfg.num_layers)])
            self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)

        def forward(self, input_ids):
            x = self.embed_in(input_ids)  # no learned position embedding (rotary only)
            for layer in self.layers:
                x = layer(x)
            return self.final_layer_norm(x)

    class ForgedGPTNeoX(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.gpt_neox = GPTNeoXModel(cfg)
            self.embed_out = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        def forward(self, input_ids):
            return self.embed_out(self.gpt_neox(input_ids))

    _FORGED_GPT_NEOX_CLASS = ForgedGPTNeoX
    return _FORGED_GPT_NEOX_CLASS


# Register at module-import time (lazy HF class load so a no-[torch] install stays importable).
try:
    from transformers import GPTNeoXForCausalLM

    from saeforge.adapters import register_adapter

    register_adapter(GPTNeoXForCausalLM, GPTNeoXAdapter())
except ImportError:  # pragma: no cover
    pass
