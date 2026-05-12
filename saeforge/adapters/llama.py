"""Llama-3 architecture adapter.

Walks an HF ``LlamaForCausalLM`` host into a projected weight dict
keyed by the same parameter names the matching native module declares.
The native module factory (`build_llama_family_module`) is shared with
the Gemma-2 adapter — Gemma-2 differs only in the four-norm-per-block
layout and the optional logit soft-cap on ``lm_head``.

Projection algebra for Llama-shaped matrices (HF ``Linear.weight``
stores as ``(out, in)``):

- ``embed_tokens.weight (V, d)`` → ``project_embed`` → ``(V, f)``.
- ``q_proj.weight (n_q_heads * head_dim, d)`` →
  ``project_residual_output`` (acts on the in-axis) → ``(d_q, f)``.
- ``k_proj`` / ``v_proj`` → same shape with ``d_kv =
  n_kv_heads * head_dim`` rows.
- ``o_proj.weight (d, n_q_heads * head_dim)`` →
  ``project_residual_input`` (acts on the first/residual axis) →
  ``(f, d_q)``.
- ``gate_proj`` / ``up_proj.weight (i, d)`` →
  ``project_residual_output`` → ``(i, f)``.
- ``down_proj.weight (d, i)`` → ``project_residual_input`` → ``(f, i)``.
- RMSNorm ``weight (d,)`` → ``project_residual_aligned`` → ``(f,)``.
- ``lm_head.weight (V, d)`` → ``project_unembed`` → ``(V, f)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from saeforge.adapters.base import ArchitectureAdapter, to_numpy
from saeforge.utils.lazy import require_extra

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from saeforge.model import NativeModelConfig
    from saeforge.projector import SubspaceProjector


class LlamaAdapter(ArchitectureAdapter):
    """Adapter for HF :class:`transformers.LlamaForCausalLM`.

    Llama-3 is the canonical instance; earlier Llama generations work
    too via the same class. ``attention_width="feature_native"`` is
    not supported on Llama / Gemma-2 in v0.2 — the per-head GQA
    structure requires more thought than the GPT-2 fused-c_attn case.
    """

    family = "llama"

    # Number of per-layer RMSNorm weights this family emits.
    _norms_per_layer: tuple[str, ...] = (
        "input_layernorm",
        "post_attention_layernorm",
    )

    def walk(
        self,
        host: Any,
        projector: "SubspaceProjector",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        if attention_width != "host":
            raise NotImplementedError(
                f"{type(self).__name__} only supports attention_width="
                f"'host' in v0.2; got {attention_width!r}. The "
                f"feature-native attention path is GPT-2-only for now."
            )

        out: dict[str, np.ndarray] = {}

        out["model.embed_tokens.weight"] = projector.project_embed(
            to_numpy(host.model.embed_tokens.weight)
        )

        for i, block in enumerate(host.model.layers):
            prefix = f"model.layers.{i}"

            # Q/K/V/O — HF Linear.weight is (out, in); project_residual_output
            # acts on the *in* (right) axis and returns (out, n_features).
            out[f"{prefix}.self_attn.q_proj.weight"] = projector.project_residual_output(
                to_numpy(block.self_attn.q_proj.weight)
            )
            out[f"{prefix}.self_attn.k_proj.weight"] = projector.project_residual_output(
                to_numpy(block.self_attn.k_proj.weight)
            )
            out[f"{prefix}.self_attn.v_proj.weight"] = projector.project_residual_output(
                to_numpy(block.self_attn.v_proj.weight)
            )
            # Qwen2 has biases on Q/K/V; Llama and Gemma-2 don't. The bias
            # lives in head_dim space (not residual space), so it passes
            # through unprojected.
            for qkv in ("q_proj", "k_proj", "v_proj"):
                b = getattr(block.self_attn, qkv).bias
                if b is not None:
                    out[f"{prefix}.self_attn.{qkv}.bias"] = to_numpy(b)
            # o_proj is (hidden, n_q_heads * head_dim); the *first* axis
            # is the residual one, so project_residual_input applies.
            out[f"{prefix}.self_attn.o_proj.weight"] = projector.project_residual_input(
                to_numpy(block.self_attn.o_proj.weight)
            )

            # SwiGLU — gate / up are (intermediate, hidden), project the
            # right axis; down is (hidden, intermediate), project the
            # left/residual axis.
            out[f"{prefix}.mlp.gate_proj.weight"] = projector.project_residual_output(
                to_numpy(block.mlp.gate_proj.weight)
            )
            out[f"{prefix}.mlp.up_proj.weight"] = projector.project_residual_output(
                to_numpy(block.mlp.up_proj.weight)
            )
            out[f"{prefix}.mlp.down_proj.weight"] = projector.project_residual_input(
                to_numpy(block.mlp.down_proj.weight)
            )

            # RMSNorm γ — residual-aligned. No β (RMSNorm has no bias).
            for norm_name in self._norms_per_layer:
                out[f"{prefix}.{norm_name}.weight"] = projector.project_residual_aligned(
                    to_numpy(getattr(block, norm_name).weight)
                )

        out["model.norm.weight"] = projector.project_residual_aligned(
            to_numpy(host.model.norm.weight)
        )

        # Tied embeddings: HF aliases lm_head.weight to embed_tokens.weight
        # at load time. We omit it from the walk so the native module's
        # constructor can do its own aliasing — re-projecting both would
        # produce subtly different results due to numpy/torch dtype
        # round-trips.
        if not getattr(host.config, "tie_word_embeddings", False):
            out["lm_head.weight"] = projector.project_unembed(
                to_numpy(host.lm_head.weight)
            )

        return out

    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> "NativeModelConfig":
        from saeforge.model import NativeModelConfig

        cfg = host.config
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        # Detect Q/K/V bias on host's first block. Llama/Gemma-2 don't have
        # them; Qwen2 does. Tested on the actual nn.Linear so we work even
        # for variants that don't expose a bias flag on the HF config.
        qkv_bias = (
            len(host.model.layers) > 0
            and host.model.layers[0].self_attn.q_proj.bias is not None
        )
        return NativeModelConfig(
            family=self.family,
            hidden_size=n_features,
            qkv_inner_size=cfg.num_attention_heads * head_dim,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            head_dim=head_dim,
            intermediate_size=cfg.intermediate_size,
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            layer_norm_epsilon=getattr(cfg, "rms_norm_eps", 1e-6),
            attention_width=attention_width,
            n_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            tied_embeddings=getattr(cfg, "tie_word_embeddings", False),
            rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
            qkv_bias=qkv_bias,
        )

    def native_module_class(self) -> type:
        return _get_forged_llama_class()

    def grad_checkpoint_targets(self, module):
        # ForgedLlama (also used by Gemma2Adapter): every block is at
        # ``module.model.layers.{i}``; the input embedding is
        # ``model.embed_tokens.weight``. Gemma-2 inherits this method
        # via Gemma2Adapter(LlamaAdapter); the four-norm-per-block
        # layout doesn't change the checkpoint targets.
        return module.model.layers, module.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Native module factory — shared between Llama and Gemma-2.
# ---------------------------------------------------------------------------


_FORGED_LLAMA_CLASS = None


def build_llama_family_module(config: "NativeModelConfig"):
    """Construct a Llama / Gemma-2 native module. Lazy-imports torch."""
    cls = _get_forged_llama_class()
    return cls(config)


def _get_forged_llama_class():
    global _FORGED_LLAMA_CLASS
    if _FORGED_LLAMA_CLASS is not None:
        return _FORGED_LLAMA_CLASS

    torch = require_extra("torch", "torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    class LlamaSelfAttention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.num_heads = cfg.num_heads
            self.n_kv_heads = cfg.n_kv_heads
            self.head_dim = cfg.head_dim
            qkv_bias = getattr(cfg, "qkv_bias", False)
            self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=qkv_bias)
            self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=qkv_bias)
            self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
            self.attn_logit_softcap = cfg.attn_logit_softcap

        def forward(self, x):
            shape_prefix = x.shape[:-1]
            q = self.q_proj(x).view(*shape_prefix, self.num_heads, self.head_dim).transpose(-3, -2)
            k = self.k_proj(x).view(*shape_prefix, self.n_kv_heads, self.head_dim).transpose(-3, -2)
            v = self.v_proj(x).view(*shape_prefix, self.n_kv_heads, self.head_dim).transpose(-3, -2)
            # GQA: repeat K/V along the head axis to match Q's heads.
            n_groups = self.num_heads // self.n_kv_heads
            if n_groups > 1:
                k = k.repeat_interleave(n_groups, dim=-3)
                v = v.repeat_interleave(n_groups, dim=-3)
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            if self.attn_logit_softcap is not None:
                cap = float(self.attn_logit_softcap)
                scores = torch.tanh(scores / cap) * cap
            seq_len = scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = (attn @ v).transpose(-3, -2).contiguous()
            out = out.view(*out.shape[:-2], self.num_heads * self.head_dim)
            return self.o_proj(out)

    class SwiGLU_MLP(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class LlamaBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            eps = cfg.rms_norm_eps if cfg.rms_norm_eps is not None else 1e-6
            self.input_layernorm = RMSNorm(cfg.hidden_size, eps=eps)
            self.self_attn = LlamaSelfAttention(cfg)
            self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=eps)
            self.mlp = SwiGLU_MLP(cfg)
            self.family = cfg.family
            # Gemma-2 wraps the attn and mlp blocks with extra norms; the
            # residual loop becomes:
            #   x = x + post_attention_layernorm(self_attn(input_layernorm(x)))
            #   x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
            # vs. Llama's:
            #   x = x + self_attn(input_layernorm(x))
            #   x = x + mlp(post_attention_layernorm(x))
            if self.family == "gemma2":
                self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=eps)
                self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=eps)

        def forward(self, x):
            if self.family == "gemma2":
                attn_out = self.self_attn(self.input_layernorm(x))
                x = x + self.post_attention_layernorm(attn_out)
                mlp_out = self.mlp(self.pre_feedforward_layernorm(x))
                x = x + self.post_feedforward_layernorm(mlp_out)
            else:
                x = x + self.self_attn(self.input_layernorm(x))
                x = x + self.mlp(self.post_attention_layernorm(x))
            return x

    class LlamaTransformer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            eps = cfg.rms_norm_eps if cfg.rms_norm_eps is not None else 1e-6
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList(
                [LlamaBlock(cfg) for _ in range(cfg.num_layers)]
            )
            self.norm = RMSNorm(cfg.hidden_size, eps=eps)
            # Hybrid-bridge insertion on the residual stream between the embed
            # and mid regions (after block 0) and between the mid and lm-head
            # regions (after block L-2). Mirrors the GPT-2 wiring in
            # ``saeforge/adapters/gpt2.py``. See the
            # ``hybrid-bridge-llama-family`` capability spec.
            self.bridges = self._build_bridges(cfg)

        @staticmethod
        def _build_bridges(cfg):
            if not getattr(cfg, "bridges", False):
                return None
            from saeforge.bridges import BridgeConfig, make_bridge

            bcfg = BridgeConfig(
                init=cfg.bridge_init,
                nonlin=cfg.bridge_nonlin,
                pre_layernorm=cfg.bridge_pre_layernorm,
                train=True,
            )
            return nn.ModuleDict(
                {
                    "emb_mid": make_bridge(cfg.hidden_size, bcfg),
                    "mid_lm": make_bridge(cfg.hidden_size, bcfg),
                }
            )

        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            n = len(self.layers)
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.bridges is not None:
                    if i == 0 and n >= 3:
                        x = self.bridges["emb_mid"](x)
                    elif i == n - 2 and n >= 3:
                        x = self.bridges["mid_lm"](x)
            return self.norm(x)

    class ForgedLlama(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.model = LlamaTransformer(cfg)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            if cfg.tied_embeddings:
                # Alias lm_head.weight to embed_tokens.weight (post-init so
                # the loader skips the lm_head slot when the walk omits it).
                self.lm_head.weight = self.model.embed_tokens.weight

        def forward(self, input_ids):
            hidden = self.model(input_ids)
            logits = self.lm_head(hidden)
            if self.config.final_logit_softcap is not None:
                cap = float(self.config.final_logit_softcap)
                logits = torch.tanh(logits / cap) * cap
            return logits

    _FORGED_LLAMA_CLASS = ForgedLlama
    return ForgedLlama


# Register at module-import time. As with the GPT-2 adapter, the HF
# class is lazy-loaded so importing the module without [torch] doesn't
# crash the package.
try:
    from transformers import LlamaForCausalLM

    from saeforge.adapters import register_adapter

    register_adapter(LlamaForCausalLM, LlamaAdapter())
except ImportError:  # pragma: no cover
    pass
