"""GPT-2 architecture adapter.

Walks an HF ``GPT2LMHeadModel`` (or bare ``GPT2Model``) into a
projected weight dict keyed by HF parameter names. The walk semantics
are unchanged from v0.1's ``SubspaceProjector.project_module``; the
code has moved here so dispatch can be plugged in.

The native module factory (``_build_gpt2_module``) and the
config-from-host builder (``GPT2Adapter.build_native_config``) also
live in this module so a future contributor adding a new architecture
has one file to clone.
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


class GPT2Adapter(ArchitectureAdapter):
    """Adapter for HF GPT-2 models (``GPT2LMHeadModel``, ``GPT2Model``).

    The walk emits the same parameter set the v0.1 ``project_module``
    produced. ``attention_width="feature_native"`` (v0.2 opt-in) routes
    ``c_attn`` / ``c_proj`` through the both-sides projection.
    """

    family = "gpt2"

    def walk(
        self,
        host: Any,
        projector: "SubspaceProjector",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        if attention_width not in ("host", "feature_native"):
            raise ValueError(
                f"attention_width must be 'host' or 'feature_native'; "
                f"got {attention_width!r}"
            )
        try:
            from transformers import GPT2LMHeadModel, GPT2Model
        except ImportError as e:
            raise ImportError(
                "GPT2Adapter.walk needs the [torch] extra; install it "
                "with `pip install sae-forge[torch]`."
            ) from e

        if isinstance(host, GPT2LMHeadModel):
            transformer = host.transformer
            lm_head_weight = host.lm_head.weight
        elif isinstance(host, GPT2Model):
            transformer = host
            lm_head_weight = None
        else:  # pragma: no cover — registry should never dispatch here
            raise NotImplementedError(
                f"GPT2Adapter.walk only handles HF GPT-2 hosts; got "
                f"{type(host).__name__}"
            )

        out: dict[str, np.ndarray] = {}

        out["transformer.wte.weight"] = projector.project_embed(
            to_numpy(transformer.wte.weight)
        )
        out["transformer.wpe.weight"] = projector.project_embed(
            to_numpy(transformer.wpe.weight)
        )

        for i, block in enumerate(transformer.h):
            prefix = f"transformer.h.{i}"
            out[f"{prefix}.ln_1.weight"] = projector.project_residual_aligned(
                to_numpy(block.ln_1.weight)
            )
            out[f"{prefix}.ln_1.bias"] = projector.project_residual_aligned(
                to_numpy(block.ln_1.bias)
            )

            c_attn_w = to_numpy(block.attn.c_attn.weight)
            c_attn_b = to_numpy(block.attn.c_attn.bias)
            c_proj_w = to_numpy(block.attn.c_proj.weight)
            c_proj_b = to_numpy(block.attn.c_proj.bias)
            if attention_width == "feature_native":
                out[f"{prefix}.attn.c_attn.weight"] = projector.project_qkv_full(c_attn_w)
                bq, bk, bv = np.split(c_attn_b, 3)
                out[f"{prefix}.attn.c_attn.bias"] = np.concatenate(
                    [
                        projector.project_residual_bias(bq),
                        projector.project_residual_bias(bk),
                        projector.project_residual_bias(bv),
                    ]
                )
                out[f"{prefix}.attn.c_proj.weight"] = projector.project_residual_full(c_proj_w)
            else:
                out[f"{prefix}.attn.c_attn.weight"] = projector.project_residual_input(c_attn_w)
                out[f"{prefix}.attn.c_attn.bias"] = c_attn_b.copy()
                out[f"{prefix}.attn.c_proj.weight"] = projector.project_residual_output(c_proj_w)
            out[f"{prefix}.attn.c_proj.bias"] = projector.project_residual_bias(c_proj_b)

            out[f"{prefix}.ln_2.weight"] = projector.project_residual_aligned(
                to_numpy(block.ln_2.weight)
            )
            out[f"{prefix}.ln_2.bias"] = projector.project_residual_aligned(
                to_numpy(block.ln_2.bias)
            )
            out[f"{prefix}.mlp.c_fc.weight"] = projector.project_residual_input(
                to_numpy(block.mlp.c_fc.weight)
            )
            out[f"{prefix}.mlp.c_fc.bias"] = to_numpy(block.mlp.c_fc.bias).copy()
            out[f"{prefix}.mlp.c_proj.weight"] = projector.project_residual_output(
                to_numpy(block.mlp.c_proj.weight)
            )
            out[f"{prefix}.mlp.c_proj.bias"] = projector.project_residual_bias(
                to_numpy(block.mlp.c_proj.bias)
            )

        out["transformer.ln_f.weight"] = projector.project_residual_aligned(
            to_numpy(transformer.ln_f.weight)
        )
        out["transformer.ln_f.bias"] = projector.project_residual_aligned(
            to_numpy(transformer.ln_f.bias)
        )

        if lm_head_weight is not None:
            out["lm_head.weight"] = projector.project_unembed(
                to_numpy(lm_head_weight)
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
        if attention_width == "feature_native":
            qkv_inner = n_features
            head_dim = n_features // cfg.n_head
        else:
            qkv_inner = cfg.n_embd
            head_dim = cfg.n_embd // cfg.n_head
        return NativeModelConfig(
            family="gpt2",
            hidden_size=n_features,
            qkv_inner_size=qkv_inner,
            num_layers=cfg.n_layer,
            num_heads=cfg.n_head,
            head_dim=head_dim,
            intermediate_size=(
                cfg.n_inner if cfg.n_inner is not None else 4 * cfg.n_embd
            ),
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.n_positions,
            layer_norm_epsilon=cfg.layer_norm_epsilon,
            attention_width=attention_width,
        )

    def native_module_class(self) -> type:
        return _get_forged_gpt2_class()

    def grad_checkpoint_targets(self, module):
        # ForgedGPT2: every transformer block lives at
        # ``module.transformer.h.{i}``; the input embedding is
        # ``transformer.wte.weight``.
        return module.transformer.h, module.transformer.wte.weight


# ---------------------------------------------------------------------------
# Native module factory — produces the same `ForgedGPT2` shape that v0.1
# `_build_torch_module` produced. Defined here so a future contributor
# adding a new architecture has one self-contained file to mirror.
# ---------------------------------------------------------------------------


_FORGED_GPT2_CLASS = None  # cached after first build


def build_gpt2_module(config: "NativeModelConfig"):
    """Construct the GPT-2-shaped native module. Lazy-imports torch."""
    cls = _get_forged_gpt2_class()
    return cls(config)


def _get_forged_gpt2_class():
    global _FORGED_GPT2_CLASS
    if _FORGED_GPT2_CLASS is not None:
        return _FORGED_GPT2_CLASS

    torch = require_extra("torch", "torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class Conv1D(nn.Module):
        """HF GPT-2 style Conv1D: y = x @ weight + bias, weight shape (in, out)."""

        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(in_features, out_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            nn.init.normal_(self.weight, std=0.02)

        def forward(self, x):
            return torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight).view(
                *x.shape[:-1], self.bias.size(0)
            )

    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.n_heads = cfg.num_heads
            self.head_dim = cfg.head_dim
            self.qkv_inner = cfg.qkv_inner_size
            self.c_attn = Conv1D(cfg.hidden_size, 3 * cfg.qkv_inner_size)
            self.c_proj = Conv1D(cfg.qkv_inner_size, cfg.hidden_size)

        def forward(self, x):
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.qkv_inner, dim=-1)
            q = q.view(*q.shape[:-1], self.n_heads, self.head_dim).transpose(-3, -2)
            k = k.view(*k.shape[:-1], self.n_heads, self.head_dim).transpose(-3, -2)
            v = v.view(*v.shape[:-1], self.n_heads, self.head_dim).transpose(-3, -2)
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            seq = scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = (attn @ v).transpose(-3, -2).contiguous()
            out = out.view(*out.shape[:-2], self.qkv_inner)
            return self.c_proj(out)

    class MLP(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.c_fc = Conv1D(cfg.hidden_size, cfg.intermediate_size)
            self.c_proj = Conv1D(cfg.intermediate_size, cfg.hidden_size)
            self.activation = cfg.activation

        def forward(self, x):
            h = self.c_fc(x)
            if self.activation == "gelu":
                h = F.gelu(h, approximate="tanh")
            elif self.activation == "relu":
                h = F.relu(h)
            else:
                raise ValueError(f"unsupported activation {self.activation}")
            return self.c_proj(h)

    class Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.ln_1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
            self.attn = CausalSelfAttention(cfg)
            self.ln_2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
            self.mlp = MLP(cfg)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class Transformer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.wpe = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
            self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
            self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
            # Bridges are inserted on the forward path between block 0 → block 1
            # (the embed/mid boundary) and between block L-2 → block L-1 (the
            # mid/lm-head boundary). v1 implements this for GPT-2 only.
            # TODO(hybrid-bridge-llama-gemma): mirror the same construction in
            # the Llama/Gemma-2 native modules (saeforge/adapters/llama.py).
            # Tracked as deferred task in
            # openspec/changes/hybrid-bridge-forge/tasks.md §15. The
            # HybridBasisBundle routing layer is family-agnostic; only the
            # native nn.Module forward needs per-family wiring.
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
            seq = input_ids.size(-1)
            pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            x = self.wte(input_ids) + self.wpe(pos)
            n = len(self.h)
            for i, block in enumerate(self.h):
                x = block(x)
                if self.bridges is not None:
                    if i == 0 and n >= 3:
                        x = self.bridges["emb_mid"](x)
                    elif i == n - 2 and n >= 3:
                        x = self.bridges["mid_lm"](x)
            return self.ln_f(x)

    class ForgedGPT2(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.transformer = Transformer(cfg)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        def forward(self, input_ids):
            hidden = self.transformer(input_ids)
            return self.lm_head(hidden)

    _FORGED_GPT2_CLASS = ForgedGPT2
    return ForgedGPT2


# Register at module-import time. The HF class is lazy-loaded so importing
# this module without [torch] doesn't crash; instead, the registration
# below is skipped and the dispatcher will surface a clean
# `NotImplementedError` if a user tries to dispatch on a GPT-2 host
# without [torch].
try:
    from transformers import GPT2LMHeadModel, GPT2Model

    from saeforge.adapters import register_adapter

    register_adapter(GPT2LMHeadModel, GPT2Adapter())
    register_adapter(GPT2Model, GPT2Adapter())
except ImportError:  # pragma: no cover — exercised only without [torch]
    pass
