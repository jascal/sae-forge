"""NativeModel — small transformer whose residual width equals the feature-basis size.

Why an in-tree implementation: a v0 forged model has residual width ``n_features``
but inherits the host's attention internal width (``n_heads * head_dim``) and MLP
inner width. Those don't generally factor as ``n_features = n_heads * head_dim``,
so stock ``GPT2LMHeadModel`` config-driven shapes don't apply. The minimal nn.Module
below preserves the host's internal widths and projects only the residual-touching
edges of every block.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from saeforge.projector import SubspaceProjector
from saeforge.utils.lazy import require_extra


@dataclass
class NativeModelConfig:
    """Architecture knobs for a forged native model.

    ``hidden_size`` is fixed by the feature basis (``basis.n_features``).
    ``qkv_inner_size`` and ``intermediate_size`` are inherited from the host.
    """

    hidden_size: int
    qkv_inner_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int = 1024
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.qkv_inner_size:
            raise ValueError(
                f"qkv_inner_size {self.qkv_inner_size} must equal "
                f"num_heads ({self.num_heads}) * head_dim ({self.head_dim})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> NativeModelConfig:
        return cls(**payload)


def _build_torch_module(config: NativeModelConfig):
    """Construct the torch nn.Module skeleton. Lazy-imports torch."""
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
        def __init__(self, cfg: NativeModelConfig):
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
        def __init__(self, cfg: NativeModelConfig):
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
        def __init__(self, cfg: NativeModelConfig):
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
        def __init__(self, cfg: NativeModelConfig):
            super().__init__()
            self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.wpe = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
            self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
            self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)

        def forward(self, input_ids):
            seq = input_ids.size(-1)
            pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            x = self.wte(input_ids) + self.wpe(pos)
            for block in self.h:
                x = block(x)
            return self.ln_f(x)

    class ForgedGPT2(nn.Module):
        def __init__(self, cfg: NativeModelConfig):
            super().__init__()
            self.config = cfg
            self.transformer = Transformer(cfg)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        def forward(self, input_ids):
            hidden = self.transformer(input_ids)
            return self.lm_head(hidden)

    return ForgedGPT2(config)


class NativeModel:
    """Forged transformer with a feature-basis-width residual stream."""

    def __init__(self, config: NativeModelConfig):
        self.config = config
        self._module = _build_torch_module(config)

    @property
    def torch_module(self):
        return self._module

    def parameters(self):
        return self._module.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._module.parameters())

    def forward(self, input_ids):
        return self._module(input_ids)

    @classmethod
    def from_host(
        cls,
        host_model_id: str,
        projector: SubspaceProjector,
        *,
        dtype: str = "float32",
        device: str = "cpu",
    ) -> NativeModel:
        """Construct a native model by projecting ``host_model_id``'s weights through ``projector``."""
        transformers = require_extra("transformers", "torch")
        host = transformers.GPT2LMHeadModel.from_pretrained(host_model_id).eval()
        weights = projector.project_module(host)
        config = _config_from_host(host, projector.basis.n_features)
        model = cls.from_projected_weights(config, weights)
        model._move(dtype=dtype, device=device)
        return model

    @classmethod
    def from_projected_weights(
        cls,
        config: NativeModelConfig,
        weights: dict[str, np.ndarray],
    ) -> NativeModel:
        """Assemble a native model from a dict of pre-projected ``np.ndarray`` weights."""
        torch = require_extra("torch", "torch")
        model = cls(config)
        state = model._module.state_dict()
        for name, arr in weights.items():
            target = name
            # HF GPT2's lm_head linear stores weight as (vocab, hidden), matching our key
            if target not in state:
                raise KeyError(f"projected key {name!r} has no slot in NativeModel state_dict")
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(state[target].dtype)
            if tensor.shape != state[target].shape:
                raise ValueError(
                    f"shape mismatch for {target}: projected {tensor.shape}, "
                    f"expected {tuple(state[target].shape)}"
                )
            state[target] = tensor
        model._module.load_state_dict(state)
        return model

    def _move(self, dtype: str, device: str) -> None:
        torch = require_extra("torch", "torch")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}; choose from {list(dtype_map)}")
        self._module.to(dtype=dtype_map[dtype], device=device)

    def save_pretrained(self, output_dir: str | Path) -> None:
        from safetensors.torch import save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        state = {k: v.contiguous() for k, v in self._module.state_dict().items()}
        save_file(state, str(output_dir / "model.safetensors"))

    @classmethod
    def load_pretrained(cls, input_dir: str | Path) -> NativeModel:
        from safetensors.torch import load_file

        input_dir = Path(input_dir)
        config = NativeModelConfig.from_dict(json.loads((input_dir / "config.json").read_text()))
        model = cls(config)
        state = load_file(str(input_dir / "model.safetensors"))
        model._module.load_state_dict(state)
        return model


def _config_from_host(host_model, n_features: int) -> NativeModelConfig:
    """Pull the host's per-block dimensions and merge with the basis-width residual."""
    cfg = host_model.config
    head_dim = cfg.n_embd // cfg.n_head
    return NativeModelConfig(
        hidden_size=n_features,
        qkv_inner_size=cfg.n_embd,
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        head_dim=head_dim,
        intermediate_size=cfg.n_inner if cfg.n_inner is not None else 4 * cfg.n_embd,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.n_positions,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
    )
