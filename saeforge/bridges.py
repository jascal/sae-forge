"""BridgeModule — learnable n_features × n_features alignment between adjacent bases.

A hybrid-bridge forge run inserts two ``BridgeModule`` instances on the forged
module's forward pass: one between the embed region and the mid region, one
between the mid region and the lm-head region. Each bridge is a square linear
map (optionally pre-LN'd, optionally non-linearly activated) on the residual
stream in basis-space (last dim ``n_features``).

The v1 default is ``init="orthogonal", nonlin="none", pre_layernorm=True``: a
LN-normalized purely-linear square map. The linear-default is intentional —
the algebraic concern is that a pure-linear bridge folds into adjacent
weights, so the bridge's benefit (if any) must come from initialization
isolation rather than added capacity. See
``openspec/changes/hybrid-bridge-forge/design.md`` § "The honest algebraic concern".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from saeforge.utils.lazy import require_extra


@dataclass
class BridgeConfig:
    """Knobs for ``BridgeModule``. See module docstring for the v1 default rationale."""

    init: Literal["orthogonal", "identity", "zero"] = "orthogonal"
    nonlin: Literal["none", "relu", "gelu"] = "none"
    pre_layernorm: bool = True
    train: bool = True


def _build_bridge_class():
    # Lazy-import triggers the friendly ImportError when [torch] is missing.
    require_extra("torch", "torch")
    import torch.nn as nn

    class BridgeModule(nn.Module):
        """Square ``n_features → n_features`` linear bridge with optional pre-LN and activation.

        Forward (in order): LN (if ``pre_layernorm``) → Linear → activation (if not ``none``).
        Last dim must equal ``n_features``; shape is preserved.
        """

        def __init__(self, n_features: int, config: BridgeConfig) -> None:
            super().__init__()
            if n_features < 2:
                raise ValueError(
                    f"BridgeModule: n_features must be >= 2; got {n_features}"
                )
            self.n_features = n_features
            self.config = config
            self.ln: nn.Module | None = (
                nn.LayerNorm(n_features) if config.pre_layernorm else None
            )
            self.linear = nn.Linear(n_features, n_features, bias=False)
            self._init_linear()
            if config.nonlin == "none":
                self.nonlin: nn.Module | None = None
            elif config.nonlin == "relu":
                self.nonlin = nn.ReLU()
            elif config.nonlin == "gelu":
                self.nonlin = nn.GELU()
            else:  # pragma: no cover — typing.Literal narrows in practice
                raise ValueError(f"BridgeModule: unknown nonlin {config.nonlin!r}")
            if not config.train:
                for p in self.parameters():
                    p.requires_grad = False

        def _init_linear(self) -> None:
            import torch as _torch
            import torch.nn as nn

            if self.config.init == "orthogonal":
                nn.init.orthogonal_(self.linear.weight)
            elif self.config.init == "identity":
                with _torch.no_grad():
                    self.linear.weight.copy_(_torch.eye(self.n_features))
            elif self.config.init == "zero":
                nn.init.zeros_(self.linear.weight)
            else:  # pragma: no cover
                raise ValueError(f"BridgeModule: unknown init {self.config.init!r}")

        def forward(self, x):
            if x.size(-1) != self.n_features:
                raise ValueError(
                    f"BridgeModule: expected last dim {self.n_features}; got {x.size(-1)}"
                )
            h = self.ln(x) if self.ln is not None else x
            h = self.linear(h)
            if self.nonlin is not None:
                h = self.nonlin(h)
            return h

    return BridgeModule


_BRIDGE_CLASS = None


def make_bridge(n_features: int, config: BridgeConfig | None = None):
    """Construct a fresh ``BridgeModule``. Lazy-imports torch on first call."""
    global _BRIDGE_CLASS
    if _BRIDGE_CLASS is None:
        _BRIDGE_CLASS = _build_bridge_class()
    return _BRIDGE_CLASS(n_features, config or BridgeConfig())


def bridge_class():
    """Return the ``BridgeModule`` class (constructing it on first call)."""
    global _BRIDGE_CLASS
    if _BRIDGE_CLASS is None:
        _BRIDGE_CLASS = _build_bridge_class()
    return _BRIDGE_CLASS
