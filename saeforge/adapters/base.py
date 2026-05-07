"""ArchitectureAdapter ABC — the contract every host-architecture
adapter implements.

An adapter knows three things about its target architecture:

1. **The walk** — which HF parameter names exist in the host model and
   how each maps onto a projected weight in the corresponding native
   module. Pure-numpy; no torch operations beyond reading the host's
   parameters via the `_to_numpy` helper.
2. **The native config** — how the host's per-block dimensions
   (hidden_size, num_heads, num_kv_heads, intermediate_size, …) lift
   into a :class:`~saeforge.model.NativeModelConfig` whose ``family``
   matches the adapter.
3. **The native module class** — the ``nn.Module`` subclass produced
   by ``NativeModel.from_projected_weights`` for this family. Returned
   lazily so importing the adapter doesn't require torch.

The dispatcher (:mod:`saeforge.adapters`) maps an HF model class to its
adapter via ``register_adapter`` at import time. Unregistered
architectures raise ``NotImplementedError`` naming the type and the
registered classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from saeforge.model import NativeModelConfig
    from saeforge.projector import SubspaceProjector


class ArchitectureAdapter(ABC):
    """Abstract contract for a host-architecture adapter.

    Subclasses set the ``family`` class attribute and implement the three
    abstract methods. Adapters live under ``saeforge/adapters/<family>.py``
    and register themselves at import time via
    :func:`saeforge.adapters.register_adapter`.
    """

    #: The native-model family identifier emitted on
    #: :class:`~saeforge.model.NativeModelConfig`. One of ``"gpt2"``,
    #: ``"llama"``, ``"gemma2"``.
    family: str = ""

    @abstractmethod
    def walk(
        self,
        host: Any,
        projector: "SubspaceProjector",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        """Project every relevant host weight; return a flat dict keyed
        by the corresponding native module's parameter names.

        The returned dict's keys SHALL be a subset of the native
        module's ``state_dict()`` keys; every value's shape SHALL match
        the corresponding native parameter's shape exactly.
        """

    @abstractmethod
    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> "NativeModelConfig":
        """Pull per-block dimensions from ``host.config`` into a
        :class:`NativeModelConfig` whose ``family`` matches ``self.family``.
        """

    @abstractmethod
    def native_module_class(self) -> type:
        """Return the ``nn.Module`` subclass used to instantiate forged
        models for this family. Lazy-imports torch.
        """


def to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to float64 numpy without requiring torch
    at import time. Mirrors the helper in ``saeforge/projector.py``.
    """
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(tensor).astype(np.float64, copy=False)
