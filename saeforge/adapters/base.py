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
    from saeforge.eval.faithfulness import FaithfulnessTarget
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

    def default_faithfulness_target(self) -> "FaithfulnessTarget":
        """Return the default faithfulness scorer for this family.

        Consulted by
        :func:`~saeforge.eval.targets._default_target_for` when no
        explicit ``ForgePipeline(faithfulness=...)`` is set. Override
        this to declare a non-KL default for a family (e.g. cosine
        for encoder-only models, per-state-step KL for SSMs).

        The default implementation returns :class:`KLTarget`, which
        matches the historical LM-family policy. ``KLTarget`` is
        imported lazily to break the
        ``saeforge.eval.targets`` → ``saeforge.adapters`` import cycle.
        """
        from saeforge.eval.targets.kl import KLTarget

        return KLTarget()

    def host_wrapped_module(self, host, basis, scale_boost: float = 1.0):
        """Construct a host-wrapped forged ``nn.Module`` for this family.

        Host-wrapped mode keeps the residual stream in basis coordinates
        at every block boundary but runs every transformer block with
        the host's exact (unprojected) weights, decoding the residual to
        ``d_model`` at block entry and re-encoding the result at block
        exit. Used by the ``forward_mode="host_wrapped"`` dispatch on
        under-complete bases where ``native_in_basis`` math is invalid.

        v1 ships GPT-2 only. Other bundled adapters raise
        ``NotImplementedError`` via this default implementation. See
        ``openspec/changes/add-host-wrapped-forge-fallback`` for the
        per-family rollout plan.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement host_wrapped_module yet. "
            f"v1 ships GPT-2 only; see openspec/changes/"
            f"add-host-wrapped-forge-fallback for the rollout plan."
        )

    def grad_checkpoint_targets(self, module):
        """Return ``(blocks, embedding_param)`` for activation checkpointing.

        ``blocks`` is the iterable of transformer blocks whose ``forward``
        should be wrapped in ``torch.utils.checkpoint.checkpoint``;
        ``embedding_param`` is the input-side parameter that needs
        ``requires_grad=True`` so the checkpointed graph has at least
        one input requiring grad (the embedding output is not itself a
        leaf tensor).

        Default raises ``NotImplementedError`` so each family must opt
        in. Pre-fix v0.3 ``_enable_grad_checkpointing`` hardcoded the
        GPT-2 layout (``module.transformer.h``,
        ``module.transformer.wte.weight``); reaching the GPT-2 branch
        with a ForgedLlama instance crashed inside the FSM and looked
        like a successful run with KL=0.0 to the caller.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grad_checkpoint_targets. "
            f"Add an override returning (blocks, embedding_param) for the "
            f"family's native-module layout."
        )


def to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to float64 numpy without requiring torch
    at import time. Mirrors the helper in ``saeforge/projector.py``.

    Goes via ``.float()`` because numpy has no native bfloat16 dtype — a direct
    ``.numpy()`` on a bf16 tensor raises ``TypeError: Got unsupported ScalarType BFloat16``.
    """
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().float().numpy().astype(np.float64, copy=False)
    return np.asarray(tensor).astype(np.float64, copy=False)
