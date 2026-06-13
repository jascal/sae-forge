"""Architecture-adapter registry.

Maps an HF transformers model class to a
:class:`saeforge.adapters.base.ArchitectureAdapter` that knows how to
walk that architecture's weights into a feature-basis projection and
build a matching :class:`saeforge.model.NativeModel`.

Public API
----------

- :func:`register_adapter` — register an adapter for a host class.
- :func:`adapter_for` — look up the adapter for a host model instance.
- :func:`registered_classes` — list every host class with a registered
  adapter (diagnostic).

The three bundled adapters (GPT-2, Llama, Gemma-2) register themselves
when this module is imported. Importing :mod:`saeforge.adapters` is
sufficient to populate the registry; downstream code that calls
:func:`adapter_for` doesn't need to think about ordering.
"""

from __future__ import annotations

from typing import Any

from saeforge.adapters.base import ArchitectureAdapter, to_numpy
from saeforge.world_model import WorldModel

# Registry as a list of (host_class, adapter) tuples — order matters
# only for first-match-wins semantics when subclass relationships exist.
_REGISTRY: list[tuple[type, ArchitectureAdapter]] = []


def register_adapter(host_class: type, adapter: ArchitectureAdapter) -> None:
    """Register ``adapter`` as the dispatcher target for ``host_class``.

    The first registered match wins (`adapter_for` iterates the registry
    and returns on the first ``isinstance`` hit). Subclassing matters:
    a subclass adapter MUST register before its parent so the more
    specific dispatch wins.
    """
    _REGISTRY.append((host_class, adapter))


def adapter_for(host_model: Any) -> ArchitectureAdapter:
    """Return the adapter for ``host_model``'s class.

    Raises ``NotImplementedError`` (with a message naming the host's
    type and every registered class) when no match is found.
    """
    for cls, adapter in _REGISTRY:
        if isinstance(host_model, cls):
            return adapter
    registered = [cls.__name__ for cls, _ in _REGISTRY]
    raise NotImplementedError(
        f"No architecture adapter registered for "
        f"{type(host_model).__name__!r}. Registered: {registered!r}. "
        f"To add a new architecture, implement saeforge.adapters."
        f"ArchitectureAdapter and call register_adapter(host_class, "
        f"adapter)."
    )


def registered_classes() -> list[type]:
    """Return the list of host classes with a registered adapter."""
    return [cls for cls, _ in _REGISTRY]


def registered_families() -> frozenset[str]:
    """Return the set of architecture families with a registered adapter.

    The world-model-protocol refactor replaces the hardcoded
    ``saeforge.model._SUPPORTED_FAMILIES`` tuple with this registry
    lookup. Useful for diagnostics and for any code that needs to
    enumerate families without going through host-class dispatch.
    """
    return frozenset(adapter.family for _, adapter in _REGISTRY)


def adapter_for_family(family: str) -> ArchitectureAdapter:
    """Return the adapter whose ``family`` attribute matches ``family``.

    Used by code paths that have only the ``NativeModelConfig.family``
    string in hand (e.g. inside the training loop, where the host class
    is already gone). Raises ``ValueError`` naming the available
    families when none match.
    """
    seen: set[str] = set()
    for _, adapter in _REGISTRY:
        if adapter.family in seen:
            continue
        seen.add(adapter.family)
        if adapter.family == family:
            return adapter
    raise ValueError(
        f"No adapter registered for family={family!r}. "
        f"Available families: {sorted(seen)!r}"
    )


# ---------------------------------------------------------------------------
# Built-in adapters — register at import time so `import saeforge.adapters`
# is enough to populate the registry.
# ---------------------------------------------------------------------------

# Imported for side effects: each module calls `register_adapter` at module
# scope. Order matters: more specific subclasses (Gemma2 extends Llama-like
# layout) register first if they share a base class.
from saeforge.adapters import gpt2 as _gpt2  # noqa: E402,F401
from saeforge.adapters import gpt_neox as _gpt_neox  # noqa: E402,F401
from saeforge.adapters import llama as _llama  # noqa: E402,F401
from saeforge.adapters import gemma2 as _gemma2  # noqa: E402,F401
from saeforge.adapters import qwen2 as _qwen2  # noqa: E402,F401
from saeforge.adapters import qwen3 as _qwen3  # noqa: E402,F401
from saeforge.adapters import qwen3_moe as _qwen3_moe  # noqa: E402,F401
from saeforge.adapters import whisper as _whisper  # noqa: E402,F401
from saeforge.adapters import esm2 as _esm2  # noqa: E402,F401


__all__ = [
    "ArchitectureAdapter",
    "WorldModel",
    "adapter_for",
    "adapter_for_family",
    "register_adapter",
    "registered_classes",
    "registered_families",
    "to_numpy",
]
