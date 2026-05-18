"""The :class:`WorldModel` protocol — sae-forge's architecture seam.

A ``WorldModel`` is anything that can be projected through
:class:`~saeforge.projector.SubspaceProjector` into a forged native
module. Today every bundled ``WorldModel`` is a transformer family
adapter (``gpt2``, ``llama``, ``gemma2``, ``qwen2``, ``qwen3``,
``qwen3_moe``, ``whisper_encoder``). Non-transformer adapters
(Mamba/SSM, RNN, diffusion U-Net, …) are explicit follow-ups against
this seam.

The bundled :class:`~saeforge.adapters.base.ArchitectureAdapter` ABC
satisfies this protocol structurally; third-party adapters MAY
implement ``WorldModel`` directly without inheriting from
``ArchitectureAdapter``. The ABC continues to be the recommended base
class for bundled-adapter style implementations that want the
inherited helpers (``grad_checkpoint_targets`` and the ``to_numpy``
import path).

The four-member contract:

- ``family: str`` — the family identifier surfaced on
  :class:`~saeforge.model.NativeModelConfig`. Used by registry
  dispatch (:func:`saeforge.adapters.adapter_for_family`) and by
  result-artifact metadata.
- ``walk(host, projector, *, attention_width="host") -> dict[str,
  np.ndarray]`` — returns a flat dict whose keys are
  ``native_module.state_dict()`` keys and whose values are the
  projected weights. Pure numpy; no torch operations beyond reading
  the host's parameters.
- ``build_native_config(host, n_features, *, attention_width="host")
  -> Any`` — returns a native-module config object whose ``family``
  attribute matches ``self.family``. The return type is ``Any`` so
  non-transformer adapters can return their own config dataclass
  instead of :class:`~saeforge.model.NativeModelConfig`; bundled
  transformer adapters return ``NativeModelConfig``.
- ``native_module_class() -> type`` — returns the ``nn.Module``
  subclass that ``NativeModel`` instantiates for this family. Called
  with a single positional ``config`` argument; adapters whose native
  modules don't fit the ``cls(config)`` calling convention provide a
  thin wrapper.
- ``default_faithfulness_target() -> FaithfulnessTarget`` — returns
  the default faithfulness scorer for this family. Consulted by
  :func:`~saeforge.eval.targets._default_target_for` when no explicit
  ``ForgePipeline(faithfulness=...)`` is set. Takes no arguments; if
  a future SSM-style adapter needs host-aware scorer selection (e.g.
  per-state-step KL), that's an additive protocol change and gets
  scoped against the first concrete non-transformer adapter rather
  than speculatively added here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from saeforge.eval.faithfulness import FaithfulnessTarget
    from saeforge.projector import SubspaceProjector


@runtime_checkable
class WorldModel(Protocol):
    """Protocol every host-architecture adapter satisfies.

    Conforming classes either inherit from
    :class:`~saeforge.adapters.base.ArchitectureAdapter` (the bundled
    base class) or define the four members structurally. See the
    module docstring for the contract.
    """

    family: str

    def walk(
        self,
        host: Any,
        projector: "SubspaceProjector",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        ...

    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> Any:
        ...

    def native_module_class(self) -> type:
        ...

    def default_faithfulness_target(self) -> "FaithfulnessTarget":
        ...
