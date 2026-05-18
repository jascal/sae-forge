# world-model-protocol Specification

## Purpose

The `world-model-protocol` capability defines `WorldModel`, the
public protocol every host-architecture adapter satisfies. It
generalises the existing `ArchitectureAdapter` ABC to a structural
contract that third-party adapters can satisfy without inheriting
from the bundled base class, and it lifts the default-faithfulness
dispatch off the hardcoded LM-vs-whisper table onto each adapter.

The seven bundled transformer adapters (`gpt2`, `llama`, `gemma2`,
`qwen2`, `qwen3`, `qwen3_moe`, `whisper_encoder`) are refactored to
conform; that is the only validation surface. Concrete
non-transformer adapters (SSM, RNN, diffusion) are explicit
follow-ups against this seam.

The capability composes with `faithfulness-target` (the
pluggable-faithfulness change): `_default_target_for(family)`
consults `adapter.default_faithfulness_target()` rather than a
hardcoded table.

## ADDED Requirements

### Requirement: `WorldModel` is a runtime-checkable Protocol

`saeforge.world_model.WorldModel` SHALL be a `@runtime_checkable`
`typing.Protocol` with four members:

- `family: str` — the family identifier matched against
  `NativeModelConfig.family` and consulted by
  `adapter_for_family(family)`. SHOULD be lowercase snake_case.
- `walk(host, projector, *, attention_width="host") -> dict[str,
  np.ndarray]` — project every relevant host parameter; return a
  flat dict whose keys are a subset of
  `native_module_class()(config).state_dict()` keys. Pure-numpy
  return; lazy torch internally.
- `build_native_config(host, n_features, *, attention_width="host")
  -> Any` — lift host config into the family's native config
  object. Return type is `Any` so non-transformer adapters can
  return their own config dataclass. The returned object MUST
  expose a `family: str` attribute matching this adapter's
  `family`.
- `native_module_class() -> type` — return the nn.Module subclass
  used to instantiate forged models for this family. Lazy torch.
  The returned class's `__init__` SHALL accept the config object
  returned by `build_native_config` as its sole positional
  argument.
- `default_faithfulness_target() -> FaithfulnessTarget` — return
  the family's default loop-gating scorer. Consulted by
  `saeforge.eval.targets._default_target_for(family)` when no
  explicit `ForgePipeline(faithfulness=...)` is set.

The `host` argument to `default_faithfulness_target` MAY be ignored.
Adapters whose default target does not depend on host state SHOULD
return a fresh instance each call (the result is immutable from
the caller's perspective).

#### Scenario: protocol is runtime-checkable

- **WHEN** `isinstance(GPT2Adapter(), WorldModel)` is evaluated
- **THEN** the result is `True`
- **AND** the same holds for every other bundled adapter
- **AND** `isinstance(object(), WorldModel)` is `False`

#### Scenario: third-party adapter satisfies the protocol structurally

- **GIVEN** a class defining the four members of `WorldModel`
  without inheriting from `ArchitectureAdapter`
- **WHEN** `isinstance(MyAdapter(), WorldModel)` is evaluated
- **THEN** the result is `True`
- **AND** `register_adapter(MyHostClass, MyAdapter())` succeeds
- **AND** `adapter_for(MyHostClass())` returns the registered
  instance

### Requirement: `ArchitectureAdapter` ABC gains a `default_faithfulness_target` classmethod

`saeforge.adapters.base.ArchitectureAdapter` SHALL gain a
`default_faithfulness_target(self) -> FaithfulnessTarget` method.
The default implementation SHALL return `KLTarget()` (LM-shape
default — matches the v0.4 hardcoded behaviour for every LM
family).

`WhisperEncoderAdapter` SHALL override the method to return
`CosineTarget()`. The other six bundled adapters (`gpt2`, `llama`,
`gemma2`, `qwen2`, `qwen3`, `qwen3_moe`) inherit the `KLTarget()`
default.

The method's docstring SHALL document that it is consulted by
`_default_target_for(family)` and that overriding is the way to
declare a non-KL default for a new family.

#### Scenario: LM-family adapters return KLTarget

- **WHEN** `GPT2Adapter().default_faithfulness_target()` is called
- **THEN** the return is an instance of `KLTarget`
- **AND** the same holds for `LlamaAdapter`, `Gemma2Adapter`,
  `Qwen2Adapter`, `Qwen3Adapter`, `Qwen3MoEAdapter`

#### Scenario: Whisper-encoder adapter returns CosineTarget

- **WHEN** `WhisperEncoderAdapter().default_faithfulness_target()`
  is called
- **THEN** the return is an instance of `CosineTarget`

### Requirement: `_default_target_for(family)` dispatches via the adapter registry

`saeforge.eval.targets._default_target_for(family)` SHALL be
re-implemented to consult the adapter registry:

```python
def _default_target_for(family: str | None) -> FaithfulnessTarget:
    try:
        adapter = adapter_for_family(family)
    except (ValueError, TypeError) as exc:
        # Preserve the v0.4 error-message shape.
        raise ValueError(
            f"_default_target_for: unsupported family {family!r}. "
            f"Supported: {sorted(registered_families())!r}. Pass an "
            "explicit ForgePipeline(faithfulness=...) target to override."
        ) from exc
    return adapter.default_faithfulness_target()
```

The hardcoded `_LM_FAMILIES` frozenset SHALL be removed. The
mapping from family → default target now lives on each adapter's
`default_faithfulness_target` method.

#### Scenario: dispatch result is byte-identical for every bundled family

- **WHEN** `_default_target_for(family)` is called for each of
  `"gpt2"`, `"llama"`, `"gemma2"`, `"qwen2"`, `"qwen3"`,
  `"qwen3_moe"`, `"whisper_encoder"`
- **THEN** the returned target's `name` is `"kl"` for the six
  LM families and `"cosine"` for `"whisper_encoder"` (same as the
  v0.4 hardcoded behaviour)

#### Scenario: unknown family raises the v0.4-style error

- **WHEN** `_default_target_for("fictional")` is called
- **THEN** `ValueError` is raised whose message contains
  `"fictional"` and the registered families set

### Requirement: `_build_torch_module` dispatches via the adapter registry

`saeforge.model._build_torch_module(config)` SHALL be re-implemented
as a single registry lookup:

```python
def _build_torch_module(config):
    cls = adapter_for_family(config.family).native_module_class()
    return cls(config)
```

The hardcoded family `if/elif` tree SHALL be removed. The Llama-
family adapter group (`llama`, `gemma2`, `qwen2`, `qwen3`,
`qwen3_moe`), which today share a single
`build_llama_family_module(config)` builder, SHALL expose
per-family `native_module_class()` returns that fit the
`cls(config)` calling convention (thin per-family subclasses or
`from_config(cls, config)` factory wrappers).

#### Scenario: torch module construction is unchanged across the seven bundled families

- **GIVEN** a `NativeModelConfig` for any bundled family
- **WHEN** `NativeModel(config)` is constructed
- **THEN** the resulting `torch_module` is structurally
  byte-identical to the pre-change construction (same `state_dict`
  keys, same parameter shapes, same forward output on the same
  inputs)
- **AND** the byte-identity test in
  `tests/test_world_model_byte_identity.py` pins this for every
  bundled family

### Requirement: family-set discovery is registry-driven

The set of supported families SHALL be derived from the adapter
registry rather than maintained as a hardcoded tuple:

- `saeforge.adapters.registered_families() -> frozenset[str]` —
  new helper returning the live set of `adapter.family` values
  across registered adapters.
- `saeforge.model._SUPPORTED_FAMILIES` SHALL be populated at module
  import from `registered_families()`. The module-level tuple
  persists for back-compat with any reader that imports it
  directly.
- `NativeModelConfig.__post_init__` SHALL validate
  `self.family` via `adapter_for_family(self.family)` (catching
  `ValueError` if the registry doesn't know the family) rather than
  via a tuple-membership check.

#### Scenario: registered_families reflects the bundled adapter set

- **WHEN** `registered_families()` is called after
  `saeforge.adapters` has been imported (which triggers the
  bundled adapter registrations)
- **THEN** the returned frozenset contains exactly `{"gpt2",
  "llama", "gemma2", "qwen2", "qwen3", "qwen3_moe",
  "whisper_encoder"}`

#### Scenario: NativeModelConfig with an unregistered family raises

- **WHEN** `NativeModelConfig(family="fictional", ...)` is
  constructed
- **THEN** `ValueError` is raised whose message names `"fictional"`
  and the registered set

### Requirement: public API exposes `WorldModel`

`saeforge.WorldModel` and `saeforge.adapters.WorldModel` SHALL both
re-export the protocol defined in `saeforge.world_model`. This
mirrors the `FaithfulnessTarget` precedent
(`saeforge.eval.FaithfulnessTarget` re-exported from
`saeforge.eval.faithfulness`).

#### Scenario: top-level import works

- **WHEN** `from saeforge import WorldModel` is executed
- **THEN** the import succeeds
- **AND** `WorldModel` is the same object as
  `saeforge.world_model.WorldModel`

