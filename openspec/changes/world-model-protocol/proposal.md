## Why

sae-forge's host-model abstraction lives across three modules that
each independently bake in the assumption "the host is a
transformer":

1. `saeforge/adapters/base.py::ArchitectureAdapter` is the *interface*
   intended to be the architecture seam, and it almost is — `walk()`
   returns a flat `dict[str, np.ndarray]` keyed by native-module
   parameter names, which is generic.
2. `saeforge/model.py::NativeModelConfig` is *not* generic. Required
   fields include `num_heads`, `head_dim`, `qkv_inner_size`,
   `intermediate_size`, `n_kv_heads`, `tied_embeddings`, `qkv_bias`,
   `qk_norm`, `num_experts`, `num_experts_per_tok`,
   `moe_intermediate_size` — every one of which is a transformer-shape
   knob. `_SUPPORTED_FAMILIES` is a hardcoded tuple containing only
   transformer families.
3. `saeforge/model.py::_build_torch_module` is a hardcoded `if family
   == "gpt2": … elif family in ("llama", …): … elif family ==
   "whisper_encoder": …` tree. Adding a new family means editing this
   function.
4. `saeforge/eval/targets/__init__.py::_default_target_for(family)`
   is a hardcoded LM-vs-whisper table. Adding a new family with a new
   default faithfulness target (e.g. per-step state KL for an SSM)
   means editing this function too.

The pluggable-faithfulness change (#46 / #47) opened the
*eval-side* abstraction — `FaithfulnessTarget` is a Protocol and
users can supply their own. The change deferred non-transformer host
support explicitly:

> **Non-transformer host-model support.** sae-forge's NativeModel /
> SubspaceProjector still assume a transformer with attention/MLP
> weight matrices to project. A WorldModel-protocol generalization
> (which would let RNNs and SSMs plug in directly) is a separate,
> larger change. Filed as follow-up `world-model-protocol`.

This change is that follow-up. It lifts the structural seam to
match what pluggable-faithfulness did for the eval seam: a single
`WorldModel` protocol whose conforming adapters carry everything
sae-forge needs to forge a model in that architecture family,
including the default faithfulness target.

**Scope-locked: protocol only.** This change does NOT ship any
concrete non-transformer adapter. The seven existing transformer
adapters (`gpt2`, `llama`, `gemma2`, `qwen2`, `qwen3`, `qwen3_moe`,
`whisper_encoder`) are refactored to conform to the new protocol;
that's the validation surface. Concrete non-transformer adapters
(Mamba/SSM, diffusion, RNN) are explicit follow-ups against this
seam.

## What Changes

### Scope

Introduce `WorldModel`, a `@runtime_checkable` `Protocol` that names
the contract every host-architecture adapter satisfies. The existing
`ArchitectureAdapter` ABC stays as the bundled-adapter base class —
it already satisfies `WorldModel` structurally — and gains one new
classmethod, `default_faithfulness_target()`. The hardcoded family
dispatch in `_build_torch_module` and the hardcoded family table in
`_default_target_for` both go away, replaced by adapter-registry
lookups.

`NativeModelConfig` stays as the transformer-shape config; its name
is retained for backwards compatibility but the docstring clarifies
that it's the *transformer* native config, and that
`WorldModel.build_native_config()` is free to return a different
dataclass for non-transformer adapters. Future non-transformer
adapters return their own config class whose `_build_torch_module`-
equivalent is the adapter's `native_module_class()`.

### New artifacts

- **`saeforge/world_model.py`** — defines the `WorldModel` protocol:

  ```python
  @runtime_checkable
  class WorldModel(Protocol):
      family: str

      def walk(
          self,
          host: Any,
          projector: "SubspaceProjector",
          *,
          attention_width: str = "host",
      ) -> dict[str, np.ndarray]: ...

      def build_native_config(
          self,
          host: Any,
          n_features: int,
          *,
          attention_width: str = "host",
      ) -> Any: ...

      def native_module_class(self) -> type: ...

      def default_faithfulness_target(self) -> "FaithfulnessTarget": ...
  ```

  The protocol's docstring documents three things explicitly:
  (1) `walk()` returns a flat dict whose keys are
  `native_module.state_dict()` keys — same as the existing
  `ArchitectureAdapter.walk` contract; (2) `build_native_config()`'s
  return type is `Any` so non-transformer adapters can return their
  own config dataclass instead of `NativeModelConfig`; (3)
  `default_faithfulness_target()` is consulted by
  `_default_target_for(family)` when no explicit
  `ForgePipeline(faithfulness=...)` is set.

- **`saeforge/adapters/base.py::ArchitectureAdapter.default_faithfulness_target()`** —
  new classmethod on the existing ABC. Default returns `KLTarget()`
  (the LM-shape default). `WhisperEncoderAdapter` overrides to
  `CosineTarget()`. Future non-transformer adapters override to
  whatever's appropriate for the family.

### Modified artifacts

- **`saeforge/model.py`**:
  - `_build_torch_module(config)` becomes a one-line dispatch:
    `return adapter_for_family(config.family).native_module_class()(config)`.
    The `if/elif` family tree is deleted.
  - `_SUPPORTED_FAMILIES` is replaced by a runtime check:
    `NativeModelConfig.__post_init__` looks up the adapter via
    `adapter_for_family(config.family)` and raises if none exists.
    The tuple stays at module level but is built from
    `registered_classes()` rather than hardcoded.
- **`saeforge/eval/targets/__init__.py`**:
  - `_default_target_for(family)` consults
    `adapter_for_family(family).default_faithfulness_target()`. The
    hardcoded `_LM_FAMILIES` table and the family→target mapping comment
    are deleted; the same routing now lives on each adapter.
  - Built-in adapters' `default_faithfulness_target()` overrides
    return the same `KLTarget()` / `CosineTarget()` instances the
    table did, so behaviour is byte-identical.
- **`saeforge/adapters/{gpt2,llama,gemma2,qwen2,qwen3,qwen3_moe}.py`**:
  - No changes (they inherit the `KLTarget()` default from the ABC).
- **`saeforge/adapters/whisper.py`**:
  - Override `default_faithfulness_target()` to return `CosineTarget()`.
- **`saeforge/adapters/__init__.py`**:
  - Export `WorldModel` from `saeforge.world_model`.
- **`saeforge/__init__.py`**:
  - Add `WorldModel` to the public API.

### New tests

- `tests/test_world_model_protocol.py` —
  - `WorldModel` is `@runtime_checkable`; every bundled adapter
    satisfies it via `isinstance`.
  - Every bundled adapter's `default_faithfulness_target()` returns
    an instance satisfying `FaithfulnessTarget`.
  - `_default_target_for("gpt2")` and the equivalent for every other
    bundled family return the same target type they returned before
    the change (byte-identity of the default-dispatch behaviour).
  - `_default_target_for("fictional")` still raises `ValueError`
    (unregistered family).
  - `NativeModelConfig.__post_init__` rejects an unknown family with
    a message naming `adapter_for_family`'s registry.
- `tests/test_world_model_byte_identity.py` —
  - For each bundled family, build a tiny synthetic host, run
    `ForgePipeline.run_synthetic(...)` pre- and post-change, hash
    the resulting `ForgeResult` fields (`n_params`,
    `round(faithfulness, 8)`, `faithfulness_target_name`, basis
    bytes), and assert the digest matches a pinned value. The
    digest is captured on this change's first CI run and asserted
    on every subsequent run — the load-bearing "no behavioural
    drift" property.

### Docs

- `docs/algorithm.md` — short subsection "The WorldModel seam"
  explaining where non-transformer adapters plug in (the protocol,
  the `default_faithfulness_target` hook, and the bullet points on
  what's transformer-shape today vs what's truly generic).
- `AGENTS.md` — gain a "Adding a new architecture family" section
  with the four-step recipe: implement `WorldModel`, register, add a
  family-specific `NativeConfig` dataclass if `NativeModelConfig`'s
  fields don't fit, override `default_faithfulness_target` if KL is
  wrong for the family.
- `CHANGELOG.md` — entry under unreleased noting the new
  `WorldModel` protocol export, the new
  `ArchitectureAdapter.default_faithfulness_target` classmethod
  (with `KLTarget()` default), and the migration of family dispatch
  off the hardcoded tables. Defaults preserve v0.4 behaviour
  byte-identically; existing users see no API change.

## Capabilities

### New Capabilities

- **`world-model-protocol`** — defines the `WorldModel` protocol,
  documents the relationship to the existing `architecture-adapters`
  capability (the bundled `ArchitectureAdapter` ABC is one way to
  satisfy `WorldModel`; third-party adapters MAY satisfy it
  structurally without inheriting), and pins the
  `default_faithfulness_target` integration with the existing
  `faithfulness-target` capability (#46).

### Modified Capabilities

- **`architecture-adapters`** — extended: every bundled adapter now
  exposes `default_faithfulness_target()`. The family-dispatch
  contract for the default faithfulness scorer moves from a hardcoded
  table in `saeforge.eval.targets` onto the adapter classes
  themselves.

### Out of Scope (deferred)

- **Concrete non-transformer adapters.** Mamba/SSM, RNN/GRU/LSTM,
  diffusion U-Net. Each is a separate follow-up against the new
  seam. The protocol is designed from existing transformer
  requirements only; we accept the risk that contact with a real
  non-transformer may surface needed protocol changes, and we
  prefer that over speculative generalisation.
- **A formal `NativeConfigProtocol`.** Today `WorldModel.build_native_config()`
  returns `Any` — concrete bundled adapters all return
  `NativeModelConfig`. A future change MAY introduce a
  `NativeConfigProtocol` whose minimal contract is `family: str` +
  `to_dict() / from_dict()` round-trip + `hidden_size: int`, once
  there is a non-transformer adapter to validate the shape against.
- **Changes to `SubspaceProjector`.** Its transformer-shape helpers
  (`project_residual_input`, `project_residual_output`,
  `project_residual_output_bias`, `project_q_k_v_o`, etc.) stay as-is.
  Non-transformer adapters are free to ignore them and implement
  their own projection algebra inside `walk()` using
  `projector.basis` directly. A future change MAY split
  `SubspaceProjector` into `TransformerSubspaceProjector` + a
  `SubspaceProjectorProtocol` once a non-transformer adapter exists
  to validate the seam.
- **Renaming `NativeModelConfig`.** The name persists for one or
  more minor versions to avoid touching every existing adapter, test,
  and example. A future change MAY rename it to
  `TransformerNativeConfig` once non-transformer config classes are
  in use.
- **Async / parallel `walk()`.** Today `walk()` is sync and
  single-host. A future change MAY add a parallel/distributed
  `walk_async()` for very large hosts.
- **CLI surface for the WorldModel registry.** No new CLI flags.
  The protocol is a Python-API extension; the existing
  `sae-forge forge` command keeps dispatching via the existing
  `adapter_for` lookup.

## Impact

- **No breaking changes at the default surface.** Every existing
  bundled adapter, test, and example keeps working. The
  hardcoded family tables in `_build_torch_module` and
  `_default_target_for` are replaced by registry lookups whose
  results are byte-identical to the old behaviour on the seven
  bundled families. The byte-identity test in
  `tests/test_world_model_byte_identity.py` pins this.
- **One new file.** `saeforge/world_model.py` (~80 lines including
  the protocol definition and module docstring).
- **One ABC method added.** `ArchitectureAdapter.default_faithfulness_target()`
  with a `KLTarget()` default. One override
  (`WhisperEncoderAdapter`) returning `CosineTarget()`.
- **Two hardcoded tables removed.**
  `saeforge.model._SUPPORTED_FAMILIES` (the explicit tuple) and
  `saeforge.eval.targets._LM_FAMILIES` (the LM membership set). Both
  replaced by `adapter_for_family` registry lookups whose contents
  are identical to the old tables today.
- **Two new test files** (~150 lines combined).
- **`AGENTS.md`** gains an "Adding a new architecture family"
  section; `docs/algorithm.md` gains a "The WorldModel seam"
  subsection.
- **No CLI changes.** No new pipeline knobs. No new metadata fields.

## Open Questions

- **Should `WorldModel` and `ArchitectureAdapter` keep separate
  identities?** Yes for now. The Protocol is the public contract;
  the ABC is the bundled-adapter base class with concrete inherited
  helpers (`grad_checkpoint_targets`, `to_numpy`). Third parties can
  satisfy `WorldModel` without inheriting from `ArchitectureAdapter`.
  Collapsing them is a future cosmetic change if it ever proves
  redundant.
- **Should `walk()`'s `attention_width` parameter survive on the
  protocol?** It is transformer-specific. Two options:
  (a) keep it in the protocol with a default of `"host"` that
  non-transformer adapters ignore (smallest change today, slight
  abstraction leak); (b) move it to a transformer-specific subprotocol
  and broaden `walk()` to take `**kwargs`. Going with (a) for v1 —
  the abstraction leak is one keyword argument and removing it now
  is speculative.
- **Should we surface the protocol from a top-level
  `saeforge.WorldModel` import?** Yes — same pattern as
  `FaithfulnessTarget` (re-exported from `saeforge.eval`). Listed
  in §"New artifacts" but worth flagging here so a reviewer can
  push back if they think it should live deeper in the namespace.
