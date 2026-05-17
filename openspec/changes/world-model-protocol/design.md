## Context

`saeforge/adapters/base.py::ArchitectureAdapter` is the existing
architecture seam. It defines:

- `family: str` — identifier matched against `NativeModelConfig.family`
  and `_default_target_for(family)`.
- `walk(host, projector, *, attention_width) -> dict[str, np.ndarray]`
  — project every relevant host weight; return a flat dict keyed by
  the corresponding native-module parameter name. Pure-numpy; lazy
  torch.
- `build_native_config(host, n_features, *, attention_width) -> NativeModelConfig`
  — lift host config into a `NativeModelConfig` whose `family` matches
  the adapter.
- `native_module_class() -> type` — return the nn.Module subclass
  for the family. Lazy torch.
- `grad_checkpoint_targets(module) -> tuple[Iterable, Parameter]` —
  optional; raises if not overridden.

Seven bundled adapters: `gpt2`, `llama`, `gemma2`, `qwen2`, `qwen3`,
`qwen3_moe`, `whisper_encoder`. Each registers itself at import time
via `register_adapter(host_class, adapter)`.

Outside the adapter layer, two pieces of code re-derive the family
list independently:

1. `saeforge/model.py::_build_torch_module(config)` is a hardcoded
   `if/elif` tree mapping `config.family` to a family-specific
   builder. Adding a family today means: register the adapter, add
   the family name to `_SUPPORTED_FAMILIES`, AND add an `elif` arm
   to `_build_torch_module`. Three places, one logical change.
2. `saeforge/eval/targets/__init__.py::_default_target_for(family)`
   is a hardcoded LM-vs-whisper table. The pluggable-faithfulness
   change (#46) added a friendly comment but the dispatch is still
   a hardcoded `frozenset` plus a single `if family ==
   "whisper_encoder"` branch.

Both of those tables are derivable from the adapter registry: every
adapter knows its `family` string, every adapter has a
`native_module_class()`, and we can ask each adapter what its
default faithfulness target is via a new classmethod.

This change does that consolidation, names the resulting contract
the `WorldModel` protocol, and stops there. No concrete
non-transformer adapters; no `SubspaceProjector` refactor; no new
config classes.

## Goals / Non-Goals

**Goals**:

- Define `WorldModel` as a `@runtime_checkable` Protocol in
  `saeforge/world_model.py`. Every bundled adapter satisfies it
  structurally without any further code change (the existing
  `ArchitectureAdapter` ABC already has the matching method
  signatures).
- Add `default_faithfulness_target()` as a classmethod on
  `ArchitectureAdapter` with a `KLTarget()` default. Override on
  `WhisperEncoderAdapter` to return `CosineTarget()`. Six other
  bundled adapters inherit the default.
- Replace `_build_torch_module(config)`'s `if/elif` tree with a
  single `adapter_for_family(config.family).native_module_class()(config)`
  call.
- Replace `_default_target_for(family)`'s hardcoded `_LM_FAMILIES`
  set with `adapter_for_family(family).default_faithfulness_target()`.
- Replace `_SUPPORTED_FAMILIES` with a property that reads the live
  adapter registry. Module-level tuple stays at v0.4-equivalent
  contents on first import for back-compat with tests that read it
  directly.
- Pin a byte-identity invariant: every bundled family produces the
  same `ForgeResult` digest pre- and post-change on the same fixture.

**Non-goals**:

- No concrete non-transformer adapter. The protocol is designed
  from existing transformer requirements alone. We accept the risk
  that contact with a real non-transformer surfaces protocol
  changes.
- No changes to `SubspaceProjector`. Its transformer-shape helpers
  stay; non-transformer adapters are free to ignore them and
  implement projection algebra inside `walk()`.
- No new `NativeConfigProtocol`. `WorldModel.build_native_config`'s
  return type is `Any` for now; we'll formalise it once a
  non-transformer adapter needs a different config class.
- No rename of `NativeModelConfig`. The name persists for at least
  this minor version.
- No CLI changes.

## Decisions

### Decision 1: `WorldModel` is a Protocol, `ArchitectureAdapter` stays an ABC

Two reasons to keep both:

1. The Protocol is the *public* contract. Third-party adapters can
   satisfy it structurally without inheriting from the bundled ABC.
   That preserves the same property pluggable-faithfulness gives
   for `FaithfulnessTarget`: no required base class, just a shape.
2. The ABC is the *bundled* base class with concrete inherited
   helpers — `grad_checkpoint_targets()`'s default-raises pattern,
   `to_numpy()` import path, future shared boilerplate. Future
   bundled adapters benefit from inheriting. Third-party adapters
   pay only for what they use.

Collapsing them into a single class would either force third parties
to inherit (worse ergonomics) or move the helpers onto the Protocol
(impossible — Protocols cannot have non-trivial method bodies).

### Decision 2: `default_faithfulness_target()` lives on the adapter, not on a separate registry

`_default_target_for(family)` today consults a hardcoded table. The
alternative — leave the table and add a `register_default_target(family,
target)` API — is strictly more state to maintain (the family
already lives on the adapter; threading a second registry adds a
synchronisation surface).

Putting the default on the adapter has three nice properties:

1. Adding a new family is one class definition. No second registration.
2. Adapter-side overrides are explicit and grep-able
   (`def default_faithfulness_target(self) -> FaithfulnessTarget:` in
   `whisper.py`).
3. Third-party adapters declare their own default at the same place
   they declare everything else about the family.

The ABC's default returns `KLTarget()` because LM is the dominant
case. `WhisperEncoderAdapter` is the only bundled override.

### Decision 3: `_build_torch_module` becomes a one-line dispatch

Today:

```python
def _build_torch_module(config: NativeModelConfig):
    if config.family == "gpt2":
        from saeforge.adapters.gpt2 import build_gpt2_module
        return build_gpt2_module(config)
    if config.family in ("llama", "gemma2", "qwen2", "qwen3", "qwen3_moe"):
        from saeforge.adapters.llama import build_llama_family_module
        return build_llama_family_module(config)
    if config.family == "whisper_encoder":
        from saeforge.adapters.whisper import build_whisper_encoder_module
        return build_whisper_encoder_module(config)
    raise ValueError(...)
```

After:

```python
def _build_torch_module(config: NativeModelConfig):
    adapter = adapter_for_family(config.family)
    return adapter.native_module_class()(config)
```

The bundled adapters already implement `native_module_class()`
returning the right class. The `build_*_module(config)` functions
each just instantiate that class with the config — moving the
construction one level out into the dispatcher trims a layer.

Adapters whose existing `native_module_class()` returns a class
without a `config`-accepting `__init__` (the Llama-family adapter
groups five families into one builder via a switch on
`config.family`) need a small refactor: each family's
`native_module_class()` returns a thin subclass that calls
`build_llama_family_module(config)` internally. Less restrictive
than a hardcoded family list inside `_build_torch_module` and the
test count stays the same.

### Decision 4: `_SUPPORTED_FAMILIES` becomes a derived property

Today it's a module-level tuple in `model.py`:

```python
_SUPPORTED_FAMILIES = ("gpt2", "llama", "gemma2", "qwen2", "qwen3", "qwen3_moe", "whisper_encoder")
```

After:

```python
def _supported_families() -> tuple[str, ...]:
    from saeforge.adapters import registered_classes
    # Bundled adapters register at import time; the registry is the
    # source of truth.
    return tuple(sorted({a.family for _, a in __import__(
        "saeforge.adapters", fromlist=["_REGISTRY"]
    )._REGISTRY}))
```

(The implementation is uglier than the snippet — encapsulation can
be improved by adding a public `registered_families()` helper to
`saeforge.adapters` and exporting it.)

`NativeModelConfig.__post_init__` uses `adapter_for_family(self.family)`
to validate. The two existing tests that import `_SUPPORTED_FAMILIES`
keep working because they get the same tuple contents.

### Decision 5: Byte-identity guarantee via fixture digest

The invariant: every bundled family produces the same `ForgeResult`
fields pre- and post-change. Strategy:

- For each family, a tiny synthetic host (existing
  `tiny_gpt2` / `tiny_llama` / etc. fixtures, or new minimal versions
  for the families that don't have them).
- Run `ForgePipeline.run_synthetic(...)` with a fixed seed.
- Hash `(n_params, round(faithfulness, 8),
  faithfulness_target_name, basis.W_dec.tobytes())`.
- Pin the digest in the test file. Regenerating it after an
  intentional behaviour change is a one-line edit and lands in the
  same commit as the behaviour change.

The digest is captured once on this change's first CI run (so the
"pre-change main" reference is the implementation PR's first green
build, not a separate snapshot). Acceptable because the only goal
is "no drift across this refactor" — not "no drift from some
historical reference".

### Decision 6: `WorldModel.build_native_config()` returns `Any`

The return type is `Any` (not `NativeModelConfig`) so future
non-transformer adapters can return their own config dataclass.
All bundled adapters return `NativeModelConfig`. Callers downstream
(`NativeModel.__init__`, `ForgePipeline._run_*_imperative`) consume
the returned config via duck-typed attribute access; the only
attribute the dispatcher really requires is `family`.

We accept the trade-off that type-checkers can't statically verify
`build_native_config` returns a concrete type. A formal
`NativeConfigProtocol` (with `family: str`, `hidden_size: int`,
`to_dict()`, `from_dict()`) is a follow-up.

### Decision 7: Migration is internal — no public API breaks

The public surface — `ForgePipeline`, `ForgeResult`, `FaithfulnessTarget`,
the seven `adapters/{family}` modules — does not change. The
`WorldModel` protocol is a new export; everything else is internals
moving from hardcoded tables to registry lookups.

## Open Questions

- **`attention_width` on the protocol.** This is a transformer-shape
  parameter. Two options: (a) keep it in the protocol signature with a
  default of `"host"` (non-transformer adapters ignore it); (b) drop
  it from the protocol and require non-transformer adapters to take
  `**kwargs`. Going with (a) — the abstraction leak is one keyword
  argument, dropping it now is speculative.
- **Where does the `WorldModel` protocol live?** Two candidates:
  `saeforge/world_model.py` (new top-level module, fits the
  `saeforge.eval.faithfulness::FaithfulnessTarget` precedent) or
  `saeforge/adapters/base.py` (already houses `ArchitectureAdapter`).
  Going with the new module — separates "the public Protocol" from
  "the bundled ABC" cleanly, matches pluggable-faithfulness.
- **Should `WorldModel` be re-exported as `saeforge.WorldModel`?**
  Yes — same pattern as `FaithfulnessTarget` (re-exported from
  `saeforge.eval`). Listed in tasks.md §1.4.

## Risks / Trade-offs

- **Risk: protocol shape doesn't survive a real non-transformer
  adapter.** Genuine. Mitigation: the change is scope-locked to the
  protocol surface. When the first non-transformer adapter lands,
  it surfaces protocol changes immediately and we can iterate
  before there are downstream users.
- **Risk: registry lookup overhead in `NativeModel.__init__`.** A
  list scan over 7 entries per native-module construction.
  Negligible (`O(7)` Python attribute reads) and `__init__` runs
  once per pipeline call.
- **Risk: the byte-identity test masks a subtle behaviour change
  that future targets/configs would surface.** Mitigation: the hash
  set includes `n_params`, `faithfulness` rounded to 8 decimals,
  and the basis bytes — covers structural and numerical drift on
  the existing surface. Drift on a new surface that the test
  doesn't cover is a known unknown.
- **Trade-off: `NativeModelConfig` name retention.** Reads slightly
  oddly once non-transformer adapters exist (a non-transformer
  adapter's config is NOT a `NativeModelConfig`). Acceptable for
  the minor-version migration window; rename is filed as a
  follow-up.

## Migration Plan

- **Existing users**: no migration. The public API is unchanged;
  the registry-driven dispatch produces byte-identical results on
  the seven bundled families.
- **Adapter authors**: optional. `default_faithfulness_target()` is
  inherited from the ABC if not overridden. Overriding it is the
  way to declare a non-KL default for a new family. Adding a new
  family stops requiring an edit to `_build_torch_module`'s
  `if/elif` tree.
- **Third parties writing a `WorldModel` adapter from scratch**:
  satisfy the four methods on the protocol; call
  `register_adapter(host_class, instance)`; you're done.
