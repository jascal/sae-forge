## 1. `WorldModel` protocol

- [ ] 1.1 Create `saeforge/world_model.py` defining the
  `@runtime_checkable` `WorldModel` Protocol with four members:
  `family: str`, `walk(host, projector, *, attention_width="host")
  -> dict[str, np.ndarray]`,
  `build_native_config(host, n_features, *, attention_width="host")
  -> Any`, `native_module_class() -> type`,
  `default_faithfulness_target() -> FaithfulnessTarget`.
- [ ] 1.2 Docstring documents: (a) `walk()` keys are
  `native_module.state_dict()` keys; (b) `build_native_config`'s
  `Any` return type is intentional — non-transformer adapters return
  their own config dataclass; (c) `default_faithfulness_target` is
  consulted by `_default_target_for(family)` when no explicit
  `ForgePipeline(faithfulness=...)` is set;
  (d) `host` MAY be ignored by `default_faithfulness_target`
  (same convention as `FaithfulnessTarget.score`).
- [ ] 1.3 Re-export `WorldModel` from `saeforge.adapters` (alongside
  `ArchitectureAdapter`).
- [ ] 1.4 Re-export `WorldModel` from `saeforge.__init__` (top-level)
  matching the `FaithfulnessTarget` precedent.

## 2. `ArchitectureAdapter.default_faithfulness_target` ABC hook

- [ ] 2.1 Add `default_faithfulness_target(self) -> FaithfulnessTarget`
  method on `ArchitectureAdapter` in `saeforge/adapters/base.py`.
  Default implementation returns `KLTarget()`. Lazy-import
  `KLTarget` to avoid a circular import on
  `saeforge.eval.targets`.
- [ ] 2.2 Add the method's docstring: documents that this hook is
  consulted by `_default_target_for(family)` and that overriding is
  the way to declare a non-KL default for a family.
- [ ] 2.3 Override on `saeforge/adapters/whisper.py::WhisperEncoderAdapter`
  to return `CosineTarget()`.
- [ ] 2.4 Confirm no other bundled adapters need overrides — the
  six LM-family adapters (`gpt2`, `llama`, `gemma2`, `qwen2`,
  `qwen3`, `qwen3_moe`) inherit the `KLTarget()` default.

## 3. Eliminate `_LM_FAMILIES` table

- [ ] 3.1 In `saeforge/eval/targets/__init__.py`, replace
  `_default_target_for(family)` body with:
  ```python
  from saeforge.adapters import adapter_for_family
  try:
      adapter = adapter_for_family(family)
  except ValueError as exc:
      raise ValueError(...) from exc
  return adapter.default_faithfulness_target()
  ```
  Preserve the existing error message style (names the offending
  family + the registered set). Drop `_LM_FAMILIES` and the family
  → target mapping comment.
- [ ] 3.2 Verify the existing test
  `tests/test_faithfulness_target_protocol.py::test_default_target_for_unknown_family_raises`
  still passes with the new error path.

## 4. Eliminate `_build_torch_module` `if/elif` tree

- [ ] 4.1 In `saeforge/model.py`, replace the `if/elif` body of
  `_build_torch_module(config)` with:
  ```python
  from saeforge.adapters import adapter_for_family
  cls = adapter_for_family(config.family).native_module_class()
  return cls(config)
  ```
- [ ] 4.2 For the Llama-family adapters (`llama`, `gemma2`, `qwen2`,
  `qwen3`, `qwen3_moe`) whose `native_module_class()` currently
  returns a class whose `__init__` doesn't take a config, add thin
  wrapper classes (or a `from_config(config)` classmethod) so they
  fit the `cls(config)` calling convention. Document this in the
  PR description — it's a real-but-small refactor.
- [ ] 4.3 Add `adapter_for_family` to `saeforge.adapters.__all__` if
  it isn't already.
- [ ] 4.4 Replace `_SUPPORTED_FAMILIES` with a helper
  `registered_families() -> frozenset[str]` exported from
  `saeforge.adapters`. The existing tuple stays as
  `_SUPPORTED_FAMILIES` for back-compat with any in-repo or external
  read; populate it once at module import from `registered_families()`.
- [ ] 4.5 `NativeModelConfig.__post_init__` validates `self.family`
  via `adapter_for_family(self.family)` (raising `ValueError` with
  the same message style on miss). The current
  `if self.family not in _SUPPORTED_FAMILIES: raise ValueError(...)`
  is replaced.

## 5. Tests

- [ ] 5.1 `tests/test_world_model_protocol.py` —
  - `isinstance(adapter, WorldModel)` is `True` for every bundled
    adapter instance.
  - `isinstance(object(), WorldModel)` is `False`.
  - Every bundled adapter's `default_faithfulness_target()` returns
    an instance satisfying `FaithfulnessTarget`.
  - The six LM-family adapters return `KLTarget`; the whisper
    adapter returns `CosineTarget`.
  - `_default_target_for(family)` returns the same type as before
    the change for each bundled family — pin the family→target-type
    mapping in a parametrised test, identical to
    `test_faithfulness_target_protocol.py::test_default_target_for_known_families`
    but exercising the registry-backed dispatcher.
  - `_default_target_for("fictional")` raises `ValueError`.
- [ ] 5.2 `tests/test_world_model_byte_identity.py` —
  - For each bundled family with a tiny fixture
    (`tiny_gpt2`, `tiny_llama`, plus whatever's available for
    gemma2/qwen2/qwen3/qwen3_moe/whisper_encoder), run
    `ForgePipeline.run_synthetic(...)` with a fixed seed.
  - Hash `(n_params, round(faithfulness, 8),
    faithfulness_target_name, basis.W_dec.tobytes())`.
  - Pin the per-family digest as a module-level constant.
  - Assert the digest matches across re-runs.
  - The first CI run captures the digests; subsequent runs guard
    against drift.
- [ ] 5.3 The full existing suite stays green. Particular attention
  to the seven adapter tests, the byte-identity test from
  pluggable-faithfulness
  (`test_faithfulness_target_protocol.py::test_pipeline_default_matches_explicit_kltarget`),
  and the whisper-encoder smoke.

## 6. Docs

- [ ] 6.1 `docs/algorithm.md` — add a "The WorldModel seam"
  subsection. Cover: what's transformer-shape vs what's truly
  generic; the four-member protocol; where non-transformer
  adapters plug in; pointer at `saeforge/adapters/base.py` for the
  bundled ABC.
- [ ] 6.2 `AGENTS.md` — add an "Adding a new architecture family"
  subsection. Four-step recipe:
  1. Implement `WorldModel` (either by inheriting from
     `ArchitectureAdapter` or structurally).
  2. Register via `register_adapter(host_class, instance)` at
     module-import time.
  3. If `NativeModelConfig`'s fields don't fit, define a
     family-specific config dataclass and return it from
     `build_native_config()`.
  4. Override `default_faithfulness_target()` if KL is wrong for
     the family.
- [ ] 6.3 `CHANGELOG.md` — under `[Unreleased]`, add a "WorldModel
  protocol" entry listing the new export, the new
  `default_faithfulness_target` hook, the registry-driven dispatch
  replacing the two hardcoded tables, and the byte-identity
  guarantee for the seven bundled families.

## 7. Validation

- [ ] 7.1 `openspec validate world-model-protocol --strict`.
- [ ] 7.2 `ruff check` clean on every modified file.
- [ ] 7.3 Full `pytest` suite green. Zero regressions on the
  seven adapter tests; the new byte-identity test pinned.
- [ ] 7.4 `python examples/forge_with_gt_alignment.py` and
  `python examples/forge_synthetic_llama.py` complete under their
  usual budgets (60s / 30s on CPU).

## 8. What this change explicitly defers

- [ ] 8.1 **Concrete non-transformer adapters** (Mamba/SSM, RNN,
  diffusion U-Net). Each is a separate follow-up against this seam.
  The protocol is designed from existing transformer requirements
  alone; we accept the risk that contact with a real non-transformer
  may surface needed protocol changes, and we prefer that over
  speculative generalisation.
- [ ] 8.2 **A formal `NativeConfigProtocol`.** `build_native_config`
  returns `Any` today. Formalising the config contract waits for a
  non-transformer adapter to validate against.
- [ ] 8.3 **`SubspaceProjector` refactor.** Its transformer-shape
  helpers stay. Non-transformer adapters MAY ignore them and
  implement projection algebra inside `walk()`.
- [ ] 8.4 **Renaming `NativeModelConfig`.** Persists through this
  minor version. A future change renames it once non-transformer
  config classes exist.
- [ ] 8.5 **Async / parallel `walk()`.** Sync, single-host.
- [ ] 8.6 **CLI surface for the registry.** Python-API only.
- [ ] 8.7 **Faithfulness-protocol extensions** (host-state-trajectory
  metrics, dynamical-systems metrics for SSMs). Lands with the
  first concrete non-transformer adapter, not here.
