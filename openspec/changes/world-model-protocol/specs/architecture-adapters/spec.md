## MODIFIED Requirements

### Requirement: every bundled adapter exposes a default faithfulness target

Every bundled `ArchitectureAdapter` SHALL expose a
`default_faithfulness_target(self) -> FaithfulnessTarget` method —
the dispatch hook consulted by
`saeforge.eval.targets._default_target_for(family)` when no explicit
`ForgePipeline(faithfulness=...)` is set. This applies to all seven
bundled adapters: `GPT2Adapter`, `LlamaAdapter`, `Gemma2Adapter`,
`Qwen2Adapter`, `Qwen3Adapter`, `Qwen3MoEAdapter`,
`WhisperEncoderAdapter`.

The `ArchitectureAdapter` ABC SHALL provide a default
implementation returning `KLTarget()`. Subclasses MAY override.
`WhisperEncoderAdapter` SHALL override to return `CosineTarget()`.
The six LM-family adapters inherit the `KLTarget()` default.

The family-dispatch contract for the default faithfulness scorer
SHALL move from a hardcoded table in `saeforge.eval.targets` onto
the adapter classes themselves. Behaviour on the seven bundled
families MUST be byte-identical to the pre-change hardcoded
dispatch — pinned by the byte-identity test in
`tests/test_world_model_byte_identity.py`.

#### Scenario: LM-family adapters return KLTarget

- **WHEN** `default_faithfulness_target()` is invoked on each of
  `GPT2Adapter`, `LlamaAdapter`, `Gemma2Adapter`, `Qwen2Adapter`,
  `Qwen3Adapter`, `Qwen3MoEAdapter`
- **THEN** the returned target is an instance of `KLTarget`
- **AND** `target.name == "kl"`
- **AND** `target.better_when == "lower"`

#### Scenario: Whisper-encoder adapter returns CosineTarget

- **WHEN** `WhisperEncoderAdapter().default_faithfulness_target()`
  is invoked
- **THEN** the returned target is an instance of `CosineTarget`
- **AND** `target.name == "cosine"`
- **AND** `target.better_when == "higher"`

### Requirement: adapter registry exposes a family-set helper

`saeforge.adapters` SHALL expose
`registered_families() -> frozenset[str]` as a public helper
returning the live set of `adapter.family` values across registered
adapters. It SHALL be the single source of truth for "which
families does this build support" and SHALL be importable as
`from saeforge.adapters import registered_families` (added to
`saeforge.adapters.__all__`). Tests, docs, and downstream tooling
consume this helper instead of re-deriving the family set.

`saeforge.model._SUPPORTED_FAMILIES` SHALL be populated at module
import from `registered_families()` rather than maintained as a
separate hardcoded tuple. The module-level tuple persists for
back-compat with any reader that imports it directly.

`NativeModelConfig.__post_init__` SHALL validate `self.family` via
`adapter_for_family(self.family)` (raising `ValueError` with a
message naming the offending family and the registered set) rather
than via a tuple-membership check.

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
