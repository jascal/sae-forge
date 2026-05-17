# faithfulness-target Specification

## Purpose

The `faithfulness-target` capability defines a pluggable scorer
protocol for `ForgePipeline`'s faithfulness signal. v0.4 hard-coded
the scorer to KL divergence for LM hosts and per-frame cosine
similarity for Whisper encoders by inlining a family check inside
the `evaluate_faithfulness` action. This capability lifts that
dispatch behind the `FaithfulnessTarget` protocol so callers can
supply their own scorer (probe accuracy, ground-truth-feature
alignment, monosemanticity, …) without modifying action code.

The default surface is unchanged: when `ForgePipeline(faithfulness=
None, ...)` (the default), the action picks `KLTarget` for LM
families and `CosineTarget` for `whisper_encoder`. Existing fixtures,
notebooks, and the CLI continue to produce byte-identical results.

## ADDED Requirements

### Requirement: `FaithfulnessTarget` protocol shape

`saeforge.eval.faithfulness.FaithfulnessTarget` SHALL be a
`@runtime_checkable` `typing.Protocol` with the following members:

- `name: str` — short identifier (e.g. `"kl"`, `"cosine"`,
  `"gt_alignment"`). Used as `ForgeResult.faithfulness_target_name`,
  in FSM transitions-log entries, and as a stable key in
  `forge_result.json` metadata. Implementations SHOULD use a stable,
  serialisation-friendly slug (lowercase, snake_case or kebab-case)
  so the value can be matched in downstream tooling without quoting
  surprises.
- `better_when: Literal["higher", "lower"]` — direction of
  improvement for the scalar score.
- `score(*, forged, host, ctx: Mapping[str, Any]) -> tuple[float,
  float]` — returns `(score, perplexity_analog)`. The first element
  is the user-visible faithfulness number; the second is a
  positive-real quantity the FSM's `perplexity < best_perplexity`
  progress check can consume. Conventions per `better_when`:
  - `"lower"`: `perplexity_analog` MUST be a positive-real
    monotonically increasing function of the score (canonical:
    `exp(score)` for KL).
  - `"higher"`: `perplexity_analog` MUST be a positive-real
    monotonically decreasing function of the score (canonical:
    `1 - score` for cosine, clamped at `0.0`).

The `host` argument MAY be ignored by an implementation. Targets that
do not consult the host (GT-alignment scorers, monosemanticity
metrics, probe-accuracy scorers reading a cached probe) SHOULD accept
`host` for protocol conformance but SHOULD NOT load or move it.
sae-forge still loads the host on the FSM path today; a
`requires_host` opt-out is tracked as a follow-up.

Implementations SHALL be sync functions. Async scoring is out of
scope for v1; if/when needed, a separate `score_async` method may
be added in a follow-up without breaking the sync surface.

Implementations SHALL accept arbitrary keys in `ctx` and SHALL NOT
mutate `ctx`. Implementations MUST raise a clear `KeyError` (or
`ValueError`) if a required ctx key is missing, naming the expected
key — silent zero-score returns from missing inputs are a debugging
hazard. Third-party targets SHOULD namespace their ctx keys (e.g.
`_myorg_input_ids`, `_gt_alignment_inputs`) to avoid collisions with
sae-forge built-ins, which use the `_eval_*` prefix.

#### Scenario: protocol is runtime-checkable

- **WHEN** `isinstance(KLTarget(), FaithfulnessTarget)` is evaluated
- **THEN** the result is `True`
- **AND** `isinstance(CosineTarget(), FaithfulnessTarget)` is `True`
- **AND** `isinstance(object(), FaithfulnessTarget)` is `False`

#### Scenario: missing ctx key raises an actionable error

- **GIVEN** `KLTarget()` and a ctx missing `_eval_input_ids`
- **WHEN** `KLTarget().score(forged=..., host=..., ctx={})` is called
- **THEN** `KeyError` is raised whose message names `_eval_input_ids`
- **AND** no host or forged forward pass is executed

### Requirement: Built-in `KLTarget` preserves v0.4 LM behaviour

`saeforge.eval.targets.KLTarget` SHALL:

- Declare `name = "kl"` and `better_when = "lower"`.
- Read `ctx["_eval_input_ids"]` and `ctx.get("device", "cpu")`.
- Delegate to `saeforge.forge._kl_from_input_ids(forged, host,
  input_ids, device=device)`.
- Return `(kl, exp(kl))`.

The returned `kl` SHALL be bit-equal to the v0.4 `_evaluate_lm`
output on the same `(forged, host, input_ids, device)` triple.

#### Scenario: KLTarget matches `_kl_from_input_ids` byte-for-byte

- **GIVEN** the toy GPT-2 fixture and `eval_input_ids` from the
  existing `test_pipeline_smoke.py` setup
- **WHEN** `KLTarget().score(forged=..., host=..., ctx=...)` is
  invoked
- **AND** `_kl_from_input_ids(...)` is invoked with the same inputs
- **THEN** the first tuple element from `KLTarget.score` equals the
  return of `_kl_from_input_ids` (bit-equal under fp32; within
  `atol=1e-9` under bf16 / fp16)
- **AND** the second tuple element equals `math.exp(first)`

### Requirement: Built-in `CosineTarget` preserves v0.4 whisper behaviour

`saeforge.eval.targets.CosineTarget` SHALL:

- Declare `name = "cosine"` and `better_when = "higher"`.
- Read `ctx["_eval_audio_features"]`,
  `ctx.get("_eval_encoder_states")`, and `ctx.get("device", "cpu")`.
- Delegate to `saeforge.audio_eval.cosine_faithfulness(forged, host,
  audio_features, precomputed_host_states=..., device=device)`.
- Return `(cosine, max(0.0, 1.0 - cosine))`.

The returned `cosine` SHALL be bit-equal to the v0.4
`_evaluate_whisper_encoder` output on the same ctx.

#### Scenario: CosineTarget matches `cosine_faithfulness` byte-for-byte

- **GIVEN** the synthetic Whisper fixture from
  `test_whisper_encoder_adapter.py`
- **WHEN** `CosineTarget().score(forged=..., host=..., ctx=...)` is
  invoked
- **THEN** the first tuple element equals
  `cosine_faithfulness(forged, host, audio_features,
  precomputed_host_states=..., device=...)`
- **AND** the second tuple element equals `1.0 - first` (clamped at
  `0.0`)

### Requirement: `evaluate_faithfulness` dispatches via the active target

`saeforge.actions.evaluate_faithfulness` SHALL:

1. Look up `target = ctx.get("_faithfulness_target")`. If present,
   use it directly.
2. Otherwise, call `_default_target_for(forged.config.family)`,
   which SHALL return `CosineTarget()` for `"whisper_encoder"` and
   `KLTarget()` for `"gpt2" | "llama" | "gemma2" | "qwen2" |
   "qwen3"`. Unknown family SHALL raise `ValueError` whose message
   names the offending family and the supported set.
3. Call `score, perplexity = target.score(forged=..., host=...,
   ctx=ctx)`.
4. Write `ctx["faithfulness"] = score`, `ctx["perplexity"] =
   perplexity`. Compute `should_continue` from a per-`better_when`
   predicate:
   - `"lower"`: existing KL semantics —
     `(score >= min_faithfulness if min_faithfulness == 0.0 else
     score <= min_faithfulness * -1) and perplexity < best_perplexity
     and current_iter + 1 < iterations`.
   - `"higher"`: existing cosine semantics —
     `score >= min_faithfulness and perplexity < best_perplexity and
     current_iter + 1 < iterations`.
5. Log the active target's `name` in the `evaluate_faithfulness`
   transitions-log entry alongside the existing `faithfulness` /
   `perplexity` / `should_continue` / `advance_stream` fields.

The action SHALL NOT consult `forged.config.family` when an explicit
target is present in ctx — the user-supplied target fully overrides
family dispatch.

#### Scenario: GPT-2 default routes to KLTarget

- **GIVEN** a ctx with no `_faithfulness_target` and a forged model
  whose `config.family == "gpt2"`
- **WHEN** the FSM reaches `evaluate_faithfulness`
- **THEN** `_default_target_for("gpt2")` is invoked and returns a
  `KLTarget` instance
- **AND** the score written to `ctx["faithfulness"]` equals the v0.4
  `_evaluate_lm` result

#### Scenario: Whisper default routes to CosineTarget

- **GIVEN** a ctx with no `_faithfulness_target` and a forged model
  whose `config.family == "whisper_encoder"`
- **WHEN** the FSM reaches `evaluate_faithfulness`
- **THEN** `_default_target_for("whisper_encoder")` is invoked and
  returns a `CosineTarget` instance
- **AND** the score written to `ctx["faithfulness"]` equals the v0.4
  `_evaluate_whisper_encoder` result

#### Scenario: User-supplied target overrides family dispatch

- **GIVEN** a ctx with `_faithfulness_target` set to a custom
  `GTAlignmentTarget(better_when="higher")` AND a forged model whose
  `config.family == "gpt2"`
- **WHEN** `evaluate_faithfulness` runs
- **THEN** `GTAlignmentTarget.score` is invoked (verified by patching
  `_kl_from_input_ids` to raise — the test passes iff it is never
  called)
- **AND** `ctx["faithfulness"]` equals the alignment score
- **AND** `should_continue` is computed under the `"higher"`
  predicate (`score >= min_faithfulness`)

#### Scenario: Unknown family without explicit target raises

- **GIVEN** a ctx with no `_faithfulness_target` and a forged model
  whose `config.family == "fictional"`
- **WHEN** `evaluate_faithfulness` runs
- **THEN** `ValueError` is raised whose message contains
  `"fictional"` and the list of supported families

### Requirement: `ForgePipeline` accepts a `faithfulness` target

`saeforge.ForgePipeline` SHALL accept a constructor kwarg
`faithfulness: FaithfulnessTarget | None = None`.

When `None` (the default), `ForgePipeline.run(...)` and
`ForgePipeline.run_synthetic(...)` SHALL produce a `ForgeResult`
whose `faithfulness` and `n_params` are bit-equal to the v0.4
implementation on the same fixture (the byte-identity test in
`tests/forge/test_pipeline_byte_identity.py` pins this via a hash
over the relevant fields).

When set, the pipeline SHALL:

- Thread the instance into the FSM ctx as
  `_faithfulness_target` from `_build_fsm_ctx`.
- On the imperative path, call `self.faithfulness.score(forged=...,
  host=..., ctx=...)` directly (skipping the legacy
  `faithfulness_kl(...)` call) and record the result via
  `faithfulness=` and `faithfulness_target_name=` on `ForgeResult`.

`__post_init__` SHALL NOT validate the target's ctx-key dependencies
(those are scorer-internal); it MAY validate the trivial built-in
case "user passed `faithfulness=KLTarget()` but `eval_prompts=[]`"
with a clear error message.

#### Scenario: default `faithfulness=None` is byte-identical to v0.4

- **WHEN** `ForgePipeline(faithfulness=None, ...).run(...)` is
  invoked on the toy GPT-2 fixture
- **THEN** the resulting `ForgeResult` hashes to the same digest as
  the v0.4 implementation on the same fixture (hash includes
  `n_params`, `round(faithfulness, 8)`, `faithfulness_target_name`,
  and `basis.W_dec.tobytes()`)

#### Scenario: custom target threaded into FSM ctx

- **GIVEN** `ForgePipeline(faithfulness=GTAlignmentTarget(labels=L),
  orchestrator="fsm", ...)`
- **WHEN** `run_synthetic(...)` is called
- **THEN** the FSM ctx built by `_build_fsm_ctx` contains a
  `_faithfulness_target` key whose value is the same
  `GTAlignmentTarget` instance
- **AND** the `evaluate_faithfulness` action invokes that instance

### Requirement: `ForgeResult` generalises faithfulness; `faithfulness_kl` is deprecated

`saeforge.forge.ForgeResult` SHALL gain two new fields:

- `faithfulness: float | None = None` — the active target's score.
- `faithfulness_target_name: str | None = None` — the active
  target's `name`.

`faithfulness_kl` SHALL be re-implemented as a property:

- Read: returns `self.faithfulness` when
  `self.faithfulness_target_name == "kl"`, else returns `None`.
  Emits `DeprecationWarning` pointing at `self.faithfulness`.
- Write: the dataclass constructor SHALL still accept
  `faithfulness_kl=` as a kwarg through a custom `__init__` (or
  `__post_init__` adapter). Setting it SHALL forward to
  `self.faithfulness` and set `self.faithfulness_target_name = "kl"`.
  Setting via the kwarg SHALL emit `DeprecationWarning`.

The deprecation window is one minor version. The next minor version
after this change removes the property and the constructor kwarg; the
removal is announced in this change's `CHANGELOG.md` entry.

#### Scenario: `faithfulness_kl` read returns the KL value and warns

- **GIVEN** `r = ForgeResult(faithfulness=0.123,
  faithfulness_target_name="kl", ...)`
- **WHEN** `r.faithfulness_kl` is read
- **THEN** the return value is `0.123`
- **AND** a `DeprecationWarning` is emitted whose message names
  `.faithfulness`

#### Scenario: `faithfulness_kl` read returns `None` when target is not KL

- **GIVEN** `r = ForgeResult(faithfulness=0.91,
  faithfulness_target_name="cosine", ...)`
- **WHEN** `r.faithfulness_kl` is read
- **THEN** the return value is `None`
- **AND** a `DeprecationWarning` is emitted

#### Scenario: `faithfulness_kl=` constructor kwarg still works

- **WHEN** `ForgeResult(faithfulness_kl=0.2, ...)` is constructed
- **THEN** a `DeprecationWarning` is emitted
- **AND** the resulting object has `faithfulness == 0.2` and
  `faithfulness_target_name == "kl"`
