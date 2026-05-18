# faithfulness-target Specification

## Purpose

Defines the pluggable scorer surface that gates the forge FSM's
refine loop. `FaithfulnessTarget` lets callers swap the
loop-gating signal (KL by default for LM hosts, cosine for
Whisper encoders, GT-alignment AUC for label-rich fixtures, or
any user-supplied scorer) without touching pipeline plumbing.
Built-in targets (`KLTarget`, `CosineTarget`, `GroundTruthTarget`)
preserve the family-dispatched defaults; `ForgePipeline.faithfulness`
threads a user-supplied target end-to-end through both the
imperative and FSM paths.

## Requirements
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


### Requirement: Built-in `GroundTruthTarget` scores forged residuals against label matrices

`saeforge.eval.targets.GroundTruthTarget` SHALL be a built-in
implementation of the `FaithfulnessTarget` protocol with:

- `name = "gt_alignment"`.
- `better_when = "higher"`.
- Constructor signature
  `__init__(self, labels: np.ndarray, *, scorer:
  Literal["auc"] = "auc", pool: Literal["mean", "max", "last"]
  = "mean", hidden_extractor: Callable | None = None)`.

The constructor SHALL validate at construction time:

- `labels.ndim == 2` (raise `ValueError` naming the offending
  shape otherwise).
- `labels.shape[0] >= 1` and `labels.shape[1] >= 1` (raise
  `ValueError` naming the offending shape otherwise).
- `scorer == "auc"` (raise `ValueError` naming the supplied
  scorer and the supported set `{"auc"}` otherwise — the
  parameter is a forward-compatibility hook for future scorers
  but only `"auc"` ships in v1).
- `pool in ("mean", "max", "last")` (raise `ValueError` naming
  the supplied pool and the supported set otherwise).

The constructor SHALL NOT import torch. `labels` SHALL be coerced
to a numpy array of float dtype at construction time so the rest
of the implementation can assume numpy throughout.

`GroundTruthTarget.score(*, forged, host, ctx)` SHALL:

1. Read `ctx["_eval_input_ids"]` (the same key consumed by
   `KLTarget`; built-in targets share the `_eval_*` ctx
   namespace). Raise `KeyError` whose message names
   `_eval_input_ids` when the key is missing or its value is
   `None`. The error SHALL fire before any forged forward pass.
2. Read `ctx.get("device", "cpu")`. The default extractor uses
   it to place inputs; user-supplied extractors are free to
   ignore it.
3. Validate `self.labels.shape[0] == input_ids.shape[0]`. Raise
   `ValueError` whose message names both shapes when they
   disagree.
4. Call the active hidden extractor on `forged` and `input_ids`.
   The active extractor is `self.hidden_extractor` when
   non-`None`, else the default extractor specified in the next
   requirement.
5. Pool the returned tensor across the sequence dimension per
   `self.pool`:
   - `"mean"`: `tensor.mean(dim=1)`.
   - `"max"`: `tensor.max(dim=1).values`.
   - `"last"`: `tensor[:, -1, :]`.
   Skip pooling if the tensor is already 2D (the extractor
   pre-pooled).
6. Detach, move to CPU, and convert to a numpy float array of
   shape `(N, hidden_size)`.
7. Compute per-feature × per-label AUC. Result shape `(hidden_size,
   M)` where `M == labels.shape[1]`.
8. Reduce: `mean_best_auc = float(auc.max(axis=0).mean())` —
   max over features (best matching feature per label), then
   mean over labels.
9. Return `(mean_best_auc, max(0.0, 1.0 - mean_best_auc))`.
   The tuple convention follows the protocol's `better_when="higher"`
   rule (`perplexity_analog` is a positive-real monotonically
   decreasing function of the score), mirroring `CosineTarget`'s
   `(cosine, max(0, 1 - cosine))`.

`GroundTruthTarget.score` SHALL NOT consult `host`. The `host`
kwarg is accepted exclusively for protocol conformance. No host
forward pass runs from within `score`; no device move on `host`
happens from within `score`.

`GroundTruthTarget` SHALL NOT have a tracked
`requires_host = False` attribute in this change. The protocol's
deferred `requires_host` opt-out (filed as a follow-up in
`saeforge/eval/faithfulness.py:55-60`) will add the attribute and
the upstream skip-host-forward optimisation together; this
change does not block on it.

#### Scenario: GroundTruthTarget satisfies the runtime-checkable protocol

- **WHEN** `isinstance(GroundTruthTarget(labels=np.eye(4)),
  FaithfulnessTarget)` is evaluated
- **THEN** the result is `True`
- **AND** the constructor did not import torch (verifiable by
  patching `sys.modules` to remove `torch` before construction)

#### Scenario: Missing _eval_input_ids raises before any forward

- **GIVEN** `target = GroundTruthTarget(labels=np.eye(4))` and
  a `forged` whose `.forward` is patched to raise `RuntimeError`
- **WHEN** `target.score(forged=forged, host=None, ctx={})` is
  called
- **THEN** `KeyError` is raised whose message names
  `_eval_input_ids`
- **AND** the patched `forward` was never called

#### Scenario: Shape mismatch between labels and input_ids is rejected

- **GIVEN** `target = GroundTruthTarget(labels=np.eye(4))`
  (i.e. `labels.shape == (4, 4)`)
- **AND** `ctx = {"_eval_input_ids": tensor of shape (5, 8)}`
- **WHEN** `target.score(forged=..., host=None, ctx=ctx)` is
  called
- **THEN** `ValueError` is raised whose message contains both
  `4` and `5`

#### Scenario: Identity-extractor fixture scores near 1.0

- **GIVEN** a `(N, M)` binary label matrix `L`
- **AND** a `hidden_extractor` that returns a tensor whose
  pooled `(N, hidden_size)` view equals `L` exactly (with
  `hidden_size >= M`, padding extra columns with noise)
- **AND** `target = GroundTruthTarget(labels=L,
  hidden_extractor=that_extractor)`
- **WHEN** `target.score(forged=..., host=None, ctx={
  "_eval_input_ids": ids})` is called
- **THEN** the returned `(score, perplexity_analog)` satisfies
  `score > 0.95`
- **AND** `perplexity_analog == max(0.0, 1.0 - score)`

#### Scenario: host is never consulted

- **GIVEN** `target = GroundTruthTarget(labels=L)` with any
  valid `L`
- **AND** a `host` whose `.forward` is patched to raise
- **WHEN** `target.score(forged=valid_forged, host=host,
  ctx=valid_ctx)` is called
- **THEN** `score` returns a `(float, float)` tuple
- **AND** the patched host `forward` was never called

### Requirement: `GroundTruthTarget`'s default hidden extractor covers LM-shape forged modules

`GroundTruthTarget` SHALL ship with a default `hidden_extractor`
that is used when the constructor is called without an explicit
`hidden_extractor=`. Given `(forged, input_ids)`, the default
extractor SHALL:

1. Attempts `forged.torch_module.transformer(input_ids)`. If
   that attribute exists and the call succeeds, the returned
   tensor is the residual stream `(batch, seq, hidden_size)`
   (the GPT-2-shape contract — see
   `saeforge/adapters/gpt2.py::ForgedGPT2`).
2. Otherwise attempts `forged.torch_module.model(input_ids)`.
   If that attribute exists, the returned tensor is the
   residual stream (the Llama/Gemma2/Qwen2/Qwen3/Qwen3_moe
   contract — see
   `saeforge/adapters/llama.py::ForgedLlama`).
3. Otherwise raises `RuntimeError` whose message names the two
   attributes attempted (`.transformer`, `.model`) and instructs
   the caller to pass an explicit `hidden_extractor=`. The
   message SHALL include the type name of
   `forged.torch_module` for debuggability.

The default extractor SHALL detach the returned tensor and move
it to CPU before returning. The default extractor SHALL NOT
assume the eval set fits in a single batch — it operates on
whatever `input_ids` tensor the FSM ctx supplied; batching is
upstream's responsibility.

The default extractor does NOT support the `whisper_encoder`
family. Users running `GroundTruthTarget` against a Whisper
forge SHALL pass an explicit `hidden_extractor=`; the default
will raise the `RuntimeError` above on attempt.

#### Scenario: Default extractor picks up GPT-2-shape `.transformer`

- **GIVEN** a fake `forged` whose `.torch_module.transformer`
  is callable and returns a known tensor `T` of shape
  `(batch, seq, hidden_size)`
- **AND** `target = GroundTruthTarget(labels=L)` (no
  `hidden_extractor=` passed)
- **WHEN** `target.score(forged=fake, host=None, ctx={
  "_eval_input_ids": ids})` is called
- **THEN** the AUC is computed against the rows of `T` (after
  pooling and numpy conversion)
- **AND** the call does NOT touch `.torch_module.model`

#### Scenario: Default extractor falls back to Llama-shape `.model`

- **GIVEN** a fake `forged` whose `.torch_module` exposes
  `.model` (callable, returning a known tensor) but not
  `.transformer`
- **AND** `target = GroundTruthTarget(labels=L)`
- **WHEN** `target.score(...)` is called
- **THEN** the AUC is computed against the `.model` output
- **AND** no `AttributeError` propagates to the caller

#### Scenario: Default extractor surfaces a clear error for exotic hosts

- **GIVEN** a fake `forged` whose `.torch_module` exposes
  neither `.transformer` nor `.model`
- **AND** `target = GroundTruthTarget(labels=L)`
- **WHEN** `target.score(...)` is called
- **THEN** `RuntimeError` is raised whose message contains
  `hidden_extractor`
- **AND** the message names both `.transformer` and `.model`
  as the attributes attempted
- **AND** the message includes the type name of
  `forged.torch_module`

### Requirement: `GroundTruthTarget` is exported from the canonical import paths

`GroundTruthTarget` SHALL be importable from:

- `saeforge.eval.targets.GroundTruthTarget` (canonical location
  alongside `KLTarget` and `CosineTarget`).
- `saeforge.eval.targets.gt_alignment.GroundTruthTarget` (the
  module-level path).
- `saeforge.eval.GroundTruthTarget` (the `eval`-namespace
  convenience re-export, matching how `KLTarget` /
  `CosineTarget` / `FaithfulnessTarget` are surfaced).

The `saeforge.eval.targets.__init__` module's `__all__` SHALL
list `"GroundTruthTarget"` alongside the existing entries. The
module docstring's count of built-in implementations SHALL be
updated from "two" to "three" and SHALL mention
`GroundTruthTarget` in the same enumeration style as
`KLTarget` / `CosineTarget`.

The `_default_target_for(family)` family-dispatch function
SHALL NOT change in this requirement. `GroundTruthTarget` is
fixture-specific opt-in only; it is not a family default for
any family.

#### Scenario: Canonical import path resolves

- **WHEN** `from saeforge.eval.targets import GroundTruthTarget`
  is executed
- **THEN** the import succeeds
- **AND** the imported symbol is the same object as
  `saeforge.eval.targets.gt_alignment.GroundTruthTarget`

#### Scenario: eval-namespace re-export resolves

- **WHEN** `from saeforge.eval import GroundTruthTarget` is
  executed
- **THEN** the import succeeds
- **AND** the imported symbol is the same object as
  `saeforge.eval.targets.GroundTruthTarget`

#### Scenario: `_default_target_for` is unchanged

- **WHEN** `_default_target_for("gpt2")` is called
- **THEN** the return value is a `KLTarget` instance (not a
  `GroundTruthTarget`)
- **AND** the same is true for every LM family listed in the
  v0.4 dispatch
- **AND** `_default_target_for("whisper_encoder")` still
  returns a `CosineTarget` instance
