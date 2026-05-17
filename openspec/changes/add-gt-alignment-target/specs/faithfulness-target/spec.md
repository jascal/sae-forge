# faithfulness-target Specification

## ADDED Requirements

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
