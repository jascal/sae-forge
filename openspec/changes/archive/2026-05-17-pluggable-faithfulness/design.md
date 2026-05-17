## Context

Today's `evaluate_faithfulness` action in
`saeforge/actions/__init__.py` dispatches on
`forged.config.family`:

- LM families (`gpt2`, `llama`, `gemma2`, `qwen2`, `qwen3`) →
  `_evaluate_lm` → `_kl_from_input_ids`. Writes
  `ctx["faithfulness"] = kl`, `ctx["perplexity"] = exp(kl)`,
  `ctx["should_continue"]` derived from
  `min_faithfulness` (negated KL convention) and a `perplexity <
  best_perplexity` progress check.
- `whisper_encoder` → `_evaluate_whisper_encoder` →
  `cosine_faithfulness`. Writes
  `ctx["faithfulness"] = cosine`,
  `ctx["perplexity"] = 1 - cosine`,
  `ctx["should_continue"]` derived from a
  `cosine >= min_faithfulness` predicate (positive convention).

Both scorers are pure functions of `(forged, host, ctx-derived
inputs) -> (score, perplexity_analog)`. The `should_continue`
predicate differs only by direction (`lower-better` vs `higher-better`).
The action's body has two parallel arms that share the same shape;
adding a third scorer means a third arm and a third
`min_faithfulness` convention to remember.

The user-visible surface today:

- `ForgeResult.faithfulness_kl: float | None` is the headline number
  on every `ForgePipeline.run(...)`. The continual-learning loop
  reads `ctx["faithfulness"]` and writes the same value back into
  the result.
- There is no Python-API way to swap the scorer. A caller who wants
  to gate the loop on a GT-alignment score has to subclass the
  action layer.

## Goals / Non-Goals

**Goals**:

- Lift the family dispatch out of the action and behind a
  `FaithfulnessTarget` protocol. `KLTarget` and `CosineTarget` are
  the two built-in implementations.
- `ForgePipeline(faithfulness=None, ...)` is byte-identical to v0.4
  on every fixture in `tests/forge/test_pipeline_smoke.py`.
- `ForgeResult.faithfulness` is a generic float keyed by the target's
  `name`. `ForgeResult.faithfulness_kl` remains as a one-minor-version
  deprecated alias.
- The user can supply a custom target as a constructor kwarg without
  touching action code. Both the imperative path and the FSM path
  honour it.

**Non-goals**:

- No additional built-in scorers in this change. Pearson, Spearman,
  monosemanticity, probe accuracy, etc. are all single-file follow-ups
  once the protocol exists.
- No CLI flag. The Python API is the integration point. A
  `--faithfulness-target` entry-point registry can come later; the
  documented use cases (notebooks, research code) all go through
  Python.
- No non-transformer host support. `NativeModel` /
  `SubspaceProjector` still assume attention/MLP weight matrices to
  project. A WorldModel-protocol generalisation is a separate change.
- No polygram-side changes. The GT-alignment scorer in the example /
  test fixture builds on a synthetic 2D mixture-of-gaussians — no
  polygram code is touched.
- No distributed faithfulness scoring. Single-host is preserved.

## Decisions

### Decision 1: Protocol returns `(score, perplexity_analog)`, not just `score`

The FSM's `_compute_advance_stream` and the refine-loop's
`perplexity < best_perplexity` progress check both consume
`perplexity` today. Hiding that field would force one of:

- The action computes the analog from the score (couples the action
  back to the target's convention — defeats the point of the protocol).
- The action drops the progress check entirely (changes loop semantics
  for the byte-identity test).

Returning both is cheap (one extra float per call) and keeps the
existing `should_continue` shape intact. Targets pick the convention
that matches `better_when`: `lower-better` returns `exp(score)` (KL
convention); `higher-better` returns `1 - score` (cosine convention).
Each convention keeps `perplexity < best_perplexity` pointing the right
direction.

### Decision 2: `better_when` is per-target, not hard-coded by name

A field on the target, not a registry keyed by `name`. Lets third-party
targets declare their direction without sae-forge knowing the name.
The `should_continue` predicate consults this field:

- `better_when="lower"`: existing KL semantics. `min_faithfulness=0.0`
  disables the gate (matches v0.4); `min_faithfulness < 0` encodes a
  max allowed score via the legacy `kl <= min_faithfulness * -1`
  predicate (preserved for byte-identity with KL).
- `better_when="higher"`: `score >= min_faithfulness` predicate
  (matches the existing cosine arm).

The legacy KL-negation convention is preserved for `better_when="lower"`
specifically because removing it would change the byte-identity of
existing FSM runs with `min_faithfulness < 0` set. Future targets that
want a less surprising "max allowed lower-better score" convention can
declare `better_when="higher"` against `-score` or roll their own
predicate.

### Decision 3: Target is passed in, not registered

The protocol is a single class, not a `(register, dispatch_by_name)`
registry. Three reasons:

1. Notebook / research code wants to pass a target instance directly;
   a registry adds a layer of indirection with no payoff.
2. Targets carry state (the GT-alignment scorer holds the cluster
   labels, a probe-accuracy scorer holds the probe). A registry would
   either lose that state or require factories.
3. A registry can be added on top later if a CLI `--faithfulness-target`
   flag becomes useful. Doing it now is speculative.

`ctx["_faithfulness_target"]` is the FSM-side handoff. `None` means
"use the family-dispatched default" — the only behaviour the v0.4
action exposes.

### Decision 4: Family dispatch survives as the default policy

When `faithfulness=None`, the action picks `KLTarget` for LM families
and `CosineTarget` for `whisper_encoder`. That is byte-identical to
v0.4's hard-coded dispatch. The byte-identity test pins this; without
it, every existing notebook / fixture / example would have to declare a
target explicitly.

The dispatch logic moves into a small helper
`saeforge.eval.targets._default_target_for(family)` so it has one
home.

### Decision 5: `ForgeResult.faithfulness_kl` deprecation strategy

Two-stage migration:

- **This change**: `ForgeResult` gains `faithfulness: float | None` and
  `faithfulness_target_name: str | None`. `faithfulness_kl` becomes a
  property that returns the value when `faithfulness_target_name ==
  "kl"`, otherwise `None`. Read access emits `DeprecationWarning` once
  per Python session (using `warnings.warn(..., DeprecationWarning,
  stacklevel=2)` — not `once`, because tests need to see the warning
  every time).
- **One minor version later**: remove the property. `faithfulness_kl`
  becomes a plain `AttributeError` on access. The user-facing migration
  is `result.faithfulness_kl` → `result.faithfulness` everywhere; the
  changelog entry calls this out.

Constructor accepts `faithfulness_kl=` as a kwarg that forwards to
`faithfulness` and sets `faithfulness_target_name="kl"` — keeps any
existing call site that constructs `ForgeResult` directly working
through the deprecation window.

### Decision 6: Built-in targets read ctx, not arguments

`KLTarget.score(forged, host, ctx)` reads `ctx["_eval_input_ids"]`
itself; `CosineTarget` reads `ctx["_eval_audio_features"]` and
`ctx["_eval_encoder_states"]`. This matches how the action layer
already plumbs eval inputs through ctx and avoids a wider protocol
signature. A third-party target that needs new inputs adds its own
ctx key on construction (no FSM topology change required).

The trade-off: targets are coupled to the ctx key names. Documented
in the protocol docstring with the recommendation to namespace
third-party keys (`_my_target_input_ids`) to avoid clashes.

### Decision 7: Imperative path constructs the target inline

`_run_real_imperative` doesn't build a ctx dict; it calls
`faithfulness_kl(...)` directly today. The minimal change is:

```python
if self.faithfulness is None:
    faithfulness = faithfulness_kl(model, host, prompts, ...)
    target_name = "kl"
else:
    score, _ = self.faithfulness.score(
        forged=model, host=host, ctx={"_eval_input_ids": ..., "device": ...},
    )
    faithfulness = score
    target_name = self.faithfulness.name
```

This keeps the imperative path's byte-identity property — when
`faithfulness=None` the call shape is exactly the v0.4 call — while
giving custom targets the same ctx contract they get on the FSM path.

### Decision 8: Byte-identity test uses a hash, not float equality

The pre-change implementation can drift in low-order fp32 bits across
torch versions / hardware. Hashing the relevant `ForgeResult` fields
(`n_params`, `faithfulness` rounded to 8 decimal places, the basis
shape, the active target name) and comparing against a pinned digest
gives a stable invariant without making the test brittle to torch
upgrade noise. The digest is captured on a pinned fixture and lives
in the test file as a constant; regenerating it after an intentional
behaviour change is a one-line edit.

## Open Questions

- **Regrow-trigger framing.** Earlier framing of this change described
  the basis loop's regrow step as KL-gated. In current code
  (`saeforge/machines/basis.orca.md`) `perform_regrowth` is gated on
  `ctx.regrow_count > 0` and (when `adaptive_regrow=True`) on
  `RegrowController.next_count(...)`. Faithfulness gates `should_continue`,
  which decides whether the *outer* refine loop re-enters `BasisMachine`
  (and therefore whether another compress↔regrow cycle runs). The
  pluggable-target tests cover the outer-loop case. If we later want
  target-aware inner-loop gating ("skip regrow when target says we're
  already converged"), that's a separate change to `BasisMachine` and
  the `should_regrow` guard. Flagging here so the next reader doesn't
  re-derive it.
- **Should `score` be `async`?** Probe-accuracy scorers that hit a
  remote eval service would benefit; today's targets are sync. Defer:
  if/when an async use case lands, we can add a `score_async` method
  alongside `score` rather than changing the existing signature.
- **Should the protocol surface a `requires_host: bool` flag?** A
  GT-alignment scorer doesn't need a host model at all — it only needs
  the forged model's encoded features and the labels. Today the FSM
  always loads the host. Loading the host unnecessarily is a real cost
  on Llama/Gemma-scale runs. Defer: the scorer can ignore the host
  argument; documenting it as optional in the protocol docstring is
  enough until someone hits the cost.

## Risks / Trade-offs

- **Two paths into faithfulness (imperative + FSM).** The byte-identity
  test pins the imperative path; an explicit FSM-path test pins the
  FSM path. Drift between them is the main hazard; the two tests
  share a fixture so any drift surfaces immediately.
- **Targets reading ctx by string key.** A typo in a third-party
  target's ctx key (`_eval_input_ids` vs `_eval_inputs`) silently
  reads `None` and the score path picks up a zero. Mitigation: built-in
  targets check the key is present and raise a clear `KeyError` with
  the expected key name. The protocol docstring recommends the same
  pattern for third-party targets.
- **Deprecation churn.** Every consumer of `ForgeResult.faithfulness_kl`
  in the repo (and in user code) sees a `DeprecationWarning`. We grep
  the repo and migrate all in-repo call sites in the same PR.
  External callers get a one-minor-version window.

## Migration Plan

Three audiences:

- **Default users** (`ForgePipeline(...).run(...)` with no
  `faithfulness=`): nothing changes. The byte-identity test pins this.
  `result.faithfulness_kl` keeps working through the deprecation window.
- **Whisper-encoder users**: nothing changes. The family dispatch picks
  `CosineTarget` automatically.
- **Custom-target users**: import a target from
  `saeforge.eval.targets`, or implement the protocol, and pass it as
  `faithfulness=`. Two doc surfaces (`docs/finetune-recipe.md` and
  `docs/advanced-fsm-options.md`) get the swap example.

`CHANGELOG.md` calls out the deprecation under the next minor version.
The removal happens one minor version later.
