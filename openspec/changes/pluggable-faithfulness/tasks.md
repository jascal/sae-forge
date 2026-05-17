## 1. Protocol and built-in targets

- [ ] 1.1 Extend `saeforge/eval/faithfulness.py` (the existing module
  housing `faithfulness_kl`) with the `FaithfulnessTarget` protocol
  (`name`, `better_when`, `score`). Co-located with `faithfulness_kl`
  so callers have one import surface and the protocol sits next to
  the function it generalises. The docstring SHALL call out: stable
  `name` slug convention; `host` may be ignored (and why); third-party
  ctx keys SHOULD use a namespaced prefix to avoid clashes with
  built-in `_eval_*` keys. Document the `(score, perplexity_analog)`
  contract and the `Mapping[str, Any]` ctx convention.
- [ ] 1.2 Create `saeforge/eval/targets/kl.py::KLTarget`. `name="kl"`,
  `better_when="lower"`. `score(...)` reads
  `ctx["_eval_input_ids"]` and `ctx.get("device", "cpu")`, delegates
  to `saeforge.forge._kl_from_input_ids`, returns `(kl, exp(kl))`.
  Raises `KeyError` if `_eval_input_ids` is missing.
- [ ] 1.3 Create `saeforge/eval/targets/cosine.py::CosineTarget`.
  `name="cosine"`, `better_when="higher"`. `score(...)` reads
  `ctx["_eval_audio_features"]` and (optionally)
  `ctx["_eval_encoder_states"]`, delegates to
  `saeforge.audio_eval.cosine_faithfulness`, returns
  `(cosine, 1.0 - cosine)`.
- [ ] 1.4 Create `saeforge/eval/targets/__init__.py`. Exports
  `KLTarget`, `CosineTarget`, and
  `_default_target_for(family: str) -> FaithfulnessTarget` (returns
  `CosineTarget()` for `"whisper_encoder"`, `KLTarget()` otherwise;
  unknown family raises `ValueError`). The protocol itself
  (`FaithfulnessTarget`) lives in `saeforge.eval.faithfulness` and is
  re-exported from `saeforge.eval` for convenience.

## 2. `evaluate_faithfulness` rewrite

- [ ] 2.1 Replace the `if family == "whisper_encoder": ... else: ...`
  body in `saeforge/actions/__init__.py::evaluate_faithfulness` with
  a single dispatch: `target = ctx.get("_faithfulness_target") or
  _default_target_for(family)`. Call `target.score(forged=..., host=..., ctx=ctx)`.
- [ ] 2.2 Move the `should_continue` predicate into a small helper
  `_predicate_from_target(target, score, ctx) -> bool` that branches
  on `target.better_when`. `lower` preserves the existing KL-negation
  convention exactly; `higher` preserves the cosine convention.
- [ ] 2.3 Delete `_evaluate_lm` and `_evaluate_whisper_encoder` from
  the action module (their bodies now live on the target classes).
  `_compute_advance_stream` is unchanged.
- [ ] 2.4 Log the active target's `name` in the
  `evaluate_faithfulness` transitions-log entry so FSM traces still
  show which scorer ran.

## 3. `ForgePipeline` plumbing

- [ ] 3.1 Add `faithfulness: FaithfulnessTarget | None = None` to
  `ForgePipeline`. Default `None` preserves v0.4 behaviour.
- [ ] 3.2 In `_build_fsm_ctx`, write `_faithfulness_target` into the
  returned dict (value: `self.faithfulness`, may be `None`). The
  action layer reads it via `ctx.get(...)`.
- [ ] 3.3 In `_run_real_imperative`, when `self.faithfulness is not
  None`, build a minimal ctx (`{"_eval_input_ids": ..., "device":
  ...}` plus whatever the target reads) and call
  `self.faithfulness.score(forged=model, host=host, ctx=ctx)`. When
  `None`, call `faithfulness_kl(...)` as today (byte-identity).
- [ ] 3.4 In `_run_synthetic_imperative`, same change as 3.3 against
  the existing `_kl_from_input_ids(...)` call.
- [ ] 3.5 Validate at `__post_init__` time: if `self.faithfulness is
  not None` and `self.eval_prompts == [] and self.eval_audio_features
  is None`, raise `ValueError` with a message naming the target's
  `name` and the missing ctx field. (A custom target may need
  different inputs entirely; only validate when the active target is
  a built-in and the corresponding eval input is empty.)

## 4. `ForgeResult` deprecation

- [ ] 4.1 Add `faithfulness: float | None = None` and
  `faithfulness_target_name: str | None = None` to `ForgeResult`.
- [ ] 4.2 Replace the `faithfulness_kl: float | None` field with a
  property: returns `self.faithfulness` when
  `self.faithfulness_target_name == "kl"`, else `None`. Property
  getter emits `DeprecationWarning` pointing at `.faithfulness`.
- [ ] 4.3 Accept `faithfulness_kl=` as a constructor kwarg in a
  custom `__init__` (since `@dataclass` won't let us alias). Forward
  to `faithfulness=` and set `faithfulness_target_name="kl"`. Emit
  `DeprecationWarning` on the kwarg path too — every test under
  4.4 (in-repo write-site migration) MUST be a counterexample to
  ensure the warning fires only on legacy call sites, not on our own
  code.
- [ ] 4.4 Update every in-repo write site to use `faithfulness=` and
  `faithfulness_target_name=` directly:
  - `_run_real_imperative` (currently writes `faithfulness_kl=`)
  - `_run_real_fsm`
  - `_run_synthetic_imperative`
  - `_run_synthetic_fsm`
  - `saeforge/forge_quality.py` (any callers of `ForgeResult`)
  - `saeforge/sweep.py` (any callers)
  Grep `faithfulness_kl=` to find any I missed.
- [ ] 4.5 Update every in-repo read site that consumes
  `result.faithfulness_kl` to use `result.faithfulness` instead.
  In-repo consumers should not trigger their own deprecation warnings.

## 5. Tests

- [ ] 5.1 `tests/eval/test_faithfulness_target_protocol.py` —
  - `KLTarget` and `CosineTarget` satisfy the protocol (runtime
    `isinstance(target, FaithfulnessTarget)` check via
    `@runtime_checkable`).
  - `KLTarget.score(...)` returns the same float as the legacy
    `_kl_from_input_ids(...)` on the same inputs.
  - `CosineTarget.score(...)` returns the same float as the legacy
    `cosine_faithfulness(...)` on the same inputs.
  - `_default_target_for("gpt2")` returns a `KLTarget` instance;
    `_default_target_for("whisper_encoder")` returns a `CosineTarget`
    instance; `_default_target_for("fictional")` raises `ValueError`.
- [ ] 5.2 `tests/forge/test_pipeline_byte_identity.py` — run
  `ForgePipeline(faithfulness=None, ...)` on the toy GPT-2 fixture;
  hash `(n_params, round(faithfulness, 8),
  faithfulness_target_name, basis.W_dec.tobytes())` and compare
  against a pinned `sha256` digest. The digest is captured once on
  pre-change `main` and asserted post-change.
- [ ] 5.3 `tests/forge/test_pipeline_with_custom_target.py` —
  - Define a `GTAlignmentTarget` in the test module that scores
    forged-feature → cluster-label alignment on a 2D
    mixture-of-gaussians fixture (built inline; ~20 lines).
    `better_when="higher"`, returns `(alignment, 1 - alignment)`.
  - Run `ForgePipeline(faithfulness=GTAlignmentTarget(labels=...),
    orchestrator="fsm", ...)` with a deliberately
    under-compressed basis (kept features < ground-truth feature
    count).
  - Assert the loop re-enters `refining` based on the GT alignment
    being below `min_faithfulness`. Verify `_kl_from_input_ids` is
    NOT called (patch it to raise; the test passes iff KL is never
    consulted).
  - Assert `result.faithfulness_target_name == "gt_alignment"` and
    `result.faithfulness` is the alignment score.
- [ ] 5.4 `tests/forge/test_forge_result_deprecation.py` —
  - `ForgeResult(faithfulness=0.1,
    faithfulness_target_name="kl").faithfulness_kl` returns `0.1`
    and emits `DeprecationWarning`.
  - `ForgeResult(faithfulness=0.9,
    faithfulness_target_name="cosine").faithfulness_kl` returns
    `None` and emits `DeprecationWarning`.
  - `ForgeResult(faithfulness_kl=0.2)` (constructor kwarg path)
    emits `DeprecationWarning` and produces an object with
    `faithfulness == 0.2`, `faithfulness_target_name == "kl"`.
- [ ] 5.5 Existing tests stay green with no flag changes.
  Specifically:
  - `tests/forge/test_pipeline_smoke.py` (whatever the v0.4 smoke is
    called)
  - `tests/test_whisper_encoder_adapter.py`
  - `tests/test_audio_eval.py`
  - `tests/training/test_distillation.py`
  Run the full `pytest` suite and confirm zero regressions.

## 6. Example

- [ ] 6.1 Write `examples/forge_with_gt_alignment.py`:
  - Build a 2D mixture-of-gaussians fixture (3 clusters, 1000
    samples) with known cluster labels.
  - Construct a synthetic SAE basis from cluster centroids (one
    feature per cluster, plus deliberate noise).
  - Build a `GTAlignmentTarget(labels=..., better_when="higher")`.
  - Run a tiny `ForgePipeline` over the synthetic basis with
    `orchestrator="fsm"`, `n_tasks=2`, `min_faithfulness=0.8`.
  - Print `result.faithfulness`, `result.faithfulness_target_name`,
    and the transitions-log summary.
  - Target wall-clock: under 60s on a CPU laptop. No HF download,
    no GPU, no polygram side-effects.
- [ ] 6.2 Register `examples/forge_with_gt_alignment.py` in
  `tests/test_examples_smoke.py` (or whatever the existing
  examples-smoke harness is named) so a CI regression breaks the
  build, not user trust.

## 7. Docs

- [ ] 7.1 `docs/finetune-recipe.md`: add a "Swapping the faithfulness
  target" subsection after "Host distillation". Cover the protocol
  signature, the two built-in targets, a one-screen custom-target
  example, and a pointer at
  `examples/forge_with_gt_alignment.py`. Note that the default
  (`faithfulness=None`) is byte-identical to v0.4.
- [ ] 7.2 `docs/advanced-fsm-options.md`: add `faithfulness` to the
  "Basis loop (inner)" knobs table (or a sibling table — pick the
  one that's about loop gating). Note that the target's
  `better_when` controls `min_faithfulness` semantics, replacing the
  hard-coded LM-vs-encoder rule.
- [ ] 7.3 `CHANGELOG.md`: under `[Unreleased]`, add a "Pluggable
  faithfulness target" entry listing the new field, the two
  built-in targets, the deprecation of
  `ForgeResult.faithfulness_kl`, and the planned removal one minor
  version later. Include an explicit before/after migration block
  for the most common consumer pattern:

  ```text
  Before:
      result = pipeline.run(...)
      print(result.faithfulness_kl)            # DeprecationWarning

  After (KL default — no code change required):
      result = pipeline.run(...)
      print(result.faithfulness)               # same value

  After (custom target):
      from saeforge.eval.faithfulness import FaithfulnessTarget
      result = ForgePipeline(faithfulness=MyTarget(), ...).run(...)
      print(result.faithfulness, result.faithfulness_target_name)
  ```

  The migration block sits at the top of the entry so anyone scanning
  for "faithfulness_kl" finds the rename inline.

## 8. Validation

- [ ] 8.1 `openspec validate pluggable-faithfulness --strict`.
- [ ] 8.2 `ruff check` clean on every modified file.
- [ ] 8.3 Full `pytest` suite green. Particular attention to:
  - the byte-identity test (5.2),
  - the whisper-encoder smoke,
  - the host-distillation tests.
- [ ] 8.4 `python examples/forge_with_gt_alignment.py` completes in
  under 60s on the 16GB Intel Mac (the cross-arch defaults-validation
  surface).

## 9. What this change explicitly defers

- [ ] 9.1 **Non-transformer host support.** `WorldModel`-protocol
  generalisation (RNNs / SSMs / diffusion U-Nets) is a separate,
  larger change tracked as follow-up `world-model-protocol`. The
  faithfulness protocol does not commit to a particular host model
  shape — it just consumes `ctx` — so it's compatible with that
  future change, but does not deliver it.
- [ ] 9.2 **Additional built-in scorers.** Pearson, Spearman,
  monosemanticity, probe accuracy, feature-coverage. Single-file
  follow-ups once the protocol lands; none of them block this
  change.
- [ ] 9.3 **Distributed faithfulness scoring** across multiple GPUs.
  Single-host preserved. A target's `score` is free to shard
  internally but the protocol doesn't require it.
- [ ] 9.4 **CLI `--faithfulness-target` flag.** Would need an
  entry-point registry. Documented use cases all go through the
  Python API.
- [ ] 9.5 **Polygram-side changes.** This change is sae-forge-internal.
  The GT-alignment example uses a self-contained synthetic basis;
  no polygram contracts move.
- [ ] 9.6 **Inner-loop target-aware regrow gating.** Today's
  `perform_regrowth` is gated on `regrow_count` / the adaptive
  controller, not on faithfulness. Lifting that gate behind the
  target is a separate change to `BasisMachine` if it ever proves
  useful. The pluggable-target change here gates only the
  outer-refine loop, via `should_continue`, matching v0.4 behaviour.
