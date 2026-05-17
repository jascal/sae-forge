## Why

v0.4 shipped the `FaithfulnessTarget` protocol and two built-in
targets — `KLTarget` (LM-family default) and `CosineTarget`
(Whisper-encoder default). The pluggable-faithfulness change
(`openspec/changes/archive/2026-05-17-pluggable-faithfulness/`)
explicitly punted a third built-in for label-rich fixtures:

> **Out of scope: ground-truth feature-alignment scorer.** This was
> proposed as a third built-in target. We've left it out of this
> change because no in-tree caller needs it yet; once a downstream
> consumer ships, the target lands as a single-file follow-up
> against the protocol.

That consumer has shipped. `jascal/sm-sae`'s
`scripts/forge_pipeline.py::GroundTruthAlignment` is a per-fixture
ground-truth scorer: forward the eval `input_ids` through the
forged model, capture the residual stream at a canonical layer,
pool across the sequence axis, then compute per-feature × per-label
AUC, max over features, mean over labels. The motivating use is
synthetic mixture-of-gaussians fixtures with known cluster IDs and
the Standard Model decay cascade, where the underlying features
are *known* and KL is at best a proxy for the quantity that
matters.

Any fixture with per-sample binary labels wants the same target:
synthetic mixtures, BERT-probe-derived datasets, the SM physics
benchmark, concept-bottleneck eval suites. Re-implementing it
across consumer repos leads to drift (the sm-sae version, for
example, hard-codes mean pooling and walks `forged.torch_module
.transformer` directly — both fine for GPT-2-shape hosts and wrong
for Llama / Gemma / Qwen-shape ones). Upstreaming a strict
generalisation with pluggable pooling and a pluggable
hidden-state extractor closes the gap and makes the scorer
discoverable via `saeforge.eval.targets.GroundTruthTarget`.

The implementation lift is small — one file, ~120 lines including
docstrings and the numpy-only AUC. The proposal is to ship it as
the third built-in target, keep the same `(score,
perplexity_analog)` and ctx-key conventions as `KLTarget` /
`CosineTarget`, and **not** wire it into family default dispatch:
GT-alignment is fixture-specific opt-in only, passed via
`ForgePipeline(faithfulness=GroundTruthTarget(labels=...))`.

## What Changes

### Scope

Add `GroundTruthTarget` as a third built-in faithfulness target.
The existing `FaithfulnessTarget` protocol does not change. The
`_default_target_for(family)` family-dispatch table does not
change — GT-alignment is never a family default. The
`evaluate_faithfulness` action does not change — it already
consults `ctx["_faithfulness_target"]` first and falls back to
family dispatch, which is the contract a user-supplied
`GroundTruthTarget` flows through.

### New artifacts

- **`saeforge/eval/targets/gt_alignment.py`** — defines
  `GroundTruthTarget`. `name = "gt_alignment"`,
  `better_when = "higher"`. Constructor signature:

  ```python
  class GroundTruthTarget:
      name = "gt_alignment"
      better_when = "higher"

      def __init__(
          self,
          labels: np.ndarray,                          # (N, M) binary
          *,
          scorer: Literal["auc"] = "auc",
          pool: Literal["mean", "max", "last"] = "mean",
          hidden_extractor: Callable[..., "torch.Tensor"] | None = None,
      ): ...

      def score(self, *, forged, host, ctx) -> tuple[float, float]: ...
  ```

  `labels` is the ground-truth alignment matrix. Row `i` is the
  label vector for eval row `i`; column `j` is one binary label
  category. Shape `(N, M)` where `N == len(eval_input_ids)` and
  `M` is the label-vocabulary size.

  `scorer` is fixed to `"auc"` in v1. The parameter exists so
  future scorers (Pearson, Spearman, monosemanticity) can land
  through the same surface without breaking signatures; they're
  out of scope for this change.

  `pool` selects the across-sequence reduction applied to the
  hidden tensor before AUC. `"mean"` matches the sm-sae default
  and is the recommended choice for residual-stream features.
  `"max"` and `"last"` cover the two next-most-common asks
  (max-activation features, last-token-conditioned features).

  `hidden_extractor` is the host-shape-aware hook for getting
  `(batch, seq, hidden_size)` residual activations out of the
  forged model. When `None`, the default extractor tries
  `forged.torch_module.transformer(input_ids)` then
  `forged.torch_module.model(input_ids)` — that covers the six
  bundled LM-shape families (gpt2 + llama/gemma2/qwen2/qwen3/
  qwen3_moe). If neither attribute exists or both raise,
  `score` raises `RuntimeError` whose message names
  `hidden_extractor=`. The whisper-encoder family is not
  covered by the default and is out of scope for v1
  (Whisper hosts shouldn't be using GT-alignment for the same
  reason cosine is the family default there).

  `score(...)`:
  1. Reads `ctx["_eval_input_ids"]` (same key as `KLTarget` —
     both consume tokenised eval prompts; built-in targets share
     the `_eval_*` namespace). Raises `KeyError` whose message
     names the key when missing, matching `KLTarget`'s pattern.
  2. Validates `labels.shape[0] == input_ids.shape[0]`. Raises
     `ValueError` naming both shapes when they disagree.
  3. Calls the resolved hidden extractor on `input_ids` to get
     `(batch, seq, hidden_size)`. Pools across `seq` per
     `pool`. Result: `(batch, hidden_size)`.
  4. Computes per-feature × per-label AUC via a numpy rank-based
     implementation (no sklearn dependency — see Decision 2).
     Result: `(hidden_size, M)`.
  5. Returns `(mean(max_over_features(auc)), max(0.0, 1.0 -
     score))`. The pair convention follows the protocol's
     `better_when="higher"` rule, mirroring `CosineTarget`'s
     `(cosine, max(0, 1 - cosine))`.
  6. Does NOT consult `host`. The `host` kwarg is accepted for
     protocol conformance (and `isinstance` checks against the
     `@runtime_checkable` protocol) but never touched. This is
     the use case the existing protocol docstring's "host MAY
     be ignored" carve-out was written for; the protocol's
     deferred `requires_host` opt-out (already filed as a
     follow-up in the protocol's source comment) would let
     sae-forge skip the host forward pass on the FSM path.

### Modified artifacts

- **`saeforge/eval/targets/__init__.py`** —
  - Add `GroundTruthTarget` to `__all__` and re-export from
    `saeforge.eval.targets.gt_alignment`.
  - Mention the new built-in in the module docstring's "two
    built-in implementations" → "three built-in implementations"
    line. `_default_target_for` is unchanged.
- **`saeforge/eval/__init__.py`** —
  - Re-export `GroundTruthTarget` so `from saeforge.eval import
    GroundTruthTarget` works alongside the existing
    `KLTarget` / `CosineTarget` re-exports.

### New tests

- **`tests/eval/test_gt_alignment_target.py`** —
  - `isinstance(GroundTruthTarget(labels=...), FaithfulnessTarget)`
    is `True` (the protocol is `@runtime_checkable`).
  - `GroundTruthTarget.name == "gt_alignment"` and
    `better_when == "higher"`.
  - Missing `ctx["_eval_input_ids"]` raises `KeyError` whose
    message names the key. No host or forged forward pass runs
    (patch `forged.forward` to raise; the test passes iff it is
    never called).
  - Shape-mismatch (labels.shape[0] != input_ids.shape[0])
    raises `ValueError` naming both shapes.
  - Identity-extractor sanity test: build a synthetic forged
    stub whose `hidden_extractor(input_ids)` returns a tensor
    whose pooled rows equal the label matrix exactly. Assert
    `score(...) > 0.95` and the `perplexity_analog` equals
    `1.0 - score` (clamped at 0).
  - Pool variants: with `pool="last"` and a hidden tensor whose
    *last* position is the labels, score is still `> 0.95`.
    With `pool="max"` and a hidden tensor with a single
    activated position per row, score is `> 0.95`.
  - Default extractor: with a fake forged module exposing
    `.transformer(input_ids) -> tensor`, the default extractor
    picks it up. With a fake exposing `.model(...)`, ditto.
    With neither, `RuntimeError` naming `hidden_extractor=`.
  - AUC implementation parity: on a 32×4 hidden / 32×3 label
    fixture, the numpy AUC matches `sklearn.metrics
    .roc_auc_score` (if sklearn is importable in the test env;
    skip otherwise) within `atol=1e-9`.
- **`tests/forge/test_pipeline_with_gt_alignment.py`** —
  - End-to-end test on a synthetic 2D mixture-of-gaussians
    fixture (3 clusters, ~100 samples) with known cluster IDs
    one-hot-encoded as the label matrix. Build a tiny synthetic
    forged model whose residual-stream output is a noisy
    transform of the cluster IDs. Run
    `ForgePipeline(faithfulness=GroundTruthTarget(labels=L),
    orchestrator="fsm", n_tasks=2, min_faithfulness=0.7)` over
    the fixture.
  - Assert `result.faithfulness_target_name == "gt_alignment"`.
  - Assert `result.faithfulness > 0.7` (the FSM gate is met).
  - Assert `_kl_from_input_ids` is never called (patch it to
    raise; the test passes iff it is never consulted on the
    GT-alignment path).

### New example

- **`examples/forge_with_gt_alignment.py`** — end-to-end demo:
  - Build a 2D mixture-of-gaussians fixture (3 clusters, 1000
    samples) with known cluster labels.
  - Construct a synthetic SAE basis (one feature per cluster
    centroid plus deliberate noise).
  - Build a `GroundTruthTarget(labels=L)`.
  - Run a tiny `ForgePipeline` with `orchestrator="fsm"`,
    `n_tasks=2`, `min_faithfulness=0.8`.
  - Print `result.faithfulness`,
    `result.faithfulness_target_name`, and the transitions-log
    summary.
  - Wall-clock target: under 60s on a CPU laptop. No HF
    download, no GPU, no polygram side-effects.
- Register in the examples-smoke harness
  (`tests/test_examples_smoke.py`, or whatever the equivalent
  is at implementation time) so a CI regression breaks the
  build rather than user trust.

### Docs

- **`docs/finetune-recipe.md`** — under the existing "Swapping
  the faithfulness target" subsection (added in
  pluggable-faithfulness), add `GroundTruthTarget` to the
  built-in target list and a one-screen example showing
  `ForgePipeline(faithfulness=GroundTruthTarget(labels=L,
  pool="mean"), ...)`. Pointer at
  `examples/forge_with_gt_alignment.py`.
- **`docs/advanced-fsm-options.md`** — add `GroundTruthTarget`
  to the list of built-in targets (alongside `KLTarget` /
  `CosineTarget`) under the `faithfulness` knob entry. Note
  that GT-alignment is fixture-specific (not a family default).
- **`CHANGELOG.md`** — under `[Unreleased]`, add a "Ground-truth
  alignment target" entry naming the new class, the supported
  pool strategies, and the example. Note that the
  pluggable-faithfulness protocol is unchanged and KL / cosine
  defaults are byte-identical.

## Capabilities

### Modified Capabilities

- **`faithfulness-target`** — gains a third built-in target.
  The protocol, the family-dispatch table, the
  `evaluate_faithfulness` action, and the
  `ForgePipeline.faithfulness` plumbing are all unchanged. The
  spec delta is exclusively ADDED requirements covering
  `GroundTruthTarget`'s shape and behavioural guarantees.

### Out of Scope (deferred)

- **Additional scorers** (Pearson, Spearman, monosemanticity,
  probe accuracy, feature-coverage). The `scorer` constructor
  argument is a hook for them, but no scorer beyond `"auc"`
  ships in v1. Each lands as a separate built-in or as a
  user-side `FaithfulnessTarget` once a concrete consumer
  materialises.
- **Multi-label-hierarchy comparison** (compare against several
  label matrices simultaneously, e.g. cluster IDs + concept
  bottleneck + probe). Single label matrix only. A wrapping
  target can combine multiple `GroundTruthTarget` instances
  externally.
- **Pooling strategies beyond `mean` / `max` / `last`.**
  Attention-weighted pooling, layer-specific pooling, learned
  pooling — all are extension points but none land here.
- **Whisper-encoder default support.** The default
  `hidden_extractor` covers the six LM-shape families. Whisper
  hosts using GT-alignment must pass an explicit
  `hidden_extractor=` and are not a v1 target.
- **`requires_host=False` protocol opt-out.** Already filed as
  a tracked follow-up in the existing protocol's source comment
  (`saeforge/eval/faithfulness.py:55-60`). `GroundTruthTarget`
  is the second motivating consumer (after `CosineTarget` on a
  labelled fixture); landing the opt-out is a separate change
  that this proposal explicitly does not block on. When it
  lands, `GroundTruthTarget` will gain `requires_host = False`
  and sae-forge can skip the host forward pass on the FSM path.
- **Family default routing.** `_default_target_for(family)`
  stays at KL-for-LM / cosine-for-whisper. GT-alignment is
  opt-in only via the `ForgePipeline(faithfulness=...)` kwarg.
- **CLI surface.** No `--faithfulness-target gt_alignment` flag.
  The label matrix is a numpy array and doesn't have a
  reasonable CLI representation; users go through the Python
  API.

## Impact

- **No breaking changes.** The protocol is unchanged; existing
  targets are unchanged; family default dispatch is unchanged;
  `ForgePipeline.faithfulness` semantics are unchanged.
- **One new file.** `saeforge/eval/targets/gt_alignment.py`
  (~120 lines including the docstring, the constructor, the
  default extractor helper, the numpy AUC, and the `score`
  method).
- **Two re-export lines added** (`targets/__init__.py` and
  `eval/__init__.py`).
- **Two new test files** (~200 lines combined).
- **One new example file** (~60 lines).
- **No dependency changes.** Numpy is already required; the
  numpy rank-based AUC ships in the target module itself. No
  sklearn dependency added.
- **No CLI changes.** No new pipeline knobs. No new metadata
  fields. `ForgeResult.faithfulness_target_name` already
  carries the active target's `name`; for GT-alignment it
  reads `"gt_alignment"`.

## Open Questions

- **Should we ship sklearn as an optional dep for AUC parity
  testing?** Going with no for v1 — the numpy rank-based AUC
  is ~15 lines and validated in the test suite. The
  `roc_auc_score` parity test skips when sklearn isn't
  installed, so contributors with sklearn locally get the
  extra assertion for free.
- **Should the default `hidden_extractor` be a per-family
  table (`gpt2` → `.transformer`, `llama`/`gemma2`/`qwen*` →
  `.model`) or the duck-typed try-`.transformer`-then-`.model`
  fallback proposed here?** Going with the fallback. A
  family-keyed table introduces a second source of truth for
  family dispatch (the first being
  `_default_target_for` / `_build_torch_module`), which we'd
  have to update for every new bundled family. The fallback
  is two `getattr`s and a clear `RuntimeError` if neither
  attribute exists — equally robust for the cases that
  exist today and zero maintenance for new families that
  follow the same naming pattern. Revisit if a future
  bundled family uses a different attribute name.
- **Should `labels` be torch tensors or numpy arrays?** Numpy.
  The label matrix is fixture metadata, not a tensor that
  participates in gradient flow; keeping it numpy avoids
  device-placement questions and matches the
  `_eval_input_ids` precedent (which is a torch tensor only
  because tokenisers return one). The default extractor
  returns a torch tensor; the target detaches and converts
  to numpy before computing AUC.
