## 1. `GroundTruthTarget` implementation

- [ ] 1.1 Create `saeforge/eval/targets/gt_alignment.py`. Module
  docstring mirrors `kl.py` / `cosine.py` style: protocol pointer,
  `better_when="higher"` convention reminder, ctx keys consumed
  (`_eval_input_ids`, `device`), the `host`-is-ignored carve-out
  pointer to `saeforge/eval/faithfulness.py:55-60`.
- [ ] 1.2 Define `GroundTruthTarget` with class attributes
  `name = "gt_alignment"` and `better_when = "higher"`.
  `__init__(self, labels, *, scorer="auc", pool="mean",
  hidden_extractor=None)`. Validate at construction time:
  `labels.ndim == 2`; `labels.shape[0] >= 1`;
  `labels.shape[1] >= 1`; `scorer == "auc"` (v1 only — other
  values raise `ValueError` naming the supported set); `pool in
  ("mean", "max", "last")`. Coerce `labels` to a numpy array of
  dtype float (any binary-castable input).
- [ ] 1.3 Implement the default hidden extractor as a private
  helper `_default_hidden_extractor(forged, input_ids)`:
  try `forged.torch_module.transformer(input_ids)`; on
  `AttributeError`, try `forged.torch_module.model(input_ids)`;
  on `AttributeError` raise `RuntimeError` whose message names
  `hidden_extractor=` and the two attributes tried.
  The helper SHALL detach the returned tensor and move it to CPU
  before returning so `score` doesn't accumulate gradient graph
  or stay on a device.
- [ ] 1.4 Implement the rank-based AUC as a private helper
  `_pairwise_auc(scores, labels)`. `scores` is `(N, F)`,
  `labels` is `(N, M)`, both binary-castable. Return `(F, M)`
  AUC matrix. Use `scipy.stats.rankdata(scores, axis=0,
  method="average")` to get average-rank ties handling (matches
  sklearn's `roc_auc_score` convention bit-for-bit). Then
  `auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) /
  (n_pos * n_neg)` vectorised over `(F, M)`. Handle the
  degenerate case where one class is missing for some label
  column by setting that column's AUC to `0.5` (chance) and
  emitting no warning — synthetic fixtures hit this often
  enough that a warning is noise. Import scipy at module top
  (not lazily) — it's a hard runtime dep now and lazy import
  would mask install issues.
- [ ] 1.5 Implement `score(*, forged, host, ctx)`:
  1. Read `ctx["_eval_input_ids"]`. Raise `KeyError` whose
     message names the key when missing or `None`, matching
     `KLTarget.score`'s pattern (see
     `saeforge/eval/targets/kl.py:39-48`).
  2. Validate `self.labels.shape[0] == input_ids.shape[0]`.
     Raise `ValueError` naming both shapes when they disagree.
  3. Call `extractor(forged, input_ids)` (using
     `self.hidden_extractor` if provided else
     `_default_hidden_extractor`). Result tensor is
     `(batch, seq, hidden_size)` or `(batch, hidden_size)` (the
     extractor MAY pre-pool, in which case `seq` is absent).
  4. Pool across `seq` per `self.pool`. `"mean"` →
     `tensor.mean(dim=1)`; `"max"` → `tensor.max(dim=1).values`;
     `"last"` → `tensor[:, -1, :]`. Skip pooling if the tensor
     is already 2D.
  5. Detach, CPU, numpy-cast to `(N, hidden_size)` float array.
  6. Call `_pairwise_auc(scores, self.labels)` →
     `(hidden_size, M)`.
  7. `mean_best_auc = float(auc.max(axis=0).mean())`.
  8. Return `(mean_best_auc, max(0.0, 1.0 - mean_best_auc))`.
- [ ] 1.6 Type annotations: `Callable[..., torch.Tensor] | None`
  for `hidden_extractor` — `Callable[..., ...]` rather than
  `Callable[[Any, Any], ...]` so user-supplied extractors can
  accept additional kwargs (e.g. `layer=`) the target itself
  doesn't pass. Imports are lazy (`from saeforge.utils import
  require_extra` then `torch = require_extra("torch", "torch")`
  inside `score`) so the `eval` package stays importable
  without torch.

## 1a. Dependency bump

- [ ] 1a.1 Add `scipy>=1.10` to `pyproject.toml::[project]
  dependencies` (alongside the existing `numpy>=1.24` and
  `safetensors>=0.4` entries). Floor is conservative —
  `scipy.stats.rankdata(axis=..., method="average")` has been
  stable since well before 1.10; the floor exists so the
  install message is unambiguous on stale environments rather
  than to track a recent feature.
- [ ] 1a.2 Update the `pyproject.toml` no-extras-install
  comment block (currently mentioning "pure-numpy basis loader
  and projector math stay on the no-extras install") to note
  that scipy now ships in the core install as well. Keep the
  intent of the comment intact: torch / transformers are still
  optional via `[torch]` / `[intel]`.

## 2. Re-exports

- [ ] 2.1 Add `GroundTruthTarget` to `saeforge/eval/targets/__init__.py`'s
  `__all__` and import line. Update the module docstring's
  built-in count from "two" to "three".
- [ ] 2.2 Re-export from `saeforge/eval/__init__.py` alongside
  `KLTarget` / `CosineTarget`.
- [ ] 2.3 Do NOT touch `_default_target_for`. GT-alignment is
  opt-in only.
- [ ] 2.4 Do NOT touch `evaluate_faithfulness` in
  `saeforge/actions/__init__.py`. The existing
  `ctx.get("_faithfulness_target")` path already routes a
  user-supplied target end-to-end.

## 3. Tests

- [ ] 3.1 `tests/eval/test_gt_alignment_target.py` —
  - `isinstance(GroundTruthTarget(labels=np.eye(4)),
    FaithfulnessTarget)` is `True`.
  - `GroundTruthTarget.name == "gt_alignment"`;
    `better_when == "higher"`.
  - Constructor validation: `labels.ndim == 1`,
    `pool="invalid"`, `scorer="pearson"` each raise
    `ValueError` whose message names the offending value.
  - `score(...)` raises `KeyError` naming `_eval_input_ids`
    when ctx is `{}`. Patch `forged.forward` to raise; the
    test passes iff the error fires before any forward.
  - Shape-mismatch: `labels.shape[0] != input_ids.shape[0]`
    raises `ValueError` naming both shapes.
  - Identity fixture: `hidden_extractor` returns a tensor
    whose pooled rows equal a known label matrix exactly.
    Assert `score(...) > 0.95` and
    `perplexity_analog == max(0, 1 - score)`.
  - Pool variants — repeat the identity test with
    `pool="last"` (labels live in the final seq position) and
    `pool="max"` (labels live in a single random position per
    row).
  - Default extractor: a fake forged stub with
    `.torch_module.transformer(input_ids)` returning a known
    tensor — default extractor picks it up. A stub with
    `.torch_module.model(...)` instead — default extractor
    picks that up. Neither — `RuntimeError` naming
    `hidden_extractor=`.
  - AUC parity (skip if sklearn unimportable): on a 32×4
    scores / 32×3 labels fixture,
    `_pairwise_auc(scores, labels)` matches
    `sklearn.metrics.roc_auc_score` within `atol=1e-12`. (The
    tighter atol vs the previous design reflects that we now
    use scipy `rankdata(method="average")`, the same convention
    sklearn uses internally — disagreement should be at
    floating-point noise, not algorithmic.)
  - AUC parity on ties (skip if sklearn unimportable): build a
    32×4 scores fixture with deliberate ties (e.g. round to 2
    decimals so multiple rows share scores), 32×3 labels.
    Parity within `atol=1e-12`. This is the regression test
    for Decision 2 — fails loudly if anyone reverts to
    ordinal ranks.
  - Degenerate AUC: a label column with all-zero entries gets
    AUC `0.5` (no warning).
- [ ] 3.2 `tests/forge/test_pipeline_with_gt_alignment.py` —
  - 2D mixture-of-gaussians fixture (3 clusters, ~100
    samples). Cluster IDs one-hot-encoded to a `(100, 3)`
    label matrix.
  - Build a tiny synthetic forged model whose residual after
    pooling is a clean (low-noise) projection of the cluster
    IDs through an orthogonal hidden_size×3 mixing matrix.
  - Run `ForgePipeline(faithfulness=GroundTruthTarget(
    labels=L), orchestrator="fsm", n_tasks=2,
    min_faithfulness=0.7, ...)`.
  - Assert `result.faithfulness_target_name == "gt_alignment"`.
  - Assert `result.faithfulness >= 0.7` (the
    `min_faithfulness` gate is met).
  - Patch `saeforge.forge._kl_from_input_ids` to raise. Test
    passes iff KL is never consulted on the GT-alignment path.
  - Wall-clock target: under 5s.
- [ ] 3.3 Existing tests stay green. Specifically:
  - `tests/eval/test_faithfulness_target_protocol.py` (the
    protocol smoke from pluggable-faithfulness).
  - `tests/forge/test_pipeline_byte_identity.py` (defaults
    are byte-identical to pre-change because
    `_default_target_for` is unchanged).
  - The full `pytest` suite. Zero regressions.

## 4. Example

- [ ] 4.1 Write `examples/forge_with_gt_alignment.py`:
  - Build a 2D mixture-of-gaussians fixture (3 clusters, 1000
    samples) with known cluster labels.
  - Construct a synthetic SAE basis from cluster centroids
    (one feature per cluster, plus deliberate Gaussian
    noise).
  - Build `target = GroundTruthTarget(labels=L)` (default
    `pool="mean"`).
  - Run a tiny `ForgePipeline` over the synthetic basis with
    `orchestrator="fsm"`, `n_tasks=2`, `min_faithfulness=0.8`,
    `synthetic=True` (no HF download).
  - Print `result.faithfulness`,
    `result.faithfulness_target_name`, and the
    transitions-log summary.
  - Wall-clock target: under 60s on a CPU laptop. Sanity-test
    on the 16GB Intel Mac (the cross-arch defaults-validation
    surface — see user-memory note).
- [ ] 4.2 Register `examples/forge_with_gt_alignment.py` in
  the existing examples-smoke harness (look for
  `tests/test_examples_smoke.py` or grep for
  `forge_with_*.py` registrations at implementation time) so
  a CI break catches regressions instead of users.

## 5. Docs

- [ ] 5.1 `docs/finetune-recipe.md`: under the existing
  "Swapping the faithfulness target" subsection, expand the
  built-in list from `KLTarget` / `CosineTarget` to include
  `GroundTruthTarget`. Add a one-screen example showing
  `ForgePipeline(faithfulness=GroundTruthTarget(labels=L,
  pool="mean"), ...)`. Pointer at
  `examples/forge_with_gt_alignment.py`.
- [ ] 5.2 `docs/advanced-fsm-options.md`: add
  `GroundTruthTarget` to the list of built-in targets under
  the `faithfulness` knob entry. Note that GT-alignment is
  fixture-specific (not a family default) and that
  `better_when="higher"` flips the `min_faithfulness`
  predicate to `score >= min_faithfulness` (already covered
  by the pluggable-faithfulness predicate helper).
- [ ] 5.3 `CHANGELOG.md`: under `[Unreleased]`, add a
  "Ground-truth alignment target" entry. Name the new class,
  the supported pool strategies (`mean` / `max` / `last`),
  the example, and an explicit "the pluggable-faithfulness
  protocol is unchanged; KL / cosine defaults are
  byte-identical" line so anyone scanning the changelog for
  breaking changes finds the reassurance inline.

## 6. Validation

- [ ] 6.1 `openspec validate add-gt-alignment-target --strict`.
- [ ] 6.2 `ruff check` clean on every modified file.
- [ ] 6.3 Full `pytest` suite green.
- [ ] 6.4 `python examples/forge_with_gt_alignment.py`
  completes in under 60s on the 16GB Intel Mac.
- [ ] 6.5 Fresh-install smoke: `pip install -e .` in a clean
  venv pulls scipy; `python -c "from saeforge.eval import
  GroundTruthTarget; import numpy as np;
  GroundTruthTarget(labels=np.eye(4))"` succeeds without any
  ImportError. Verify on both the 16GB Intel Mac (Python 3.11,
  scipy wheel availability) and the M4 box (Apple Silicon
  scipy wheel).

## 7. What this change explicitly defers

- [ ] 7.1 **Additional scorers** (Pearson, Spearman,
  monosemanticity, probe accuracy, feature-coverage). The
  `scorer="auc"` parameter is the hook; concrete
  alternatives land as their own changes when there's a
  consumer.
- [ ] 7.2 **Multi-label-hierarchy comparison.** Single label
  matrix only.
- [ ] 7.3 **Pooling strategies beyond `mean` / `max` /
  `last`.** Attention-weighted, layer-specific, and learned
  pooling are all extension points but none land here.
- [ ] 7.4 **Whisper-encoder default support.** Default
  extractor covers LM-shape families only. Whisper users pass
  an explicit `hidden_extractor=`.
- [ ] 7.5 **`requires_host=False` protocol opt-out.** Tracked
  follow-up in `saeforge/eval/faithfulness.py:55-60`.
  `GroundTruthTarget` is a motivating second consumer; the
  opt-out is a separate change.
- [ ] 7.6 **Family default routing.**
  `_default_target_for(family)` is unchanged.
  GT-alignment is opt-in via the `ForgePipeline(faithfulness=
  ...)` kwarg.
- [ ] 7.7 **CLI surface.** No
  `--faithfulness-target gt_alignment` flag. Label matrices
  don't have a reasonable CLI representation.
