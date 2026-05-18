## Context

The pluggable-faithfulness change shipped the `FaithfulnessTarget`
protocol and two built-in scorers in v0.4. It explicitly deferred
a third built-in for label-rich fixtures pending a downstream
consumer. That consumer (`jascal/sm-sae`'s `GroundTruthAlignment`)
is now in production and the same target shape is being
reinvented in adjacent repos.

This design note covers the two non-obvious decisions in the
`GroundTruthTarget` shape: the default hidden-state extractor,
and the AUC implementation choice. Everything else
(`name = "gt_alignment"`, `better_when = "higher"`, ctx key
reuse, return-tuple convention) follows the existing
`KLTarget` / `CosineTarget` patterns directly and isn't a
decision worth re-litigating.

## Goals / Non-Goals

**Goals:**
- Strict generalisation of the sm-sae implementation: pluggable
  pooling and pluggable hidden extractor.
- Minimal new dependency surface. One added runtime dep
  (`scipy>=1.10`) for `scipy.stats.rankdata`; no sklearn.
- Default extractor that works across the six LM-shape families
  (gpt2, llama, gemma2, qwen2, qwen3, qwen3_moe) without a
  per-family table.
- `host`-free scoring path. The protocol already permits it; the
  implementation must actually not touch `host`.

**Non-Goals:**
- Multi-label-hierarchy scoring.
- Scorers beyond AUC.
- Whisper-encoder default coverage.
- Async / distributed scoring.
- A new spec capability — this is a strict extension of the
  existing `faithfulness-target` capability.

## Decisions

### Decision 1 — Default `hidden_extractor` uses duck typing, not a family table

The forged module surface across the six bundled LM-shape
families splits two ways:

- **GPT-2 shape** (`saeforge/adapters/gpt2.py::ForgedGPT2`):
  `self.transformer(input_ids)` returns post-`ln_f` hidden
  states `(batch, seq, hidden_size)`. `self.lm_head` then
  projects to vocab.
- **Llama shape** (`saeforge/adapters/llama.py::ForgedLlama` —
  shared by llama, gemma2, qwen2, qwen3, qwen3_moe):
  `self.model(input_ids)` returns post-norm hidden states with
  the same shape.

Two options for the default extractor:

**Option A** — family-keyed table:

```python
_EXTRACTOR_BY_FAMILY = {
    "gpt2": lambda f, ids: f.torch_module.transformer(ids),
    "llama": lambda f, ids: f.torch_module.model(ids),
    "gemma2": lambda f, ids: f.torch_module.model(ids),
    "qwen2": lambda f, ids: f.torch_module.model(ids),
    "qwen3": lambda f, ids: f.torch_module.model(ids),
    "qwen3_moe": lambda f, ids: f.torch_module.model(ids),
}
```

**Option B** — duck-typed fallback:

```python
def _default_hidden_extractor(forged, input_ids):
    module = forged.torch_module
    if hasattr(module, "transformer"):
        return module.transformer(input_ids)
    if hasattr(module, "model"):
        return module.model(input_ids)
    raise RuntimeError(
        "GroundTruthTarget could not locate a residual-stream attribute "
        f"on forged.torch_module (type={type(module).__name__}). "
        "Tried `.transformer` (GPT-2 shape) and `.model` (Llama shape). "
        "Pass hidden_extractor=... explicitly."
    )
```

**Going with Option B.** Reasons:

- A family-keyed table is a second source of truth for family
  dispatch (the first lives in `saeforge/model.py::_build_torch_module`
  and `saeforge/eval/targets/__init__.py::_default_target_for`).
  Every new bundled family would have to update three tables
  instead of one entry plus a `register_adapter` call. The
  `world-model-protocol` change in flight (`openspec/changes/
  world-model-protocol/`) is explicitly trying to reduce that
  multi-source-of-truth surface, not grow it.
- Two `getattr` checks cover every existing case and every
  plausible future bundled family that follows the same naming
  pattern (HF's `*ForCausalLM` wrappers do — `transformer` for
  GPT-2 lineage, `model` for the post-Llama transformers
  ecosystem).
- When the heuristic fails, the error message names
  `hidden_extractor=` explicitly. That's the right escape hatch:
  exotic hosts (Whisper, future SSM adapters, third-party
  forge modules) supply their own extractor and don't depend on
  the default.

Risks accepted: a future bundled family that uses neither
attribute name would need the heuristic widened (one new
`hasattr` branch) or the user would have to pass an explicit
extractor. Both are acceptable; the cost of pre-emptive
generality (Option A) outweighs the cost of one future
heuristic widening.

### Decision 2 — AUC via `scipy.stats.rankdata`, no sklearn

The rank-based AUC formula `auc = (sum_of_positive_ranks -
n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)` is well-known and
gives bit-identical results to `sklearn.metrics.roc_auc_score`
*provided the ranks use average-rank tie handling* (the
convention sklearn uses internally). A pure-numpy implementation
via `np.argsort(np.argsort(...))` returns ordinal ranks, which
drift from sklearn whenever scores tie across classes.

For continuous residual activations from a well-trained host,
ties are rare. But the first motivating consumer
(`jascal/sm-sae`'s GT-alignment on the SM physics fixture) runs
on discrete-cluster labels and may hit ties during early training
where many features sit near saturation. We don't want sm-sae to
adopt the upstream target only to discover a silent drift from
its prior local-numpy result on day one.

Two options for fixing the ties:

**Option A** — ship pure-numpy with ordinal ranks, document the
caveat, swap to `rankdata` reactively when a consumer reports
drift.

**Option B** — add `scipy>=1.10` as a runtime dep and call
`scipy.stats.rankdata(..., method="average")` in the helper.

**Going with Option B.** Reasons:

- The first consumer is exactly the case that hits the bug. The
  "wait for a real report" lever has nothing left to learn — the
  report is queued.
- scipy is widely available and is a transitive dep of nearly
  every scientific-Python project a sae-forge user is likely to
  already have installed (it ships in the dev venv via
  scikit-learn, and is on the default conda / standard
  data-science install footprints).
- The helper shrinks: a 6-line `_pairwise_auc` using `rankdata`
  replaces the ~15-line pure-numpy version, and the AUC parity
  test against sklearn moves from "skip if missing" reassurance
  to a meaningful equivalence claim (still gated by
  `importorskip` since sklearn itself isn't a dep).
- `scipy>=1.10` is a soft floor; `rankdata` has been stable
  since at least scipy 1.0. Anything plausibly installed today
  satisfies it.

Risks accepted: one new ~80MB runtime dep for a 6-line helper.
The alternative — re-implementing average-rank ties in pure
numpy — is doable in ~10 lines but reinvents what scipy ships,
and any subtle disagreement with scipy's convention shows up as
a sklearn-parity-test failure that's harder to debug than the
dep cost is to absorb.

The test suite includes a `roc_auc_score` parity check guarded
by `pytest.importorskip("sklearn")` so contributors with sklearn
locally get the extra assertion at no cost; CI doesn't require
it.

### Decision 3 — `host` is ignored, not validated

The protocol docstring already says `host` MAY be ignored
(`saeforge/eval/faithfulness.py:55-60`). `GroundTruthTarget`
exercises that carve-out. The signature accepts `host` for
protocol conformance (and for `isinstance` checks against the
`@runtime_checkable` protocol), but `score` never reads it.

We do NOT raise when `host` is `None`. We do NOT log a warning.
The protocol's `requires_host=False` opt-out (filed as a
follow-up on the protocol itself) is what eventually lets
sae-forge skip the host *forward pass* upstream of this target;
until that lands, the wasted host forward is an upstream
inefficiency, not something this target should be papering over.

### Decision 4 — Labels are numpy, not torch

The label matrix is fixture metadata. It doesn't participate in
gradient flow, doesn't need device placement, and conceptually
lives next to the eval-data CSV more than next to the model
weights. Coercing it to numpy at construction time avoids the
torch import on the constructor path (so building the target
doesn't pull in torch when the user only wanted to register it
on a pipeline).

The hidden-state side IS a torch tensor (since the forged
forward returns one). The target detaches, moves to CPU, and
converts to numpy before the AUC. That conversion is once per
`score` call, on a `(N, hidden_size)` tensor, so the cost is
trivial.

## Risks / Trade-offs

- **Heuristic extractor masks a different bug.** If a future
  `ForgedX` exposes `.transformer` but it returns logits (not
  residual stream), the heuristic silently scores against the
  wrong tensor. Mitigation: the integration test
  (3.2 in `tasks.md`) asserts a numerical floor on a known
  fixture; a wrong-tensor regression breaks that test.
- **AUC ties.** Resolved up-front via Decision 2 —
  `scipy.stats.rankdata(method="average")` matches sklearn's
  default convention. Risk now is the dep cost (one new runtime
  dep) rather than silent drift on tie-heavy fixtures.
- **Pooling defaults to mean.** Mean-pooling discards
  sequence-position structure; for some fixtures `last` or
  `max` would score higher. Mitigation: the constructor exposes
  `pool=` and the docs / example call out the choice
  explicitly.
- **Labels and eval-set ordering.** The `(N, M)` label matrix
  must be row-aligned with the eval set. The target validates
  shape but cannot validate *identity*. Mitigation: the
  docstring spells out the ordering contract; the example
  builds both in lockstep.

## Migration Plan

This change introduces a new built-in but doesn't change the
default behaviour of any existing pipeline. Migration is
purely opt-in:

- No-op for callers who don't set `faithfulness=` — KL stays
  the LM default, cosine stays the Whisper default.
- New callers wire `faithfulness=GroundTruthTarget(labels=L)`
  on `ForgePipeline` construction and gate the FSM loop on
  AUC instead of KL.
- Downstream consumers currently re-implementing the target
  (sm-sae, etc.) replace their local copy with the upstream
  import. The behavioural drift between the sm-sae version's
  defaults and `GroundTruthTarget`'s defaults is:
  pooling becomes pluggable (was hardcoded mean — same default),
  hidden extractor becomes pluggable (was hardcoded
  `forged.torch_module.transformer` — now duck-typed, broadens
  family coverage). Both are strictly broader behaviour; no
  downstream caller has to change anything to get
  sm-sae-equivalent results.

## Open Questions

- **Should the example use sm-sae's SM physics fixture or stay
  on the synthetic mixture-of-gaussians?** Going with the
  synthetic fixture in the example. The SM physics fixture
  depends on sm-sae's data generation pipeline; the synthetic
  fixture is self-contained, fast, and doesn't pull a foreign
  repo into sae-forge's example surface. sm-sae can keep using
  the upstream target against its native fixture in its own
  examples.
- **Should `GroundTruthTarget` cache the hidden extraction
  across loop iterations?** No. The FSM's outer-refine loop
  re-runs `evaluate_faithfulness` against a re-forged model
  each iteration, so the cache would invalidate on every call.
  Caching would make sense for a target that re-uses the same
  forged-model state across multiple scoring calls, but that's
  not how the loop is shaped today.
- **Does `pool="last"` need a "pad-aware" variant?** Probably,
  eventually. For now the default extractor doesn't propagate
  attention masks and the eval set is short enough that
  ignoring padding is fine. A future `pool="last_nonpad"` is
  out of scope; the workaround is to truncate `input_ids` to
  the actual sequence length before passing through.
