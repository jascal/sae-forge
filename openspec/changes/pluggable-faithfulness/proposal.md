## Why

sae-forge's faithfulness signal is currently hard-wired to KL divergence
between the host's and the forged model's next-token distributions. The
metric is `faithfulness_kl`. It is read in two places:

1. The single-shard `ForgePipeline._run_real_imperative` reports it as
   the headline number on `ForgeResult.faithfulness_kl`. The FSM path
   produces the same number via the `evaluate_faithfulness` action and
   stores it in `ctx["faithfulness"]`.
2. The continual-learning / inner-refine loop uses it as the
   loop-progression gate. `_evaluate_lm` writes `should_continue` from
   `min_faithfulness` and a `perplexity < best_perplexity` progress
   check; the `RefineMachine.evaluating` transitions consume that flag
   to decide whether to re-enter `refining` (which re-invokes
   `BasisMachine` for another compress↔regrow cycle) or exit.

KL is the right default for LM hosts. It is the wrong signal — and
sometimes meaningless — for several real use cases callers have already
run into:

- **Encoder-only / non-LM forges.** `whisper-encoder-eval` already
  needed a parallel scorer (`cosine_faithfulness`) because per-frame
  hidden states aren't probability distributions. The dispatch lives
  inside the action as a family check; adding a third scorer means
  another `if family == ...` arm.
- **Synthetic / ground-truth-labelled SAEs.** When the underlying
  features are known (toy mixture-of-gaussians fixtures, the
  `polygram` synthetic-basis test bench), KL is a proxy for the actual
  thing we care about — recovery of the GT feature assignment.
  Optimising KL on the loop while the GT-alignment metric stagnates is
  a documented pattern in our continual-learning runs.
- **Domain-specific scorers.** Probing accuracy, monosemanticity,
  Pearson correlation on a labelled probe — every one of these has
  been requested as a "can we also gate the loop on …" question and
  every one of them is a five-line scorer plus a forty-line dispatch
  patch today.

The lift is small. `_kl_from_input_ids` and `cosine_faithfulness` are
already pure functions of `(forged, host, eval_inputs) -> float`. The
load-bearing change is lifting the dispatch out of the action and
behind a `FaithfulnessTarget` protocol that callers can register or
pass directly. Existing behaviour stays byte-identical: KL is the
default target for LM families, cosine for whisper_encoder.

## What Changes

### Scope

Introduce a `FaithfulnessTarget` protocol that returns
`(score: float, perplexity_analog: float)` from a `(forged, host, ctx)`
triple. `ForgePipeline` accepts an optional
`faithfulness: FaithfulnessTarget | None = None`. When `None` (the
default), behaviour is byte-identical to v0.4 — the action's existing
family dispatch picks `KLTarget` (LM families) or `CosineTarget`
(whisper_encoder). When set, the action invokes the user-provided
target and skips family dispatch entirely.

`ForgeResult` gains a generic `faithfulness: float | None` field keyed
by the target's `name`. `ForgeResult.faithfulness_kl` becomes a
deprecated alias: reads still return the same value when the active
target is `"kl"`, and emit `DeprecationWarning` pointing at
`.faithfulness`. Removed in the version after next.

### New artifacts

- **`saeforge/eval/faithfulness.py`** (existing module — extended) —
  defines the `FaithfulnessTarget` protocol alongside the existing
  `faithfulness_kl` function. Co-locating the protocol with the
  KL implementation it generalises keeps a single import surface
  (`from saeforge.eval.faithfulness import FaithfulnessTarget,
  faithfulness_kl`) and avoids the import-clutter of a sibling
  `target.py`:

  ```python
  class FaithfulnessTarget(Protocol):
      name: str               # "kl" / "cosine" / "gt_alignment" / ...
      better_when: Literal["higher", "lower"]
      def score(
          self,
          *,
          forged,
          host,
          ctx: Mapping[str, Any],
      ) -> tuple[float, float]: ...
  ```

  The second tuple element is the "perplexity analog" the FSM's
  progress-check (`perplexity < best_perplexity`) already consumes —
  for KL it's `exp(kl)`, for cosine it's `1 - cosine`, for a
  GT-alignment scorer it's `1 - alignment`. Targets pick the convention
  that matches their `better_when` so the LM-shaped progress check
  keeps pointing the right way.

  Minimal custom-target sketch (full version in
  `examples/forge_with_gt_alignment.py`):

  ```python
  from saeforge.eval.faithfulness import FaithfulnessTarget

  class GTAlignmentTarget:
      name = "gt_alignment"
      better_when = "higher"

      def __init__(self, labels):
          self._labels = labels  # ground-truth cluster assignment

      def score(self, *, forged, host, ctx):
          # `host` is ignored — GT alignment doesn't need a teacher.
          features = forged.encode(ctx["_gt_alignment_inputs"])
          alignment = _cluster_alignment(features, self._labels)
          return float(alignment), float(1.0 - alignment)

  # Plug into the pipeline:
  pipeline = ForgePipeline(
      basis=basis,
      projector=projector,
      host_model_id="gpt2",
      faithfulness=GTAlignmentTarget(labels=cluster_ids),
      # …
  )
  ```

  `_gt_alignment_inputs` is the third-party ctx key the target reads —
  the protocol docstring recommends namespacing such keys (e.g.
  `_myorg_input_ids`) to avoid collisions with built-ins.

- **`saeforge/eval/targets/kl.py`** — `KLTarget(FaithfulnessTarget)`,
  `name="kl"`, `better_when="lower"`. Delegates to
  `_kl_from_input_ids` and exponentiates. Reads
  `ctx["_eval_input_ids"]`.
- **`saeforge/eval/targets/cosine.py`** — `CosineTarget`,
  `name="cosine"`, `better_when="higher"`. Delegates to
  `cosine_faithfulness`. Reads `ctx["_eval_audio_features"]` and
  `ctx["_eval_encoder_states"]`.
- **`saeforge/eval/targets/__init__.py`** — exports the two built-in
  targets and `_default_target_for(family)`. The protocol itself is
  imported from `saeforge.eval.faithfulness`.

### Modified artifacts

- **`saeforge/actions/__init__.py::evaluate_faithfulness`** — replaces
  the inline family `if/else` with a single dispatch:
  1. If `ctx.get("_faithfulness_target") is not None`, use it.
  2. Else select the built-in target by `forged.config.family`:
     LM families → `KLTarget`, `whisper_encoder` → `CosineTarget`,
     unknown family → `ValueError` (existing behaviour preserved).
  The `should_continue` predicate consults `target.better_when` so
  `min_faithfulness` semantics generalise: `lower` keeps the existing
  KL-negation convention; `higher` uses the cosine-style
  `score >= min_faithfulness` convention. Both are documented on the
  target docstring rather than buried in the action body.
- **`saeforge/forge.py::ForgePipeline`** — new field
  `faithfulness: FaithfulnessTarget | None = None`. Threaded into the
  FSM ctx as `_faithfulness_target` from `_build_fsm_ctx`. The
  imperative path calls `target.score(forged=..., host=..., ctx=...)`
  directly when set, falling back to the existing
  `faithfulness_kl(...)` call when `None` (byte-identity preserved).
- **`saeforge/forge.py::ForgeResult`** — adds `faithfulness: float |
  None` and `faithfulness_target_name: str | None`. Keeps
  `faithfulness_kl` as a property that returns `faithfulness` when
  `faithfulness_target_name == "kl"`, otherwise returns `None`, in
  both cases emitting `DeprecationWarning` on access. The constructor
  still accepts `faithfulness_kl=` as a kwarg (forwarded to
  `faithfulness`) so existing callers keep working for one minor
  version.
- **`saeforge/machines/refine.orca.md`** — docstring-only update:
  notes that `should_continue` is now target-aware via
  `better_when`. No FSM topology change.

### New tests

- `tests/eval/test_faithfulness_target_protocol.py` — protocol shape,
  `KLTarget` and `CosineTarget` round-trip the existing helpers
  byte-identically.
- `tests/forge/test_pipeline_byte_identity.py` —
  `ForgePipeline(faithfulness=None, ...)` produces a `ForgeResult`
  whose `faithfulness` and `n_params` hash bit-equal to the same
  pipeline pre-change on the toy GPT-2 fixture. Asserted via
  `hashlib.sha256` over the relevant fields, not float equality.
- `tests/forge/test_pipeline_with_custom_target.py` — builds a
  `GTAlignmentTarget` against a 2D mixture-of-gaussians fixture with
  known cluster labels, runs one basis-loop iteration with a
  deliberately under-compressed input, asserts the loop continues
  (re-enters `refining`) based on the GT-alignment score and that the
  KL value is not consulted (verified by patching `_kl_from_input_ids`
  to raise).
- `tests/forge/test_forge_result_deprecation.py` —
  `ForgeResult.faithfulness_kl` emits `DeprecationWarning` once per
  access; returns `None` when the active target is not `"kl"`.

### New examples

- **`examples/forge_with_gt_alignment.py`** — end-to-end demo on a
  synthetic 2D mixture-of-gaussians SAE with known cluster labels. Runs
  in under a minute on CPU; no HF download, no GPU. Demonstrates
  passing `faithfulness=GTAlignmentTarget(labels=...)` and reading the
  resulting `ForgeResult.faithfulness` (and `faithfulness_target_name`)
  back.

### Docs

- `docs/finetune-recipe.md` — new "Swapping the faithfulness target"
  subsection after the existing "Host distillation" block, with the
  protocol signature, the two built-in targets, and a pointer at
  `examples/forge_with_gt_alignment.py`.
- `docs/advanced-fsm-options.md` — adds a row to the "Basis loop
  (inner)" knobs table documenting `faithfulness` and noting the
  target's `better_when` is what makes `min_faithfulness` semantics
  per-target.
- `CHANGELOG.md` — under the next minor (v0.5 or v0.6, depending on
  what lands first): the deprecation of `ForgeResult.faithfulness_kl`
  in favour of the generic `faithfulness` / `faithfulness_target_name`
  pair, and the addition of the `FaithfulnessTarget` protocol with two
  built-in targets.

## Capabilities

### New Capabilities

- **`faithfulness-target`** — defines the `FaithfulnessTarget` protocol,
  the dispatch precedence (`ctx["_faithfulness_target"]` overrides
  family dispatch), the `(score, perplexity_analog)` return contract,
  and the `better_when` → `min_faithfulness` semantics. Built-in
  `KLTarget` and `CosineTarget` are conformance-tested implementations
  of the protocol.

### Out of Scope (deferred)

- **Non-transformer host support.** `NativeModel` and
  `SubspaceProjector` still walk attention/MLP weight matrices to
  build the forged model. A `WorldModel`-protocol generalisation
  (which would let RNNs / SSMs / diffusion U-Nets plug in directly) is
  a separate, larger change. Filed as follow-up `world-model-protocol`.
- **Additional built-in scorers** (Pearson, Spearman,
  monosemanticity, probe accuracy, feature-coverage). The point of
  this change is the protocol; once it lands, additional scorers are
  trivial follow-ups that don't need their own openspec changes.
- **Distributed faithfulness scoring** across multiple GPUs. Current
  behaviour is single-host; preserve that. A `score` implementation
  that internally shards is allowed by the protocol but not provided.
- **Polygram-side changes.** This change is sae-forge-internal. The
  GT-alignment scorer in tests and `examples/` does not require
  polygram modifications — the polygram-side `LearnedKnobAssignment`
  contract is untouched.

## Impact

- **No breaking changes at the default surface.**
  `ForgePipeline(faithfulness=None, ...)` is byte-identical to v0.4 on
  the imperative path and the FSM path. The byte-identity test pins
  this.
- **One-version deprecation window** for `ForgeResult.faithfulness_kl`.
  Reads still work, return the same value when the active target is
  `"kl"`, and emit `DeprecationWarning`. Scheduled for removal one
  minor version after the change lands.
- **`evaluate_faithfulness` action body shrinks** — the
  `_evaluate_lm` / `_evaluate_whisper_encoder` helpers move to
  `saeforge/eval/targets/{kl,cosine}.py` as `Target.score`
  implementations. The action becomes a ~15-line dispatcher.
- **New files** in `saeforge/eval/target.py`,
  `saeforge/eval/targets/{__init__,kl,cosine}.py`,
  `tests/eval/test_faithfulness_target_protocol.py`,
  `tests/forge/test_pipeline_with_custom_target.py`,
  `tests/forge/test_forge_result_deprecation.py`,
  `examples/forge_with_gt_alignment.py`.
- **No CLI changes.** The protocol is a Python-API extension. CLI
  callers continue to get the family-dispatched default. A future
  `--faithfulness-target` flag is possible but explicitly out of
  scope here (would need an entry-point registry; the Python API is
  enough for the documented use cases).

## Open Questions

- **Regrow-trigger framing.** Earlier framing of this change described
  the basis loop's regrow step as KL-gated. In current code
  (`saeforge/machines/basis.orca.md`) `perform_regrowth` is gated on
  `ctx.regrow_count > 0`, not on faithfulness — faithfulness gates
  whether the *outer* refine loop re-enters `BasisMachine` (via
  `should_continue`), which is what determines whether another
  compress↔regrow cycle runs. The pluggable-target test covers the
  outer-loop case (target's score gates `should_continue` and the
  refine loop re-enters `refining`). If we later want target-aware
  inner-loop gating (e.g. "skip regrow when target says we're
  already converged"), that's a separate change to `BasisMachine`.
- **Should `FaithfulnessTarget.score` accept a `device` argument or
  read it from `ctx`?** Built-in targets read `ctx["device"]` today;
  the protocol matches that to avoid duplicating it in every signature.
  If a third-party target wants to override, it can read `ctx` itself.
- **Should the deprecation warning fire on `ForgeResult.faithfulness_kl
  = ...` (write) as well as read?** Probably yes for symmetry, but
  serialisation round-trips through `ForgeResult.from_dict` would then
  warn on every load. Defer: warn on read only, document the symmetric
  case in the docstring.
