## Why

Today a forge run uses a **single** `FeatureBasis` and a single
`SubspaceProjector` (`saeforge/projector.py`). Every host weight in
every layer — token embeddings, every attention block, the lm_head —
is projected through the same `n_features × d_model` decoder. That
single basis was extracted from activations at one capture layer
(typically deep-mid: `transformer.h.8` for GPT-2, layer 12 for
Gemma-2-2B). The basis is therefore optimal for the residual
*distribution at that one layer*, and only that one layer.

Two empirical pain points have accumulated against this assumption:

1. **Initial KL is high and layer-choice-sensitive.** The
   `project_auto_scale_boost` memory and `forge_layer_choice_gemma`
   investigation both surfaced the same shape: picking the basis
   layer is one of the highest-leverage knobs in the pipeline,
   single-basis runs at the "wrong" layer plateau worse, and the
   `scale_boost="auto"` heuristic still leaves a large factor of
   activation-magnitude headroom on the table. A single basis can
   only be optimal for one position in the model.

2. **The hybrid-bridge prototype run (M4-only, May 10–11 session)
   suggested a large faithfulness improvement** from anchoring
   *three* bases at the natural information-flow boundaries —
   embedding output (layer 0), mid-stream (layer N/2), pre-lm_head
   (layer L-1) — and inserting small learnable bridge matrices
   between adjacent bases on the forged module's forward path. The
   prototype was a single ad-hoc Gemma-2-2B M4 run; the numbers
   (KL=11.81 below the `ln(V)=12.45` uniform-noise floor; ~300× wall-
   time speedup vs single-basis to the same plateau) have not been
   reproduced, cross-architecture-verified, or shipped. This change
   moves the mechanism from M4 prototype to a shipped, opt-in,
   cross-architecture-validated forge path with conservative defaults
   chosen against the GPT-2 tier on Intel.

This change does **not** assume the prototype numbers replicate. It
ships the mechanism behind a default-off toggle, pins the algebraic
contract, and adds the validation harness — Intel/GPT-2 on the
cross-architecture defaults surface, with M4/Gemma-2-2B as a follow-
on confirmation. If the mechanism fails to clear single-basis on
GPT-2 with reasonable defaults, the toggle stays off and the design
is the dispositive negative result.

## What Changes

### Scope

Add an optional three-basis hybrid forge path to `ForgePipeline`.
When `hybrid_bridge=True`, the pipeline:

1. Loads up to three `FeatureBasis` instances — `basis_embed`,
   `basis_mid` (the existing `basis`), `basis_lm_head` — each
   trained at a different host capture layer with the same
   `n_features`.
2. Routes each per-layer weight in `SubspaceProjector.project_module`
   through the basis whose anchor layer matches that weight's role
   (embedding rows / pre-block-0 → embed basis; blocks
   `[1 .. L-2]` → mid basis; final block / lm_head → lm-head basis).
3. Inserts two `BridgeModule` parameter blocks (`bridge_emb_mid`,
   `bridge_mid_lm`) into the forged `NativeModel`'s forward path at
   the two basis boundaries. Each bridge is an `n_features ×
   n_features` learnable matrix (default zero-mean orthogonal init)
   wrapped by an optional LayerNorm + non-linearity (default
   `LayerNorm` + identity — linear bridge, see `design.md` on why
   the linear-only variant is the v1 default).
4. Trains the two bridges (and optionally the three projection
   matrices) during the existing fine-tune stage. No new optimizer,
   no new schedule — bridges are added to the param group already
   plumbed by `forge-finetune-recipe`.

When `hybrid_bridge=False` (the default), zero new code paths
execute and the existing single-basis behavior is byte-identical to
pre-change. The byte-equivalence acceptance gate
(`test_imperative_and_fsm_byte_equivalent`) continues to be the
load-bearing check.

### New artifacts

- **`saeforge/bridges.py`** — new module. `BridgeModule(nn.Module)`:
  a tiny torch module with a single `n_features × n_features`
  `Linear` (configurable: pre-LN, post-activation), exposing
  `.from_init(...)` factory methods for orthogonal / identity / zero
  initialization. ~80 lines. Pure-torch — lazy-imports `torch`.
- **`saeforge/hybrid_basis.py`** — new module. `HybridBasisBundle`
  dataclass bundling the three bases + the layer-boundary indices
  (`embed_layer`, `mid_layer`, `lm_head_layer`). Validates that the
  three bases share `d_model` and `n_features`. Provides `.basis_for_layer(idx)`
  returning the basis whose region contains `idx`. ~120 lines.
- **`saeforge/adapters/_hybrid.py`** — adapter helper. Wraps the
  per-architecture adapter (`GPT2Adapter` / `Gemma2Adapter` /
  `LlamaAdapter`) so `walk(host, projector, *, hybrid=HybridBasisBundle)`
  routes each emitted key through the bundle-selected basis. The
  three per-architecture adapters get one new optional kwarg; the
  routing logic is shared.
- **`tests/test_hybrid_bridge.py`** — unit tests for `BridgeModule`,
  `HybridBasisBundle.basis_for_layer` boundaries, and the routing
  invariant (every projected weight came from exactly one basis
  according to its layer-region).
- **`tests/integration/test_hybrid_bridge_gpt2.py`** — end-to-end
  GPT-2 forge: build three small bases (e.g. layers 0 / 6 / 11),
  run a short forge, assert pre-FT KL is finite, post-FT KL is
  finite-and-not-NaN, weights round-trip safetensors. This is the
  cross-architecture defaults-validation surface running on Intel.
- **`scripts/compare_single_vs_hybrid_gpt2.py`** — a one-shot
  comparison harness (not part of pytest). Runs single-basis and
  hybrid forges on `gpt2` with the same `n_features`, eval prompts,
  and seeds; emits a per-step KL table. This is the artifact that
  decides whether to default `hybrid_bridge` on for any subsequent
  release.

### Modified artifacts

- **`saeforge/forge.py`** — `ForgePipeline` gains four optional
  fields (all defaulting to v0 behavior):
  `hybrid_bridge: bool = False`,
  `basis_embed: FeatureBasis | None = None`,
  `basis_lm_head: FeatureBasis | None = None`,
  `bridge_config: BridgeConfig = field(default_factory=BridgeConfig)`.
  `__post_init__` validates that `hybrid_bridge=True` requires both
  extra bases to be present and to share `d_model` / `n_features`
  with `self.basis`. When `hybrid_bridge=False` and the extra bases
  are `None`, no change to existing code paths.
- **`saeforge/projector.py`** — `SubspaceProjector.project_module`
  gains a `hybrid: HybridBasisBundle | None = None` keyword arg.
  When `None`, the existing single-basis dispatch is taken verbatim.
  When provided, dispatches through `saeforge.adapters._hybrid`.
- **`saeforge/model.py`** — `NativeModel.from_projected_weights`
  gains an optional `bridges: dict[str, BridgeModule] | None = None`
  kwarg. When `None`, the forged module's `forward` is unchanged.
  When provided, the two bridge modules are registered as submodules
  and called at the two boundary layer indices on the residual.
- **`saeforge/cli.py`** — new flags on the `forge` subparser:
  `--hybrid-bridge`, `--basis-embed PATH`, `--basis-lm-head PATH`,
  `--bridge-init {orthogonal,identity,zero}` (default `orthogonal`),
  `--bridge-nonlin {none,relu,gelu}` (default `none`).
- **`docs/forge_layer_choice.md`** (existing) — adds a "Hybrid /
  multi-basis" subsection pointing at the comparison harness and the
  Intel/GPT-2 baseline numbers once they exist.
- **`CHANGELOG.md`** — `## [Unreleased]` `### Added` entry.

### Out of scope (deferred)

- **More than 3 bases / arbitrary-layer chains.** The proposal sticks
  to the embed / mid / lm-head triple because those are the host
  model's three structural boundaries (embedding table on one side,
  unembedding on the other, the transformer stack in between).
  Arbitrary chaining (e.g. one basis every 4 layers) is tracked as
  follow-up `multi-anchor-forge`.
- **Non-linear bridges as default.** The linear bridge is v1 default
  because it preserves the proposal's algebraic interpretation as a
  basis-to-basis subspace alignment. A ReLU / GELU bridge is a
  small-MLP that adds non-linearity; supported via the CLI flag but
  not the default. See `design.md`.
- **Sharing bridges across host architectures (transfer).** The M4
  write-up speculated bridges could be reusable across host
  architectures; we make no such claim. Bridges are per-forge-run
  artifacts.
- **Frozen-basis training.** Bridges-only training (freeze the three
  bases, train only the bridges) is an obvious ablation but not the
  default. Tracked as `bridges-only-finetune`.
- **Tied-embedding hosts.** Hosts that tie token embedding to
  lm_head (GPT-2 with `tie_word_embeddings=True`, Llama-family) make
  the embed-basis and lm-head-basis algebraically constrained.
  v1 explicitly *errors out* with a clear message in this case; the
  Gemma-family (untied) is the supported path. Tracked as
  `hybrid-bridge-tied-embeddings`.

## Capabilities

### New Capabilities

- **`hybrid-bridge-forge`** — defines the three-basis routing
  contract (which weight comes from which basis), the bridge module
  contract (shape, init, optional non-linearity), the cross-basis
  shape compatibility requirement (`d_model` and `n_features` match
  across the three bases), the tied-embedding refusal, and the
  byte-equivalence-when-disabled scenario.

### Modified Capabilities

- **`subspace-projector`** — one MODIFIED requirement adds an
  optional `hybrid` dispatch arm to `project_module`. When the arm
  is unused the existing single-basis behavior is preserved
  byte-identically.

## Impact

- **No public API breakage.** Single-basis `ForgePipeline.run()`,
  CLI invocations, and on-disk artifacts are unchanged for the
  default `hybrid_bridge=False` case. New fields and flags are
  additive.
- **No FSM topology change.** The three-machine hierarchy
  (`stream` / `refine` / `basis`) is untouched. Hybrid routing lives
  inside the existing `project_to_subspace` action — same state, same
  guard, same target — by passing the hybrid bundle through ctx.
- **Test surface.** ~25 new tests across `test_hybrid_bridge.py` and
  the integration harness. The existing byte-equivalence gate
  (`test_imperative_and_fsm_byte_equivalent`) continues to pass
  unmodified.
- **Validation surface.** The Intel/GPT-2 comparison harness
  (`scripts/compare_single_vs_hybrid_gpt2.py`) becomes the
  cross-architecture defaults-decision artifact. The M4-only
  Gemma-2-2B reproduction is tracked as a follow-up in
  `tasks.md` §10.

## Sequencing

- **Depends on:** `architecture-adapters` (already on `main`) for
  the per-architecture walking surface that the hybrid routing
  helper wraps. `subspace-projector` (already on `main`) for the
  baseline projection algebra.
- **Independent of:** `adaptive-regrow` (basis-loop concern;
  hybrid-bridge is a projection-time concern). `forge-whisper-encoder`
  (different architecture adapter; orthogonal).
- **Single PR.** The mechanism is small enough (~600 net LOC) that
  staging would be more overhead than the diff itself. The byte-
  equivalence gate plus the Intel/GPT-2 integration test are the
  shipping criteria.
