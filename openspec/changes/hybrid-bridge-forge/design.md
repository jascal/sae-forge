# Design: hybrid-bridge-forge

## What this is, in one paragraph

A forge run with `hybrid_bridge=True` loads three `FeatureBasis`
instances captured at three different host layers (embed / mid /
lm-head), routes each host weight through the basis whose anchor
layer matches that weight's structural role, and inserts two small
learnable `n_features × n_features` bridge matrices into the forged
module's forward pass at the basis boundaries. Bridges are trained
during the existing fine-tune stage. When the toggle is off, the
forge runs single-basis exactly as before — byte-identical.

## Naming and field semantics

| Field | Type | Default | Semantics |
|---|---|---|---|
| `hybrid_bridge` | bool | `False` | Master toggle. When `False`, no new code runs. |
| `basis` | `FeatureBasis` | required | **Existing.** Acts as the *mid* basis under hybrid. |
| `basis_embed` | `FeatureBasis \| None` | `None` | Basis used for layer-0 host weights. Required when `hybrid_bridge=True`. |
| `basis_lm_head` | `FeatureBasis \| None` | `None` | Basis used for final-block + lm_head host weights. Required when `hybrid_bridge=True`. |
| `bridge_config` | `BridgeConfig` | factory | Dataclass: `init: str = "orthogonal"`, `nonlin: str = "none"`, `pre_layernorm: bool = True`, `train: bool = True`. |

The existing `basis` field is **not renamed** to `basis_mid` — the
single-basis path keeps its current name so every existing config /
test / CLI call continues to work. Under `hybrid_bridge=True`,
`self.basis` plays the role of `basis_mid` by convention.

## Three bases, three regions

The three host layer regions and their basis assignments:

| Region | Host weights | Basis |
|---|---|---|
| **Embed** | Token embedding rows (`wte.weight`), positional embeddings (`wpe.weight`), block 0's ln_1 / attn / mlp / ln_2 | `basis_embed` |
| **Mid** | Blocks `1 .. L-2` (ln_1 / attn / mlp / ln_2 for each) | `basis` (mid) |
| **LM-head** | Block `L-1`'s ln_1 / attn / mlp / ln_2, final `ln_f`, `lm_head.weight` | `basis_lm_head` |

The boundary indices (`embed_layer=0`, `mid_layer=1`,
`lm_head_layer=L-1`) are derived from the host model's `n_layer` —
not user-configurable in v1, to keep the routing contract tight. A
follow-up (`multi-anchor-forge`) opens these up.

For GPT-2 (`n_layer=12`): block 0 → embed, blocks 1–10 → mid,
block 11 + lm_head → lm-head.

For Gemma-2-2B (`n_layer=26`): block 0 → embed, blocks 1–24 → mid,
block 25 + lm_head → lm-head. (The M4 prototype's `embed_basis_layer=0,
basis_layer=12, lm_head_basis_layer=25` triple lines up with this
exactly.)

## Bridges: two matrices, on the residual stream

Two `BridgeModule` instances are inserted into `NativeModel.forward`
at the two region boundaries:

- **`bridge_emb_mid`**: applied to the residual stream *after* the
  embed-region's last block emits its output, *before* the first
  mid-region block reads it. Shape `(n_features, n_features)`.
- **`bridge_mid_lm`**: applied between the last mid-region block and
  the first lm-head-region block. Shape `(n_features, n_features)`.

```python
def forward(self, input_ids):
    x = self.embed_to_residual(input_ids)          # (B, T, n_features)
    x = self.block_0(x)                            # embed-region
    x = self.bridge_emb_mid(x)                     # NEW
    for blk in self.blocks_mid:                    # blocks 1..L-2
        x = blk(x)
    x = self.bridge_mid_lm(x)                      # NEW
    x = self.block_lastminus1(x)                   # lm-head-region
    x = self.ln_f(x)
    return self.lm_head(x)
```

`BridgeModule`'s forward is (in order, all configurable):

```python
def forward(self, x):                              # (..., n_features)
    if self.pre_layernorm:
        x = self.ln(x)
    x = self.linear(x)                             # (n_features, n_features)
    if self.nonlin is not None:
        x = self.nonlin(x)
    return x
```

Default config (`init="orthogonal"`, `nonlin="none"`, `pre_layernorm=True`):
a single LN-normalized `n×n` linear map. With `nonlin="none"` this
is **purely linear**.

## The honest algebraic concern

A linear bridge between two bases is mathematically a
`(n_features × n_features)` linear map composed with two decode/encode
projections; the composition is a single linear `(d_model × d_model)`
transformation of the residual. That single transformation could in
principle be folded into the adjacent block's weights, so a
linear-only bridge **adds no algebraic capacity over a directly-trained
adjacent-block weight perturbation**.

Two reasons to ship the linear default anyway:

1. **Initialization geometry.** The bridge starts from
   *orthogonal-init*, isolated from the adjacent block's learned
   weights. The fine-tune sees a clean, separately-initialized
   parameter that can absorb subspace-alignment error without
   perturbing every other weight in the adjacent block. Empirically
   (M4 prototype) this is where the "filter / cleaner" effect comes
   from — not from the bridge having strictly more capacity, but from
   its initialization and gradient path being isolated. This is the
   testable hypothesis: if isolation matters, a linear bridge with
   orthogonal init outperforms equivalent unfrozen direct fine-tuning
   to the same step budget.

2. **Toggleability.** A linear bridge can be **folded into adjacent
   weights at save time** for a no-op deployment path (zero inference
   overhead, zero extra params on disk). A non-linear bridge cannot.
   The v1 design preserves this option (see "Save-time fold" below).

If the linear-bridge isolation hypothesis fails to clear single-basis
on the Intel/GPT-2 baseline, the design's negative result is
itself the artifact — and the `--bridge-nonlin {relu,gelu}` CLI flag
is the immediate-next experiment without a code change.

## FSM placement: no topology change

Hybrid dispatch happens entirely inside the existing
`project_to_subspace` action (`BasisMachine`'s `projected ←
compressed` transition). The action reads
`ctx.get("hybrid_bridge_bundle")` — when present, calls
`projector.project_module(host, hybrid=bundle)`; when absent, the
single-basis call path is taken verbatim.

The `BridgeModule` instances are constructed in the same action and
written to ctx as `bridges = {"emb_mid": ..., "mid_lm": ...}`. The
subsequent `fine_tune_model` action picks up `ctx["bridges"]` and
adds their parameters to the optimizer's param group.

No new state. No new event. No new guard. No new transition. The
Mermaid diagram is unchanged; the drift CI passes without doc edits.

## Save-time fold (deferred but anticipated)

When `bridge_config.nonlin == "none"` and `bridge_config.pre_layernorm
== False`, each linear bridge `B` can be folded into the adjacent
block by composing it with the next residual-input projection
matrix. v1 does not implement this — bridges are saved as separate
parameters in the forged safetensors — but the spec pins that this
fold is *possible* for the linear default, so the follow-up
(`hybrid-bridge-save-time-fold`) doesn't have to reopen the design.

A non-linear bridge or a pre-LN'd bridge cannot be folded; they ship
as live forward-pass modules in the forged `NativeModel`.

## Tied-embedding refusal

GPT-2 and Llama-family hosts default to `tie_word_embeddings=True`:
the same matrix backs both `wte.weight` and `lm_head.weight`. Under
hybrid forging, those two would have to project through
*different* bases (embed vs lm-head). The most honest thing to do at
v1 is refuse the configuration loudly:

```python
if host_config.tie_word_embeddings and hybrid_bridge:
    raise ValueError(
        "hybrid_bridge=True is incompatible with tied embeddings "
        "(host_config.tie_word_embeddings=True). The embed and "
        "lm_head bases would have to project the same matrix "
        "through different feature spaces. Use a host with "
        "untied embeddings (Gemma-2 family) or disable "
        "hybrid_bridge."
    )
```

Implications for the Intel validation surface: GPT-2 ships with
`tie_word_embeddings=True` by default. To run the Intel/GPT-2
integration test the fixture **must** explicitly construct the host
with `tie_word_embeddings=False` (i.e. an untied GPT-2 — the host
model still works, the model card lies about a constraint that's
purely a save-time tie). This is a known papering-over for the
defaults-validation surface and documented in the integration test's
module docstring.

The principled fix is `hybrid-bridge-tied-embeddings` (follow-up):
either share the embed and lm-head bases (degrades to two-basis
hybrid) or impose an equality constraint on the bridge product.
Both are out of scope here.

## Cross-architecture validation tiering

Defaults for the four configurable knobs (`bridge_init`,
`bridge_nonlin`, `pre_layernorm`, `train`) are picked on a tier
ladder rather than declared up front:

| Tier | Host model | Hardware | Owner | Status |
|---|---|---|---|---|
| **T0** | `tiny_gpt2` (test fixture) | CPU | This change | Pre-merge: must pass |
| **T1** | `gpt2` (untied embeddings) | Intel Mac CPU | This change | Pre-merge: comparison harness runs |
| **T2** | `gpt2-medium`, `gpt2-large` | Intel Mac CPU | This change | Pre-merge if RAM allows; otherwise post-merge |
| **T3** | `gemma-2-2b` | M4 Apple Silicon | Follow-up (user) | Reproduce M4 prototype numbers |
| **T4** | Larger Gemma / Llama untied variants | External CUDA | Follow-up (community) | Validation request to external contributors with NVIDIA/CUDA boxes once T0–T2 ship |

The shipping criterion is T0+T1 green. T2 is best-effort on Intel
(the user's 16GB box may not have headroom for `gpt2-large` + three
bases simultaneously; if so, T2 moves to "post-merge"). T3 and T4
are explicitly delegated — T3 to the M4 box, T4 to a community
validation request once the mechanism is on `main` and the
comparison harness reproduces locally.

The defaults declared in v1 (`init="orthogonal", nonlin="none",
pre_layernorm=True`) are the *best guess from prototype + literature*
and explicitly subject to revision when T1 numbers arrive. If T1
shows `nonlin="none"` underperforms `nonlin="relu"`, the default
flips in a follow-up patch — not via a re-spec.

## Determinism

The bridge modules are torch parameters with seed-controlled
initialization (orthogonal init uses `torch.nn.init.orthogonal_`,
which honors `torch.manual_seed`). Two forge runs with identical
seed + config produce byte-identical bridges *and* byte-identical
forged weights — same guarantee the rest of the forge already gives.

The byte-equivalence test under `hybrid_bridge=False` is preserved
trivially because no hybrid code path is exercised: the new optional
fields are `None`, the new toggle is `False`, every existing call
site sees the same arguments it sees today.

## Risks and rejected alternatives

### Rejected: bridges as a non-linearity-mandatory module

Forcing `nonlin != "none"` would side-step the algebraic-fold
concern but make save-time fold impossible and would mask the
real question — *does isolation alone explain the prototype's
improvement, or is the non-linearity load-bearing?* The linear
default is the cleaner experiment.

### Rejected: arbitrary-layer chaining (k bridges, k+1 bases)

More bases, more bridges, more degrees of freedom — but no clear
structural anchor for *where* to put them past the three
embed/mid/lm-head boundaries. v1 sticks to the three structural
boundaries. `multi-anchor-forge` is the follow-up.

### Rejected: bridge transfer across host architectures

The M4 write-up speculated bridges might transfer across hosts —
i.e. a Gemma-2-2B bridge could initialize a Llama-3-8B forge. This
is a per-shape requirement (`n_features` must match) and a
distributional assumption (mid-residual statistics align across
families) we have no evidence for. v1 makes no transfer claim.

### Risk: orthogonal init is wrong for some host

`torch.nn.init.orthogonal_` requires the parameter to be 2D and at
least as wide as it is tall, which a square `n_features × n_features`
matrix is. Edge case: `n_features=1` (degenerate; the projector
already rejects this). All other shapes are valid. Documented in
the `BridgeModule` docstring.

### Risk: pre-LN with `nonlin="none"` is still a no-op fold

A LN-then-linear bridge cannot be folded into adjacent weights at
save time (LN is not linear). The `pre_layernorm=True` default
*does* eliminate the save-time-fold optimization for the linear
case. v1 keeps `pre_layernorm=True` as the default anyway, because
*training stability* of an `n×n` linear map without normalization
is the more pressing risk than save-time-fold (which is a
nice-to-have follow-up, not a v1 requirement). The CLI flag lets
users override.

## Why land it now

Three queued investigations (`forge_layer_choice_gemma`,
`project_kl_nonmonotonic`, the M4 May 10–11 session) all converge on
the same root cause: a single-layer basis is the wrong abstraction
for a model that does qualitatively different work at its
boundaries. Hybrid bridges are the smallest possible structural
intervention that respects this. Landing the mechanism now gives
the cross-architecture validation surface a stable target; landing
the M4 prototype numbers separately gives the *defaults* surface a
stable target. Conflating the two has slowed iteration.
