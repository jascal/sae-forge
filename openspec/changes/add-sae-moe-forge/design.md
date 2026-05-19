# Design — `add-sae-moe-forge`

## Context

Polygram 0.9.0 (PR #87) publishes `cluster_experts` /
`ExpertDictionary` as the polygram-side surface for clustering SAE
features into routable experts. The polygram module docstring
defines a clean split:

> Trained MLP router — belongs in sae-forge (where torch lives).
> Bio-specific scoring — downstream.

This proposal is sae-forge picking up the polygram baton. The
"baton" is a tuple `(Dictionary, decoder_vectors, n_experts,
feature→expert map)`; the sae-forge handover is to produce a torch
`Module` that routes residual activations through the expert
partition and reconstructs.

## Two abstractions, kept separate

sae-forge already has MoE-related code: `saeforge/adapters/qwen3_moe.py`
and `NativeModelConfig.num_experts/num_experts_per_tok/…`. These
forge a *host* MoE transformer (the host has router+experts INSIDE
each transformer block; the forge projects each expert's MLP weights
through a flat feature basis). The new sae-moe-forge capability is
**not the same abstraction** and **does not share machinery** with
qwen3_moe.

- `qwen3_moe` (existing): host-block-level MoE. Routing decision is
  host's own. The forge respects it.
- `sae-moe-forge` (new): SAE-level MoE. Routing decision is *new*
  and made over expert clusters of SAE features. The host knows
  nothing about it.

Keeping them separate is a deliberate design choice — collapsing
them would force every Qwen3-MoE host through the new SAE-side
routing or vice versa. Independent in v1; the composition is a
queued follow-up (`add-moe-as-residual-stream-layer`).

## Why "sub_dictionary" experts first

Three expert implementations were considered (and the prompt that
seeded this proposal asked for all three):

1. **`sub_dictionary`** — each expert is a slice of the existing
   SAE decoder. No new parameters; pure routing of existing
   computation.
2. **`tiny_mlp`** — each expert is a small 1–2-layer MLP distilled
   from the cluster. New parameters; requires a distillation
   training pass.
3. **`residual_block`** — each expert is a (LayerNorm + MLP + skip)
   block, deeper than tiny_mlp. Even more new parameters.

The v1 pick is `sub_dictionary` for three reasons:

- **Zero new parameters.** The forge IS a pure projection of the
  existing SAE into a routed form. No training needed. This is
  the smallest possible MVP — exactly the falsifiable-correctness
  shape the host-wrapped work proved fast to validate.
- **Faithfulness baseline is well-defined.** When `k_experts =
  n_experts`, sub-dictionary MoE collapses to the flat SAE
  byte-identically. That gives a structural correctness check
  that has nothing to do with cluster quality.
- **Sparsity gain is honest.** Compute saving is exactly
  `k_experts / n_experts` (in counted decoder-row touches), not a
  function of distillation quality.

`tiny_mlp` and `residual_block` need their own validation
infrastructure (per-cluster MSE during distillation, parameter
budgets, regularisation). Landing them alongside sub_dictionary
multiplies the v1 surface 3× without proportionally more learning.
They're staged as follow-ups, named in the proposal's "out of
scope" section so the queue is visible.

## Why the heuristic router first

Same reasoning applies. The polygram heuristic
(`ExpertDictionary.route(activations, top_k)` — sum activations per
expert, top-k) is:

- **Zero trainable parameters.** Inherits polygram's deterministic
  routing.
- **Inherits future polygram improvements.** If polygram lands a
  better routing heuristic (e.g., coactivation-method block
  formation), `ForgedMoE` picks it up via the polygram surface
  with no code change here.
- **A sane lower bound for the trained router to beat.** If
  `add-moe-trained-router` (queued) lands, the heuristic is the
  baseline the learned router must outperform on the same
  acceptance gate.

`linear` and `mlp` routers introduce trainable params + a loss
specification + load-balancing concerns + initialisation choices.
Each adds an axis of "did we get this right" — for v1 we want
none. The heuristic is what polygram explicitly intended to ship.

## Encoder is unchanged in v1

The MoE forge applies to the *decoder* side. The encoder
(`W_enc`, `b_enc` from the SAE checkpoint) still runs an N-feature
linear map per token to produce the feature activations the router
consumes. For SAEs where the encoder is the bottleneck (rare —
encoders are typically smaller than decoders for over-complete
SAEs), this gain is partial. Documented; the encoder-side MoE
proposal is `add-moe-encoder-side`, deferred until decoder-side
is validated end-to-end on a real run.

## Cost analysis (v1, sub_dictionary + heuristic)

For an SAE with `N=n_features`, `E=n_experts`, `k=k_experts`,
`d=d_model`, batch token count `M`:

| Stage | Flat SAE | ForgedMoE (k of E) |
|---|---|---|
| Encoder (`act @ W_enc`) | `2*M*d*N` | `2*M*d*N`  *(unchanged)* |
| Decoder (full) | `2*M*N*d` | n/a |
| Routing (`features @ expert_idx_map`) | n/a | `M*N` (sum-by-expert) + sort *(O(M*E*log(E)))* |
| Per-expert decode (`f_e @ W_dec_e`) | n/a | sum over selected experts: `2*M*(k/E)*N*d` |
| **Total decode FLOPs** | `2*M*N*d` | `2*M*N*d * k/E + O(M*N)` |

For `E=16, k=2`: decode FLOPs ≈ 12.5% of flat (modulo the small
routing term). For `E=8, k=2`: 25% of flat.

The router cost (`O(M*N + M*E*log(E))`) is negligible at typical
`E ≤ 64` and `N` in the thousands — it's a sum-by-key plus a
small sort.

## Alternatives considered

### A. Train experts and router jointly from scratch

Treat the cluster assignment as just a hint and learn a full new
router + per-expert decoder from the host's residual stream. This
is what most MoE training pipelines do.

Rejected because:

1. **Loses the polygram alignment.** Polygram's clustering is the
   *interpretability claim* — each expert is a coherent feature
   group. Re-training the router can drift away from that.
2. **Out of scope for v1.** Joint training is a much bigger
   project than the sub_dictionary + heuristic baseline. It's
   `add-moe-trained-router` (queued) at minimum; doing it before
   establishing the heuristic baseline means no comparison surface.

### B. Replace the encoder, not the decoder

Route on tokens *before* the encoder, so each expert has both its
own encoder and decoder. Saves more compute (encoder cost too).

Rejected for v1 because:

1. **Router input is the residual, not the SAE features.** This
   changes the routing-input contract entirely; the polygram
   heuristic operates on feature activations, which require the
   encoder. Pre-encoder routing means a new routing surface that
   doesn't reuse `ExpertDictionary.route`.
2. **Larger correctness surface.** Encoder splitting affects
   feature firing in non-obvious ways. The decoder-side MoE has a
   clean "collapses to flat SAE when k=E" invariant; the encoder-
   side version doesn't.

`add-moe-encoder-side` is queued for after decoder-side validation.

### C. Per-layer expert assignment

In a forged transformer, each layer's residual stream gets its own
MoE forge. Different layers cluster differently.

Out of v1 scope because:

1. v1 ForgedMoE is *standalone* — not yet inserted into a forged
   transformer. The integration is `add-moe-as-residual-stream-layer`.
2. Even then, per-layer assignment is one more axis of design
   choice that we should make based on what the standalone module
   teaches us.

## Open questions deferred to follow-up

- **Whether ForgedMoE should be fine-tunable.** v1 isn't. The
  natural way to make it trainable is to add a learned router on
  top (which IS `add-moe-trained-router`); the experts themselves
  are slices of the SAE decoder and have no separate parameters
  to train.

- **What the right `expert_load()` aggregation is.** v1 returns
  "fraction of token-slots per expert" on the most recent forward.
  Other aggregations (entropy of routing distribution, max load
  ratio across experts) belong in the trained-router proposal
  where load-balancing matters for training stability.

- **Multi-K composition.** A common request: "give me a Matryoshka-
  style nested MoE so I can choose k at inference time." Requires
  trained-router infrastructure; queued as `add-moe-matryoshka`.

- **Insertion as a residual-stream layer.** `forward_mode=
  "moe_routed"` or similar — replace a chosen transformer layer
  with `forged_residual -> MoE_forward -> next_layer_residual`.
  Strong product story but requires the standalone module to be
  validated first. `add-moe-as-residual-stream-layer`.
