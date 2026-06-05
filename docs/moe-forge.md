# sae-moe-forge — routing a compressed SAE into a mixture-of-experts

`forge_to_moe` turns a polygram-compressed SAE basis into a **routed
mixture-of-experts** (`ForgedMoE`): each expert is a coherent cluster of
SAE features, and a per-token router fires only the top-k most-relevant
experts. The per-token decode cost drops to roughly `k_experts /
n_experts` of the flat SAE while the feature-level interpretability of
the underlying dictionary is preserved.

This is the sae-forge half of the polygram `cluster_experts` baton (see
polygram's `experts.py` docstring: *"Trained MLP router — belongs in
sae-forge, where torch lives. Bio-specific scoring — downstream."*).

## When to use it

- **Best on clusterable bases** — coherence-trained SAEs, or
  polygram-compressed SAEs produced with
  `BlockFormation(strategy="cosine", cosine_threshold >= 0.3)`. There,
  routing is ~free (the 2026-05-19 prototype measured routed-vs-flat MSE
  at `0.12x` the flat SAE's own error on a synthetic 4-cluster basis).
- **Still functional on near-isotropic bases**, with an *advisory*
  faithfulness diagnostic rather than a guarantee — every `ForgedMoE`
  exposes a `coherence_diagnostic` so the basis-quality signal is visible
  up front. On the near-isotropic GPT-2 L8 K=211 jbloom basis the
  prototype measured ≈ `4.6x` — usable, with the user trading some
  faithfulness for the sparsity gain.

The split is structural, not a bug: routing faithfulness is bounded by
how cosine-coherent the decoder clusters actually are.

## Quickstart

```python
from saeforge import FeatureBasis, forge_to_moe

basis = FeatureBasis.from_polygram_checkpoint("sae.compressed.safetensors")

# Auto-cluster path: reloads the polygram Dictionary from the basis
# checkpoint and clusters internally.
moe = forge_to_moe(basis, k_experts=2, coherence_threshold=0.3)

residual = ...                      # (batch, seq, d_model) host activations
recon = moe(residual)               # (batch, seq, d_model) routed reconstruction
experts = moe.route(residual)       # (batch, seq, k_experts) selected expert ids

print(moe.coherence_diagnostic.to_dict())
print(moe.faithfulness_report(residual).to_dict())
```

Supply an explicit polygram `ExpertDictionary` to skip the internal
clustering (e.g. when you clustered with custom parameters):

```python
import polygram

ed = polygram.cluster_experts(dictionary, decoder_vectors=basis.W_dec,
                              method="cosine", coherence_threshold=0.3)
moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=2)
```

If the basis was built directly (no `polygram_checkpoint_path`) and you
pass no `expert_dictionary`, `forge_to_moe` raises `ValueError` naming the
queued `add-moe-explicit-cluster-construction` follow-up that would
auto-cluster from `basis.W_dec` alone.

## The `ForgedMoE` module

A `torch.nn.Module` with **zero trainable parameters** in v1 — every
tensor is a buffer, so `.to(device)` moves the whole module and
`.parameters()` is empty.

| Method | Returns |
|---|---|
| `forward(residual, *, track_load=False)` | `(*batch, d_model)` routed reconstruction |
| `route(residual)` | `(*batch, k_experts)` int64 selected expert ids, best first |
| `expert_load()` | `(n_experts,)` token-slot fractions from the last `track_load=True` forward, else `None` |
| `encode(residual)` | `(*batch, n_features)` feature activations (the `pinv(W_dec)` encoder) |
| `faithfulness_report(host_residual)` | `FaithfulnessReport(routed_vs_flat_mse, flat_vs_host_mse, ratio)` |
| `.coherence_diagnostic` | `CoherenceDiagnostic(median_intra_cluster_cosine, max_intra_cluster_cosine, ...)` |
| `.config` | `ForgedMoEConfig` — the frozen contract surface |
| `save_pretrained(dir)` / `load_pretrained(dir)` | self-contained round-trip |

### The encoder

Features are produced by the basis pseudo-inverse `pinv(W_dec)` — the
same convention `SubspaceProjector` uses on the input side, and the
encoder the acceptance prototype measured its bands against. Using the
SAE's native `W_enc`/`b_enc` (with its activation function) is deferred
to the queued `add-moe-encoder-side` proposal.

### The sub-dictionary expert

Each expert is a **deterministic slice** of `W_dec` — the rows whose
features the polygram clustering assigned to that expert. No new
parameters, no distillation. Two consequences make the contract clean:

- **`k_experts = n_experts` collapses to the flat SAE byte-for-byte**
  (Band A: MSE/coord `<= 1e-5`). This correctness check is independent
  of cluster quality.
- **The sparsity gain is honest**: `effective_decode_cost` counts the
  decoder-row touches, which equal `k / E` of the flat cost for uniform
  clusters (Band B).

The v1 forward is a single masked matmul (`(features * active_feature_mask)
@ W_dec`) — mathematically identical to summing per-expert sub-decodes
because the partition is disjoint, with no Python loop over experts. A
gather-based sparse kernel that realises the wall-clock saving (not just
the counted-cost saving) is a queued follow-up.

## v1 scope and the follow-up roadmap

v1 ships exactly one expert type and one router; everything else raises a
clean `NotImplementedError` naming the queued proposal:

| Knob | v1 | Deferred (→ proposal) |
|---|---|---|
| `expert_type` | `"sub_dictionary"` | `"tiny_mlp"` → `add-moe-tiny-mlp-experts`; `"residual_block"` → `add-moe-residual-block-experts` |
| `router_type` | `"polygram_heuristic"` | `"linear"`, `"mlp"` → `add-moe-trained-router` |

Also queued: `add-moe-matryoshka` (nested experts, needs a trained
router first), `add-moe-steering` (`.steer(expert_ids, strength)`),
`add-moe-encoder-side` (route before the encoder), and
`add-moe-as-residual-stream-layer` (insert `ForgedMoE` into a forged
transformer's residual stream). These are intentionally out of v1 so the
correctness surface stays small — the same falsifiable-MVP cadence the
host-wrapped forge used.

The `qwen3_moe` adapter is a **different** abstraction: it forges a host
transformer that *already has* MoE blocks. `sae-moe-forge` adds a routing
layer on top of a flat SAE. They do not share machinery; composition is
the queued `add-moe-as-residual-stream-layer`.

## Acceptance bands

The capability spec (`openspec/specs/sae-moe-forge/spec.md`) gates on
four bands measured on a 256-token calibration batch:

- **Band A — fidelity collapse** (`k = n_experts` → flat SAE, MSE `<= 1e-5`).
- **Band B — sparsity gain** (counted decode-cost ratio in `[2/E ± 0.05]`).
- **Band C — faithfulness**, split into *strict* (clusterable basis:
  routed-vs-flat `<= 0.5x` flat-vs-host) and *advisory* (any basis: ratio
  reported, never gating).
- **Band D — round-trip stability** (`config.to_dict()/from_dict()` plus a
  reloaded module reproducing the same reconstruction).

See `openspec/changes/add-sae-moe-forge/smoke-results.md` for the
per-fixture prototype numbers and the cosine-pair survey that motivates
the strict/advisory split.
