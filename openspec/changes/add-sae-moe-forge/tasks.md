# Tasks — `add-sae-moe-forge`

## Prototype landed 2026-05-19

Prototype (`scripts/prototype_sae_moe_forge.py`) ran two fixtures:
GPT-2 L8 K=211 (near-isotropic) and a synthetic clusterable basis
(4 clusters of 32). Results in `smoke-results.md`. Headline:

- **Mechanical bands (A, B, D) pass universally.** The forge
  produces structurally correct output on any basis it accepts.
- **Faithfulness (Band C) is basis-dependent.** Clusterable
  bases: routing ~free (0.12× on synthetic). Near-isotropic
  bases: ~4–5× the flat projection loss.

Three proposal revisions landed back into `proposal.md` and
`spec.md` based on this:

1. Band C split into strict (clusterable) and advisory
   (any basis). The original single-tolerance `5×` bound was
   right at the edge on the K=211 fixture.
2. `forge_to_moe` will default `coherence_threshold=0.0` so the
   v1 surface works on isotropic bases without producing
   degenerate singleton clusters.
3. v1 ships a `coherence_diagnostic` field on every forged MoE
   (median intra-cluster cosine + max) so users see the
   basis-quality signal up front.

## Cadence

Same pattern as `add-host-wrapped-forge-fallback`:

1. Capability spec (this proposal landed)
2. Prototype + smoke-results.md (BEFORE production code lands)
3. Critical-path production implementation
4. Follow-up proposals queued separately

The prototype is the gate. If the falsifiable acceptance bands in
proposal.md don't pass, revise the proposal before any production
code lands. The host-wrapped revision (acceptance gate changed
after the prototype showed non-nested basis non-monotonicity) is
the precedent.

## 1. Capability spec

- [ ] 1.1 Create `openspec/specs/sae-moe-forge/spec.md` with the
      requirements + scenarios from this change dir's
      `specs/sae-moe-forge/spec.md`. Same pattern as
      `forge-forward-mode`.

## 2. Prototype (gate for everything below)

- [x] 2.1 `scripts/prototype_sae_moe_forge.py`: load the K=211
      jbloom basis, cluster with `polygram.cluster_experts`, build
      a minimal `SubDictionaryExpertSet` + `PolygramHeuristicRouter`
      in the script (not yet productionised), and report the four
      acceptance bands per the spec:
      - Reconstruction-fidelity collapse at `k=E`
      - Sparsity gain at `k=2, E≥4`
      - Reconstruction-degradation at `k=2`
      - Round-trip stability
- [x] 2.2 Write `openspec/changes/add-sae-moe-forge/smoke-results.md`
      with the measurements. If any acceptance band fails, revise
      `proposal.md` and re-run before continuing.  *(Done — Band C
      revised to strict/advisory split; revisions back-propagated
      to `proposal.md` and `spec.md`.)*
- [x] 2.3 Probe basis-clusterability survey
      (`scripts/probe_polygram_clustering.py`) to demonstrate the
      basis-quality dependency and inform the `coherence_threshold=0.0`
      default chosen in 2.2.
- [ ] 2.4 **Priority: high.** Add a real clustered-SAE fixture to
      the smoke gate before production code lands. Candidates:
      - econ-sae's supervised SAE (per [[project_fix_scale_boost_smoke]]
        polygram's cluster count saturates at the supervised concept
        count; cosine cluster structure is real).
      - Any polygram-compressed SAE produced with
        `BlockFormation(strategy="cosine", cosine_threshold>=0.3)`
        during compression.
      Why this is high-priority: the synthetic clusterable fixture
      proves the design works under contrived conditions; a real
      clustered SAE proves Band C-strict passes in production-
      relevant regimes. Without it, Band C-strict is only validated
      against a fixture I built, and reviewers can reasonably ask
      "but does this hold on a real one?" Plumbing the fixture
      through requires either an external download path or an
      on-box artifact this PR doesn't bundle today.

## 3. `FeatureBasis.polygram_checkpoint_path`

- [ ] 3.1 Add `polygram_checkpoint_path: str | None = None` to
      `FeatureBasis`. Populated by `from_polygram_checkpoint` from
      its `checkpoint_path` argument.
- [ ] 3.2 `to_dict` / `from_dict` round-trip.

## 4. `saeforge/_moe/sub_dictionary.py`

- [ ] 4.1 `class SubDictionaryExpertSet(nn.Module)`. Stores
      `(n_experts,)` parallel lists of:
      - `expert_feature_ids`: `(n_features_e,)` int64 tensor
        identifying which basis-feature rows belong to expert e.
      - `expert_W_dec`: `(n_features_e, d_model)` float32 tensor —
        the slice of `basis.W_dec` for this expert.
- [ ] 4.2 `forward(features, top_k_experts) -> reconstruction`.
      Vectorised across experts; uses index gathering rather than
      a per-expert python loop.
- [ ] 4.3 Property `effective_decode_cost` returning the counted
      decoder-row touches per token. Used by the sparsity-gain
      acceptance test.

## 5. `saeforge/_moe/routers.py`

- [ ] 5.1 `class PolygramHeuristicRouter`. Constructor accepts
      the `ExpertDictionary` and stores the `_feature_to_expert`
      map as a buffer. Stateless w.r.t. trainable params.
- [ ] 5.2 `route(features, top_k) -> top_k_experts`. Implements
      `ExpertDictionary.route` vectorised in torch (the polygram
      function is per-vector numpy; we want batched torch). Asserts
      it matches polygram's per-vector route for the same input
      within float tolerance.

## 6. `saeforge/moe.py`

- [ ] 6.1 `class ForgedMoEConfig` (frozen dataclass): `n_features`,
      `d_model`, `n_experts`, `k_experts`, `expert_type`,
      `router_type`, `source_basis_checkpoint`. `to_dict` /
      `from_dict` round-trip.
- [ ] 6.2 `class ForgedMoE(nn.Module)`. Constructor takes the
      `ExpertSet`, the `Router`, and the source basis's encoder
      (`W_enc`, `b_enc`) — kept as buffers in v1.
- [ ] 6.3 `forward(residual, *, track_load=False) -> reconstruction`.
- [ ] 6.4 `route(residual)` exposes the per-token routing without
      decoding (diagnostic surface).
- [ ] 6.5 `expert_load()` returns the most-recent forward's
      per-expert load (None when `track_load` was never set).

## 7. `forge_to_moe` entry point

- [ ] 7.1 New module-level function in `saeforge/moe.py`:
      `forge_to_moe(basis, expert_dictionary=None, *,
      k_experts=2, expert_type="sub_dictionary",
      router_type="polygram_heuristic", coherence_threshold=0.3,
      max_features_per_expert=None) -> ForgedMoE`.
- [ ] 7.2 When `expert_dictionary is None`: reload polygram
      `Dictionary` from `basis.polygram_checkpoint_path` via
      `polygram.load_sae_safetensors`, then call
      `polygram.cluster_experts`. Surface a clear error when the
      path is unavailable.
- [ ] 7.3 Validate `expert_type` and `router_type` against the
      legal sets; raise `NotImplementedError` for queued values
      (`tiny_mlp`, `residual_block`, `linear`, `mlp`) pointing at
      the named follow-up proposals.
- [ ] 7.4 Construct `SubDictionaryExpertSet` from the
      `ExpertDictionary` partition and `basis.W_dec`. Construct
      `PolygramHeuristicRouter` from `expert_dictionary`.
- [ ] 7.5 Assemble `ForgedMoE` and return.

## 8. `__init__.py` exports

- [ ] 8.1 Export `ForgedMoE`, `ForgedMoEConfig`, `forge_to_moe`
      from the package top-level.

## 9. Tests

- [ ] 9.1 `tests/test_moe_forge.py` — covers:
      - `forge_to_moe(basis)` constructs without `expert_dictionary`
        kwarg when basis has a checkpoint path (uses polygram
        clustering internally).
      - `forge_to_moe(basis, expert_dictionary=ED)` accepts an
        explicit ExpertDictionary.
      - `k_experts=n_experts` collapses to flat-SAE reconstruction
        within 1e-5 MSE per coordinate.
      - `k_experts=2, n_experts=4` produces a `(B, T, n_experts=4)`
        routing tensor with exactly 2 unique active expert IDs per
        token.
      - `expert_type="tiny_mlp"` raises `NotImplementedError`
        naming `add-moe-tiny-mlp-experts`.
      - `router_type="linear"` raises `NotImplementedError` naming
        `add-moe-trained-router`.
      - Round-trip `ForgedMoEConfig.to_dict()` →
        `ForgedMoEConfig.from_dict()`.
- [ ] 9.2 `tests/test_moe_forge_polygram_alignment.py` — verifies
      the torch router's vectorised result equals
      `ExpertDictionary.route(activations, top_k)` per-vector for
      the same input on a 32-token batch.

## 10. Docs

- [ ] 10.1 Add `docs/moe-forge.md` (~150 lines): when to use
      sae-moe-forge, the v1 scope, the expert/router types
      available, the follow-up roadmap. Cross-reference
      polygram's `experts.py` docstring for the architectural
      split.
- [ ] 10.2 README addition: one paragraph under "Status" listing
      `add-sae-moe-forge` as the new capability.

## 11. Out-of-band: ExpertSet → safetensors

- [ ] 11.1 `ForgedMoE.save_pretrained(path)` writes config + the
      expert_feature_ids partition + W_enc/b_enc buffers (not the
      full W_dec — that's re-sliced from `source_basis_checkpoint`
      on load).
- [ ] 11.2 `ForgedMoE.load_pretrained(path)` round-trip; requires
      `source_basis_checkpoint` to be reachable.
