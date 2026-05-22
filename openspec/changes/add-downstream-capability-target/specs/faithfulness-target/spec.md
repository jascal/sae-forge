# faithfulness-target Specification (delta)

## ADDED Requirements

### Requirement: `DownstreamCapabilityTarget` built-in target

`saeforge.eval.targets.DownstreamCapabilityTarget` SHALL be a built-in
`FaithfulnessTarget` that scores per-feature × per-label AUC through a
caller-supplied downstream-task encoder.

Constructor signature:

```python
DownstreamCapabilityTarget(
    *,
    encoder: Callable[[Tensor], Tensor],  # d_model -> latent_width
    labels: np.ndarray,                    # (N_items, V), binary-castable
    aggregator: Literal["pool_then_encode", "encode_then_pool"] | Callable = "pool_then_encode",
    min_prevalence: int = 0,
    decode_via_basis: bool = True,
)
```

Constraints:

- `encoder` SHALL be callable; the target SHALL NOT introspect for an
  `.encode` / `.forward` method. Users supplying an `nn.Module`
  whose `forward` returns `(reconstruction, latents)` MUST wrap it
  (e.g. `lambda x: encoder(x)[1]`). The target's docstring SHALL
  document this with an example for bio-sae's `_ReferenceSAE` shape.
- `labels` SHALL be a 2-D numpy array; SHALL be coerced to `float64`
  at construction time; SHALL have at least one row and one column.
  Construction SHALL raise `ValueError` on `ndim != 2`, `shape[0] < 1`,
  or `shape[1] < 1`.
- `aggregator` SHALL be one of the two named strings or a callable
  `(latents: Tensor, residue_indices: Tensor | None) -> Tensor`. The
  named-string set SHALL be exactly `{"pool_then_encode",
  "encode_then_pool"}`; other strings SHALL raise `ValueError` at
  construction time.
- `min_prevalence` SHALL be a non-negative integer. When > 0, the
  target SHALL drop label columns whose positive-class count is
  below the threshold at score time; the surviving column set SHALL
  be recomputed on each `score()` call (so the same target instance
  works across forge runs with different eval subsets).
- `decode_via_basis` SHALL default `True` and SHALL gate whether the
  forged hidden states are projected back to `d_model` via the basis
  before the encoder is called. Setting `False` is for users whose
  encoder already operates in basis coordinates; that case SHALL NOT
  invoke `pinv(basis_encode)` or read `ctx["basis"]`.

`name` SHALL be `"downstream_capability"`; `better_when` SHALL be
`"higher"`.

### Requirement: `DownstreamCapabilityTarget.score()` contract

The `score(*, forged, host, ctx)` method SHALL conform to the
`FaithfulnessTarget` protocol:

- `forged`: the forged `NativeModel` whose `torch_module` is called
  with `(input_ids,)` and returns `(batch, seq_len, n_features)`.
  CLS / EOS bookkeeping tokens SHALL be stripped at positions 0
  and -1 before downstream processing.
- `host`: ignored by this target (`better_when="higher"` GT-style
  targets MAY ignore host per the protocol's `host-MAY-be-ignored`
  carve-out). The target SHALL accept it for protocol conformance
  and SHALL NOT call `host(...)`.
- `ctx`: SHALL be a mapping containing `_eval_input_ids` (required;
  shape `(N_items, max_seq_len)` tensor) and optionally
  `ctx["basis"]` (an explicit `FeatureBasis` for exact `W_dec`
  recovery; see `add-downstream-capability-target/design.md`
  Decision 2). Other ctx keys SHALL be ignored.

The pipeline inside `score()` SHALL be:

```
1. For each row i in _eval_input_ids:
   a. forged_module(row_i) -> (1, L, n_features)
   b. Strip CLS / EOS: h_basis = ... [0, 1:-1, :]
   c. If decode_via_basis:
        Recover W_dec from ctx["basis"].W_dec (if set) OR
        from pinv(forged_module.basis_encode) (cached on first use).
        h_d = h_basis @ W_dec   # (L, d_model)
      Else:
        h_d = h_basis           # caller's encoder reads basis coords
   d. If aggregator == "pool_then_encode":
        z_i = encoder(h_d.mean(0, keepdim=True))  # (1, latent_width)
      Elif aggregator == "encode_then_pool":
        z_per_residue = encoder(h_d)              # (L, latent_width)
        z_i = z_per_residue.mean(0, keepdim=True) # (1, latent_width)
      Else: aggregator(h_d, None)
2. Stack z_i across rows -> Z: (N_items, latent_width)
3. Apply min_prevalence filter to labels -> Y_filt: (N_items, V_filt)
4. Compute per-feature × per-label AUC via the Mann-Whitney rank-sum
   identity (vectorised chunked matmul, same as
   biosae.sae.evaluation.score_against_ground_truth and
   GroundTruthTarget). Mean over labels of max-over-features AUC.
5. Return (score, perplexity_analog):
      score             = mean_best_auc
      perplexity_analog = max(0.0, 1.0 - score)
```

The target SHALL NOT mutate any of `forged`, `host`, or `ctx`. The
implementation SHALL cache the recovered `W_dec` keyed on
`id(forged_module)` so repeated `score()` calls on the same forge
amortise the one-time `pinv` cost.

### Requirement: Default-target dispatch unchanged

`DownstreamCapabilityTarget` SHALL NOT be returned by
`_default_target_for(family)` for any family. It is opt-in only:
callers MUST instantiate and pass via `ForgePipeline(faithfulness=...)`.

This matches the existing `GroundTruthTarget` policy (fixture-specific,
never family-defaulted). The reason: capability targets need a
caller-supplied encoder + labels, which family dispatch can't supply.

### Requirement: `pinv(basis_encode)` warning

When `decode_via_basis=True` and the target recovers `W_dec` via
`pinv(forged_module.basis_encode)`, if
`numpy.linalg.matrix_rank(basis_encode) < n_features`, the target
SHALL emit a `UserWarning` naming the rank and recommending the
caller pass `ctx["basis"]` explicitly. The warning SHALL fire once
per `id(forged_module)` (matching the cache key) — not once per
`score()` call.

### Requirement: Module exports

`saeforge.eval.targets.__init__` SHALL re-export
`DownstreamCapabilityTarget`. The top-level `saeforge.__init__`
SHALL re-export it under the same name. Imports SHALL be lazy in
the sense that constructing a `DownstreamCapabilityTarget` SHALL
NOT import torch — torch is imported inside `score()` via the
existing `require_extra` path.
