## Why

The v0 forged model has a feature-basis-width residual stream (k) but
inherits the host's attention internal width (`n_heads × head_dim =
host d_model`). The QKV-output space, head_dim, and attention scores
all live in d-space, not k-space.

This is a deliberate v0 tradeoff documented in
[`docs/algorithm.md`](../../../docs/algorithm.md) §10.2: it preserves
host attention mechanics (softmax, head splitting, positional
handling) byte-for-byte, which is exactly why the v0 identity-basis
sanity check shows KL ≈ 0. The cost is that the forged model is only
*half* feature-native — residual is, attention isn't.

The spec's §4 calls for **every** dimension to be k-wide:
`W_Q_new = E.T @ W_Q @ B.T` projects both sides, so QKV-output, head_dim,
and attention scores all live in feature space. That's the model where
the interpretability-by-construction story is fully realized:
attention scores answer "which features did this feature attend to,"
rather than "which residual-space directions did this projected
residual attend to."

This change adds feature-native attention as an opt-in v0.2 mode and
sets up the v1.0 transition where it becomes default.

## What Changes

### NativeModel

- Add `attention_width: Literal["host", "feature_native"] = "host"` to
  `NativeModelConfig`. When `"feature_native"`, the attention internal
  width equals `hidden_size` (k).
- When `attention_width == "feature_native"`, validate that `hidden_size
  % num_heads == 0` (the standard transformer constraint applied to
  the basis width). Default `head_dim = hidden_size // num_heads`.
  When the constraint fails, raise `ValueError` whose message names the
  offending values and suggests adjusting `num_heads`.
- The torch module's `CausalSelfAttention` becomes width-agnostic — it
  already is; only the per-block dimensions change. No new module code.

### SubspaceProjector

- Add `project_residual_full(W: (d, d)) -> (k, k)` computing `D @ W @ E`
  (both sides projected). Used for `c_proj` (attention output → residual).
- Add `project_qkv_full(W: (d, 3d)) -> (k, 3k)` projecting each of the
  Q / K / V output blocks separately (three independent applications of
  E to the d-wide output sub-spaces, then concatenated). Used for
  `c_attn`.
- `project_module(host_model, *, attention_width="host")` gets the new
  kwarg. When `"feature_native"`, c_attn uses `project_qkv_full`,
  c_proj uses `project_residual_full`, and the QKV-output bias projects
  via three independent `project_residual_bias` calls. Everything else
  (embeddings, MLP, layer norms, lm_head) is unchanged from v0.

### ForgePipeline

- Add `attention_width: Literal["host", "feature_native"] = "host"`
  field. Default stays `"host"` for v0.2 to preserve byte-equivalence
  with v0 outputs. v1.0 flips the default.
- `_config_from_host` accepts an `attention_width` kwarg and sets
  `qkv_inner_size = n_features` when feature-native.

### CLI

- Add `--feature-native-attention` flag to `sae-forge forge` that sets
  `attention_width="feature_native"`.

## Capabilities

### New Capabilities

- `feature-native-attention`: An opt-in v0.2 mode where the forged
  model's attention internal width equals `n_features`, projecting
  c_attn / c_proj on both sides per the algorithm spec's §4. Retains
  the host's `num_heads` by default, requiring `n_features %
  num_heads == 0`; raises a clear `ValueError` otherwise.

### Modified Capabilities

- `subspace-projector`: Gains `project_residual_full` and
  `project_qkv_full`. `project_module` gets an `attention_width`
  kwarg.
- `native-model`: `NativeModelConfig` gains the `attention_width`
  field.
- `forge-pipeline`: `ForgePipeline` gains the `attention_width` field
  and threads it through `_config_from_host`.
- `algorithm-foundation`: `docs/algorithm.md` §10.2 is updated —
  feature-native attention is no longer a "v1 target," it's the v0.2
  opt-in. The deviation note becomes "v0.2 ships both modes;
  attention_width='host' is the default for backward compatibility."

## Impact

- `saeforge/projector.py`: ~30 new lines for the two new helpers and
  the `project_module` kwarg.
- `saeforge/model.py`: ~10 lines for the `NativeModelConfig` field
  and the `_config_from_host` switch.
- `saeforge/forge.py`: ~5 lines threading `attention_width` through.
- `saeforge/cli.py`: one new flag.
- `tests/test_subspace_projector.py`: 3 new tests covering the new
  helpers and the kwargged `project_module`.
- `tests/test_native_model.py`: 1 new test for the
  divisibility constraint.
- `tests/test_feature_native_attention.py`: new file with
  end-to-end tests:
  - feature-native forge runs to `done`
  - identity-basis sanity check **still holds** (KL ≈ 0) when k = d,
    proving the algebra is right
  - faithfulness on a non-trivial basis is non-negative and finite
    (no claim that it beats v0; that's an empirical research question
    once compression quality is high)
  - host-mode and feature-native-mode produce different forged
    weights on the same non-trivial basis (regression check)

- `docs/algorithm.md`: §10.2 rewritten to describe the v0.2 opt-in
  rather than a v1 deviation; the cross-references to `subspace-projector`
  and `forge-outer-loop-fsm` capability specs stay.
- `README.md`: Add a "Feature-native attention (v0.2)" subsection
  pointing at the new flag.
- `AGENTS.md`: Update the v0.1 → v1.0 milestone breakdown to include
  v0.2 as the feature-native-attention opt-in.

## Migration path

- **v0.1 (shipped)**: `attention_width="host"` is the only mode.
- **v0.2 (this change)**: both modes ship; `"host"` stays the default
  for backward compat; `"feature_native"` is opt-in via flag.
- **v1.0 (future)**: `"feature_native"` becomes the default. The
  `"host"` mode stays available behind an explicit
  `--legacy-host-attention` flag for one milestone, then is removed.

The v1.0 transition is its own OpenSpec change
(`feature-native-attention-default`), not part of this scope.

## Out of scope

- **Re-projecting MLP intermediate widths.** §4's
  `W_in_new = E.T @ W_in @ B.T` is dimensionally inconsistent
  (it would require `h_ff = d`); the right reading is that MLP
  intermediate stays `h_ff` because it isn't residual-aligned. v0
  already does this; v0.2 keeps it. A separate `feature-native-mlp`
  change could explore projecting MLP gate / up to a k-wide
  intermediate, but that's a research question (does fine-tuning
  recover the MLP capacity?), not a spec-conformance question.
- **Encoder Eᵀ vs `pinv(W_dec)`.** The other v0 deviation in
  `docs/algorithm.md` §10.1 (using `pinv(W_dec)` instead of the SAE's
  trained encoder) is independent of attention width and is tracked
  separately. Both deviations can be addressed independently.
