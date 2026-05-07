## 1. SubspaceProjector

- [ ] 1.1 Add `project_residual_full(W: (d, d)) -> (k, k)` computing `D @ W @ E` (both sides projected via the basis pseudoinverse). Document the new identity in the module docstring's projection-algebra block
- [ ] 1.2 Add `project_qkv_full(W: (d, 3d)) -> (k, 3k)` that splits the QKV-output into Q, K, V blocks via `np.split(W, 3, axis=1)`, applies `project_residual_full` to each, and concatenates the results back along axis=1
- [ ] 1.3 Add `attention_width: Literal["host", "feature_native"] = "host"` kwarg to `project_module`; when `"feature_native"`, wire `c_attn` weight via `project_qkv_full`, `c_attn` bias via three `project_residual_bias` calls (one per Q / K / V block), and `c_proj` weight via `project_residual_full`
- [ ] 1.4 Document the attention-width contract in the module docstring with the three new identities from `design.md`

## 2. NativeModel

- [ ] 2.1 Add `attention_width: Literal["host", "feature_native"] = "host"` field to `NativeModelConfig`
- [ ] 2.2 In `NativeModelConfig.__post_init__`, when `attention_width == "feature_native"`, validate that `hidden_size % num_heads == 0`; raise `ValueError` whose message names `hidden_size`, `num_heads`, and the suggested fix (set `num_heads` to a divisor of `hidden_size`)
- [ ] 2.3 In `_config_from_host`, accept `attention_width` kwarg; when `"feature_native"`, set `qkv_inner_size = n_features` and `head_dim = n_features // n_heads` (using the host's `n_head`)
- [ ] 2.4 Verify the existing `_build_torch_module` is width-agnostic (no hardcoded `host n_embd` references); it already is, since `qkv_inner_size` drives the per-block math

## 3. ForgePipeline

- [ ] 3.1 Add `attention_width: Literal["host", "feature_native"] = "host"` field; default stays `"host"` in v0.2 to preserve the v0.1 byte-equivalence safety net
- [ ] 3.2 Thread `attention_width` through both `_run_synthetic_imperative` (call `projector.project_module(host, attention_width=self.attention_width)` and `_config_from_host(host, n_features, attention_width=self.attention_width)`) and `_run_synthetic_fsm` (carry it on the ctx so `project_to_subspace` action can pick it up)
- [ ] 3.3 In the FSM `project_to_subspace` action, read `ctx.get("attention_width", "host")` and thread it into `project_module` and `_config_from_host`

## 4. CLI

- [ ] 4.1 Add `--feature-native-attention` flag to `sae-forge forge` (and the future `sae-forge forge --fsm`); sets `attention_width="feature_native"` on the constructed `ForgePipeline`

## 5. Tests

- [ ] 5.1 `tests/test_subspace_projector.py`: 3 new tests
  - `project_residual_full` round-trips on identity basis: when `W_dec = I`, `project_residual_full(W) == W`
  - `project_qkv_full` produces `(k, 3k)` shape and the three k-wide blocks match three independent `project_residual_full(W_q)` / `(W_k)` / `(W_v)` calls
  - `project_module(host, attention_width="feature_native")` produces `c_attn.weight` of shape `(k, 3k)` and `c_proj.weight` of shape `(k, k)`
- [ ] 5.2 `tests/test_native_model.py`: 1 new test
  - `NativeModelConfig(attention_width="feature_native", hidden_size=8, num_heads=3, ...)` raises `ValueError` whose message contains `"hidden_size"` and `"num_heads"` (8 % 3 != 0)
- [ ] 5.3 New `tests/test_feature_native_attention.py`:
  - End-to-end `run_synthetic` with `attention_width="feature_native"` reaches `done`, writes the artifact tree, KL non-negative
  - **Identity-basis sanity check**: KL < 1e-3 when `W_dec = np.eye(d)` (the spec's correctness signal — algebra preserves the host exactly when the basis spans the residual)
  - Regression: host-mode and feature-native-mode produce different SHA-256 of `forged/model.safetensors` on a non-trivial random basis
  - Fine-tune still runs (4 AdamW steps; final loss <= initial)

## 6. Docs

- [ ] 6.1 Update `docs/algorithm.md` §10.2: the deviation note becomes "v0.2 ships both modes; `attention_width='host'` is the default for backward compatibility. v1.0 flips the default and removes `host` mode in v1.1." Cross-reference this change
- [ ] 6.2 Add a "Feature-native attention (v0.2)" subsection to `README.md` explaining the new flag, the divisibility constraint, and the faithfulness tradeoff
- [ ] 6.3 Update `AGENTS.md`: add v0.2 (`feature-native-attention`) to the milestone breakdown

## 7. OpenSpec scaffolding

- [x] 7.1 `openspec/changes/feature-native-attention/proposal.md`
- [x] 7.2 `openspec/changes/feature-native-attention/design.md`
- [x] 7.3 `openspec/changes/feature-native-attention/tasks.md` (this file)
- [x] 7.4 `openspec/changes/feature-native-attention/specs/feature-native-attention/spec.md`
