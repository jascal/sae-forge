## 1. NativeModelConfig: MoE fields

- [ ] 1.1 Add four new fields to `NativeModelConfig` in `saeforge/model.py`:
  - `num_experts: int = 0`
  - `num_experts_per_tok: int = 0`
  - `moe_intermediate_size: int = 0`
  - `norm_topk_prob: bool = True`
- [ ] 1.2 Add `__post_init__` validation: when `num_experts > 0`, require
  `num_experts_per_tok > 0`, `num_experts_per_tok <= num_experts`,
  `moe_intermediate_size > 0`. Clear `ValueError` naming the failing pair
- [ ] 1.3 Add `"qwen3_moe"` to `_SUPPORTED_FAMILIES` and to the
  `_build_torch_module` dispatch (routes to `build_llama_family_module`)

## 2. Qwen3MoEMLP + Qwen3Expert in the Llama-family factory

- [ ] 2.1 Inside `_get_forged_llama_class` in `saeforge/adapters/llama.py`,
  define `Qwen3Expert(nn.Module)`: a single SwiGLU MLP with
  `moe_intermediate_size` inner dim. Same shape as `SwiGLU_MLP` but
  parameterized differently
- [ ] 2.2 Define `Qwen3MoEMLP(nn.Module)`: router (`nn.Linear(hidden, num_experts, bias=False)`)
  + `nn.ModuleList[num_experts] of Qwen3Expert`. Forward: softmax→topk→
  optional renorm→naive per-expert loop with `index_add_` per design.md
- [ ] 2.3 Modify `LlamaBlock.__init__` to branch on `cfg.num_experts > 0`:
  dense path constructs `SwiGLU_MLP`, MoE path constructs `Qwen3MoEMLP`.
  Forward at the block level is unchanged
- [ ] 2.4 Confirm `SwiGLU_MLP` and `Qwen3MoEMLP` share the same
  `(B, T, hidden) -> (B, T, hidden)` forward signature so the block's
  forward doesn't need to branch

## 3. Walker contract: MoE block

- [ ] 3.1 In `LlamaAdapter.walk` (or a Qwen3MoE override — see design.md
  for the "shared walker, host-attribute-gated" pattern decision),
  detect MoE blocks by `hasattr(block.mlp, "gate")` and
  `hasattr(block.mlp, "experts")`
- [ ] 3.2 For MoE blocks: emit `mlp.gate.weight` via
  `project_residual_input` (router reads residual). Emit each
  `mlp.experts.{e}.gate_proj.weight` and `mlp.experts.{e}.up_proj.weight`
  via `project_residual_input` (right axis is residual). Emit each
  `mlp.experts.{e}.down_proj.weight` via `project_residual_output`
  (left axis is residual)
- [ ] 3.3 For dense blocks (the existing path), do nothing different —
  the `hasattr` guards make the MoE emission inert
- [ ] 3.4 Confirm no host weight is emitted twice (MoE blocks have no
  `mlp.gate_proj` etc.; dense blocks have no `mlp.gate` etc.; the two
  paths are mutually exclusive on a real host)

## 4. Qwen3MoEAdapter

- [ ] 4.1 New module `saeforge/adapters/qwen3_moe.py`. `Qwen3MoEAdapter`
  inherits from `Qwen3Adapter` with `family = "qwen3_moe"`
- [ ] 4.2 Override `build_native_config`: call `super().build_native_config()`,
  then `replace(base, family="qwen3_moe", num_experts=..., num_experts_per_tok=...,
  moe_intermediate_size=..., norm_topk_prob=...)`. Source values from
  `host.config.num_experts`, `host.config.num_experts_per_tok`,
  `host.config.moe_intermediate_size`, `host.config.norm_topk_prob`
  (fall back to True if missing)
- [ ] 4.3 Inherit `walk` from LlamaAdapter via Qwen3Adapter (no override —
  the host-attribute-gated MoE emission in step 3 handles it)
- [ ] 4.4 Register `Qwen3MoeForCausalLM` in the `try`/`except` block —
  silent skip when transformers < 4.51

## 5. Adapter registration

- [ ] 5.1 Append `from saeforge.adapters import qwen3_moe as _qwen3_moe`
  to `saeforge/adapters/__init__.py`

## 6. ForgePipeline: moe_strategy

- [ ] 6.1 Add `moe_strategy: Literal["preserve", "collapse", "top_n"] = "preserve"`
  to `ForgePipeline`
- [ ] 6.2 Add `moe_keep_n: int = 0` to `ForgePipeline`
- [ ] 6.3 `__post_init__` validation: `moe_strategy="top_n"` requires
  `moe_keep_n > 0`. `moe_strategy="collapse"` requires no extra knob
- [ ] 6.4 The forge run path threads `moe_strategy` into the walker (which
  uses it to decide between per-expert emission, averaging, or
  top-N pruning) and into `build_native_config` (which sets
  `num_experts` to 0 for `collapse`, to `moe_keep_n` for `top_n`)
- [ ] 6.5 `moe_strategy="top_n"` raises `NotImplementedError` with a clear
  pointer to the `moe-expert-calibration` follow-up

## 7. CLI

- [ ] 7.1 Add `--moe-strategy {preserve,collapse,top_n}` to
  `saeforge/cli.py`, default `preserve`
- [ ] 7.2 Add `--moe-keep-n N`. Mutually-required with
  `--moe-strategy top_n`
- [ ] 7.3 Thread both into the `ForgePipeline(...)` constructor

## 8. Conftest fixture: synthetic small Qwen3-MoE

- [ ] 8.1 New `tiny_qwen3_moe_untied` fixture in `tests/conftest.py`:
  `Qwen3MoeForCausalLM` with 128-d residual, 3 layers, 4 experts, top-2,
  moe_intermediate_size=128, untied embeddings. Skip-gates via
  `pytest.importorskip("transformers", minversion="4.51")`

## 9. Unit tests: tests/integration/test_qwen3_moe_adapter.py

- [ ] 9.1 Module-level `pytest.importorskip("transformers", minversion="4.51")`
- [ ] 9.2 `test_qwen3_moe_dispatches_to_qwen3_moe_adapter` — `adapter_for(host).family == "qwen3_moe"`
- [ ] 9.3 `test_qwen3_moe_walker_emits_gate_and_experts` — every block
  has `mlp.gate.weight` and `mlp.experts.{0..3}.{gate,up,down}_proj.weight`
  in the walk output. Shapes match the projection rules in design.md
- [ ] 9.4 `test_qwen3_moe_walker_omits_dense_mlp_keys` — no
  `mlp.gate_proj.weight` etc. in the output (those don't exist on a
  MoE host)
- [ ] 9.5 `test_qwen3_moe_native_config_sets_moe_fields` —
  `build_native_config` returns a config with `family="qwen3_moe"`,
  `num_experts=4`, `num_experts_per_tok=2`, `moe_intermediate_size=128`,
  `qk_norm=True` (inherited from Qwen3), `qkv_bias=False` (inherited)
- [ ] 9.6 `test_dense_families_keep_num_experts_zero` — sanity gate that
  Llama / Qwen2 / Qwen3-dense / Gemma-2 native configs all have
  `num_experts=0`
- [ ] 9.7 `test_qwen3_moe_forged_block_has_moe_mlp` — the forged
  module's `model.layers[0].mlp` is a `Qwen3MoEMLP` instance with
  `gate` (Linear) and `experts` (ModuleList of length 4)
- [ ] 9.8 `test_qwen3_moe_forward_finite` — forward pass on a small
  input produces finite logits
- [ ] 9.9 `test_qwen3_moe_routing_matches_host` — compare top-K routing
  decisions between host and forged module on the same input.
  Top-K *indices* should match within fp tolerance (top-K weights
  will drift slightly due to projection)

## 10. Integration test: tests/integration/test_hybrid_bridge_qwen3_moe.py

- [ ] 10.1 Module-level `pytest.importorskip("transformers", minversion="4.51")`
- [ ] 10.2 Mirror `test_hybrid_bridge_qwen3.py`:
  - `TestT0TinyQwen3MoESmoke` — bridges in state_dict; q_norm/k_norm
    present; expert MLPs round-trip; forward finite
  - `TestByteEquivalenceWhenDisabled` — `hybrid_bridge=False` leaves
    state_dict without bridge keys but with full MoE plumbing intact

## 11. Compression modes

- [ ] 11.1 Implement `preserve` mode (the default; per-expert
  projection as described in step 3)
- [ ] 11.2 Implement `collapse` mode: walker helper that averages
  expert weights before projection; forged config has `num_experts=0`
  and `intermediate_size=moe_intermediate_size`. Add a CHANGELOG/
  docs note marking this mode as "experimental — averages experts;
  produces a model that thinks like the average expert"
- [ ] 11.3 Implement `top_n` placeholder: raises `NotImplementedError`
  pointing at the `moe-expert-calibration` follow-up
- [ ] 11.4 Tests for each: `test_moe_preserve_strategy_matches_host_expert_count`,
  `test_moe_collapse_strategy_produces_dense_forged`,
  `test_moe_top_n_strategy_raises_not_implemented`

## 12. NVIDIA smoke script

- [x] 12.1 `scripts/smoke_qwen3_moe.py` — bundled in this PR per
  reviewer request. Loads `Qwen/Qwen3-30B-A3B-Base` via
  `device_map="auto"` and runs the end-to-end forge pipeline.
  Self-documented; clear failure modes for missing transformers /
  missing CUDA / wrong adapter family / non-finite logits
- [ ] 12.2 Document the script in `docs/qwen3_moe_nvidia.md` (new):
  hardware requirements (≥80GB GPU recommended), expected runtime
  (~5-10 minutes for a single forward pass on A100), what success
  looks like (exits 0 with `SMOKE OK`), failure-mode triage
- [ ] 12.3 The script is bundled in the PROPOSAL PR (this one) so it's
  available immediately. Until the adapter ships in the impl PR, the
  script will fail gracefully at the family check with "FAIL: expected
  family=qwen3_moe, got llama (Qwen3MoEAdapter not registered yet)"

## 13. Docs

- [ ] 13.1 `CHANGELOG.md` `## [Unreleased]` `### Added` entry: "Qwen3-MoE
  architecture adapter with three compression modes (preserve, collapse,
  top_n placeholder). Requires `transformers >= 4.51`. End-to-end
  validation requires NVIDIA ≥80GB; synthetic small-MoE adapter tests
  cover the M4 surface"
- [ ] 13.2 `docs/qwen3_moe_nvidia.md` (new): smoke script usage,
  hardware requirements, expert-utilization diagnostic interpretation

## 14. OpenSpec scaffolding

- [x] 14.1 `openspec/changes/qwen3-moe-support/proposal.md`
- [x] 14.2 `openspec/changes/qwen3-moe-support/design.md`
- [x] 14.3 `openspec/changes/qwen3-moe-support/tasks.md` (this file)
- [x] 14.4 `openspec/changes/qwen3-moe-support/specs/qwen3-moe-support/spec.md`
- [ ] 14.5 Run `openspec validate qwen3-moe-support --strict`

## 15. Pre-merge gates (for the impl PR that follows this proposal)

- [ ] 15.1 `pytest -q` passes on Intel — all Qwen3-MoE tests skip via
  importorskip
- [ ] 15.2 GitHub Actions CI green
- [ ] 15.3 Synthetic-small-MoE adapter tests pass on M4 (T2)
- [ ] 15.4 **NVIDIA T3 smoke (`scripts/smoke_qwen3_moe.py`) output pasted
  into the impl PR description before merge.** Load-bearing.

## 16. Deferred follow-ups

- [ ] 16.1 **`moe-expert-calibration`** — calibration utility for
  the `top_n` strategy. Run host on a corpus, log per-expert
  activation frequency, return top-N expert indices per layer
- [ ] 16.2 **`moe-fused-dispatch`** — replace the naive
  `for e in range(num_experts)` loop with a fused scatter-add path.
  Significant perf win on NVIDIA for large num_experts
- [ ] 16.3 **`qwen3-moe-aux-loss`** — load-balancing aux loss during
  forge fine-tune, if routing collapse becomes empirically observable
- [ ] 16.4 **`moe-sliding-window`** — replicate Qwen3-MoE's
  sliding-window attention for long-context fidelity. Currently
  out-of-scope (accepts `ε_attn` per `docs/algorithm.md` §5)
