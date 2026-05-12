# qwen3-moe-support Specification

## Purpose

The `qwen3-moe-support` capability adds the `Qwen3MoeForCausalLM`
architecture to the forge's supported families. Qwen3-MoE replaces
Qwen3 dense's SwiGLU MLP with a router (`mlp.gate`) + N independent
expert MLPs (`mlp.experts.{i}.{gate,up,down}_proj`), where the router
selects top-K experts per token and weighted-sums their outputs.

The capability pins the walker's per-expert + router projection
contract, the four new `NativeModelConfig` MoE fields, the forged MoE
MLP's forward-pass contract (softmax-then-topk, optional renormalization,
per-expert loop with `index_add_`), the three compression modes
(`preserve` / `collapse` / `top_n`), and the NVIDIA-tier smoke-script
contract.

This capability inherits Q/K-norm and untied-bias semantics from
`qwen3-dense-support` (via `Qwen3MoEAdapter(Qwen3Adapter)`) and
hybrid-bridge support from `hybrid-bridge-llama-family` (bridges sit
at residual-stream boundaries, orthogonal to MLP structure).

## ADDED Requirements

### Requirement: Qwen3MoEAdapter detects MoE fields from host config

`Qwen3MoEAdapter.build_native_config(host, n_features)` SHALL produce a `NativeModelConfig` with `family="qwen3_moe"` and the four MoE fields populated from the host's HF config. Specifically:

- `num_experts` ← `host.config.num_experts`
- `num_experts_per_tok` ← `host.config.num_experts_per_tok`
- `moe_intermediate_size` ← `host.config.moe_intermediate_size`
- `norm_topk_prob` ← `host.config.norm_topk_prob` (default `True` when
  absent)

The inherited fields (`qk_norm`, `qkv_bias`, `head_dim`, `n_kv_heads`,
`rms_norm_eps`, etc.) SHALL be populated by the parent
`Qwen3Adapter` / `LlamaAdapter` chain. Qwen3-MoE inherits
`qk_norm=True` and `qkv_bias=False` from Qwen3 dense.

#### Scenario: Qwen3-MoE host produces correct MoE config

- **GIVEN** a `Qwen3MoeForCausalLM` instance with
  `num_experts=128`, `num_experts_per_tok=8`, `moe_intermediate_size=768`
- **WHEN** `Qwen3MoEAdapter().build_native_config(host, n_features=256)` is invoked
- **THEN** the returned config has `family="qwen3_moe"`
- **AND** `num_experts == 128`
- **AND** `num_experts_per_tok == 8`
- **AND** `moe_intermediate_size == 768`
- **AND** `qk_norm` is True (inherited from Qwen3)
- **AND** `qkv_bias` is False (Qwen3 dense doesn't have biases)

#### Scenario: non-MoE families keep num_experts=0

- **GIVEN** a `LlamaForCausalLM` / `Qwen2ForCausalLM` / `Qwen3ForCausalLM`
  (dense) instance
- **WHEN** the relevant adapter's `build_native_config` is invoked
- **THEN** the returned config has `num_experts == 0` (regression gate
  ensuring the new field defaults preserve dense behavior)

### Requirement: walker emits MoE-block keys for hosts with mlp.gate + mlp.experts

`LlamaAdapter.walk(host, projector)` SHALL detect MoE blocks at runtime by `hasattr(block.mlp, "gate") and hasattr(block.mlp, "experts")`. For each such block, the walker SHALL emit:

- `model.layers.{i}.mlp.gate.weight` projected via `project_residual_input` (the gate reads the residual stream; same projection rule as `q_proj`/`k_proj`/`v_proj`). Output shape: `(num_experts, n_features)`.
- For each expert `e` in `range(num_experts)`:
  - `model.layers.{i}.mlp.experts.{e}.gate_proj.weight` projected via `project_residual_input`. Output shape: `(moe_intermediate_size, n_features)`.
  - `model.layers.{i}.mlp.experts.{e}.up_proj.weight` projected via `project_residual_input`. Output shape: `(moe_intermediate_size, n_features)`.
  - `model.layers.{i}.mlp.experts.{e}.down_proj.weight` projected via `project_residual_output`. Output shape: `(n_features, moe_intermediate_size)`.

For dense blocks (no `mlp.gate` submodule), the MoE emission path SHALL be inert and the existing dense walker keys (`mlp.gate_proj.weight` etc.) SHALL be emitted as before. The two paths SHALL be mutually exclusive on any real host (no host has both dense and MoE MLPs in the same block).

#### Scenario: synthetic small Qwen3-MoE walker emits expected keys

- **GIVEN** a synthetic Qwen3-MoE host with `num_hidden_layers=3`,
  `num_experts=4`
- **WHEN** `LlamaAdapter().walk(host, projector)` is invoked
- **THEN** every layer `i ∈ {0,1,2}` has key `model.layers.{i}.mlp.gate.weight` in the output
- **AND** every (layer, expert) pair `i ∈ {0,1,2}`, `e ∈ {0,1,2,3}` has all three of `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`
- **AND** no key matches `model.layers.*.mlp.gate_proj.weight` (the dense MLP keys don't exist on a MoE host)

#### Scenario: dense walker output unchanged

- **GIVEN** a `LlamaForCausalLM` / `Qwen2ForCausalLM` / `Qwen3ForCausalLM`
  (dense) instance
- **WHEN** `LlamaAdapter().walk(host, projector)` is invoked
- **THEN** no key in the output starts with `model.layers.*.mlp.gate.weight` or `model.layers.*.mlp.experts.`
- **AND** the keyset is byte-identical to the pre-this-change dense walker output

### Requirement: forged MoE MLP applies softmax-then-topk routing with optional renormalization

The `Qwen3MoEMLP` class inside `build_llama_family_module` SHALL implement the forward pass per the HF Qwen3MoE reference: compute gate logits via `nn.Linear(hidden, num_experts, bias=False)`, apply softmax in fp32 over the expert axis, take top-K, optionally renormalize the top-K weights to sum to 1 (gated by `cfg.norm_topk_prob`), and route each token to its top-K experts with the weighted sum accumulated via `index_add_`.

The implementation MAY use a naive `for e in range(num_experts)` loop for v1. A fused scatter-add path is a separate optimization (`moe-fused-dispatch`).

The expert MLP (`Qwen3Expert`) SHALL be a SwiGLU implementation with `cfg.moe_intermediate_size` inner dim:
`F.silu(gate_proj(x)) * up_proj(x)` → `down_proj(...)`.

#### Scenario: forged Qwen3-MoE block constructs gate + experts

- **GIVEN** a `NativeModelConfig` with `family="qwen3_moe"`,
  `num_experts=4`, `num_experts_per_tok=2`, `moe_intermediate_size=128`,
  `hidden_size=128`
- **WHEN** `build_llama_family_module(cfg)` is instantiated and
  `model.layers[0].mlp` is inspected
- **THEN** `mlp.gate` is `nn.Linear(128, 4, bias=False)`
- **AND** `mlp.experts` is an `nn.ModuleList` of length 4
- **AND** each expert has `gate_proj`, `up_proj`, `down_proj` submodules
  with the right shapes

#### Scenario: routing weights sum to 1 when norm_topk_prob is True

- **GIVEN** a forged Qwen3-MoE module with `norm_topk_prob=True`
- **WHEN** an instrumented forward pass logs the top-K weights for each
  token
- **THEN** each token's top-K weights sum to 1.0 within fp32 tolerance

#### Scenario: dense-family forward unchanged

- **GIVEN** a `NativeModelConfig` with `num_experts=0` (any non-MoE family)
- **WHEN** the module is instantiated
- **THEN** `model.layers[0].mlp` is a `SwiGLU_MLP` instance (the existing
  dense MLP), not a `Qwen3MoEMLP`
- **AND** the forward pass output is bit-identical to the pre-this-change
  forward for the same input

### Requirement: ForgePipeline supports preserve, collapse, and top_n MoE strategies

`ForgePipeline.moe_strategy: Literal["preserve", "collapse", "top_n"]` SHALL default to `"preserve"`. When set to:

- `"preserve"`: every host expert is projected independently. The forged config has `num_experts=host.config.num_experts`. Behavior fidelity is full.
- `"collapse"`: expert weights are averaged into a single dense MLP per layer; the router is removed. The forged config has `num_experts=0` and `intermediate_size=moe_intermediate_size` (the per-expert size). Behavior is degraded; documented as "experimental — averages experts."
- `"top_n"`: in v1, this SHALL raise `NotImplementedError` with a message pointing at the `moe-expert-calibration` follow-up issue. The validation (`__post_init__` requires `moe_keep_n > 0`) is shipped so the follow-up doesn't need to reopen the contract.

`ForgePipeline.__post_init__` SHALL validate that `moe_strategy="top_n"`
requires `moe_keep_n > 0` (else `ValueError`).

#### Scenario: preserve strategy keeps expert count

- **GIVEN** a Qwen3-MoE host with `num_experts=128` and
  `ForgePipeline(moe_strategy="preserve", ...)`
- **WHEN** the forge run produces a native config
- **THEN** `forged_cfg.num_experts == 128`
- **AND** `forged_cfg.num_experts_per_tok == host.config.num_experts_per_tok`

#### Scenario: collapse strategy produces dense forged

- **GIVEN** a Qwen3-MoE host and `ForgePipeline(moe_strategy="collapse", ...)`
- **WHEN** the forge run produces a native config
- **THEN** `forged_cfg.num_experts == 0`
- **AND** `forged_cfg.family == "qwen3"` (downgraded from `"qwen3_moe"`
  since the forged behavior is dense)
- **AND** `forged_cfg.intermediate_size == host.config.moe_intermediate_size`
- **AND** the walker emits averaged `mlp.gate_proj.weight` /
  `up_proj.weight` / `down_proj.weight` keys instead of per-expert keys

#### Scenario: top_n strategy raises NotImplementedError

- **GIVEN** `ForgePipeline(moe_strategy="top_n", moe_keep_n=8, ...)`
- **WHEN** the forge run starts
- **THEN** `NotImplementedError` is raised
- **AND** the message contains `"moe-expert-calibration"` (the
  follow-up issue name)

### Requirement: NVIDIA smoke script exists and is runnable on the target tier

A `scripts/smoke_qwen3_moe.py` script SHALL exist in the repository, exit with code `0` and `SMOKE OK` final-line when run against a real `Qwen3MoeForCausalLM` on an appropriately-sized NVIDIA GPU (≥80GB recommended). The script SHALL:

1. Verify `transformers >= 4.51` (Qwen3-MoE availability); exit code 2 with a clear message otherwise.
2. Verify CUDA availability; exit code 2 with a clear message otherwise.
3. Load the host via `device_map="auto"` and `dtype=torch.bfloat16`.
4. Confirm `adapter_for(host).family == "qwen3_moe"`.
5. Walk the host; confirm the emitted dict has the expected key shape
   (`num_experts * 3 + 1` MLP keys per layer, plus all inherited Qwen3
   attention keys).
6. Build the forged native module; confirm each block's `mlp` is a
   `Qwen3MoEMLP` with `gate` and `experts` of correct sizes.
7. Run one forward pass; confirm output shape and finite logits.
8. Optionally (under `--log-expert-utilization`), log top-K routing
   decisions per layer and compare to host's routing decisions on the
   same input.

The script SHALL accept `--host-model` to substitute any future
Qwen3-MoE release; `--n-features` to vary the basis size;
`--device-map` to override the default `"auto"`.

#### Scenario: script fails gracefully on environments missing prerequisites

- **GIVEN** an Intel Mac box with `transformers < 4.51`
- **WHEN** `python scripts/smoke_qwen3_moe.py` is invoked
- **THEN** the script exits with code 2
- **AND** stderr contains a clear message naming the
  `transformers >= 4.51` requirement
- **AND** no host load or projection is attempted

#### Scenario: script SMOKE OK on NVIDIA

- **GIVEN** an NVIDIA box with `transformers >= 4.51`,
  CUDA available, and ≥80GB GPU memory
- **WHEN** `python scripts/smoke_qwen3_moe.py` is invoked with the
  default `--host-model Qwen/Qwen3-30B-A3B-Base`
- **THEN** the script exits with code 0
- **AND** stdout ends with `SMOKE OK`
- **AND** the printed forged-module summary confirms 128 experts per
  layer (or whatever the host's actual `num_experts` is)
