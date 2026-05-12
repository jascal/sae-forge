# qwen3-dense-support Specification

## Purpose

The `qwen3-dense-support` capability adds the `Qwen3ForCausalLM`
architecture to the forge's supported families. Qwen3 dense is
Llama-shaped (SwiGLU MLP, RMSNorm, GQA, RoPE, no Q/K/V biases) plus
one structural addition: per-head RMSNorm on Q and K, applied between
the per-head reshape and the scaled dot-product. The capability pins
the walker's pass-through emission of `q_norm` / `k_norm` weights,
the `qk_norm` field auto-detection rule in `Qwen3Adapter`, and the
forged attention block's forward contract when `cfg.qk_norm=True`.

This capability inherits hybrid-bridge support from
`hybrid-bridge-llama-family` (Qwen3 routes through
`build_llama_family_module`) and Q/K/V-bias auto-detection from
`LlamaAdapter.build_native_config` (Qwen3 dense has no biases →
`qkv_bias=False` is auto-set, no Qwen3-specific bias code).

## ADDED Requirements

### Requirement: Qwen3Adapter detects qk_norm from the host's first attention block

`Qwen3Adapter.build_native_config(host, n_features)` SHALL produce a `NativeModelConfig` with `family="qwen3"` and `qk_norm=True` iff `host.model.layers[0].self_attn.q_norm` is a non-None submodule. When the attribute is absent, `qk_norm` SHALL be False.

Detection SHALL run on the actual `nn.Module` (not on the HF config
flag), to be robust across HF transformers versions and Qwen3
variants where the config attribute name may differ.

#### Scenario: Qwen3 host produces qk_norm=True

- **GIVEN** a real `Qwen3ForCausalLM` instance whose
  `model.layers[0].self_attn.q_norm` is a `Qwen3RMSNorm` submodule
- **WHEN** `Qwen3Adapter().build_native_config(host, n_features=64)` is invoked
- **THEN** the returned config has `family="qwen3"`
- **AND** `qk_norm` is True
- **AND** `qkv_bias` is False (Qwen3 has no Q/K/V biases — inherited
  from the LlamaAdapter auto-detection)

#### Scenario: Llama host produces qk_norm=False (regression gate)

- **GIVEN** a real `LlamaForCausalLM` instance
- **WHEN** `LlamaAdapter().build_native_config(host, n_features=64)` is invoked
- **THEN** the returned config has `qk_norm=False`
- **AND** no Qwen3-specific behavior leaks into the Llama path

### Requirement: walker emits q_norm and k_norm pass-through when host has them

`LlamaAdapter.walk(host, projector)` SHALL emit, for every block index `i` and for every `qk` in `("q_norm", "k_norm")` where `getattr(host.model.layers[i].self_attn, qk, None) is not None`, a key `model.layers.{i}.self_attn.{qk}.weight` with the host's tensor unprojected. The tensor's shape SHALL be `(head_dim,)` — head-dim aligned, not residual-aligned.

When the attribute is absent (Llama / Gemma-2 / Qwen2 hosts), no
key SHALL be emitted. The presence-or-absence of these keys
SHALL exactly mirror the host's actual submodule structure.

#### Scenario: Qwen3 walker emits q_norm and k_norm at every block

- **GIVEN** a Qwen3 host with `num_hidden_layers=4`
- **WHEN** `LlamaAdapter().walk(host, projector)` is invoked
- **THEN** the returned dict contains keys
  `model.layers.{0..3}.self_attn.q_norm.weight` and
  `model.layers.{0..3}.self_attn.k_norm.weight`
- **AND** each value is a 1-D array of length `head_dim`

#### Scenario: Llama walker does not emit q_norm or k_norm

- **GIVEN** a Llama host (no `q_norm` submodule)
- **WHEN** `LlamaAdapter().walk(host, projector)` is invoked
- **THEN** no key in the returned dict matches `*.self_attn.q_norm.weight`
- **AND** no key in the returned dict matches `*.self_attn.k_norm.weight`
- **AND** the keyset is byte-identical to the pre-this-change walk

### Requirement: forged attention block applies q_norm / k_norm when cfg.qk_norm is True

The `LlamaSelfAttention` class inside `build_llama_family_module` SHALL, when `cfg.qk_norm=True`, construct two `RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps or 1e-6)` submodules at `self.q_norm` and `self.k_norm`. The `forward` method SHALL apply `self.q_norm(q)` and `self.k_norm(k)` between the per-head reshape (`q.view(..., num_heads, head_dim).transpose(-3, -2)`) and the scaled dot-product (`scores = q @ k.transpose(-2, -1) / sqrt(head_dim)`).

When `cfg.qk_norm=False` (the default), `self.q_norm` and `self.k_norm`
SHALL be `None`, the forward path SHALL skip the conditional, and the
output SHALL be byte-identical to the pre-this-change forward.

#### Scenario: forged Qwen3 attention block applies qk_norm

- **GIVEN** a `NativeModelConfig` with `family="qwen3"`, `qk_norm=True`,
  `head_dim=32`, `rms_norm_eps=1e-6`
- **WHEN** `build_llama_family_module(cfg)` is instantiated and its
  `model.layers[0].self_attn` is inspected
- **THEN** `self_attn.q_norm` is an `RMSNorm` with `eps=1e-6` and
  parameter shape `(32,)`
- **AND** `self_attn.k_norm` is the same shape
- **AND** an instrumented forward call records both `q_norm` and
  `k_norm` being applied exactly once per block per token

#### Scenario: Llama / Gemma-2 / Qwen2 forged attention skips qk_norm

- **GIVEN** a `NativeModelConfig` with `family ∈ {"llama","gemma2","qwen2"}`
  and `qk_norm=False` (the default)
- **WHEN** the module is instantiated
- **THEN** `self_attn.q_norm` is `None`
- **AND** `self_attn.k_norm` is `None`
- **AND** the forward pass produces output bit-identical to the
  pre-this-change forward for the same input

### Requirement: Qwen3 hybrid-bridge works via shared Llama-family factory

Qwen3 hosts SHALL inherit working hybrid-bridge support from the `hybrid-bridge-llama-family` capability (the bridge insertion is in `LlamaTransformer.__init__` / `forward`, which Qwen3 builds through). No Qwen3-specific bridge code is added.

The `hybrid-bridge-llama-family` family-coverage requirement
("every Llama-family host that supports `hybrid_bridge` has an
integration test") SHALL be satisfied for Qwen3 by adding
`tests/integration/test_hybrid_bridge_qwen3.py`, structurally
identical to the existing Qwen2 file.

#### Scenario: Qwen3 hybrid forge produces bridges in state_dict

- **GIVEN** an untied 4-layer Qwen3 fixture and a shape-compatible
  hybrid bundle
- **WHEN** the forged Qwen3 module is constructed via
  `NativeModel.from_projected_weights(cfg_with_bridges_and_qk_norm, weights)`
- **THEN** `model.state_dict()` contains keys prefixed
  `model.bridges.emb_mid.` and `model.bridges.mid_lm.`
- **AND** the state_dict also contains
  `model.layers.{i}.self_attn.q_norm.weight` and the k_norm analog
  at every layer
- **AND** save/load round-trips both groups bit-for-bit

### Requirement: Qwen3Adapter registration is silent under transformers < 4.51

`saeforge/adapters/qwen3.py` SHALL register `Qwen3ForCausalLM` inside a `try` / `except ImportError` block. When the host environment cannot import `Qwen3ForCausalLM` (transformers < 4.51, or a custom build without Qwen3), the registration SHALL be silently skipped — no warning, no error, no side effect.

This SHALL allow the `[intel]` extras install (transformers capped at
< 4.50) to load `saeforge.adapters` cleanly without Qwen3 support.
The `registered_classes` API SHALL omit Qwen3 in that environment.
Adapter dispatch on a Qwen3 host (impossible from such an environment
because the host class itself cannot be imported) is therefore not
exercisable from `[intel]`.

#### Scenario: import is silent under old transformers

- **GIVEN** a venv with `transformers < 4.51` installed
- **WHEN** `import saeforge.adapters.qwen3` is executed
- **THEN** the import succeeds with no warnings
- **AND** the `Qwen3Adapter` class is defined (it doesn't depend on
  the HF class for its definition, only for registration)
- **AND** `saeforge.adapters.registered_classes()` does not contain
  `Qwen3ForCausalLM`

#### Scenario: register cleanly under new transformers

- **GIVEN** a venv with `transformers >= 4.51`
- **WHEN** `import saeforge.adapters.qwen3` is executed
- **THEN** `Qwen3ForCausalLM` is in `saeforge.adapters.registered_classes()`
- **AND** `adapter_for(Qwen3ForCausalLM(...))` returns a `Qwen3Adapter`
  instance with `family == "qwen3"`
