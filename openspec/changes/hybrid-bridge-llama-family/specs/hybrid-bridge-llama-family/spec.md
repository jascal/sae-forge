# hybrid-bridge-llama-family Specification

## Purpose

The `hybrid-bridge-llama-family` capability extends the `hybrid-bridge-forge`
mechanism (#18, on `main`) to the Llama-family native module factory
(`saeforge/adapters/llama.py::build_llama_family_module`). It pins the
bridge-insertion contract for every host architecture that builds through
that factory: Llama, Gemma-2, Qwen2, and any future family added to
`_SUPPORTED_FAMILIES` whose native module routes through
`build_llama_family_module`.

This capability does not redefine the bridge mechanism itself — that
contract lives in `hybrid-bridge-forge` and is family-generic. This
capability closes the *implementation* gap where `cfg.bridges=True` was
silently dropped for non-GPT-2 hosts.

## ADDED Requirements

### Requirement: LlamaTransformer constructs bridges when cfg.bridges is True

`saeforge.adapters.llama.LlamaTransformer.__init__` SHALL construct two `BridgeModule` instances when `cfg.bridges` is True, exposed as `self.bridges` of type `nn.ModuleDict` with keys `emb_mid` and `mid_lm`. Each bridge is constructed from `cfg.hidden_size` and a `BridgeConfig` populated from `cfg.bridge_init` / `cfg.bridge_nonlin` / `cfg.bridge_pre_layernorm`. When `cfg.bridges` is False, `self.bridges` SHALL be `None` and no bridge parameters appear in the module's `state_dict`.

The bridge construction SHALL mirror the GPT-2 implementation
(`saeforge.adapters.gpt2.ForgedGPT2.Transformer._build_bridges`) in shape
and semantics. Differences in submodule attribute paths
(`transformer.bridges.*` vs `model.bridges.*`) follow each host's HF naming
convention and are not part of this capability's contract.

#### Scenario: bridges present when cfg.bridges is True

- **GIVEN** a `NativeModelConfig` with `family="llama"`, `bridges=True`,
  `bridge_init="orthogonal"`, `bridge_nonlin="none"`,
  `bridge_pre_layernorm=True`, and `num_layers=4`
- **WHEN** `build_llama_family_module(cfg)` is invoked and an instance is
  constructed
- **THEN** the resulting module's `transformer.bridges` (the inner
  `LlamaTransformer.bridges` attribute) is an `nn.ModuleDict`
- **AND** it contains keys `emb_mid` and `mid_lm`
- **AND** each value is a `BridgeModule` instance with linear weight of
  shape `(hidden_size, hidden_size)`
- **AND** the module's `state_dict` contains entries with keys starting
  with `model.bridges.emb_mid.` and `model.bridges.mid_lm.`

#### Scenario: bridges absent when cfg.bridges is False

- **GIVEN** a `NativeModelConfig` with `family="llama"` and `bridges=False`
- **WHEN** an instance is constructed
- **THEN** `model.bridges` is `None`
- **AND** the module's `state_dict` contains no keys prefixed with
  `model.bridges.`
- **AND** the module's forward pass produces output bit-identical to the
  pre-this-change Llama-family forward (regression gate)

### Requirement: LlamaTransformer.forward calls bridges at indices 0 and L-2 when present

`LlamaTransformer.forward` SHALL apply `self.bridges["emb_mid"]` after block `0` and `self.bridges["mid_lm"]` after block `L-2`, where `L = len(self.layers)`. Both insertions SHALL be gated by `L >= 3`; for `L < 3` the forward path SHALL be byte-identical to the no-bridge path (the bridges are constructed but not called).

The bridge call sites SHALL be inside the existing per-block loop,
between `x = layer(x)` and the next iteration. No new tensor allocation
or device movement is introduced; the bridge operates on the residual
stream in-place in the algebraic sense (creates a new tensor as
PyTorch idiom, but matches the existing memory pattern).

#### Scenario: bridges applied at correct indices on a 4-layer host

- **GIVEN** an instantiated `LlamaTransformer` with `len(self.layers)==4`
  and `self.bridges` populated
- **WHEN** `forward(input_ids)` is invoked and instrumented to record
  every `BridgeModule.forward` call
- **THEN** `emb_mid` is called exactly once (after block 0's forward
  returns)
- **AND** `mid_lm` is called exactly once (after block 2's forward
  returns; block 3 is L-1, the lm-head-region block)
- **AND** the calls happen in the order `emb_mid` → block-1 → block-2 →
  `mid_lm` → block-3

#### Scenario: bridges skipped on too-shallow host

- **GIVEN** an instantiated `LlamaTransformer` with `len(self.layers)==2`
  and `self.bridges` populated (an edge case: bundle construction would
  have rejected this, but a manually-built config could reach here)
- **WHEN** `forward(input_ids)` is invoked
- **THEN** neither `emb_mid` nor `mid_lm` is called
- **AND** the output is identical to the forward pass with `bridges=None`

### Requirement: every Llama-family host that supports hybrid_bridge has an integration test

Every host family routed through `build_llama_family_module` and supported under `_SUPPORTED_FAMILIES` SHALL have a corresponding integration test exercising the end-to-end hybrid-bridge forge path. As of this change, the in-scope
families are `llama` and `qwen2`, each with its own test file
under `tests/integration/`. `gemma2` is explicitly deferred (M4
follow-up) but the capability requires the test to be added when
Gemma-2 bridge support is exercised against real Gemma-2 weights.

The required test cases per family are:

1. Construction with bridges produces a `state_dict` containing
   `model.bridges.*` entries.
2. Forward pass produces finite logits.
3. `save_pretrained` + `load_pretrained` preserves bridges bit-for-bit.
4. Tied-embedding host with `hybrid_bridge=True` raises the documented
   error.
5. `hybrid_bridge=False` produces a `state_dict` byte-identical to the
   pre-this-change forge for the same host.

Adding a new family routed through `build_llama_family_module` (a future
`qwen3-dense-support` host, for example) SHALL be accompanied by an
integration test file with the same five cases against an in-memory
fixture of that family. CI is the gate; the test file's absence is a
review-time block.

#### Scenario: Llama integration test asserts state-dict bridge keys

- **GIVEN** an untied 4-layer `tiny_llama` fixture and a
  shape-compatible hybrid bundle
- **WHEN** the forged Llama module is constructed via
  `NativeModel.from_projected_weights(cfg_with_bridges=True, weights)`
- **THEN** `model.state_dict()` contains at least one key prefixed
  `model.bridges.emb_mid.`
- **AND** at least one key prefixed `model.bridges.mid_lm.`
- **AND** the corresponding tensors have shape
  `(hidden_size, hidden_size)`

#### Scenario: Qwen2 integration test confirms qkv_bias + bridges combination

- **GIVEN** an untied 4-layer Qwen2 fixture (with qkv_bias=True) and a
  shape-compatible hybrid bundle
- **WHEN** the forged Qwen2 module is constructed
- **THEN** the `state_dict` contains BOTH the Q/K/V bias entries
  (`model.layers.0.self_attn.q_proj.bias` etc.) AND the bridge entries
  (`model.bridges.emb_mid.*`, `model.bridges.mid_lm.*`)
- **AND** save + load round-trips both groups bit-for-bit

### Requirement: GPT-2 hybrid-bridge surface is unchanged by this change

`saeforge.adapters.gpt2.ForgedGPT2.Transformer.forward` and its bridge insertion at indices `0` and `L-2` (under the `transformer.bridges.*` key prefix) SHALL be unchanged by this capability. The existing `tests/integration/test_hybrid_bridge_gpt2.py` suite SHALL continue to pass without modification.

#### Scenario: GPT-2 integration suite passes unmodified

- **GIVEN** the pre-this-change `tests/integration/test_hybrid_bridge_gpt2.py`
- **WHEN** the suite is run against the post-this-change tree
- **THEN** every test passes without source modification
- **AND** no test in that file is skipped or xfailed
