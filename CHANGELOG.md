# Changelog

All notable changes to sae-forge are tracked here. v0 entries land as
their corresponding OpenSpec change is archived.

## [0.2.1] — 2026-05-07

### Fixed

- **`NativeModel.save_pretrained` / `load_pretrained` round-trip on
  tied-embedding hosts** ([PR #5](../../pull/5)). The `ForgedLlama`
  constructor aliases `lm_head.weight` to `model.embed_tokens.weight`
  when `config.tied_embeddings` is True (Gemma-2 default + tied
  Llama configs), but `safetensors.torch.save_file` rejects
  shared-storage tensors. The fix drops `lm_head.weight` from the
  saved state_dict when tied; `load_pretrained` reconstructs the
  alias via the constructor and relaxes `load_state_dict(strict=False)`
  for the missing slot. Without this fix the Gemma-2-2B forge crashed
  at stage 4 save, after polygram + projection had already succeeded.

- **`examples/forge_gemma2_2b.py` SAE filename templating**
  ([PR #5](../../pull/5)). The previous hard-coded `average_l0_71`
  doesn't exist for layer 12 of `google/gemma-scope-2b-pt-res`
  (layer 12 publishes `{22, 41, 82, 176, 445}`). New `--l0` flag
  (default 82) templates into the `SAE_FILE_TEMPLATE` path.

## [0.2.0] — 2026-05-07

### Added (multi-architecture-support)

- **`saeforge/adapters/` package** — registry-based dispatch from HF
  model class to a `ArchitectureAdapter` whose contract is `walk` +
  `build_native_config` + `native_module_class`. Bundled adapters cover
  `GPT2LMHeadModel`, `GPT2Model`, `LlamaForCausalLM`, and
  `Gemma2ForCausalLM`. Unregistered architectures raise
  `NotImplementedError` naming the offending type and the registered
  set.
- **Llama-3 / Llama-2 support** — Q/K/V/O proj, SwiGLU MLP
  (gate/up/down), GQA via `num_key_value_heads`, RMSNorm γ, optional
  tied embeddings.
- **Gemma-2 support** — Llama-shaped + the two extra per-layer
  RMSNorms (`pre_feedforward_layernorm`, `post_feedforward_layernorm`)
  and post-`lm_head` `tanh(x / cap) * cap` soft-capping. Sliding-window
  alternating attention is NOT replicated in v0.2 (accepted as
  `ε_attn` per `docs/algorithm.md` §5).
- **`examples/forge_synthetic_llama.py`** — runs the full Llama
  forge pipeline against a tiny synthetic host with no HF token
  requirement; useful for CI and laptops.
- **Tests** — 22 new tests in `tests/test_architecture_adapters.py`
  covering registry dispatch, walker shape audits (incl. GQA), tied
  embeddings, four-norm Gemma-2 layout, soft-cap config passthrough,
  the no-randomly-initialised-weight invariant, and family-field
  validation. Plus `test_examples_smoke.py` (synthetic-Llama
  end-to-end smoke + Gemma-2 skip-if-unreachable) and
  `test_forge_pipeline_unregistered_arch.py`.

### Changed (Breaking — multi-architecture-support)

- **`NativeModelConfig.family: str` is now required** with no default.
  Valid values are `"gpt2"`, `"llama"`, `"gemma2"`. The pre-change
  config silently produced a GPT-2-shaped module for any inputs;
  forcing an explicit family removes the silent footgun. Callers
  migrate by adding `family="gpt2"` to existing `NativeModelConfig(...)`
  calls.
- **`NativeModelConfig` gains `n_kv_heads`, `tied_embeddings`,
  `rms_norm_eps`, `final_logit_softcap`, `attn_logit_softcap`** for
  the Llama / Gemma-2 paths. Defaults preserve the GPT-2 behaviour
  (`n_kv_heads=None` collapses to `num_heads` at `__post_init__`;
  the soft-caps default to `None` and are no-ops).
- **`SubspaceProjector.project_module`** now dispatches via the
  adapter registry instead of a hard-coded GPT-2 walker. The GPT-2
  walk semantics are unchanged. Unregistered architectures raise a
  registry-aware `NotImplementedError`; the v0.1 `"GPT-2"`-prefixed
  message is gone.
- **`ForgePipeline.run`** loads the host via
  `transformers.AutoModelForCausalLM.from_pretrained` (was
  `GPT2LMHeadModel.from_pretrained`). Non-GPT-2 hosts now load as
  their actual class — the pre-change path silently produced a
  randomly-initialised GPT-2 for any non-GPT-2 host and is the bug
  this change fixes.

### Out of scope

- **Pythia / GPT-NeoX** — deferred; needs a parallel Q/K/V upstream
  addition in polygram.
- **Gemma-2 sliding-window alternating attention** — replicating the
  exact attention pattern is future work; the native module uses the
  standard causal mask everywhere.

## [0.1.0] — 2026-05-07

### Added (forge-polygram-tuning-passthrough)

- Three typed polygram-tuning fields on `ForgePipeline`:
  `compression: CompressionConfig | None`,
  `epoch_compression: EpochCompressionConfig | None`,
  `regrow: RegrowConfig | None`. Each round-trips through the FSM
  context as a JSON-friendly dict (`cfg.to_dict()` →
  `<Config>.from_dict(ctx[key])`).
- `ForgePipeline.from_dict(data)` classmethod for YAML/JSON config
  loading; emits `UserWarning` for unknown top-level keys.
- New CLI flags: `--coverage-target`, `--cosine-threshold`,
  `--max-compress-iterations`, `--regrow-count`, `--regrow-layer`,
  `--regrow-strategy`. Long-tail tuning lives behind
  `ForgePipeline.from_dict`.
- `docs/forge_config_example.yaml` showing the
  `ForgePipeline.from_dict(yaml.safe_load(...))` shape end-to-end.
- `tests/test_polygram_tuning_passthrough.py` (15 tests) and
  `tests/test_cli.py` (5 tests).

### Changed (Breaking — forge-polygram-tuning-passthrough)

- **Removed flat `compression_strategy` and `rep_selection` fields
  on `ForgePipeline`.** Passing either now raises `TypeError` at
  construction. Migrate to
  `compression=CompressionConfig(strategy=..., rep_selection=...)`.
- **`regrow_count > 0` requires explicit `regrow=RegrowConfig(...)`.**
  `__post_init__` raises `ValueError` otherwise.
- **`perform_regrowth` action requires `ctx["regrow"]`** when
  `regrow_count > 0`. The previous `ctx.get("regrow_layer", 10)` and
  `ctx.get("host_model_id") or "gpt2"` fallbacks were removed in
  lock-step with polygram 0.1.0 dropping the matching defaults from
  `Regrower.from_compression_report`.
- Pinned `polygram>=0.1.0` (was `>=0.0.1`).

### Migration

- Replace `ForgePipeline(compression_strategy="merge",
  rep_selection="scale_aware", ...)` with
  `ForgePipeline(compression=CompressionConfig(strategy="merge",
  rep_selection="scale_aware"), ...)`.
- Callers with `regrow_count > 0` now must pass
  `regrow=RegrowConfig(model_name=<host>, layer=<int>)`. Layer is
  host-specific and no longer has a GPT-2 default.

### Internal

- Two pre-existing CI tests (`test_forge_pipeline_run_requires_host_model_id`,
  `test_project_module_unsupported_arch_raises`) gated with
  `pytest.importorskip("torch")` so the no-extras CI install stays
  green.

### Added

- Repository scaffolding: `pyproject.toml`, `README.md`, `AGENTS.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, CI workflow,
  `saeforge/` package skeleton with stub `FeatureBasis`,
  `SubspaceProjector`, `NativeModel`, `ForgePipeline`, and `cli.main`.
- OpenSpec change `bootstrap-package` defining the v0 milestone.
