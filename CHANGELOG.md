# Changelog

All notable changes to sae-forge are tracked here. v0 entries land as
their corresponding OpenSpec change is archived.

## [0.2.4] — 2026-05-07

### Added

- **`SubspaceProjector(scale_boost="auto")`** ([PR #8](../../pull/8))
  resolves to `min(1.0, d_model / n_features)` — a defensible
  starting heuristic for over-complete bases (`n_features > d_model`).
  For under/equal-complete bases the heuristic returns `1.0`
  (identity-preserving). Existing positive-float values are
  unchanged; the default remains `1.0`.
- **`--scale-boost` CLI flag** on `examples/forge_gemma2_2b.py` and
  `examples/forge_synthetic_llama.py` (both default to `"auto"`).
  `examples/forge_gpt2_real_sae.py` adds a `scale_boost` function
  parameter (notebook-driven, no argparse).

### Fixed

- **Silent footgun on over-complete bases** ([PR #8](../../pull/8)).
  Empirical anchor surfaced during a Gemma-2-2B forge attempt:
  GPT-2 (`d_model=768`) with a 1024-feature basis required
  `scale_boost ≈ 0.25` for stable training; the default `1.0` was
  too large and silently produced NaNs / saturated softmax / KL
  explosion. Construction now emits a `UserWarning` when
  `n_features > d_model` and `scale_boost == 1.0`, naming the
  empirical anchor and pointing at `"auto"` or a hand-picked
  value as the next step. Suppressed when an explicit numeric
  or `"auto"` is supplied — no scolding when the user acted
  intentionally.

## [0.2.3] — 2026-05-07

### Fixed

- **Grad checkpointing crashed on Llama / Gemma-2 hosts**
  ([PR #7](../../pull/7)). `saeforge/training/loop.py:_enable_grad_checkpointing`
  hardcoded GPT-2 submodule names (`module.transformer.h`,
  `module.transformer.wte.weight`); ForgedLlama (used by both
  `family="llama"` and `family="gemma2"`) exposes
  `module.model.layers` and `module.model.embed_tokens.weight`. Any
  `--grad-checkpoint` run on a non-GPT-2 host raised
  `'ForgedLlama' object has no attribute 'transformer'` inside the
  FSM. Fix: adapter-driven layout via a new
  `ArchitectureAdapter.grad_checkpoint_targets(module)` method with
  per-family overrides; `_enable_grad_checkpointing` dispatches via
  a new `adapter_for_family(family_str)` registry helper.

- **FSM failures surfaced as silent KL=0.0 returns**
  ([PR #7](../../pull/7)). When an FSM action raised, the failure was
  swallowed into `final_state: "failed"` and `ForgePipeline.run()`
  returned a `ForgeResult` with `n_params=0`, `faithfulness_kl=0.0`,
  exit code 0 — no diagnostic signal. Fix: new
  `saeforge.ForgeFailed` exception (subclass of `RuntimeError`) with
  `error_message`, `transitions_log`, and `extras` attached; both
  FSM dispatch paths (`_run_real_fsm`, `_run_synthetic_fsm`) raise
  it after `run_machine` when the trailing transition is `log_error`.

### Added

- **`saeforge.ForgeFailed`** exception ([PR #7](../../pull/7)) —
  re-exported from the top-level package; subclass of `RuntimeError`
  so existing exception handlers don't change shape.
- **`saeforge.adapters.adapter_for_family(family_str)`** helper —
  for code paths that have only the `NativeModelConfig.family`
  string in hand (e.g. inside the training loop, where the host
  class is already gone).
- **`ArchitectureAdapter.grad_checkpoint_targets(module)`** —
  abstract-with-default-NotImplementedError on the ABC; per-family
  overrides return `(blocks, embedding_param)` for activation
  checkpointing.

## [0.2.2] — 2026-05-07

### Fixed

- **Fine-tune recipe now runs on real-host `ForgePipeline.run()`**
  ([PR #6](../../pull/6)). Since v0.3 forge-finetune-recipe landed,
  `run()` against a real HF host (`host_model_id` set) silently
  dropped every `finetune_*` field on the floor — the recipe was
  wired into the FSM action only, but `run()` always took the
  imperative path. The headline `examples/forge_gemma2_2b.py`
  documented a 1k-step fine-tune flow that had never executed.
  `run()` now branches on `self.orchestrator`:
  `"fsm"` routes through a new `_run_real_fsm` mirroring the
  synthetic FSM path; `"imperative"` (the default) emits a
  `UserWarning` when `finetune_corpus` is set so the silent skip
  cannot recur. `examples/forge_gemma2_2b.py` sets
  `orchestrator="fsm"` when `--steps > 0`.

### Added

- **`ForgePipeline.run(finetune_iterator=...)`** ([PR #6](../../pull/6))
  — pre-built iterator bypasses the `AutoTokenizer + datasets`
  round-trip the recipe action would do via `finetune_corpus`.
  Mirrors the existing `run_synthetic` kwarg.

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
