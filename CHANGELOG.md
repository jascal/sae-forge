# Changelog

All notable changes to sae-forge are tracked here. v0 entries land as
their corresponding OpenSpec change is archived.

## [Unreleased]

### Added (add-pareto-sweep-driver)

- **Pareto sweep driver.** New `saeforge sweep-pareto` CLI subcommand
  and `ForgePipeline.sweep_pareto()` method that forge across per-K
  materialised SAE checkpoints produced by
  `polygram compress --pareto --pareto-materialize`. Optionally spans
  multiple labelled encodings (e.g. MPS vs Rung4) — pass
  `--encoding LABEL:PATH` repeatedly. Emits one JSONL row per
  `(encoding, target_n_features_kept)` capturing kept-feature count,
  downstream KL, perplexity, fine-tune loss, and elapsed seconds.
  The load-bearing primitive for Axis 4 of polygram's rung-viability
  methodology — end-to-end downstream confirmation that the Axis 1
  compression-coverage lift cashes out in forged-model KL space.
- **Three lifecycle states per row.** *Success* (forge ran),
  *frontier-only* (`--frontier-only` flag, no forge), and
  *row failure* (forge raised). Downstream consumers filter on
  `error_message is None` before reading metric fields. Failure rows
  are recorded with `error_message` populated; the sweep continues to
  the next row.
- **Resumable.** `frontier.jsonl` is append-only; rerunning the sweep
  skips already-completed `(label, K)` pairs. Truncated last lines
  (mid-write crashes) are detected, dropped, and rewritten on the
  next invocation. No lockfiles or sentinel files.
- **`--frontier-only` mode** emits manifest-derived columns only
  (`target_n_features_kept`, `n_features_kept_actual`,
  `pareto_reached_target`) without invoking the forge — cheap
  exploratory triage. Pipe through `jq` to find candidate K values
  before committing forge compute. Falls back to non-zero-row counting
  on the SAE checkpoint when `pareto.json` is absent.
- **`ParetoFrontierRow` dataclass** exported from `saeforge`, with
  `to_json_dict` / `from_json_dict` round-trip. Schema documented in
  the `pareto-sweep` capability spec.
- **Polygram pin bumped to `>=0.4.0`.** The new
  `CompressionConfig.target_n_features_kept` and `score_field` fields
  flow through the existing `_ConfigMixin.to_dict/from_dict` ctx
  round-trip in `polygram-tuning-passthrough` with no sae-forge-side
  code change — `Compressor` dispatches to `plan_with_target` when
  the field is set.
- **No FSM change.** The sweep is a flat Python loop; each row's
  forge call uses the existing `StreamMachine → RefineMachine →
  BasisMachine` hierarchy. The driver hot-swaps `pipeline.basis` and
  `pipeline.projector` per row via a context manager that restores
  the originals afterwards.
- Tests: `tests/test_sweep.py` — 27 tests covering row validation +
  JSON round-trip, manifest parsing, checkpoint enumeration (both
  `pareto/` subdir and flat layouts), multi-K sweep, resumability,
  multi-encoding, per-row failure isolation, retry-on-next-sweep,
  frontier-only with and without manifest, CLI argument parsing,
  and a `--frontier-only` end-to-end CLI smoke.

### Added (add-host-distillation-finetune-loss)

- **Host distillation in fine-tune.** `TrainingConfig` gains
  `distill_alpha` (default 1.0 = pure LM-CE, byte-identical to
  v0.3) and `distill_temperature` (default 2.0). When
  `distill_alpha < 1.0`, the loss becomes
  `α·CE(corpus) + (1-α)·τ²·KL(host ‖ forged)` — Hinton-style
  soft-label distillation with the same KL direction as
  `faithfulness_kl` (so the training objective matches the eval
  metric). The host forward runs under `torch.no_grad()` in the
  same autocast context as the student.
- **`ForgePipeline` exposes the same knobs** as
  `finetune_distill_alpha` / `finetune_distill_temperature`,
  threading them into the per-step `TrainingConfig` via the
  existing ctx-build path.
- **`α=1.0` is zero-cost.** When `distill_alpha >= 1.0` the host
  forward is skipped entirely; pre-change pipeline tests
  pass unchanged.
- **`run_finetune` rejects `host=None` + `α<1.0` at the top of
  the function** before any batches are consumed, so the
  misconfiguration can't waste work.
- Docs: new "Host distillation" section in
  `docs/finetune-recipe.md`. Tests:
  `tests/test_distillation.py` (14 tests covering field
  validation, byte-identity at `α=1.0`, gradient-flow at
  `α=0.5`, host-unchanged invariant, `α=0.0` pure-KD path,
  pipeline kwargs plumbing).

### Added (forge-whisper-encoder)

- **Whisper-encoder forging — first non-causal-LM architecture in the
  registry.** New `WhisperEncoderAdapter` walks the encoder of either
  `WhisperForConditionalGeneration` or `WhisperModel` into the
  projected weight dict the matching native module consumes. The
  decoder is out of scope for v0.4 (tracked as `forge-whisper-decoder`).
- **`ForgedWhisperEncoder` native module.** Pre-LN block layout
  matching HF Whisper, GELU MLP, MHA (no GQA). The conv stem
  (`conv1`/`conv2`) and `embed_positions` are frozen-copied from the
  host bit-for-bit — ε_conv accounting per `docs/algorithm.md` §10.5.
  A `basis_encode` buffer carries the d → f bridge
  (`projector.basis.pseudoinverse() * scale_boost`) at the conv-stem
  → first-block boundary; state-dict-resident but not a parameter, so
  the no-randomly-initialised-weights invariant applies cleanly.
- **`NativeModelConfig.output_kind`** — new field, defaults to
  `"logits"`. Accepts `"encoder_states"` for the Whisper-encoder
  family. `vocab_size` now defaults to `0` and is gated by
  `output_kind`. Cross-constraints enforced at construction. Existing
  LM callers see byte-identical behaviour.
- **`saeforge.audio_eval.cosine_faithfulness`** — per-frame cosine
  similarity between forged encoder states and host states projected
  through the forge's own `basis_encode` buffer. Optional
  `precomputed_host_states` kwarg skips the host forward when the FSM
  has pre-captured states.
- **Family-aware `evaluate_faithfulness` dispatch.** LM families go
  through `_kl_from_input_ids` verbatim (FSM byte-equivalence net
  green); `whisper_encoder` goes through `cosine_faithfulness`. The
  `faithfulness` ctx field carries the family-appropriate scalar;
  `perplexity` carries `1 - cosine` for encoder so the existing
  `perplexity < best_perplexity` progress check keeps the right
  direction. `min_faithfulness` is reinterpreted per family (KL
  negation for LM; positive cosine threshold for encoder).
- **`ForgePipeline.eval_audio_features` and `eval_encoder_states`.**
  Pipeline-level fields plumbed through `_build_fsm_ctx`. Mutually
  exclusive with `eval_prompts` at construction. The
  `eval_encoder_states` field is the audio-side analog of pre-
  tokenised `_eval_input_ids` — when set, the host forward is
  skipped inside the FSM.
- **`saeforge.audio_data.synthetic_mel_features`** — pure-numpy
  sine-sweep + Gaussian noise synthesiser producing
  `(batch, 80, n_frames)` tensors shaped like Whisper input. Used
  by the synthetic example + tests; no `[audio]` extra required.
- **`sae-forge forge --audio-features-path FILE.pt`** — CLI flag,
  argparse-level mutually exclusive with `--eval-prompts`. Loads a
  `torch.save`'d tensor and passes it through to
  `ForgePipeline.eval_audio_features`.
- **`[audio]` pyproject extra** pinning `librosa>=0.10`. Optional —
  only the real-audio `.wav`/`.flac` mel-extraction path needs it.
  Added to `[all]`.
- **New examples and docs.** `examples/forge_whisper_synthetic.py`
  runs the full pipeline on a tiny synthetic Whisper without HF
  download or audio files. `docs/audio-forge.md` is the user-facing
  reference; `docs/algorithm.md` §10.5 documents the algorithmic
  surface (output_kind, vocab_size=0, the d→f bridge, ε_conv).
- **Spec correction in the same change.** The architecture-adapters
  spec delta for Whisper originally listed q/k/v_proj.weight as
  `(f, d)` and out_proj.weight as `(d, f)`; under HF
  `nn.Linear (out, in)` convention these need to be `(d, f)` and
  `(f, d)` respectively. The `(d,)` `q_proj.bias` alongside the
  original `(f, d)` `q_proj.weight` was self-inconsistent (Linear
  bias must match the first weight axis). Spec now matches the
  implementation and HF convention.

### Added (qwen3-dense-support)

- **Qwen3 dense architecture adapter.** `Qwen3Adapter` inherits from
  `Qwen2Adapter` and stamps `family="qwen3"` + auto-detects the
  per-head Q/K RMSNorm (`qk_norm=True`). The shared `LlamaAdapter.walk`
  now emits `q_norm`/`k_norm` weights as head-dim-aligned pass-through
  whenever the host has those submodules (host-attribute-gated, no-ops
  for Llama / Gemma-2 / Qwen2). The Llama-family `LlamaSelfAttention`
  conditionally constructs `RMSNorm(head_dim)` on Q and K when
  `cfg.qk_norm=True` and applies them between projection-reshape and
  SDPA. Qwen3 inherits hybrid-bridge support automatically via the
  shared `build_llama_family_module` factory. Requires
  `transformers >= 4.51`; the `[intel]` extra is capped at `<4.50` and
  silently skips Qwen3 registration.

### Added (hybrid-bridge-llama-family)

- **Hybrid-bridge insertion into the Llama-family native module
  forward path.** `LlamaTransformer` now constructs `BridgeModule`
  instances when `cfg.bridges=True` and applies them at block indices
  `0` and `L-2` in its per-block loop, mirroring the GPT-2 wiring.
  Closes the half-built state shipped in #18 where `hybrid_bridge=True`
  on a Llama / Gemma-2 / Qwen2 host accepted the flag, projected the
  weights through three bases, and then silently dropped the bridges
  on the forward pass. Llama, Gemma-2, and Qwen2 hybrid forges now
  work end-to-end. Default-off behavior is byte-identical to today.

### Changed (hierarchical-fsm)

- **FSM refactored into a three-machine hierarchy** —
  `saeforge/machines/sae_forge.orca.md` (the v0.2 flat ten-state
  machine) is replaced with three composed sub-machines under the
  same directory: `stream.orca.md` (outermost, shard handling),
  `refine.orca.md` (middle, per-shard convergence), and
  `basis.orca.md` (innermost, compress/regrow loop). Composition
  uses `orca_runtime_python`'s native `- invoke:` directive +
  `parse_orca_md_multi`. Internal-only refactor: no public API,
  CLI, on-disk artifact, or runtime-behavior change. The
  byte-equivalence acceptance gate
  (`test_imperative_and_fsm_byte_equivalent`) is green.
- `transitions_log` entries gain a `machine_path` field
  (`"stream"` / `"stream/refine"` / `"stream/refine/basis"`) for
  debugging — additive; existing readers that ignore unknown keys
  are unaffected.
- Failure propagation records a new `error_origin_machine` ctx
  field (deepest origin wins) alongside the unchanged
  `error_message` — additive; the byte-equivalence test filters it.

### Added (hierarchical-fsm)

- `saeforge.machines.visualize.to_mermaid` — auto-generates a
  `stateDiagram-v2` block from the parsed hierarchy. Embedded in
  `docs/advanced-fsm-options.md`; `tests/fsm/test_diagram_drift.py`
  asserts the doc matches the live emit so drift can't land.
- `sae-forge inspect --fsm-diagram` — CLI flag that emits the
  Mermaid diagram to stdout. Mutually exclusive with the
  `checkpoint` positional argument.
- `tests/fsm/` test package with sub-machine topology checks,
  multi-shard hierarchy integration, the runtime compound-state
  probe, and the diagram-drift gate.

### Fixed (hierarchical-fsm)

- `saeforge.actions.scan_activations` referenced a non-existent
  `basis.directions` attribute on the `protect_top_k > 0` path
  (the attribute is `basis.W_dec`). Surfaced by the new
  `tests/fsm/test_load_and_scan_ordering.py` — the only test that
  exercises this path with a real basis. One-line correction.

## [0.3.0] — 2026-05-09

### Added (forge-continual-learning-loop)

- **Three-loop FSM topology** ([PR #11](../../pull/11)) layered on top
  of the v0.1 single-shard pipeline:
  - **Stream loop** — `evaluated → loaded` re-entry to consume the
    next shard. Triggered by `task_trigger` (one of `labeled` /
    `token_budget` / `loss_delta`).
  - **Refine loop** — preserved v0.1 `evaluated → compressed`
    re-entry for same-shard convergence.
  - **Basis loop** — new `compressed ↔ regrown` self-loop for
    `inner_refine_passes` rounds before exiting to `projected`.
- **New `activations_scanned` state** between `loaded` and
  `compressed`, hosting the `scan_activations` action that scores
  features and selects a protected set when `protect_top_k > 0`. True
  no-op (no basis load, no torch import) under the v0.2.0-default
  `protect_top_k = 0`.
- **Protected features** — structural EWC analogue at the basis
  level. `compress_with_polygram` post-filters the
  `ValidationReport` so protected indices cannot be merged or
  removed by Polygram's Compressor. The do-not-remove kwarg is the
  preferred long-term path; the workaround is documented in
  `tasks.md` §10.4 and tracked for upstreaming.
- **Replay buffer + MixedIterator** — new `saeforge.training.replay`
  module exposing `ReplayBuffer` (three policies: `reservoir` /
  `recent_window` / `per_task`) and `MixedIterator` with
  deterministic 100-cycle replay scheduling. Pure Python, no torch
  dependency at module import.
- **TaskStream abstraction** — new `saeforge.training.task_stream`
  module with `LabeledTaskStream`, `TokenBudgetTaskStream`,
  `LossDriftTaskStream`, plus a process-local registry mapping
  ``task_iterator_id`` strings to live stream instances.
- **12 new `ForgePipeline` fields**: `n_tasks`, `task_trigger`,
  `token_budget_per_task`, `loss_delta_threshold`,
  `inner_refine_passes`, `protect_top_k`, `protect_score`,
  `activation_buffer_size`, `replay_ratio`, `replay_buffer_size`,
  `replay_policy`, `task_stream`. All default to v0.1-equivalent
  values.
- **Construction-time validation** for the new continual-learning
  knobs — invalid combinations (e.g. `replay_ratio > 0` with
  `replay_buffer_size = 0`, or `replay_policy="per_task"` with
  `task_trigger != "labeled"`) raise `ValueError` at
  `ForgePipeline(...)` time, not at run.
- **`docs/advanced-fsm-options.md`** — user-facing reference covering
  the three-loop topology, every new context field, every new CLI
  flag, the three `task_trigger` modes, the three `protect_score`
  strategies, the three `replay_policy` strategies, plus a worked
  recipe per pattern (per-task / protected-features / drift-triggered).
- **24 new tests** — `tests/test_replay_buffer.py` (11),
  `tests/test_task_stream.py` (7), and
  `tests/test_continual_learning_loop.py` (6 stub-driven FSM-level
  tests covering basis-loop / stream-loop / refine-loop preservation
  / stream-dominance contract).

### Changed (forge-continual-learning-loop)

- **FSM uses orca-runtime-python rich guard grammar directly**.
  `refine_same_shard` is now the orca expression
  `ctx.advance_stream == false and ctx.should_continue == true`
  evaluated by the runtime; previously the v0.1 design called for
  precomputing flat-bool flags in Python actions. Three ctx fields
  (`next_basis_step`, `refine_same_shard`, `terminate_run`) and the
  hardcoded `_NEXT_EVENT_FOR_STATE` map are gone — the runtime and
  the parsed `MachineDef.transitions` are now the source of truth
  for control flow.
- **Machine state count: 9 → 10** (added `activations_scanned`).
  Updated `test_machine_loads_and_has_nine_states` →
  `test_machine_loads_and_has_ten_states` per the spec's MODIFIED
  requirement.
- **`README.md`** — Status section now lists the recent landed
  openspec changes; new "Continual learning" Quickstart subsection
  shows the knobs + `LabeledTaskStream` wiring; ambiguous v0.x
  version labels dropped from the How-it-works callouts.
- **`AGENTS.md`** — orca-lang dependency contract section updated
  to document the rich-guard pattern and link to the
  continual-learning advanced-options doc.

### Backwards compatibility

- **No breaking changes.** Defaults preserve v0.1 byte-identical
  behavior. The `test_imperative_and_fsm_byte_equivalent` safety net
  passes unchanged. All 20 existing FSM tests pass.

### Out of scope (deferred follow-ups)

- True activation-driven `protect_score` (current 0.3.0 ships a
  direction-L2 stub; activation-driven scoring needs host-model
  residual capture).
- Polygram `do_not_remove` kwarg upstream — the
  `ValidationReport` post-filter is the workaround until then.
- Per-loop-level scan tuning, feature-axis sampling, raw trigger
  signal exposure in ctx, basis-size growth across tasks, per-task
  evaluation matrix, token-level replay buffer, and CLI flags for
  the new continual knobs are tracked in
  `openspec/changes/forge-continual-learning-loop/tasks.md` §12.

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
