# sae-forge — Agent Orientation

## What this repo is

sae-forge is the practical bridge from Polygram-compressed sparse
autoencoders to small, semantically-native transformers. It does **not**
train SAEs, define a compression algorithm, or fork Polygram — it
consumes Polygram artifacts and projects a host model's weights into the
SAE feature basis, producing a transformer whose residual stream *is*
the surviving feature space.

## Workflow: OpenSpec-driven

All non-trivial work lands through OpenSpec changes in `openspec/changes/`.
The v0 milestone is staged as five changes:

1. `bootstrap-package` — pyproject, smoke test, CI, this file. Process-only.
2. `feature-basis` — `FeatureBasis` loader for Polygram compressed
   checkpoints; pure-numpy inspection surface; pseudoinverse cache.
3. `subspace-projector` — `SubspaceProjector` weight projection math
   (embed, QKV, MLP, unembed) with optional `scale_boost`.
4. `native-model` — `NativeModel` HF-compatible small transformer
   skeleton; `from_host` constructor that consumes a `SubspaceProjector`.
5. `forge-pipeline` — `ForgePipeline` imperative orchestrator +
   faithfulness eval + first worked example (`examples/forge_gpt2_toy.py`).

The v0.1 milestone replaces the imperative orchestrator with an
orca-lang FSM and expands sae-forge's scope to drive the full compress
→ regrow → project → fine-tune → eval loop:

6. `forge-outer-loop-fsm` — orca-lang FSM in
   `saeforge/machines/sae_forge.orca.md` driving a nine-state machine;
   `ForgePipeline` gains an `orchestrator="fsm"` opt-in; new `[orca]`
   extra; CI gains `orca verify`. Polygram's `Compressor` and
   `Regrower` become FSM actions, not externally-staged inputs.

The v0.2 and v0.3 milestones converge the implementation onto the
`docs/algorithm.md` spec and unlock real fine-tuning:

7. `feature-native-attention` (v0.2) — `attention_width` knob on
   `NativeModelConfig` / `ForgePipeline`; `"host"` default preserves
   v0.1; `"feature_native"` opts into the both-sides-projected
   c_attn / c_proj where attention internals also become k-wide. CLI
   flag `--feature-native-attention`.
8. `forge-finetune-recipe` (v0.3, drafted) — replaces the v0.1 4-step
   smoke fine-tune with a real recipe in `saeforge.training` (cosine
   LR + warmup, gradient clipping, optional gradient checkpointing,
   bf16/fp16 autocast, periodic eval, periodic saves, structured loss
   tracking). Local-corpus-first; offline-safe by spec.

Each change has `proposal.md` (why + scope), optional `design.md`
(rationale, alternatives, open questions), `tasks.md` (checklist), and
`specs/<capability>/spec.md` (delta requirements + scenarios). Validate
with `openspec validate <change-name>` before working on it; archive via
`openspec archive` when done.

## Polygram dependency contract

- Pinned at `polygram>=0.9.0`, the published distribution under the
  `orcalang` user on PyPI. 0.9.0 promotes `cluster_experts` /
  `ExpertDictionary` to the public surface (PR #87), unlocking the
  planned MoE-from-SAE forging path. Bump the floor whenever a
  Polygram release adds a field sae-forge depends on (the scale-aware
  compression work from PR #34 has been in since 0.0.1:
  `rep_selection="scale_aware"`, `strategy="merge"`, per-cluster
  `merged_norm`, roll-up `scale_compression_ratio`). Older outputs
  without merged-norm fields fall back to row-norm — handled by
  `FeatureBasis`'s loader but not encouraged in v0.
- The stable input contract is the `.safetensors` checkpoint + companion
  `compression_report.json` produced by `polygram compress` /
  `polygram compress-epoch`. `FeatureBasis.from_polygram_checkpoint` is
  the single canonical entry point.
- Polygram is **not** vendored. If you need a fix in Polygram (new field
  on `CompressionReport`, new rep-selector, etc.), open it there.

## orca-lang dependency contract

- Pinned at `orca-runtime-python>=0.1.27` behind the `[orca]` extra
  (PyPI distribution name; module name `orca_runtime_python`). Earlier
  releases ship stubbed guard / action handlers — 0.1.27 is the first
  with working `register_action` and full guard expression evaluation.
  The package called "orca-lang" in the original v0.1 spec does **not**
  exist on PyPI; orca-runtime-python is the actual classical-orca
  Python runtime.
- Required only for the FSM-driven forge path landing in v0.1
  (`forge-outer-loop-fsm`). The default imperative `ForgePipeline.run`
  from v0 does **not** require it.
- Lazy-import `orca_runtime_python` inside `saeforge.orchestrator`,
  never at package import time. `import saeforge` MUST succeed without
  `orca-runtime-python` installed.
- The canonical machine ships at `saeforge/machines/sae_forge.orca.md`
  as package data. Load it via `importlib.resources.files`, never via
  filesystem path resolution.
- orca-lang's guard grammar is intentionally restricted to comparison
  + boolean composition (`==`, `!=`, `<`, `>`, `<=`, `>=`, `and`,
  `or`, `not`, parens, null checks, var-on-RHS). Arithmetic and
  complex predicates belong in actions. The v0.2 FSM uses the rich
  guard grammar directly — e.g. `refine_same_shard` is the orca
  expression `ctx.advance_stream == false and ctx.should_continue
  == true`, evaluated by the runtime — rather than precomputing
  flat-bool flags in Python. Only predicates that genuinely need
  Python (loss-window deltas, task-trigger dispatch) live in the
  action layer.
- **Continual learning** (v0.2): a three-loop topology layered onto
  the same FSM — stream loop (per shard), refine loop (per-shard
  convergence), basis loop (compress↔regrow refinement). All knobs
  default to v0.1-equivalent values; opt in by setting any of
  `n_tasks > 1`, `inner_refine_passes > 1`, `protect_top_k > 0`, or
  `replay_ratio > 0` on `ForgePipeline`. See
  [`docs/advanced-fsm-options.md`](docs/advanced-fsm-options.md) for
  the full knob reference, decision-tree for picking a `task_trigger`,
  and worked recipes per pattern.
- This is **classical** orca-lang. q-orca-lang (the quantum extension)
  is never imported on the default forge path; `--quantum-aware` only
  influences which Polygram `confirmer` is selected inside
  `compress_with_polygram`.
- **Adaptive regrow** (v0.5, opt-in): when the fixed `--regrow-count`
  isn't right across a multi-shard run, `--adaptive-regrow` activates
  a controller in the basis loop that grows the basis toward
  `--n-features-target` based on the polygram-side `n_features_kept`
  signal. The controller is a pure function
  (`saeforge.basis.RegrowController`) wrapped in a composed action
  (`saeforge.actions.adapt_and_regrow`) sitting on the existing
  `compressed → regrown` transition — zero topology drift. Defaults
  preserve v0.2 byte-equivalence. Full knob reference and tuning
  guidelines in
  [`docs/advanced-fsm-options.md`](docs/advanced-fsm-options.md)
  under "Adaptive regrow".

## Torch dependency contract

- The `[torch]` extra is required for `NativeModel`, `SubspaceProjector`
  on real host models, and any fine-tuning. The pure-numpy basis loader
  and projector math (small synthetic hosts) stay on the no-extras
  install — that boundary matters for CI and for Polygram-only users
  inspecting compressions before forging.
- Lazy-import torch inside the modules that need it, never at package
  import time. `import saeforge` MUST succeed without torch.

## Audio architecture support (v0.4 forge-whisper-encoder)

- Encoder-only forging of Whisper variants is supported via
  `family == "whisper_encoder"`. The decoder forge (cross-attention,
  vocab head, beam search) is deliberately out of scope and tracked
  as a separate change (`forge-whisper-decoder`). Encoder-only is
  the natural unit because the polygram-side Whisper SAEs target
  encoder residuals.
- `evaluate_faithfulness` dispatches on `forged.config.family`:
  LM families (gpt2 / llama / gemma2 / qwen2 / qwen3) go through
  the existing `_kl_from_input_ids` path verbatim;
  `whisper_encoder` goes through
  `saeforge.audio_eval.cosine_faithfulness` instead. The
  `faithfulness` ctx field carries the family-appropriate scalar
  (KL for LM, cosine for encoder); `perplexity` is `exp(KL)` for
  LM and `1 - cosine` for encoder so the existing
  `perplexity < best_perplexity` progress check keeps the right
  direction in both paths.
- `min_faithfulness` is reinterpreted per family. LM uses the v0.1
  negation convention (`min_faith=-0.05` means "max KL = 0.05").
  Encoder uses the natural positive convention
  (`min_faith=0.95` means "min cosine = 0.95"). The default `0.0`
  is uniformly permissive across both.
- The conv stem (`conv1`, `conv2`) and positional embeddings stay
  at `d_model` width (frozen-copied — ε_conv per
  `docs/algorithm.md` §5). The d → f bridge from the conv-stem
  output into the SAE-basis residual stream lives in a
  `basis_encode` buffer on `ForgedWhisperEncoder`, set by the
  adapter walk from `projector.basis.pseudoinverse() *
  scale_boost`. Buffer not parameter, so it doesn't participate
  in gradient checkpointing or the no-randomly-initialised-weights
  invariant.
- Real-audio mel extraction needs `librosa` (the `[audio]` extra);
  the synthetic-fixture path
  (`saeforge.audio_data.synthetic_mel_features`) and the
  CLI `--audio-features-path` (pre-extracted tensors) work
  without it.

## File layout

```
saeforge/                  Python package
  __init__.py              version + public re-exports
  basis.py                 FeatureBasis                    (added in change 2)
  projector.py             SubspaceProjector               (added in change 3)
  model.py                 NativeModel                     (added in change 4)
  forge.py                 ForgePipeline + ForgeResult     (added in change 5)
  cli.py                   `sae-forge` console script
  eval/                    faithfulness eval helpers
  utils/                   shared helpers (lazy imports, IO, logging)
tests/                     pytest suite (mirrors package layout)
examples/                  worked examples                 (first added in change 5)
openspec/                  spec-driven changes + capability specs
.github/workflows/         CI
```

## Local dev

```bash
pip install -e ".[dev,torch,polygram]"
pytest
ruff check saeforge tests examples
```

## Conventions

- Only `numpy` and `safetensors` are mandatory runtime deps. Torch,
  transformers, and Polygram are optional extras. `import saeforge` MUST
  succeed without any of them.
- No emojis in code or generated artifacts.
- Default to no comments unless the *why* is non-obvious.
- Match the polygram CLI verb-first style: `sae-forge forge`,
  `sae-forge inspect`, etc. — not flag-driven dispatch.
- Hardware-sensitive paths (per-layer streaming, fp16 / bf16 selection)
  live behind explicit knobs on `ForgePipeline`, never as auto-detection.
