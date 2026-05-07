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

Each change has `proposal.md` (why + scope), optional `design.md`
(rationale, alternatives, open questions), `tasks.md` (checklist), and
`specs/<capability>/spec.md` (delta requirements + scenarios). Validate
with `openspec validate <change-name>` before working on it; archive via
`openspec archive` when done.

## Polygram dependency contract

- Pinned at `polygram>=0.4` — the first release with scale-aware
  compression (PR #34: `rep_selection="scale_aware"`, `strategy="merge"`,
  per-cluster `merged_norm`, roll-up `scale_compression_ratio`). Older
  Polygram outputs are missing the merged-norm fields and would force
  sae-forge to fall back to original-only norms; we treat that as
  unsupported in v0 to keep the basis-fidelity story coherent.
- The stable input contract is the `.safetensors` checkpoint + companion
  `compression_report.json` produced by `polygram compress` /
  `polygram compress-epoch`. `FeatureBasis.from_polygram_checkpoint` is
  the single canonical entry point.
- Polygram is **not** vendored. If you need a fix in Polygram (new field
  on `CompressionReport`, new rep-selector, etc.), open it there.

## orca-lang dependency contract

- Pinned at `orca-lang>=0.5` behind the `[orca]` extra. Required only
  for the FSM-driven forge path landing in v0.1
  (`forge-outer-loop-fsm`). The default imperative `ForgePipeline.run`
  from v0 does **not** require it.
- Lazy-import `orca_lang.runtime` inside `saeforge.orchestrator`, never
  at package import time. `import saeforge` MUST succeed without
  `orca-lang` installed.
- The canonical machine ships at `saeforge/machines/sae_forge.orca.md`
  as package data. Load it via `importlib.resources.files`, never via
  filesystem path resolution.
- CI runs `orca verify` against the shipped machine. Static
  verification failure blocks the merge — that is the whole reason we
  chose an FSM here.
- This is **classical** orca-lang. q-orca-lang (the quantum extension)
  is never imported on the default forge path; `--quantum-aware` only
  influences which Polygram `confirmer` is selected inside
  `compress_with_polygram`.

## Torch dependency contract

- The `[torch]` extra is required for `NativeModel`, `SubspaceProjector`
  on real host models, and any fine-tuning. The pure-numpy basis loader
  and projector math (small synthetic hosts) stay on the no-extras
  install — that boundary matters for CI and for Polygram-only users
  inspecting compressions before forging.
- Lazy-import torch inside the modules that need it, never at package
  import time. `import saeforge` MUST succeed without torch.

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
