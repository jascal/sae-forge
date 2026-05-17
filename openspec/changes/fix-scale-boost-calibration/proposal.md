# Fix `scale_boost` Calibration — landing as forge-magnitude diagnostics

Originally proposed as a measure-not-guess `scale_boost` calibration
mode (`scale_boost="calibrate"`). The 2026-05-16 smoke gate
([[project_fix_scale_boost_smoke]]) **falsified the proposal's
premise**: three successive proxies for the forge's faithfulness KL
were tried and all picked the wrong `scale_boost`. The change as
shipped is **diagnostics-only** — it surfaces the magnitude/anomaly
signals that explain WHY a sweep produced bad forge KL, but does
not attempt to auto-pick `scale_boost`.

## Why

Two unfixed bugs motivated the original proposal (preserved as
[[project_kl_nonmonotonic]], [[project_auto_scale_boost]]):

1. **KL is non-monotone in rank at the default `scale_boost=1.0`.** On
   the smoke-reproduced regime (GPT-2 layer 8, jbloom SAE sliced to
   1024 features, HEA_Rung2 n_qubits=10), `faithfulness_kl` grows
   8.21 → 86.39 across K ∈ {25, 103, 163, 211} — a 78-nat blow-up
   that should not exist if the projection retained any meaningful
   information.
2. **`scale_boost="auto"` is a safety net, not a calibration.** The
   shipped `min(1.0, d_model/n_features)` heuristic is basis-shape-
   aware but basis-content-blind.

The original mechanism would have run a five-point geometric grid
sweep matching residual-stream std in log space. The smoke gate
disproved this, and two follow-up proxy designs (layer-L shortcut
KL; real-end-of-network KL via a closure through the remaining
transformer blocks) ALSO failed against the forge's actual KL
because the forge's KL measures a fully-projected NativeModel — a
fundamentally different operation from any residual-perturbation
proxy.

## What Changes (shipped)

### Diagnostic row fields on `ParetoFrontierRow`

Two new optional fields, both default `None`, populated when the
sweep runs with `--magnitude-diagnostics`:

- `logit_std_ratio: float | None` — forged logit-std ÷ host logit-std
  on the calibration corpus, computed via a layer-L shortcut
  (`host_residual @ host_unembed`). A diagnostic of magnitude-matching
  quality independent of forge KL.
- `top1_anomalous: bool | None` — True if the forged model's mode
  top-1 next-token prediction on the calibration prompts is in a
  curated anomalous-token set (SolidGoldMagikarp family, unicode-
  fragment BPE).

These compose with (do not duplicate) the structural diagnostics from
`add-forge-quality-diagnostics`: quality-tier is a *static* basis
property, magnitude diagnostics are *dynamic* projection properties.

### CLI: `--magnitude-diagnostics`

New `sweep-pareto` flag accepting `tokens:N` (use a built-in
calibration corpus capped at N tokens) or `prompts:PATH` (JSONL with
`{"text": ...}` per line). When set, every row's
`logit_std_ratio` and `top1_anomalous` are populated. Requires
`--layer` to be set.

### CLI: `--rank-monotonicity-check`

New `sweep-pareto` flag. After the sweep completes, the driver
verifies that within each encoding label, `faithfulness_kl` is
monotone non-increasing in `n_features_kept_actual` up to a tolerance
(default 0.1 nats). Violations print a stderr advisory listing the
offending tuples. **Advisory only** — no refusal; the analyst
decides whether to act.

### `forge_quality.advise_magnitude_diagnostics`

Post-sweep stderr advisory that surfaces, per row with diagnostics:

- `logit_std_ratio` per (label, K)
- `[!] anomalous-token canary fired on encoding=L, K=K` for each
  row with `top1_anomalous=True`

Composes with the existing pre-flight `advise_sweep_quality`.

### Module: `saeforge.calibration`

New module exposing:

- `load_calibration_corpus(host_model_id, layer, n_tokens, prompts_path)`
  — host residual activations at `layer`.
- `load_host_unembed(host_model_id)` — host lm_head weight.
- `compute_host_logit_std`, `compute_forged_logit_std`,
  `top1_is_anomalous` — pure-numpy diagnostic helpers.
- `ANOMALOUS_TOKEN_IDS` — curated glitch-token set per tokenizer.

### Out of scope, deliberately

- **Auto-picking `scale_boost`.** Disproved by the smoke gate. The
  `"auto"` heuristic remains the only programmatic resolver. Users
  who need a tuned `scale_boost` pick a literal value and document
  why.
- **`compute_forge_kl` / `forward_kl` / `prepare_calibration` /
  `SubspaceProjector(scale_boost="calibrate")`.** These existed in
  pre-merge drafts to support the calibrate mode and were removed
  once the mode was dropped.

## Falsifiable acceptance gate

Pre-merge smoke (2026-05-16, Intel 16GB) — see
`smoke-results.md` in this change dir for the full audit trail:

- Baseline arm (`scale_boost=1.0`) reproduces the documented blow-up:
  K=25 → K=203 produces KL 8.21 → 86.39, +78 nats. **PASSED.**
- `--magnitude-diagnostics` populates `logit_std_ratio` and
  `top1_anomalous` on every row. **PASSED.**
- `--rank-monotonicity-check` advisory fires correctly on the
  baseline arm, naming the offending K-pairs. **PASSED.**
- (Originally proposed) `--scale-boost-calibrate` produces monotone
  non-increasing KL. **DROPPED** — no proxy tested could deliver this.

## What this change is NOT

This is not a fix for the underlying KL blow-up. The blow-up is
structural: it happens inside the projected NativeModel's forward
pass, where stacked projections compound direction errors across all
12 GPT-2 layers. Diagnosing it via post-mortem row fields is what
this change adds. **Actually fixing it** is a separate proposal:
characterise which projected layer(s) drive the amplification.

## Capabilities

### Modified Capabilities

- `pareto-sweep`: `ParetoFrontierRow` gains `logit_std_ratio` and
  `top1_anomalous`. `sweep-pareto` CLI gains `--magnitude-diagnostics
  VALUE` and `--rank-monotonicity-check`. Existing rows / invocations
  byte-identical when the new flags aren't supplied.

### Added Capabilities

- None at the projector level (the originally-proposed
  `scale-boost-calibration` capability was dropped).

## Impact

- **Modified**: `saeforge/sweep.py` (`ParetoFrontierRow` two new
  fields, `sweep_pareto` two new kwargs, two new advisories);
  `saeforge/forge.py` (pass-through); `saeforge/cli.py` (two new
  flags); `saeforge/forge_quality.py`
  (`advise_magnitude_diagnostics`); `saeforge/projector.py`
  (unchanged behaviour; `"calibrate"` mode no longer accepted).
- **New module**: `saeforge/calibration.py` — calibration corpus +
  unembed loaders, pure-numpy diagnostic helpers, anomalous token
  set. Numpy-only for the helpers; `transformers` lazy-loaded for
  the loaders.
- **No breaking changes**: row schema extension forward-compatible;
  no behaviour change unless the new flags are set.
- **Dependencies**: no new external dependencies.

## Risk note

The diagnostic flag adds one forward pass through the host model on
the calibration corpus per sweep — bounded, deterministic, not
per-row. Cost is small (~hundreds of tokens × one GPT-2 forward).
The advisory is plain stderr text, no behaviour change on the
forge.
