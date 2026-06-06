## Why

The shipped `composition-subspace-preserve` capability (two-basis forge,
v0.13.0) defines the composition subspace `U_C` as the principal subspace
of the host attention's **aggregate** QK/OV read+write geometry over the
capture layers — the dominant singular directions of stacked
`[W_Q^h | W_K^h]` and `W_V^h W_O^h`. The two-basis spec hedged that the
mechanism was unproven at scale and that the negative would be the
dispositive result.

It is now resolved, against an **alive** forge (real GPT-2, single-layer
residual forge of the induction-feeding layer, blocks otherwise host —
the only configuration that does not collapse to uniform output; whole-
model single-basis forge of GPT-2 dies at every scale). Three preserve
variants at matched 20-dimensional budget, measuring the circuit-specific
excess (induction-predictable KL minus complement KL):

| preserve | circuit excess | global KL | overlap w/ writers |
|---|---|---|---|
| none (single-basis) | +0.643 | 3.30 | — |
| **reader geometry** (the shipped `U_C`) | +0.602 (−6%) | 3.58 | — |
| **writer OV-output** (this change) | **−0.068 (−111%)** | 4.02 | — |
| attribution `∂loss/∂residual` | +0.733 (worse) | 2.02 (best) | **0.05** |

Three findings, each load-bearing:

1. **The shipped `U_C` preserves the wrong directions.** Aggregate
   reader-layer geometry does essentially nothing for circuit fidelity
   (−6%). The fragile signal a forge smears is the **predecessor-write**
   — the OV *output* of the prev-token writer heads (e.g. GPT-2 `4.11` at
   layer 4) — which the reader-layer geometry never even contains.

2. **Preserving the writers' OV-output eliminates the circuit tax**
   (−111%; the induction-specific excess goes to ~0) at the same budget.
   The corrected `U_C` is the orthonormalised union of the **OV-output
   row spaces** of the circuit's **writer heads**, not the aggregate
   read/write geometry of the reader layers.

3. **There is no functional shortcut.** A label-free attribution
   subspace (`∂circuit-loss/∂residual`) is **0.05 overlap with the
   writer subspace — nearly orthogonal** — and fails to protect the
   circuit. The gradient finds the *generically output-important*
   directions (hence the best global KL) not the *circuit-mechanistic*
   ones. **Loss-sensitivity ≠ circuit-mechanism.** Circuit-faithful
   forging therefore *requires* behavioral identification of the
   circuit's writer heads — there is no gradient/aggregate substitute.

This change redefines `U_C` as **writer OV-output preservation** and adds
the minimal behavioral writer-identification (`composition_heads`
becomes the circuit *writer* heads, or a preset that detects them). It is
honest about the cost: writer-output preserve trades ~+0.7 global KL for
circuit fidelity, and global-fidelity vs circuit-fidelity are different
subspaces (overlap 0.05) — one small preserved subspace cannot buy both.

Evidence and the falsified alternatives live in the `lm-sae` consumer:
`scripts/two_basis_single_layer.py` (commits `7f69f93` writer-output,
`1d56b52` attribution-falsified, `32711ed` the alive-forge / reader
falsification).

## What Changes

### Scope

Redefine the composition subspace `U_C` from **aggregate reader-layer
QK/OV geometry** to **the OV-output subspace of the circuit's writer
heads**, and make `composition_heads` mean *which writer heads to
preserve* (explicit list, or a behavioral preset). The
assertion-preserve (`U_A`) half, the projector arm, the
circuit-faithfulness metric, the byte-equivalence-when-disabled
contract, and the default-off posture are all unchanged.

### New artifacts

- **`saeforge/circuit_heads.py`** — behavioral writer identification.
  `prev_token_heads(host, corpus, *, top_k)` returns the heads with the
  highest Δ=1 (previous-token) attention on a calibration corpus (the
  induction feeders); `duplicate_token_heads(...)` the same-token movers.
  ~120 lines; one forward pass with `attn_implementation="eager"`. This
  is the minimal idiom detector; richer idioms are a follow-up.
- **`saeforge/composition_subspace.py`** — add
  `extract_writer_subspace(host, *, writer_heads, rank)` returning a
  `CompositionSubspace` whose columns are the orthonormalised union of
  the writer heads' OV-output row spaces (`rowspace(W_V^h W_O^h)`),
  top-`rank` by singular value. The existing
  `extract_composition_subspace` (reader geometry) is retained as
  `mode="reader-geometry"` for comparison/ablation and is documented as
  the falsified-weaker option.
- **`tests/test_circuit_heads.py`** / **`tests/test_writer_subspace.py`**
  — prev-token detection recovers a known mover on a tiny fixture;
  writer subspace orthonormality, rank, and the "preserved subspace
  reproduces the writers' OV output to tolerance" invariant.

### Modified artifacts

- **`saeforge/composition_subspace.py`** — `extract_composition_subspace`
  gains `mode: Literal["writer-output","reader-geometry"] =
  "writer-output"`. Default flips to writer-output (the validated
  mechanism); `reader-geometry` is opt-in and carries a docstring note
  that it does not protect circuits.
- **`saeforge/forge.py`** — `ForgePipeline.composition_heads` semantics
  change: a list of `(layer, head)` writer heads, or a preset string
  (`"prev-token"` / `"duplicate-token"` / `"all"`). When a preset, the
  pipeline calls `circuit_heads` on the eval corpus to identify them.
  `"all"` keeps the legacy reader-geometry path (documented as weaker).
  `_build_augmented_basis` builds `U_C` via `extract_writer_subspace`.
- **`saeforge/cli.py`** — `--composition-heads` accepts the presets;
  `--composition-mode {writer-output,reader-geometry}` (default
  `writer-output`).
- **`docs/two_basis_forge.md`** — replace the `U_C` definition with the
  writer-output mechanism + the alive-forge evidence table + the
  no-functional-shortcut result.
- **`CHANGELOG.md`** — `## [Unreleased]` `### Changed`.

### Out of scope (deferred)

- **Cross-layer circuit-subspace bundles / full circuit graphs.** v1
  preserves the writer heads' OV output; reading the full
  writer→reader composition graph to preserve only the live edges is
  the `circuit-subspace-bundle` follow-up.
- **Idiom library beyond prev-token / duplicate-token.** Copy /
  name-mover / copy-suppression / successor detectors (the output
  idioms) need task-specific behavioral probes and are deferred to
  `circuit-idiom-library`.
- **Attribution / gradient preservation.** Falsified for *circuit*
  fidelity here (overlap 0.05); retained only as the documented
  global-fidelity option, not implemented as a `U_C` source.
- **Writer-head weighting.** v1 takes a uniform orthonormalised union of
  the writer heads' OV-output. Weighting by attention mass or by each
  writer's contribution to the circuit metric is deferred to
  `writer-weighted-uc`.
- **Whole-model GPT-2 forge.** Out of reach for single-basis (dies);
  needs `hybrid-bridge-forge` (multi-layer bases) composed with this —
  tracked as `hybrid-bridge-two-basis`.

## Capabilities

### Modified Capabilities

- **`composition-subspace-preserve`** — the `U_C` extraction requirement
  is MODIFIED: `U_C` is the orthonormalised union of the **writer heads'
  OV-output row spaces**, where the writer heads are explicit or
  behaviorally identified (`composition_heads`), not the aggregate
  reader-layer QK/OV geometry. A second MODIFIED requirement pins
  `composition_heads` as the writer-head selector (list or preset). The
  byte-equivalence-when-disabled and circuit-faithfulness requirements
  are unchanged.

## Impact

- **No default-on behaviour change for disabled forge.** `U_C` only
  exists when `composition_preserve=True`; default-off remains
  byte-identical.
- **Behaviour change when `composition_preserve=True`.** `U_C` now
  preserves writer OV-output (validated to protect circuits) instead of
  reader geometry (validated not to). This is a *correctness* change to
  an opt-in path, not an API break; `composition_heads` gains preset
  values and a `(layer, head)` list form.
- **Honest cost surfaced.** The run report notes the circuit-vs-global
  fidelity trade (writer-output costs global KL); the comparison harness
  gains the writer-vs-reader-vs-attribution row.

## Sequencing

- **Depends on:** `composition-subspace-preserve` (shipped in v0.13.0)
  for the augmented-basis machinery and the circuit-faithfulness metric.
- **Composes with:** `hybrid-bridge-forge` (multi-layer bases) — a
  whole-model GPT-2 forge would preserve writer OV-output per anchor;
  tracked separately as `hybrid-bridge-two-basis`.
- **Single PR.** ~300 net LOC (the writer subspace + the behavioral
  detector + the knob rewiring). The byte-equivalence gate and a
  writer-output integration test are the shipping criteria.
