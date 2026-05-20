## Why

`add-forge-quality-diagnostics` made the *structural* forge-feasibility signal legible in `ParetoFrontierRow` (`host_d_model`, `basis_rank`, `quality_ratio`, `quality_tier`). That answers "can this basis span the host residual stream." It does *not* answer "how many distinct concepts does this dictionary actually encode" — which is the question polygram's compressor already has data on, sitting unused in `compression_report.json` and partially exposed in `FeatureBasis.metadata` but never piped to the row.

The empirical case comes from `econ-sae`'s Phase 7.2 forge sweep. With matched encoding capacity (Rung5, cap=128), polygram compressed two SAEs that targeted the same substrate:

| substrate                | clusters | zeroed | loners | redundancy |
|--------------------------|----------|--------|--------|------------|
| Phase 1.6 attn (unsupervised) | 7   | 62     | 59     | 48% |
| Phase 6.2 dual-head (supervised) | 6 | 88   | 34     | 69% |

Across a Rung3 → Rung4 → Rung5 capacity sweep on the supervised SAE, cluster count grew 2 → 3 → 6 and saturated at exactly the 6 supervised concepts. **Cluster count is a direct readout of how many distinct concepts the compressed dictionary encodes**, and `n_zeroed / kept` is a direct readout of how concentrated those concepts are.

These metrics are essentially free in sae-forge: `FeatureBasis.from_polygram_checkpoint` already loads them into `metadata` (`n_features_kept`, `n_clusters`). The compression report on disk has the rest (`n_zeroed`, strategy). This proposal pipes them through to `ParetoFrontierRow` and adds one derived ratio plus an advisory line for the saturation case where the next encoding rung would unlock more cluster resolution.

This pairs with `add-forge-quality-diagnostics` as the *content* counterpart to *structure*: rank-ratio tells you whether the basis can span d_model; cluster count tells you how many concepts live in the basis. An analyst reading a frontier.jsonl row wants both.

## What Changes

### `ParetoFrontierRow` gains four polygram diagnostic fields

All optional, all default `None` (backwards-compat for existing `frontier.jsonl` consumers):

- **`polygram_n_clusters: int | None`** — number of distinct concept clusters polygram's compressor identified in the dictionary. Sourced from `compression_report.json` (`n_clusters`), already exposed by `FeatureBasis.metadata["n_clusters"]`.
- **`polygram_n_zeroed: int | None`** — number of dictionary slots polygram zeroed as redundant during compression. Sourced from `compression_report.json` (`n_zeroed`). Falls back to `None` for reports that pre-date the field (older polygram outputs).
- **`polygram_redundancy_ratio: float | None`** — derived: `n_zeroed / (n_clusters + n_zeroed)` when both are present; `None` when either is missing. The single number a frontier plot should colour rows by to surface "concept concentration."
- **`polygram_encoding_capacity: int | None`** — the encoding's cap (Rung3=16, Rung4=32, Rung5=128, HEA_Rung2(n)=2ⁿ). Resolved from the encoding spec already passed to the sweep, not from the report (some reports don't record cap).

All four fields are populated when the polygram compression report is loadable. When the report is missing, malformed, or pre-dates a field, the corresponding row fields are `None` and the sweep proceeds normally.

### New module `saeforge/polygram_diagnostics.py`

Thin helper module:

- `load_polygram_report(checkpoint_path) -> dict | None` — wraps the existing `FeatureBasis._locate_report` logic to fetch the report dict without loading the whole basis; returns `None` on any failure (missing report, JSON error, unreadable file). Logs at INFO.
- `compute_redundancy_ratio(n_clusters, n_zeroed) -> float | None` — returns `n_zeroed / (n_clusters + n_zeroed)` when both ≥ 0 and their sum > 0; `None` otherwise.
- `resolve_encoding_capacity(encoding_spec) -> int | None` — given an encoding spec string from the sweep (`"rung3"`, `"rung5"`, `"hea_rung2(n=6)"`), returns the cap. `None` for unrecognised encodings.

This module is the single place the spec test suite asserts behaviour against; the sweep driver calls it once per row.

### Pre-flight advisory extension: cluster-count saturation

When `add-forge-quality-diagnostics`' advisory already runs and `polygram_n_clusters == polygram_encoding_capacity` for the *largest* K in a sweep's manifest, the advisory SHALL append a one-line note:

> Note: polygram_n_clusters (N) equals encoding capacity (N) — the encoding may be saturated. Consider re-running polygram compress with a larger encoding (Rung4 → Rung5, or HEA_Rung2(n_qubits=N+1)) to see whether additional concepts are present.

The note is purely informational — no refusal, no behaviour change. It mirrors econ-sae's Rung3 → Rung4 → Rung5 progression where cluster count grew 2 → 3 → 6 and saturated at 6 on the last bump.

### Out of scope, deliberately

- **Computing `n_clusters` / `n_zeroed` from a non-polygram-compressed SAE.** These are polygram-specific metadata; for non-polygram inputs the fields stay `None`. sae-forge does not re-derive them from `W_dec`.
- **Predictive KL from polygram diagnostics.** This proposal surfaces structural concept-count metrics; it does not estimate post-forge KL from them. That's a research project.
- **Multi-encoding sweep automation.** The existing `--encoding rung3,rung4,rung5` plumbing already supports running the same SAE against multiple encodings (one row per encoding). This proposal does not add a new `--polygram-capacity-sweep` flag — the existing surface is enough.
- **Polygram-side report-schema changes.** Any field that older polygram reports don't expose stays `None`. No upstream PR is required.
- **`saturated` tier label.** Don't extend `QualityTier` with a polygram-saturation member; the advisory note is enough. Keeping the tier vocabulary tied to *structural* rank ratio preserves the clean separation from `add-forge-quality-diagnostics`.

## Capabilities

### Modified Capabilities

- `pareto-sweep`: `ParetoFrontierRow` gains four new optional fields (`polygram_n_clusters`, `polygram_n_zeroed`, `polygram_redundancy_ratio`, `polygram_encoding_capacity`); the pre-flight advisory gains one optional saturation note appended after the existing rank-tier message. Existing rows / invocations byte-identical when the polygram report is absent or pre-dates these fields.

## Impact

- **New module**: `saeforge/polygram_diagnostics.py` — `load_polygram_report`, `compute_redundancy_ratio`, `resolve_encoding_capacity`, `format_saturation_note`.
- **Modified**:
  - `saeforge/sweep.py` — `ParetoFrontierRow` gains the four new fields; `_process_row` populates them from the report path bound to the per-row basis.
  - `saeforge/forge_quality.py` (from `add-forge-quality-diagnostics`) — `advise_sweep_quality` appends the saturation note when the polygram report shows `n_clusters == capacity` at the largest K. Existing return shape unchanged (still `str | None`).
  - `saeforge/__init__.py` — export `load_polygram_report`, `compute_redundancy_ratio`.
- **No breaking changes**: row schema extension is forward-compatible (existing readers see `null`); no behaviour change unless a polygram report supplies the relevant fields.
- **Dependencies**: none new. Polygram report shape is the existing public contract documented in `AGENTS.md` ("Polygram dependency contract"). Older polygram outputs that don't include `n_zeroed` get `None` for the relevant fields — no upstream PR required.

## Risks

- **`n_zeroed` may be absent from older polygram reports.** The spec marks this field optional and falls back to `None`. Forward-compat behaviour is the same as the unset case.
- **`n_clusters` semantics depend on polygram's compressor strategy.** `merge` vs other strategies may report different values; that's a polygram-side contract, surfaced here as-is. The proposal does not interpret the number — it surfaces it. The README addition makes clear that the metric is "as reported by polygram" and points to the polygram docs for definitional details.
- **Capacity resolution depends on encoding spec parsing.** `HEA_Rung2(n_qubits=N)` is currently the only parametric encoding; `resolve_encoding_capacity` parses it conservatively (returns `None` on unknown forms). Spec includes a scenario for the unknown-encoding fallback.
- **Saturation advisory is heuristic.** Cluster count equalling capacity *can* mean true saturation, or simply that the SAE happens to have exactly `capacity` distinct concepts. The advisory is informational and doesn't refuse; the wording explicitly uses "may be saturated" rather than "is saturated."
