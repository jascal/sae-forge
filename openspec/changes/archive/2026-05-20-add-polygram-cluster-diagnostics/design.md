## Context

`add-forge-quality-diagnostics` (already landed) gave `ParetoFrontierRow` four structural-rank fields. That answers "can the basis span the host residual stream." Reading a frontier.jsonl row, an analyst can now filter `quality_tier == "good" or "saturated"` and see only rows with a fighting chance of producing a useful forge.

What that doesn't answer: **how many distinct concepts does the dictionary encode?** Two SAEs with identical `basis_rank` and identical `quality_tier` can have wildly different concept structure — one could encode 6 clean concepts plus 40 redundant copies, the other 40 distinct concepts with no redundancy. The forge consequences are completely different.

`econ-sae`'s Phase 7.2 demonstrated this concretely. Supervised SAE training (Phase 6.2 dual-head + focal loss) produced a dictionary with 6 clusters / 88 zeroed / 34 loners at Rung5 cap=128; an unsupervised attention-only SAE at the same encoding produced 7 clusters / 62 zeroed / 59 loners. Same `basis_rank` to within rounding; very different *concept concentration*. Rung-sweep on the supervised SAE showed cluster count grew 2 → 3 → 6 as encoding capacity went 16 → 32 → 128, then saturated at 6 (the supervised concept count).

The data to surface this is already loaded. `FeatureBasis.metadata` carries `n_clusters` and `n_features_kept` from the compression report. The report file is locatable next to every polygram safetensors. We just don't pipe these onto the row.

**This proposal pipes the polygram-side concept-structure metrics onto `ParetoFrontierRow` and adds one advisory line when cluster count saturates against encoding capacity.**

## Goals / Non-Goals

**Goals:**
- Expose `polygram_n_clusters`, `polygram_n_zeroed`, `polygram_redundancy_ratio`, `polygram_encoding_capacity` per row.
- Make the cross-encoding sweep (`--encoding rung3,rung4,rung5`) the analyst's tool for measuring cluster-count saturation, with no new CLI flag required.
- Add one informational advisory line when `n_clusters == encoding_capacity` at the sweep's largest K — the empirical signal that more encoding capacity might find more concepts.
- Backwards compatibility: older `frontier.jsonl` files and polygram reports without the optional fields still load and emit; the new fields default to `None`.

**Non-Goals:**
- Computing concept count from `W_dec` directly. The polygram compressor is the source of truth; sae-forge surfaces what's reported.
- Refusing degenerate concept setups. Cluster count is a description, not a gate. The existing `--quality-floor` knob (rank-ratio-based) is the only refusal surface.
- Predicting post-forge KL from cluster count. Possible follow-up; out of scope here.
- A `polygram_tier` categorical (analogous to `quality_tier`). Resisted — cluster count is already a small integer; analysts can `jq 'select(.polygram_n_clusters >= 5)'` directly without a tier vocabulary.
- Upstream polygram changes. Some older reports don't expose `n_zeroed`; that's a `None` in the row, not a hard error or an upstream PR.

## Decisions

### Decision 1 — Surface what polygram reports, don't recompute

`n_clusters` is read from `compression_report.json` (`report["n_clusters"]`). `n_zeroed` is read from the same report. We do not re-derive these by re-running clustering on `W_dec`. Reasons:

- Polygram is the source of truth for its own compression strategy; recomputing would diverge from the polygram-side definition the moment the strategy changes.
- Re-running clustering at sweep time is expensive (clusters depend on the rep-selector + scale_boost + cancellation rounds polygram already executed).
- Falling back to `None` for missing fields is cheap and preserves backwards compatibility with older polygram outputs.

**Alternative considered**: re-derive `n_clusters` from `W_dec` via Gram-matrix block analysis. Rejected — would silently disagree with polygram for the same dictionary, undermining the row's interpretability.

### Decision 2 — Compute one derived ratio (`redundancy_ratio`), expose raw counts otherwise

The proposal exposes raw `n_clusters` and `n_zeroed`. The single derived field is `redundancy_ratio = n_zeroed / (n_clusters + n_zeroed)`. We do not also compute `loner_count` or `cluster_density`.

- Raw counts let analysts compute anything they want; `jq` arithmetic on integers is trivial.
- One ratio is the headline-plot field that pairs with `quality_ratio` from the structural diagnostics.
- More derived fields would multiply the schema surface without adding information.

**Alternative considered**: expose `loner_count = n_features_kept - n_clusters` as a fourth field. Rejected — easily derivable when `n_features_kept` is already present elsewhere in the row (existing `n_features_kept_actual`), and `n_features_kept` semantics differ subtly between polygram strategies.

### Decision 3 — Encoding capacity resolved from spec string, not from report

`polygram_encoding_capacity` is computed by parsing the encoding spec the sweep was invoked with (`rung3` → 16, `rung4` → 32, `rung5` → 128, `hea_rung2(n_qubits=6)` → 64). The report itself sometimes omits the cap. Parsing the spec is the deterministic source.

Unknown encodings parse to `None`; the row's capacity field is `None`; the saturation advisory does not fire. This degrades gracefully for forward-compatibility with new encodings polygram may add.

**Alternative considered**: read `encoding_capacity` from the report. Rejected — older reports don't have it; parsing is deterministic and lab-frame-correct.

### Decision 4 — Saturation note is appended to the existing advisory, not a new advisory

The pre-flight `advise_sweep_quality` already returns a `str | None`. When the polygram-side check detects `n_clusters == capacity` at the largest K, the function appends one extra line to the already-formed advisory (or builds a single-line advisory if no rank-tier message was warranted).

- One advisory surface keeps the CLI behaviour predictable — analysts read one block, not two.
- The note appends only when `compute_redundancy_ratio` and `resolve_encoding_capacity` both return non-None values; otherwise it's silent.
- The wording uses *may be saturated* (not *is saturated*) because the empirical case is "more capacity may find more concepts" — not a hard claim.

**Alternative considered**: a separate `--polygram-capacity-sweep` CLI that auto-runs Rung3/4/5 in one command. Rejected (out of scope) — the existing `--encoding rung3,rung4,rung5` already does this; the advisory's job is to prompt the analyst to run that sweep, not to do it automatically.

### Decision 5 — Pipe via the basis loader, not a fresh report read in the sweep

`FeatureBasis.metadata` already carries `n_clusters` and `n_features_kept` from the report parse that `from_polygram_checkpoint` performs. The sweep driver reads these from the basis after loading, plus opens the report once more to read `n_zeroed` (the missing field). This means:

- Zero new file IO on the hot path for sweeps where the report has been read already (the basis loader caches it).
- One extra JSON read per encoding to fetch `n_zeroed` — the existing `_locate_report` helper finds the path; `json.load` is microseconds.

**Alternative considered**: extend `FeatureBasis.metadata` with `n_zeroed` so the sweep doesn't reach into the report a second time. Rejected (deferred) — touches the basis-loader contract and would require coordinating with `feature-basis` capability changes; the sweep-side read is cheap and isolated.

## Risks / Trade-offs

- **Polygram report schema drift.** If polygram renames `n_zeroed` or changes its semantics, the field becomes `None` until sae-forge adapts. The fallback path is silent (logged at INFO, not raised). Spec explicitly tolerates this.
- **`n_clusters` definition is polygram's, not ours.** Different polygram strategies report different numbers for the same `W_dec`. We surface what polygram says; the README explicitly disclaims and links to polygram's docs.
- **Saturation advisory is a heuristic prompt.** Cluster count equalling capacity is a *signal* to consider more capacity, not a proof more concepts exist. The wording reflects this; no refusal.
- **No predictive value (yet).** We don't claim "high `redundancy_ratio` ⇒ low post-forge KL." That's a research question. The diagnostic is descriptive; correlation analysis is future work.
- **Cross-encoding row count multiplies.** Existing `--encoding rung3,rung4,rung5` already produces 3× rows; analysts comfortable with that surface inherit the same shape here. No new explosion.
