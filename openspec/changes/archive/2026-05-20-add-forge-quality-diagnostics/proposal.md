## Why

`sae-forge`'s high-level goal is to forge native models that are excellent — small transformers whose forward pass produces logits close to a host model's, in a SAE-derived feature basis. "Excellent" means *useful*: KL between forged and host logits should be on the order of ≤1.0 for the forged model to inherit meaningful capabilities.

Today the tool has **no opinion about forge feasibility**. PR #33's live Axis-4 smokes produced clean exit codes, well-formed JSONL frontier rows, and `faithfulness_kl` values around 6.99–7.08. Those KL values are catastrophically high (≈ near-random output) — every forge in the sweep was producing a near-useless model. The frontier rows looked fine; the impl was correct; but the *forge quality* was uniformly garbage and there was no in-row signal flagging that.

The root cause was a structural mismatch the sweep tooling didn't surface: GPT-2-small has a 768-dim residual stream, and the smoke compressed a 32-feature SAE down to bases of rank 1–4. There is no way to span a 768-dim residual with a rank-1 basis. The forge step ran honestly, fine-tuned what it could, and produced the best rank-1 model possible — which is still ~near-random. The tool ran a 5-minute experiment whose outcome was foreseeable from the input shape alone.

`add-forge-quality-diagnostics` makes that foreseeability **a first-class signal in the row schema and a pre-flight advisory**. Analysts get `basis_rank`, `host_d_model`, `quality_ratio`, and `quality_tier` per row, and the sweep CLI warns up front when a setup looks degenerate. Doesn't change forge behaviour; doesn't refuse setups; just surfaces what a domain expert already knew the moment they saw `K=4, host=gpt2`.

This is the prerequisite for trusting `kl_attribution` / rep_selection improvements (polygram-side follow-up). Rep selection is a quality knob in **good regimes**; until the analyst can identify which rows are in a good regime, swapping reps is rearranging deck chairs.

## What Changes

### `ParetoFrontierRow` gains four diagnostic fields

All optional, all default `None` (backwards compat for existing `frontier.jsonl` consumers):

- **`host_d_model: int | None`** — host transformer's residual stream width (= `hidden_size` from `AutoConfig.from_pretrained(host_model_id)`). Cached once per sweep; cheap config-only fetch with no weight load.
- **`basis_rank: int | None`** — numerical rank of the kept-features basis matrix `W_dec[surviving_rows, :]`, computed via `numpy.linalg.matrix_rank` with the default tolerance. Reflects the *actual* span of the basis, which can be less than `n_features_kept_actual` if surviving features are linearly dependent.
- **`quality_ratio: float | None`** — `basis_rank / host_d_model`. The single number a frontier plot should colour rows by; values near 1.0 are saturated, values << 0.1 are degenerate.
- **`quality_tier: str | None`** — categorical label from the ratio, one of `"saturated"`, `"good"`, `"undersized"`, `"degenerate"`. Thresholds documented in the spec; tweakable via `--quality-tier-thresholds`.

Default tier thresholds (configurable):
- `saturated`: `quality_ratio >= 1.0`
- `good`: `0.5 <= quality_ratio < 1.0`
- `undersized`: `0.0625 <= quality_ratio < 0.5` (host_d_model/16 to host_d_model/2)
- `degenerate`: `quality_ratio < 0.0625`

These are heuristic defaults from a small empirical base (the GPT-2 + jbloom-SAE smokes) — useful for triage, not load-bearing for the algorithm. Analysts who want stricter cutoffs override via the CLI.

### Pre-flight advisory in `sweep-pareto`

Before the first forge call, the driver SHALL:

1. Resolve `host_d_model` once via `transformers.AutoConfig.from_pretrained(host_model_id).hidden_size`. Cached for the sweep duration.
2. For each `--encoding` argument, examine the SAE checkpoint with the smallest target K (the smallest-rank planned basis), compute its `basis_rank`, and derive `quality_tier`.
3. If ANY encoding's smallest-K tier is `"degenerate"` or `"undersized"`, print a stderr advisory listing the affected encoding + K, the computed ratio, and a suggested K floor (the smallest K whose `basis_rank >= host_d_model / 2`, derived from the manifest's per-K cluster counts).
4. The advisory is **informational only by default** — the sweep proceeds. Analysts opt in to refusal via `--quality-floor RATIO`.

### New CLI flag: `--quality-floor RATIO`

When set, the sweep refuses if any K's projected `quality_ratio` falls below the floor. Default: not set (advisory-only). Suggested usage: `--quality-floor 0.5` for "I only want sweeps where every row is at least in the `good` tier."

### Optional: `--quality-tier-thresholds`

`--quality-tier-thresholds saturated:1.0,good:0.5,undersized:0.0625` lets the user override the default thresholds. Power-user knob; rarely needed.

### Out of scope, deliberately

- **`basis_rank` calculation via SVD or condition number rather than `matrix_rank`.** The default `matrix_rank` is cheap and adequate; switching to a condition-number-aware tier is a follow-up.
- **Predictive KL estimation.** This proposal surfaces structural diagnostics (rank, ratio); it does not estimate the post-forge KL from those numbers. That's a research project.
- **Auto-recommending K floors based on the host.** The advisory *prints* a suggested floor; it doesn't rewrite the user's `--pareto` list.
- **Tier-based forge skipping.** Even `degenerate` rows still run a forge (and the resulting KL row is informative for the analyst). `--quality-floor` controls refusal explicitly; nothing skips silently.
- **Pre-flight checks for the eval prompt corpus** (e.g., warn if `--eval-prompts` has fewer than N tokens). Worth doing later as a separate diagnostic; out of scope here.

## Capabilities

### Modified Capabilities

- `pareto-sweep`: `ParetoFrontierRow` gains four new optional fields; `sweep-pareto` CLI gains the pre-flight advisory, `--quality-floor`, and `--quality-tier-thresholds` flags. Existing rows / invocations byte-identical when the new fields aren't requested or when the host fit is `good`/`saturated`.

## Impact

- **New module**: `saeforge/forge_quality.py` — `QualityTier` enum, `compute_basis_rank(W_dec) -> int`, `classify_quality(basis_rank, host_d_model, thresholds) -> QualityTier`, `resolve_host_d_model(host_model_id) -> int`, `advise_sweep_quality(encodings, manifests, host_d_model) -> str | None` (returns the stderr advisory text, or None when no warning is warranted).
- **Modified**:
  - `saeforge/sweep.py` — `ParetoFrontierRow` gains four new fields with `None` defaults; `sweep_pareto` accepts optional `quality_floor: float | None = None` and `quality_thresholds: QualityThresholds | None = None` kwargs; `_process_row` populates the new fields from the actual loaded basis.
  - `saeforge/cli.py` — new `--quality-floor`, `--quality-tier-thresholds` flags on `sweep-pareto`; `_cmd_sweep_pareto` calls `resolve_host_d_model` + `advise_sweep_quality` before the sweep loop.
  - `saeforge/forge.py` — `ForgePipeline.sweep_pareto` gains pass-through kwargs.
  - `saeforge/__init__.py` — export `QualityTier`, `QualityThresholds`.
- **No breaking changes**: row schema extension is forward-compatible (existing readers see `null` for the new fields); no behaviour change unless the new flags are set.
- **Dependencies**: `transformers` already pulled in by the `[torch]` / `[intel]` extras; no new deps. The `AutoConfig.from_pretrained` call requires network access on first use per host (cached afterward by the HF cache).
