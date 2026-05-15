## Context

PR #33's Axis-4 smokes (2026-05-14, GPT-2 + jbloom SAE, N=8 and N=32) produced clean exit codes, well-formed `frontier.jsonl`, and `faithfulness_kl` values of 6.50–7.08. The impl was correct; every spec scenario passed. But the *forge quality* was uniformly catastrophic — KL≈7 corresponds to a forged model whose output is essentially decoupled from the host's. Useful forges have KL on the order of ≤1.0.

The mismatch was visible in retrospect: GPT-2-small has a 768-dim residual stream; the smokes compressed 8- or 32-feature SAEs down to bases of rank 1–4. No rank-1 basis can span a 768-dim residual. The forge step ran honestly and produced the best rank-1 model possible, which is still near-random output.

The frontier row schema today carries `faithfulness_kl` but no signal about *whether that KL is interpretable*. Analysts looking at the JSONL see four numbers; without expert prior they have no way to know whether 6.99 means "this is a bad forge" or "this is a normal KL for this setup."

**This proposal makes forge feasibility legible in the row schema and at sweep-time, without changing forge behaviour.**

## Goals / Non-Goals

**Goals:**
- Surface `basis_rank` and `host_d_model` as numeric fields on every frontier row, alongside the existing `faithfulness_kl`.
- Surface a categorical `quality_tier` so analysts can filter trivially: `jq 'select(.quality_tier == "good" or .quality_tier == "saturated")'`.
- Print a pre-flight advisory when the sweep configuration looks degenerate, with a suggested K floor.
- Optional strict mode (`--quality-floor RATIO`) that refuses degenerate sweeps before they pay forge cost.
- Backwards compatibility: existing `frontier.jsonl` consumers see `null` for the new fields.

**Non-Goals:**
- Estimating post-forge KL from structural inputs. The diagnostic is `rank/d_model`, not `predicted_kl`.
- Refusing forge runs by default. The default is advisory-only; refusal is opt-in.
- Eval-prompt-corpus quality checks (different axis; deserves its own proposal).
- Rewriting the user's `--pareto` list to enforce the floor. The CLI prints a suggested floor; the user retains agency.
- Changing forge mechanics. Same `SubspaceProjector`, same `ForgePipeline.run`, same FSM. Pure diagnostics layer.

## Decisions

### Decision 1 — `numpy.linalg.matrix_rank` with default tolerance

`basis_rank` is computed via `numpy.linalg.matrix_rank(W_dec_kept, tol=None)` where `W_dec_kept` is the surviving features' decoder rows. Default tolerance uses the machine-precision-aware heuristic numpy already provides; no manual epsilon.

**Alternative considered**: SVD-based rank with a configurable tolerance. Rejected — adds a knob without empirical evidence the default is wrong; can be added later if needed.

**Alternative considered**: count non-zero rows in `W_dec_kept` (simpler than rank). Rejected — would miss the case where two kept features have linearly dependent decoder rows (rare but possible), and gives a misleadingly high rank.

### Decision 2 — Cache `host_d_model` once per sweep via `AutoConfig`

`transformers.AutoConfig.from_pretrained(host_model_id).hidden_size` fetches only the config (network-fast, cached by HF), no weight load. Done once before the forge loop; reused across all rows. The cost is one HTTP fetch on the first run (cached afterward).

**Alternative considered**: read d_model from the host model after it loads in the first forge. Rejected — the pre-flight advisory has to fire BEFORE the first forge, so we need d_model up front.

**Alternative considered**: take `--host-d-model N` as an explicit CLI flag. Rejected — the config lookup is so cheap there's no reason to force the user to know this. Could be added as an override flag if `AutoConfig` ever fails for an unsupported host.

### Decision 3 — Four-tier categorical with heuristic thresholds

Tiers and default thresholds:

| Tier | Ratio range | Heuristic meaning |
|------|-------------|-------------------|
| `saturated` | `>= 1.0` | Basis can fully span the host residual stream; forge bounded only by fine-tune budget |
| `good` | `[0.5, 1.0)` | Substantial coverage; forge can plausibly inherit most host capabilities |
| `undersized` | `[0.0625, 0.5)` | Basis covers some-but-not-most of residual; expect KL >> 0 even after fine-tune |
| `degenerate` | `< 0.0625` | Rank too low to span residual; KL will be near catastrophic, results not interpretable as "what does this SAE forge to" |

The `0.0625` lower bound (= 1/16) is a rule-of-thumb breakpoint motivated by **the PR #33 N=32 Rung4 smoke**: bases of rank 1 against GPT-2's 768-dim residual produced KL ≈ 6.99 (= near-random output entropy for a small vocab). That's `ratio = 1/768 ≈ 0.0013` — two orders of magnitude below 0.0625, well into the "the forged model is decoupled from the host" regime. The boundary is set high enough above that pathological floor that any setup tagged `degenerate` is empirically known to produce uninterpretable forges; below `0.0625` is "you're effectively forging a constant model." The `0.5` good/undersized boundary is symmetric (half-coverage); the `1.0` saturated boundary is the obvious "basis can in principle span the full residual" cutoff. All three are heuristic defaults from a small empirical base (one host family); the spec marks them as adjustable via `--quality-tier-thresholds`, and revisiting after cross-host data is explicit follow-up work.

The wording `degenerate` describes *the structural rank ratio*, not the validity of the run. Exploratory low-rank smokes (like the PR #33 ones that surfaced this very problem) remain valid as impl validation — they prove the sweep mechanics work even when the basis is doomed. The advisory message format explicitly states this so users running intentional low-rank experiments don't feel chastised.

**Why discrete tiers and not just the ratio**: analysts using `jq` for frontier triage want a single string filter, not a ratio threshold to remember. The numeric ratio is also emitted for plotting.

**Alternative considered**: continuous score from 0–1 instead of tiers. Rejected — useful for plots but not for triage filters. Both are emitted (ratio is the continuous version).

### Decision 4 — Advisory by default, refusal opt-in

The pre-flight check prints stderr advisories when the setup looks degenerate but does NOT exit non-zero. Refusal requires `--quality-floor RATIO`.

**Why default-advisory**: many legitimate users want to explore the degenerate regime (e.g., this very impl PR's smokes were valuable even though they were structurally garbage — they validated the impl). Hard-refusal-by-default would create friction for legitimate exploration.

**Alternative considered**: refuse by default with `--allow-degenerate` to opt out. Rejected — punishes the common cheap-smoke use case.

**Alternative considered**: emit a structured machine-readable warning (e.g., a `WARNING` field on every row). Rejected — adds row-schema noise; the categorical `quality_tier` already encodes the warning.

### Decision 5 — Advisory is per-encoding, examines the SMALLEST K only

The advisory loop runs once per `--encoding` argument and examines only the smallest-K materialised SAE (because that's the worst-case basis rank for the sweep). If that K is `good`/`saturated`, the whole encoding's sweep is fine; no advisory printed. If degenerate, the advisory names the encoding and K, computes the suggested floor (the smallest K from the manifest whose basis rank exceeds `host_d_model / 2`), and prints once.

**Why smallest K only**: ratio is monotonically increasing in `n_features_kept_actual` (more rows in `W_dec_kept` → rank can only go up). If the smallest K is fine, all larger K's are at least as fine.

**Alternative considered**: examine every K and emit a per-K advisory line. Rejected — noisy; the smallest-K check is sufficient.

### Decision 6 — Diagnostics computed pre-forge, populated post-forge

`basis_rank` is computed when the SAE checkpoint is loaded inside `_basis_swap` (which already reads `W_dec` for the projector). `host_d_model` is cached at sweep start. `quality_ratio` and `quality_tier` are derived from these two values. All four are populated on the `ParetoFrontierRow` regardless of whether the forge succeeded (including row failures, where `faithfulness_kl` is null but the diagnostic fields are still meaningful).

**Why pre-forge values on post-forge rows**: the diagnostic is about *the structural setup* the forge inherited, not about *what the forge produced*. Failure rows are exactly where diagnostics matter most — they tell you whether the failure was forge-recoverable or structurally doomed.

### Decision 7 — `--quality-tier-thresholds` is a power-user override

The threshold defaults are heuristics; analysts running specific research may want different cutoffs. The CLI accepts `--quality-tier-thresholds saturated:1.0,good:0.5,undersized:0.0625` to override, but the flag is documented as rarely-needed and the defaults are the load-bearing values for the categorical labels.

**Why expose at all**: research integrity — analysts who report a `quality_tier` distribution in a paper need to be able to document and tweak the thresholds.

## Risks / Trade-offs

- **Heuristic tier thresholds may be wrong for non-GPT-2 hosts.** The `0.5` good/undersized boundary and `0.0625` degenerate boundary come from a small empirical base. Spec marks them as defaults; configurable. Tightening or loosening based on cross-host data is a future refinement.

- **`matrix_rank` can be misleading for noisy weight matrices.** If `W_dec_kept` has very small singular values that numerically pass the rank test, the reported rank may overstate the actual span. The default tolerance is reasonable for SAE-derived decoders; spec notes this as a known limitation.

- **`AutoConfig.from_pretrained` requires network access on first use.** Subsequent runs hit the HF cache. Failure to fetch (offline, gated model) means `host_d_model` is `None` and all the derived fields are `None`. The advisory falls back to "could not resolve host d_model" warning instead of refusing the sweep — diagnostic is informational, not load-bearing.

- **Non-residual-stream hosts and non-transformer hosts.** The `hidden_size` lookup is calibrated for residual-stream LMs (GPT-2, Llama, Gemma, Pythia). For exotic hosts — Whisper encoder (`d_model` field, may differ), encoder-decoder architectures (separate encoder/decoder widths), CV transformers (`hidden_size` exists but residual-vs-channel interpretation differs), non-transformer hosts (no `hidden_size` at all) — the resolved `host_d_model` may be missing, mismatched, or misleading. The driver SHALL fall back to `host_d_model = None` (and skip the advisory) when `AutoConfig.from_pretrained(...).hidden_size` raises `AttributeError`. For successful-but-questionable resolutions, the spec recommends a one-line caveat in the advisory: "host d_model resolved as N via AutoConfig; interpretation as residual-stream width assumes a standard transformer architecture." Sae-forge's per-host adapter registry (`saeforge.adapters`) may grow per-host overrides in a follow-up; out of scope here.

- **`quality_tier='degenerate'` is loaded language.** Some users will hit this on intentional low-rank smokes (like PR #33's runs) and feel chastised. The wording in stderr and docs SHALL make clear that `degenerate` describes the rank ratio, not the value of the run — exploratory smokes legitimately operate there.

- **No predictive accuracy for `quality_tier`.** We do not yet have data to claim "rows tagged `good` produce KL < 1.0." The tier is a *structural* signal; the analyst correlates tier with post-forge KL as they collect data. Future work could fit an empirical KL-vs-ratio model.
