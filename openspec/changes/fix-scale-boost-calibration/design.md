# Design — `fix-scale-boost-calibration`

## Context

`SubspaceProjector.scale_boost` controls the magnitude of the encode
operation (`x @ E * scale_boost`) when mapping a host's residual stream
into the n-feature basis. PR #8 / 0.2.4 (2026-05-08) introduced
`scale_boost="auto"` resolving to `min(1.0, d_model / n_features)`,
plus a `UserWarning` when an over-complete basis is constructed with
the literal 1.0.

Two empirical findings motivated this change
([[project_kl_nonmonotonic]], [[project_auto_scale_boost]]):

1. On GPT-2 layer 8 with polygram-default knobs and the jbloom SAE,
   `faithfulness_kl` is non-monotone in kept-feature count under the
   default `scale_boost=1.0`: KL grows 8.2 → 86 across K ∈ {25, 103,
   163, 211} (smoke-reproduced 2026-05-16; ~78-nat blow-up).
2. On a 1.32× over-complete GPT-2 basis, `"auto"` picks
   `scale_boost=0.758` and produces KL 153.5; manual best at
   `scale_boost=0.01` gives KL 6.71.

The original proposal added a `scale_boost="calibrate"` mode that
would auto-pick the right value from a five-point grid. The shipped
change is **strictly weaker than that** — it adds diagnostic surface
that surfaces the magnitude/anomaly signals related to the failure
modes, but does not pick `scale_boost`. The reason is the proposal's
mechanism was empirically falsified mid-implementation, and no proxy
for the forge's faithfulness KL we tested could find the right
`scale_boost` cheaply.

## Goals / Non-Goals

**Goals:**

- Surface a `logit_std_ratio` diagnostic on every `ParetoFrontierRow`
  when opt-in `--magnitude-diagnostics` runs.
- Surface a `top1_anomalous` canary on every diagnostic row.
- Print a post-sweep stderr advisory listing per-row diagnostics +
  any anomalous-canary fires.
- Provide a `--rank-monotonicity-check` advisory that detects the
  documented non-monotonicity pattern across a sweep's K grid.

**Non-Goals:**

- Auto-picking `scale_boost`. Disproved by the smoke gate.
- Calibrating any other projector parameter.
- Refusing sweeps based on monotonicity violations. Advisory only.
- Modifying `SubspaceProjector` behaviour. The `"auto"` /
  literal-float paths are byte-identical to pre-change.

## Decisions

### Decision 1 — Diagnostics only; no auto-picking

The 2026-05-16 smoke gate ([[project_fix_scale_boost_smoke]]) ran
three successive proxies for the forge's faithfulness KL:

1. **Residual-stream std-matching** (the original proposal's
   Decision 2). Anti-correlated with KL on the tested regime —
   KL-optimal `scale_boost` had `logit_std_ratio` of 0.27–0.49 across
   K, not ≈ 1.0. Std-matching picks `sb=1.0` and produces KL identical
   to the broken baseline.
2. **Layer-L logit shortcut KL** (compute KL using
   `residual @ lm_head` for both host and forged, skipping the rest
   of the network). Closer to right but still wrong: it picked
   `sb=0.5` as best where real forge KL says `sb=0.25` is best by
   a factor of ~80×. Adding `0.5` to the grid made forge KL strictly
   worse at every K.
3. **Real end-of-network KL via a closure through the remaining
   transformer blocks + final layer norm + lm_head**. This is the
   "real KL" of a residual perturbation at layer L — but it ALSO
   diverges from the forge's `faithfulness_kl`. The forge's KL
   measures a fully-projected `NativeModel` (every layer's weights
   touched), whose forward pass compounds direction errors across
   all 12 GPT-2 layers. A one-shot residual perturbation can't see
   that compounding.

So the chase for a "cheap proxy for forge KL" came up empty: any
proxy that doesn't actually run the forge gets the wrong answer.
Real forge-level calibration would cost ~5× the forge per row, a
significant scope expansion. We chose to ship only the diagnostic
surface (which is independently useful) and defer the auto-picking
question to a separate proposal that addresses the root cause —
which is structural in the projected NativeModel, not a magnitude
issue.

### Decision 2 — Calibration corpus: built-in token-capped sample, prompt-path override

Default diagnostic-corpus uses a built-in fixed-seed token-capped
sample (~1024 tokens of generic English from a wikipedia-style
corpus). Override path: `--magnitude-diagnostics prompts:PATH`
accepts a JSONL with `{"text": ...}` lines.

The built-in corpus is deterministic. Token count default 1024 is a
starting point — large enough for stable logit-std estimates,
small enough for negligible compute.

### Decision 3 — Two diagnostic flags, not one combined boolean

`logit_std_ratio` (float) and `top1_anomalous` (bool) are emitted
separately. They detect related-but-distinct failure modes:

- High `logit_std_ratio` (≫1) AND `top1_anomalous=True`: classic
  blow-up — magnitude saturated AND argmax landed on a rare-vocab
  anomalous token (`���`, `cloneembedreportprint`).
- Low `logit_std_ratio` (≪1) AND `top1_anomalous=False`: classic
  collapse — magnitudes squashed, predictions are coherent but boring
  (` the` everywhere).
- `logit_std_ratio ≈ 1.0` AND `top1_anomalous=False`: well-calibrated
  on magnitude (but doesn't guarantee low forge KL — see Decision 1).
- `logit_std_ratio ≈ 1.0` AND `top1_anomalous=True`: rare edge case
  worth surfacing — magnitudes match host but the argmax went
  anomalous. Likely an angle-of-projection problem.

A single combined boolean would lose this triangulation.

### Decision 4 — Rank-monotonicity check is advisory, not gating

Adjacent K pairs within the same encoding label with
`KL[high] - KL[low] > 0.1` print a stderr advisory. The sweep
continues. The 0.1-nat tolerance is generous (grid-noise comfortable);
the goal is to catch the 8.2 → 86 pattern, not micro-fluctuations.

**Alternative considered**: refuse on violation. Rejected — the
analyst may deliberately be sweeping at sb=1.0 to *verify* the
documented blow-up.

## Architecture Sketch

```
saeforge/calibration.py (new)
├── load_calibration_corpus(host_model_id, layer, n_tokens=1024, prompts_path=None) → np.ndarray
├── load_host_unembed(host_model_id) → np.ndarray
├── ANOMALOUS_TOKEN_IDS: dict[tokenizer_name, set[int]]
├── compute_host_logit_std(host_acts, host_unembed) → float          (row field)
├── compute_forged_logit_std(host_acts, projector, host_unembed) → float (row field)
└── top1_is_anomalous(host_acts, projector, host_unembed, anomalous_set) → bool

saeforge/projector.py
├── SubspaceProjector  (UNCHANGED behaviour)
│   ├── scale_boost: Union[float, str] = 1.0   (only "auto" or float)
│   └── _auto_scale_boost() → float

saeforge/sweep.py (modified)
├── ParetoFrontierRow
│   ├── logit_std_ratio: float | None  (new)
│   └── top1_anomalous: bool | None  (new)
├── sweep_pareto(..., magnitude_diagnostics, rank_monotonicity_check)  (new kwargs)
├── _maybe_advise_rank_monotonicity(rows)  (new)
└── advise_magnitude_diagnostics  (lives in forge_quality.py)

saeforge/cli.py (modified)
└── sweep-pareto subparser
    ├── --magnitude-diagnostics VALUE  (new)
    └── --rank-monotonicity-check     (new)

saeforge/forge_quality.py (modified)
└── advise_magnitude_diagnostics(rows)  (post-sweep stderr advisory)
```

## Smoke gate result (2026-05-16)

Pre-merge smoke run on Intel 16GB. Captured in `smoke-results.md` in
this change dir (the audit trail).

- Baseline arm (`scale_boost=1.0`) reproduces the documented blow-up
  on the sliced jbloom SAE / HEA_Rung2(n_qubits=10): KL 8.21 → 86.39
  across K ∈ {25, 103, 163, 203}. **PASSED.**
- `--magnitude-diagnostics` populates `logit_std_ratio` and
  `top1_anomalous` on every row. The advisory prints the per-row
  ratios. **PASSED.**
- `--rank-monotonicity-check` fires correctly with the right
  format (K-pair, KL_low, KL_high, delta). **PASSED.**
- Originally proposed `--scale-boost-calibrate` gate (monotone
  non-increasing KL after calibration): **DROPPED**. Three proxies
  tried, all picked wrong sb.

## Sequencing and dependencies

This change is purely additive — no upstream proposals are blocked
by it. Downstream proposals that depend on monotone KL behaviour
(probing refinement, block-structured A/B at calibrated KL) need a
**different** fix: root-cause work on the projected NativeModel,
not a magnitude knob. That work is its own proposal.
