# Smoke gate results — `fix-scale-boost-calibration`

Run 2026-05-16 on Intel 16GB MBP. Target: jbloom GPT-2 layer-8 SAE
sliced to top-1024-by-norm (the full 24576-feature SAE doesn't fit
HEA_Rung2 cap at any tractable n_qubits), HEA_Rung2 n_qubits=10,
K ∈ {25, 103, 163, 211}.

## Headline

**The proposal's auto-calibration premise was falsified.** Three
successive proxies for the forge's `faithfulness_kl` were tried and
all picked the wrong `scale_boost`. The change merged as
**diagnostics-only**: it surfaces the magnitude/anomaly signals that
explain WHY a sweep produced bad forge KL, but does not attempt to
auto-pick `scale_boost`. The structural fix for the underlying
blow-up is deferred to a separate proposal.

## Setup

- jbloom GPT-2 layer-8 SAE, sliced to top-1024 features by decoder
  norm. Saved as `smoke_fix_scale_boost/jbloom_l8_n1024.safetensors`
  in the smoke run directory (not committed — see `.gitignore`).
- HEA_Rung2 with `n_qubits=10` (cap=1024).
- K targets: 25, 103, 163, 211. Compressed bases:

| K target | actual kept | basis_rank | quality_tier |
|----------|-------------|------------|--------------|
| 25       | 25          | 34         | degenerate   |
| 103      | 103         | 177        | undersized   |
| 163      | 163         | 287        | undersized   |
| 211      | 203         | 523        | good         |

(The K=211 target was not reached by polygram; surviving features
landed at 203. All ranks are below `d_model=768`.)

## Baseline arm (`scale_boost=1.0`)

Reproduced the documented blow-up:

| K (actual) | faithfulness_kl |
|------------|------------------|
| 25         | 8.21             |
| 103        | 31.31            |
| 163        | 27.31            |
| 203        | **86.39**        |

Delta K=25→K=203: **+78.18 nats**. The `--rank-monotonicity-check`
advisory fired correctly with two adjacent-pair violations.

## Proxy 1 — Residual-stream std-matching (the original proposal)

The original `scale_boost="calibrate"` mechanism: pick `sb` minimising
`|log(forged_residual_std) - log(host_residual_std)|`. Empirical
result: calibrate picked sb=1.0 at every K — bit-identical to the
broken baseline.

| K (kept) | best sb (manual sweep) | KL at best sb | logit_std_ratio at best | what proxy 1 picked |
|----------|-------------------------|----------------|--------------------------|---------------------|
| 34       | 0.10                   | 5.98           | 0.27                    | sb=1.0 (ratio≈0.39 ← closest to 1.0) |
| 177      | 0.10                   | 5.14           | 0.37                    | sb=1.0 |
| 287      | 0.10                   | 4.74           | 0.41                    | sb=1.0 |
| 523      | 0.10                   | 4.23           | 0.49                    | sb=1.0 |

The KL-optimal `scale_boost` has `logit_std_ratio` between 0.27–0.49
across K, NOT ~1.0. Std-matching is anti-correlated with KL in this
regime. **Proxy 1 falsified.**

## Proxy 2 — Layer-L logit shortcut KL

Switched to a "real KL" formulation but using a layer-L shortcut:
both host and forged logits computed as `residual @ lm_head`,
skipping the rest of the network. Empirical result: at K=25, calibrate
picked sb=0.25 (real forge KL 6.14) — 4× better than baseline. But:

| K (kept) | shortcut-KL says best sb | shortcut KL at that sb | real forge KL at that sb |
|----------|----------------------------|------------------------|----------------------------|
| 34       | sb=0.5                    | 1.83                   | 6.19                       |
| 177      | sb=0.5                    | 5.41                   | 15.73                      |
| 287      | sb=0.5                    | 4.25                   | 11.09                      |
| 523      | sb=0.5                    | 0.41                   | **41.89**                  |

Adding `sb=0.5` to the grid (the new shortcut-KL minimum) made real
forge KL strictly worse at every K. Shortcut KL systematically
underestimates how through-layer amplification compounds direction
errors. **Proxy 2 falsified.**

## Proxy 3 — Real end-of-network KL via remaining transformer blocks

Built a closure that runs the round-tripped residual through layers
L+1..N + final LN + lm_head, producing real end-of-network logits.
Verified bit-exact against the host's actual `lm_head` output.

| K (kept) | proxy 3 says best sb | KL at sb=1.0 (this proxy) | forge faithfulness_kl at sb=1.0 |
|----------|----------------------|----------------------------|----------------------------------|
| 34       | sb=5.0+              | 5.58                       | 8.21                             |
| 177      | sb=5.0+              | 3.97                       | 31.31                            |
| 287      | sb=5.0+              | 2.77                       | 27.31                            |
| 523      | sb=2.0               | 1.17                       | 86.39                            |

Proxy 3 picked sb=1.0 at every K (closest to the unbounded optimum
within the legal grid). On the forge: KL identical to baseline at
K=25–163, and 86.39 at K=203 — **70× larger than what proxy 3 saw**.

The forge's KL measures a fully-projected NativeModel (every layer's
weights touched). A one-shot residual perturbation at layer L doesn't
see the stacked-projection compounding across all 12 layers.
**Proxy 3 falsified.**

## Conclusion

No cheap proxy for forge KL exists in this design space. To target
forge KL directly would require running the full projection +
NativeModel forward per grid point — ~5× the cost of a single forge.
That's its own proposal.

The shipped change keeps everything that is independently useful:

- **Two row fields** (`logit_std_ratio`, `top1_anomalous`) — populated
  by `--magnitude-diagnostics`, surface magnitude collapse / blow-up
  / SolidGoldMagikarp signatures.
- **`--rank-monotonicity-check`** — flags adjacent K pairs where KL
  rises across an encoding sweep.
- **`advise_magnitude_diagnostics`** — post-sweep stderr summary.

The structural fix for the documented blow-up is deferred. It lives
in the projected NativeModel's stacked-layer forward pass —
characterise which layer(s) are driving the amplification, address
that directly. That's a separate proposal.

## Pattern note

When a proposal claims X is a "cheap proxy for forward KL", check
empirically whether the proxy actually tracks KL on the regime the
production code will see. Two instances now of "cheap proxy"
intuitions being wrong — direct measurement at sweep scale is cheap
(softmax+KL is milliseconds, dwarfed by forge), so the cost argument
that justifies most proxies is itself usually wrong. Saved as
`feedback_target_the_real_loss` in user memory.
