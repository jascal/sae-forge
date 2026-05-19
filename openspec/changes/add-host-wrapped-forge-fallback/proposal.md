# Add host-wrapped forge fallback for under-complete bases

## Why

`fix-scale-boost-calibration`'s 2026-05-16 smoke gate documented a
structural KL blow-up on the GPT-2 layer-8 reference sweep:
`faithfulness_kl` grows 8.21 → 86.39 across K ∈ {25, 103, 163, 211}
at `scale_boost=1.0` ([[project_kl_nonmonotonic]],
[[project_fix_scale_boost_smoke]]). That work tested three KL proxies
and shipped diagnostics-only, with the structural fix deferred: "the
blow-up is structural — it happens inside the projected NativeModel's
forward pass, where stacked projections compound direction errors
across all 12 GPT-2 layers."

A 2026-05-18 layer-amplification diagnostic
(`scripts/diagnose_layer_amplification.py`) localised the mechanism:

1. **Block 0 is the sole amplifier.** Forged residual norms post-
   block-0 jump 0.40× → 2.92× → 8.39× → **52.3×** host norms across
   K=25 → 211. All 12 subsequent blocks roughly preserve that magnitude
   without further amplification; `via_host_kl` saturates at ~10–15
   nats from layer 1 onward.

2. **The driver is `||pseudoinverse(W_dec)||_2`.** Every projected
   residual-aligned parameter (`ln_*.weight`, `ln_*.bias`,
   `attn.c_proj.weight/bias`, `mlp.c_proj.bias`) inherits the
   pseudoinverse's max singular value. At K=25 → 211: `pinv_max_sv`
   grows 1.71 → 9.30, and every encoded parameter's
   forged-to-host-norm ratio tracks it within ±0.1.

3. **The root cause is a category error.** `LayerNorm.weight` and
   `LayerNorm.bias` are *per-coordinate gains*, not vectors in
   residual space. The current code projects them with
   `encode(γ) = γ @ pinv`, which is mathematically equivalent to
   treating a per-coord gain as if it were a residual point — there
   is no isomorphism between per-coord gains in host space and
   per-coord gains in basis space when the basis is non-orthonormal.
   **LayerNorm does not commute with basis change.**

A direct fix attempt (`scripts/validate_host_native_ln.py`) replaced
forged LayerNorm with `decode → LN_host → encode` and produced a
non-uniform improvement: K=211 KL dropped 89.9 → 64.4, but K=25/103/
163 KLs grew 4–22 nats. The decode/LN-host arm normalizes over the
rank-deficient projection `decode(z) = z @ W_dec`, whose variance
estimate diverges from a "true" host residual's. The `c_proj` and
residual-bias paths still inherit the `||pinv||_2` amplification —
fixing LayerNorm alone is insufficient.

**The deeper finding is that "native transformer in basis space" is
not an exact reformulation of the host transformer for any
non-orthonormal basis.** Every nonlinearity (LayerNorm's variance,
attention softmax) sees basis-space statistics that differ from
host-space statistics. The current `saeforge` design implicitly
assumes high-fidelity bases (quality_tier ∈ {`good`, `saturated`})
where the gap is small. For under-complete bases (`undersized`,
`degenerate`) the gap compounds across layers and produces the
documented blow-up.

This change adds a second forward implementation — `forward_mode=
"host_wrapped"` — that runs every transformer block host-native:
residual stream stays in basis space, but at each block boundary the
residual is decoded to d_model, the host's exact transformer block
runs (with its original weights, nonlinearity, and norm statistics),
and the result is re-encoded. Forge KL is guaranteed monotone in
basis rank by construction: every additional kept feature reduces the
decode/encode round-trip error and tightens the residual-stream
bottleneck.

Defaults dispatch via basis quality tier:

- `quality_tier ∈ {good, saturated}` → `native_in_basis` (the current
  forward pass, which works in this regime and preserves the "small
  forged transformer" parameter count). Native-in-basis remains the
  *highlighted path*; this is the regime polygram-clustered SAEs land
  in.
- `quality_tier ∈ {undersized, degenerate}` → `host_wrapped` (the
  new fallback). Compute equals host inference. Parameter count
  equals host parameter count plus the basis matrices. Provided so
  the library degrades gracefully on under-complete inputs rather
  than emitting blown-up KL.

## What Changes

### New `NativeModelConfig.forward_mode` field

Optional, default `"auto"`. Accepts:

- `"auto"` — dispatch by basis quality tier (`good`/`saturated` →
  `native_in_basis`; `undersized`/`degenerate` → `host_wrapped`).
  Default. Logs the chosen mode at INFO once per
  `NativeModel.from_host` call.
- `"native_in_basis"` — force the current forward (existing v0.5.1
  behaviour; byte-identical when the call would have selected this
  mode anyway).
- `"host_wrapped"` — force the new fallback.

Backwards-compatible: callers that don't set `forward_mode` get
`"auto"`, which selects `"native_in_basis"` exactly when
`quality_tier` is `good`/`saturated`. The smoke regime
(`quality_tier ∈ {degenerate, undersized}` at every K) would now
select `host_wrapped` automatically and produce monotone KL — the
behaviour change is intentional and lands behind a `--force-forward-
mode` CLI escape hatch for reproducibility of the documented blow-up.

### New native-module forward path: `_HostWrappedForward`

A family-agnostic wrapper module that stores:

- Frozen references to the host transformer's `wte`, `wpe`,
  `transformer.h` (or equivalent for non-GPT-2 families), `ln_f`,
  `lm_head`. Host weights are **not projected** in this mode.
- `W_dec` and `pseudoinverse(W_dec)` as registered buffers (shape
  `(n_features, d_model)` and `(d_model, n_features)`).
- The optional `scale_boost` scalar (defaults to `1.0` in
  host-wrapped mode — there's no rank-deficiency-driven blow-up to
  shrink-compensate for).

Forward (causal-LM, GPT-2-style):

```
x_host = host_wte(input_ids) + host_wpe(pos)        # (B, T, d_model)
z = x_host @ pinv * scale_boost                     # encode to basis
for block in host_transformer.h:
    x_host = z @ W_dec                              # decode
    x_host = block(x_host)                          # host-native block
    z = x_host @ pinv * scale_boost                 # re-encode
x_host = z @ W_dec
x_host = host_ln_f(x_host)
return host_lm_head(x_host)                         # (B, T, vocab)
```

The residual stream `z` is in basis coordinates at every block
boundary — the interpretability contract is preserved. Every per-op
nonlinearity sees host-native statistics.

Non-LM families (Whisper encoder) wrap the encoder's host transformer
analogously, with the same encode/decode pattern at every block
boundary.

### Per-family adapter responsibilities

Adapters MUST grow one new method:

```python
def host_wrapped_module(
    self,
    host_model,
    basis: FeatureBasis,
    scale_boost: float,
) -> nn.Module:
    """Construct a host-wrapped forged module for this family."""
```

The bundled adapters (GPT-2, Llama, Gemma-2, Qwen2, Qwen3,
Qwen3-MoE, Whisper-encoder) SHALL each implement this. v1 lands
GPT-2 only; the other adapters MAY raise `NotImplementedError` with
a clear message until follow-up. The `architecture-adapters` spec
gains a delta describing the new method.

### `NativeModel.from_host` dispatches on `forward_mode`

When the resolved mode is `host_wrapped`:

1. Load the host model.
2. Resolve quality tier (compute `basis_rank`, classify).
3. Build the host-wrapped module via the adapter; do NOT call
   `projector.project_module`.
4. Skip `from_projected_weights`; instead, store the host modules
   directly.

When the resolved mode is `native_in_basis`: existing path, no
change.

### CLI: `--forward-mode`

`forge` and `sweep-pareto` subcommands gain a `--forward-mode`
argument accepting `auto`/`native_in_basis`/`host_wrapped`. Default
`auto`. Threaded into `NativeModelConfig`. The `sweep-pareto` driver
records the resolved mode per row in a new optional
`ParetoFrontierRow.forward_mode_resolved: str | None` field.

### `ParetoFrontierRow` schema extension

One new optional field, default `None`, forward-compatible with
existing readers:

- `forward_mode_resolved: str | None` — the forward mode actually
  used (`"native_in_basis"` or `"host_wrapped"`). `None` for rows
  produced before this change or for invocations that bypassed the
  config field. Populated when the row's forge succeeded.

### Out of scope, deliberately

- **Modifying the `native_in_basis` math.** The documented amplification
  is a property of running the basis-projected ops outside the regime
  the projection is valid in. The fix is to *not run them there*; the
  math itself is unchanged in its good-tier regime. Tikhonov-regularized
  pseudoinverse / scale-normalized LN parameters are deferred follow-ups
  in their own openspec proposals if they turn out to materially tighten
  the marginal regime.
- **Removing the `scale_boost` knob.** It still affects encode magnitudes
  in `native_in_basis` mode. In `host_wrapped` mode it has minimal effect
  (since every block decodes/re-encodes — global scaling cancels at the
  decode step). The default stays `1.0`.
- **Re-instrumenting `add-forge-quality-diagnostics`' advisory.** The
  existing pre-flight advisory still fires its undersized/degenerate
  warning. The new dispatch reads the same quality tier but does NOT
  refuse — it just routes to the fallback. The advisory wording stays as
  shipped.
- **Continual-learning / fine-tune integration.** The forge-finetune-
  recipe and continual-learning FSM operate on the forged module's
  parameters. In `host_wrapped` mode those parameters are the host's
  (frozen by default). Fine-tuning a host-wrapped forge means
  fine-tuning the host weights — a different objective. v1 ships
  host-wrapped mode with fine-tune *disabled* (raises if
  `run_finetune` is called on a `host_wrapped` module); the
  forge-finetune extension for this mode is a follow-up proposal.
- **`bridges=True` in host-wrapped mode.** Hybrid bridges insert
  learnable layers between projected blocks. Host-wrapped has no
  projected blocks. v1 raises a clear error if `bridges=True` and
  `forward_mode="host_wrapped"`.

## Falsifiable acceptance gate

A 2026-05-18 prototype of the host-wrapped forward
(`scripts/prototype_host_wrapped_forward.py`) ran ahead of this
proposal landing and characterised what host-wrapped actually
delivers. Acceptance gates are calibrated to those measurements
rather than to aspirational pre-experiment numbers.

On the documented smoke regime (GPT-2 layer-8 jbloom SAE sliced to
1024 features, HEA_Rung2 n_qubits=10, K ∈ {25, 103, 163, 211},
`scale_boost=1.0`):

| K   | n_feat | KL `native_in_basis` | KL `host_wrapped` | reduction |
|-----|--------|----------------------|-------------------|-----------|
| 25  | 34     | 9.65                 | 9.58              | ~0%       |
| 103 | 177    | 37.91                | 9.65              | **3.9×**  |
| 163 | 287    | 29.23                | 12.48             | **2.3×**  |
| 211 | 523    | 89.91                | 15.42             | **5.8×**  |

- **Baseline arm** (`--forward-mode native_in_basis`): reproduces
  the documented blow-up trajectory (matches the smoke-results.md
  reference numbers within tolerance).
- **Fallback arm** (`--forward-mode host_wrapped`) SHALL satisfy:
  - At every K, `host_wrapped` KL ≤ `native_in_basis` KL.
  - K=211 KL strictly tighter than K=211 baseline (15.4 vs 89.9 in
    the prototype — gate: < 25 nats).
  - The rank-dependent amplification (where native KL grows with
    K's basis rank) SHALL be absent: no K-pair adjacent ΔKL > 10
    nats in the host_wrapped arm.
- **Auto arm** (`--forward-mode auto`): all four K values dispatch
  to `host_wrapped` (since `quality_tier` is `degenerate`/`undersized`
  at every K). KL trajectory matches the fallback arm byte-identically.
- **Good-tier sanity** (synthetic basis: `n_features = d_model = 768`,
  orthonormal `W_dec`) SHALL produce `host_wrapped` KL < 0.5 nats —
  confirming the fallback is mathematically exact when decode/encode
  is identity. Prototype result: KL ≈ 0 (float-precision zero).

### What host-wrapped does NOT deliver

The prototype shows host-wrapped KL on the smoke regime is *not*
monotone in K, and the proposal does *not* claim it should be. The
four smoke bases are non-nested — they're four independent polygram
K-target outputs, each picking a different subset of the SAE's
features. Forge KL in host-wrapped mode is bounded by each basis's
individual residual-stream approximation quality, which varies
non-monotonically across these particular bases. This is a property
of the bases, not the forward path.

Host-wrapped removes the **rank-dependent amplification** (the
catastrophic blow-up that scales with `||pinv||_2`). It does not, and
cannot, fix the irreducible basis-approximation error that limits
under-complete forge quality. Improving that requires improvements to
the basis itself (more features, better compression algorithm) — out
of scope for this change.

The smoke regime is preserved as a regression suite for the
*native_in_basis* arm (gated behind `--force-forward-mode
native_in_basis`) so future polygram-side changes can still surface
the documented amplification.

## Capabilities

### Added Capabilities

- `forge-forward-mode` (new) — defines the `forward_mode` field, the
  `auto` dispatch logic, and the host-wrapped forward contract.

### Modified Capabilities

- `architecture-adapters` — adapter interface gains
  `host_wrapped_module(host_model, basis, scale_boost) -> nn.Module`.
  v1: GPT-2 only; others raise `NotImplementedError` with a clear
  follow-up pointer.
- `pareto-sweep` — `ParetoFrontierRow` gains optional
  `forward_mode_resolved: str | None`. CLI `sweep-pareto` gains
  `--forward-mode VALUE`. Existing rows / invocations byte-identical
  when the new flag isn't supplied.
- `subspace-projector` — `NativeModelConfig` gains `forward_mode:
  str = "auto"`. Existing configs construct unchanged (defaulting to
  `auto`, which selects `native_in_basis` on good/saturated bases —
  byte-identical to v0.5.1 on those bases).

## Impact

- **New module**: `saeforge/forward_mode.py` — `resolve_forward_mode(
  basis, requested) -> Literal["native_in_basis", "host_wrapped"]`
  and the `_HostWrappedModule` factory (family-agnostic shell;
  delegates to `adapter.host_wrapped_module` for the per-family
  block).
- **New per-family files**: `saeforge/adapters/_host_wrapped/<family>.py`
  hosting the family-specific host-wrapped module class. v1 ships
  `gpt2.py` only.
- **Modified**:
  - `saeforge/model.py` — `NativeModelConfig.forward_mode`,
    `NativeModel.from_host` dispatches.
  - `saeforge/adapters/base.py` — interface adds
    `host_wrapped_module`.
  - `saeforge/adapters/gpt2.py` — implements `host_wrapped_module`.
  - `saeforge/adapters/{llama,gemma2,qwen2,qwen3,qwen3_moe,whisper}.py`
    — stub `host_wrapped_module` that raises `NotImplementedError`
    with a follow-up pointer.
  - `saeforge/sweep.py` — `ParetoFrontierRow.forward_mode_resolved`;
    `sweep_pareto` threads the kwarg.
  - `saeforge/cli.py` — `--forward-mode` argument on `forge` and
    `sweep-pareto` subparsers.
  - `saeforge/forge.py` — pipeline `ForgePipeline` accepts
    `forward_mode`.
- **No breaking changes**: row schema extension forward-compatible;
  the existing forward path is unchanged on good/saturated bases.
- **Dependencies**: no new external dependencies.

## Risks

- **Behaviour change for under-complete-basis users.** Anyone
  currently running with `quality_tier ∈ {undersized, degenerate}` at
  the default `scale_boost=1.0` and consuming the (currently blown-up)
  KL would see a different number after this lands. The blown-up KL
  is documented as a bug, not a feature, but the change in resolved
  forward_mode is observable. Mitigation: `--force-forward-mode
  native_in_basis` is a 1-flag escape hatch; the smoke regression
  preserves the documented numbers.
- **Compute cost in host-wrapped mode.** Equal to host inference plus
  ~24 small matmuls per forward (decode + encode at each of 12
  blocks). For GPT-2-small this is <5% over host. At Gemma-2-2B the
  per-block matmuls are larger (`d_model=2304`) — still <10% over
  host. Documented in the new capability spec.
- **Fine-tune disabled in v1 host-wrapped mode.** Documented in
  proposal and surfaced as a clean runtime error. The forge-finetune
  extension for this mode is a queued follow-up; not a v1 blocker.
- **Per-family rollout staged.** v1 lands GPT-2 only. Llama/Gemma-2/
  Qwen2/Qwen3/Whisper raise a clean `NotImplementedError` when
  `forward_mode` resolves to `host_wrapped`, pointing at the queued
  follow-up. The `auto` default is fine here because the existing
  examples (`forge_gemma2_2b.py`, `forge_synthetic_llama.py`) use
  good-tier bases, so they don't trigger the fallback.
