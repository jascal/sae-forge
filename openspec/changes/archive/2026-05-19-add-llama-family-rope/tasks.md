# Tasks — `add-llama-family-rope`

## Cadence

Same pattern as `add-host-wrapped-forge-fallback` and
`add-sae-moe-forge`:

1. Capability spec delta (this proposal landed)
2. Prototype + smoke-results.md (BEFORE production code lands)
3. Critical-path production implementation
4. Re-measurement on M4 + flagship-demo runbook update
5. Re-evaluation of queued per-family `host_wrapped` rollouts

The prototype is the gate. If gates (1)–(4) in proposal.md don't
pass, revise the proposal before production code lands. The
`add-host-wrapped-forge-fallback` revision (Band C strict/advisory
split after the prototype showed non-nested-basis non-monotonicity)
is the precedent.

## 1. Capability spec delta

- [ ] 1.1 Update `openspec/specs/architecture-adapters/spec.md` with
      the new RoPE requirements per Llama-family adapter. Add a
      table of positional-encoding modes per family. Add scenarios
      for: (a) Llama-family forged module is position-sensitive at
      `rope_mode="standard"`; (b) Llama-family forged module is
      position-invariant at `rope_mode="none"` (the regression
      arm); (c) GPT-2 forged module is unchanged.

## 2. Prototype (gate for everything below)

- [ ] 2.1 `scripts/prototype_llama_rope.py`: build a tiny synthetic
      Llama config (2 layers, hidden_size=64, n_heads=4, vocab=512,
      rope_theta=10000.0) on Intel. Implement RoPE inline in the
      script. Forge against a synthetic basis with both
      `rope_mode="none"` and `rope_mode="standard"`. Report:
      - Logit difference between `rope_mode="none"` and pre-fix
        main (must be ~0 — confirms the only behaviour change is
        the rotation step).
      - Logit difference between `rope_mode="standard"` token-
        reordered inputs (must be > 1e-3 — confirms rotation IS
        applied).
      - Round-trip stability of the new config fields.
- [ ] 2.2 Write `openspec/changes/add-llama-family-rope/smoke-results.md`
      with the measurements. If any gate fails, revise `proposal.md`
      and re-run before continuing.

## 3. `NativeModelConfig` plumbing

- [ ] 3.1 Add `rope_mode: str = "standard"` to `NativeModelConfig`.
      `__post_init__` validates `rope_mode ∈ {"standard", "none"}`.
      Emits a `UserWarning` when `rope_mode="none"` is set on a
      Llama-family config (pointing at the regression-diff use case
      so it doesn't accidentally ship to production).
- [ ] 3.2 Add `rope_theta: float = 10000.0`, `rope_scaling: dict |
      None = None`, `partial_rotary_factor: float = 1.0` to
      `NativeModelConfig`.
- [ ] 3.3 `to_dict` / `from_dict` round-trip the four new fields;
      `from_dict` tolerates payloads missing them.

## 4. `saeforge/_positional/rope.py`

- [ ] 4.1 New module exposing `apply_rotary_pos_emb(q, k, cos, sin)
      -> (q_rot, k_rot)` and `compute_rope_cache(seq_len, head_dim,
      theta, partial_factor, device, dtype) -> (cos, sin)`. Pure
      torch; lazy-imports torch.
- [ ] 4.2 The `apply_rotary_pos_emb` SHALL match HF's reference
      implementation up to numeric tolerance (~1e-5) on a small
      sanity input. Pinned by a unit test in
      `tests/test_rope.py`.

## 5. `LlamaSelfAttention.forward` integration

- [ ] 5.1 In `saeforge/adapters/llama.py`, modify
      `LlamaSelfAttention.forward` to apply
      `apply_rotary_pos_emb(q, k, cos, sin)` after Q/K
      projection-and-reshape, before the optional Q/K norm (Qwen3)
      and the scaled dot-product. When `cfg.rope_mode == "none"`,
      skip the rotation step.
- [ ] 5.2 In `LlamaTransformer.forward`, precompute `(cos, sin)`
      from `position_ids = torch.arange(seq_len)` and the config's
      `rope_theta` / `partial_rotary_factor` once per forward. Pass
      through to each block's attention.
- [ ] 5.3 Raise `NotImplementedError` with a clean message when
      `rope_scaling is not None and rope_scaling.get("type") not in
      (None, "default")`. Point at `add-rope-scaling-types` as the
      follow-up.

## 6. `LlamaAdapter.build_native_config` plumbing

- [ ] 6.1 Populate `rope_theta` from `host.config.rope_theta`
      (default to 10000.0 if absent).
- [ ] 6.2 Copy `host.config.rope_scaling` verbatim to
      `config.rope_scaling`.
- [ ] 6.3 Populate `partial_rotary_factor` from
      `host.config.partial_rotary_factor` when the attribute exists
      (Qwen3 family); default to 1.0 otherwise.

## 7. `ForgeResult.positional_encoding` diagnostic

- [ ] 7.1 Add `positional_encoding: str | None = None` to
      `ForgeResult`. Validate against
      `{"absolute_projected", "rotary", "none_skipped"} | {None}`.
- [ ] 7.2 In `ForgePipeline._run_real_imperative` and
      `_run_synthetic_imperative`: after building `model`, derive
      the value:
      - `gpt2` family → `"absolute_projected"`
      - Llama-family with `rope_mode="standard"` → `"rotary"`
      - Llama-family with `rope_mode="none"` → `"none_skipped"`
      - `whisper_encoder` → `"sinusoidal"` (already wired in conv
        stem)
- [ ] 7.3 Surface in `forge_result.json` via the existing
      result-write path. Surface in
      `examples/forge_gemma2_2b.py`'s `run_summary.json` under
      `forge.positional_encoding`.

## 8. Adapter assertion test

- [ ] 8.1 `tests/test_positional_encoding_assertion.py` — parametrise
      over every bundled adapter family. For each:
      - Build a tiny synthetic host of the family on Intel (no HF
        download). For Llama-family hosts: configure with
        `rope_theta=10000.0`.
      - Forge against a synthetic basis at default
        `rope_mode="standard"`.
      - Assert: forwarding `[1, 2, 3]` vs `[3, 2, 1]` at positions
        `[0, 1, 2]` produces last-token logits that differ in L2
        by at least `1e-3` (only possible if positional info is
        applied).
      - Re-forge with `rope_mode="none"` and assert the same
        comparison produces L2 difference < `1e-5` (no positional
        info, attention is order-equivariant).
- [ ] 8.2 For families where Q/K-norm is active (Qwen3 / Qwen3-MoE),
      the assertion still holds — the per-head norm doesn't restore
      positional sensitivity.

## 9. `docs/algorithm.md` correction

- [ ] 9.1 Edit `docs/algorithm.md:205-207` to distinguish GPT-2
      (absolute-projected via `wpe`) from Llama-family (RoPE
      applied per this change).
- [ ] 9.2 Add `ε_rope` to the documented forge error budget
      alongside `ε_attn`.
- [ ] 9.3 After this change is archived, update the doc to
      cross-reference
      `openspec/changes/archive/<date>-add-llama-family-rope/` for
      the audit trail.

## 10. At-scale re-measurement on M4

- [ ] 10.1 The M4 owner re-runs `examples/forge_gemma2_2b.py`
      against the same Gemma Scope SAE + layer + L0 + n-features
      that produced today's 13.19 KL baseline.
- [ ] 10.2 Commit the post-fix numbers into the smoke-results.md
      gate (5) row.
- [ ] 10.3 Update `docs/flagship-gemma2-2b-demo.md` acceptance
      bands with the measured pre/post numbers.

## 11. Re-evaluation of per-family `host_wrapped` queue

- [ ] 11.1 With post-RoPE Gemma KL in hand: write a short
      `re-evaluation.md` under `openspec/changes/archive/<date>-
      add-llama-family-rope/` describing whether the queued
      `add-host-wrapped-{llama,gemma2,qwen2,qwen3,qwen3_moe}`
      proposals are still load-bearing. Two outcomes likely:
      - **RoPE alone closes the gap**: per-family `host_wrapped`
        gets deprioritized; the `forge-forward-mode` spec's
        family-rollout requirement gets a footnote noting that
        Llama-family hosts don't typically need the fallback when
        the basis is good-tier.
      - **LayerNorm-pinv still dominates**: per-family
        `host_wrapped` stays on the queue; the impl proposal for
        each family gets written.
- [ ] 11.2 The decision is documented but NOT acted on in this
      change — that's a separate proposal cycle.

## 12. Out of scope (named follow-ups)

- [ ] 12.1 `add-rope-scaling-types` — Linear / dynamic / yarn /
      longrope scaling for long-context variants (Llama 3.1+,
      Qwen3-128K). Raises `NotImplementedError` from the forward
      in v1.
- [ ] 12.2 `add-positional-encoding-on-pareto-frontier-row` —
      bubble the `positional_encoding` field into
      `ParetoFrontierRow` for systematic sweeps. Probably folds
      into the queued `--forward-mode on sweep-pareto` follow-up
      since both add row schema.
- [ ] 12.3 `add-rope-perf-tuning` — pre-compute the `(cos, sin)`
      table as a buffer on `LlamaTransformer` rather than per
      forward, if benchmarks show the rebuild matters.
