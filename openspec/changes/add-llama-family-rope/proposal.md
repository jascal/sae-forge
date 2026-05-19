# Add RoPE to the Llama-family forge

> **Status**: bug-fix proposal. The Llama-family forged attention
> module has been shipping without rotary positional encoding since
> the family was added — this affects Llama, Gemma-2, Qwen2, Qwen3,
> and Qwen3-MoE. The five adapters' docstrings claim "GQA, RoPE"
> support; `docs/algorithm.md` claims positional handling is
> "identical to the host." Neither matches the code. This proposal
> lands the missing capability and surfaces the resolved positional
> mode on `ForgeResult` so the next silent skip can't recur.

## Why

A 2026-05-19 at-scale Gemma-2-2B run on M4 produced
`faithfulness_kl = 13.19` on the four short `EVAL_PROMPTS` in
`examples/forge_gemma2_2b.py`. For Gemma's 256K-vocab tokenizer,
`ln(V_gemma) ≈ 12.45` — so the forge is only **0.74 nats above
the uniform-over-vocab baseline**, an essentially uninformative
result. Investigation traced the cause to the Llama-family forged
attention module:

- **`saeforge/adapters/llama.py:311-338`** — `LlamaSelfAttention.forward`
  goes from `q_proj` / `k_proj` / `v_proj` directly to
  `scores = q @ k.transpose(-2, -1)` with no rotation step.
- **`saeforge/adapters/llama.py:479-489`** — `LlamaTransformer.forward`
  feeds embeddings straight into the layer stack with no positional
  encoding setup. There is no `position_ids` tensor anywhere in the
  forward path.
- **`grep -rn 'rotary|rotate_half|apply_rotary|freqs_cis|inv_freq|rope' saeforge/`**
  returns zero hits under `adapters/`. RoPE is structurally absent.

The four adapters that inherit from `LlamaAdapter` (Gemma-2 via
`Gemma2Adapter(LlamaAdapter)`, Qwen2, Qwen3, Qwen3-MoE) all share
this defect. The fifth adapter — GPT-2 (`saeforge/adapters/gpt2.py`)
— is correct: GPT-2 uses absolute positional embeddings via `wpe`,
which the adapter projects through `pinv` and the forged
`Transformer.forward` adds to the residual at entry (`x = self.wte(input_ids) + self.wpe(pos)`).
So the bug is Llama-family-specific, but it affects every non-GPT-2
forge the library has produced.

Per-claim falsification against the shipped code:

| Source | Claim | Reality |
|---|---|---|
| `saeforge/adapters/qwen2.py:3` (docstring) | "Qwen2 forge — GQA + SwiGLU + **RoPE**" | RoPE not implemented |
| `saeforge/adapters/qwen3.py:3` (docstring) | "Qwen3 forge — Qwen2 + Q/K-norm + **RoPE**" | RoPE not implemented |
| `saeforge/adapters/qwen3_moe.py:9` (docstring) | "Qwen3-MoE — Qwen3 + router + experts; **RoPE** inherited from Qwen3" | RoPE not implemented |
| `docs/algorithm.md:205-207` | "Keeps attention mechanics (softmax, head splitting, **positional handling**) identical to the host." | Positional handling is absent in the Llama-family forge |

The recent `add-host-wrapped-forge-fallback` smoke gate (#58, KL 89.9
→ 15.4 on GPT-2 K=211) did NOT exercise this bug because GPT-2 has
the working `wpe` projection. A future `add-host-wrapped-gemma2`
would conflate two effects — the LayerNorm-pinv fix from
`forge-forward-mode` and the missing RoPE — which is the structural
reason this proposal precedes any per-family host-wrapped rollout.

The host-side fix is well-understood and standard. The polygram
upstream is positionally agnostic — this is entirely sae-forge-side.

## What Changes

### Required positional encoding contract per family

Each Llama-family adapter SHALL declare its positional encoding
behavior. The bundled families fall into one category in v1:

- **`"rope"`**: applies rotary positional embedding to Q and K after
  projection-and-reshape, before the scaled dot-product. Configured
  by `rope_theta` (host `config.rope_theta`) and, when set on the
  host, `rope_scaling` (v1 supports `type="default"` only; other
  types raise `NotImplementedError` pointing at
  `add-rope-scaling-types`).

| Family | Host class | Positional mode | v1 status |
|---|---|---|---|
| `gpt2` | `GPT2LMHeadModel` | `"absolute_projected"` (wpe) | Already correct |
| `llama` | `LlamaForCausalLM` | `"rope"` | **New: ships v1** |
| `gemma2` | `Gemma2ForCausalLM` | `"rope"` | **New: ships v1** |
| `qwen2` | `Qwen2ForCausalLM` | `"rope"` | **New: ships v1** |
| `qwen3` | `Qwen3ForCausalLM` | `"rope"` | **New: ships v1** |
| `qwen3_moe` | `Qwen3MoeForCausalLM` | `"rope"` | **New: ships v1** |
| `whisper_encoder` | `WhisperModel.encoder` | `"sinusoidal"` (already wired via the conv-stem positional embedding) | Already correct; no change |

### New `NativeModelConfig.rope_mode` field

Optional, default `"standard"`. Accepts:

- `"standard"` — applies RoPE per the host's `rope_theta` /
  `rope_scaling` config. The default for Llama-family
  `build_native_config` outputs.
- `"none"` — skips RoPE entirely. Reproduces the buggy pre-fix
  behaviour exactly; used for regression diffing and to validate
  the impl is the *only* source of behaviour change in the fix PR.
  Emits a `UserWarning` at construction time on Llama-family
  configs (since this is a known-bad regime).

`__post_init__` validates that `rope_mode in {"standard", "none"}`
and raises `ValueError` naming the legal values otherwise. The
field is ignored on non-Llama families (GPT-2 doesn't read it;
Whisper-encoder doesn't read it).

### New `NativeModelConfig` plumbing

- `rope_theta: float = 10000.0` — base for the rotary frequency.
  Populated by `LlamaAdapter.build_native_config` from
  `host.config.rope_theta`.
- `rope_scaling: dict | None = None` — copied verbatim from
  `host.config.rope_scaling`. v1 raises `NotImplementedError` from
  the forward when `rope_scaling is not None and rope_scaling.type
  not in ("default", None)`.
- `partial_rotary_factor: float = 1.0` — fraction of head_dim to
  rotate. Populated from `host.config.partial_rotary_factor` when
  the attribute exists (Qwen3 family); defaults to 1.0 (full
  rotation) for Llama, Gemma-2, Qwen2.

### `LlamaSelfAttention.forward` gains RoPE application

After Q/K projection and reshape, before the optional Q/K norm
(Qwen3) and the scaled dot-product:

```python
q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
```

`cos` and `sin` are precomputed once per forward in
`LlamaTransformer.forward` from `position_ids = arange(seq_len)`
and the config's `rope_theta` / `partial_rotary_factor`.

When `cfg.rope_mode == "none"`, the rotation step is skipped — the
forward returns to the pre-fix behaviour byte-identically.

### New `ForgeResult.positional_encoding` diagnostic field

`Literal["absolute_projected", "rotary", "none_skipped"] | None`,
default `None`. Populated by `ForgePipeline._run_real_imperative`
and `_run_synthetic_imperative` from the resolved family and
`rope_mode`:

- `"absolute_projected"` — GPT-2 family (`wpe` projected through
  `pinv`)
- `"rotary"` — Llama-family with `rope_mode="standard"`
- `"none_skipped"` — Llama-family with `rope_mode="none"` (regression
  diff arm); also emitted by Whisper-encoder when the conv-stem
  positional embedding has been suppressed (n/a in v1; reserved for
  future)

The field surfaces in `ForgeResult` and in `run_summary.json` in
`examples/forge_gemma2_2b.py` so silent skips are immediately
visible in any future audit. The pre-fix run would have reported
`"none_skipped"` if this field had existed — making the bug a
60-second discovery rather than a 6-hour KL post-mortem.

### Adapter assertion test

`tests/test_positional_encoding_assertion.py` walks every registered
adapter. For each, builds a tiny synthetic host of the corresponding
family and exercises:

- When the host's config exposes `rope_theta` (Llama-family) AND
  `rope_mode != "none"` on the forged config, the forged module's
  forward output SHALL differ on inputs that differ only in token
  position. Specifically: `out(input_ids[1, 2, 3]) ≠ out(input_ids[3, 2, 1])`
  modulo the masked-LM-style causal-attention layout.
- When `rope_mode = "none"`, the forged output SHALL be invariant to
  position swap. This is the regression-diff arm and pins the
  pre-fix behaviour.
- For Whisper-encoder, the conv-stem sinusoidal embedding is the
  positional source — the assertion is structurally different and
  scoped out (already covered by existing tests).

The test catches future regressions where someone adds a new
Llama-family adapter and forgets to wire RoPE.

### `docs/algorithm.md:205-207` correction

The paragraph claiming "Keeps attention mechanics (softmax, head
splitting, positional handling) identical to the host" SHALL be
updated to:

- Distinguish between GPT-2 (absolute positional embeddings preserved
  via `wpe` projection) and Llama-family (RoPE, as added by this
  proposal).
- Add an `ε_rope` term to the documented forge error budget alongside
  the existing `ε_attn` (Gemma sliding-window) entry.
- Reference this change's archived path post-merge.

## Falsifiable acceptance gate

### Mechanical (universal, Intel-runnable)

1. **`rope_mode="none"` regression arm**: forge a tiny synthetic
   Llama config (2 layers, hidden_size=64, 4 heads, vocab=512) with
   `rope_mode="none"`. Logits SHALL match the pre-change main
   commit byte-identically on the same input. This is the
   "rotation is the *only* source of behaviour change" gate.

2. **`rope_mode="standard"` position sensitivity**: same fixture
   with `rope_mode="standard"`. For input tokens `[1, 2, 3]` vs
   `[3, 2, 1]` at positions `[0, 1, 2]`, the last-token logits
   SHALL differ by at least `1e-3` in L2. (At `rope_mode="none"`
   they would be identical — attention is order-equivariant
   without positional info.)

3. **Adapter assertion test passes** on every registered adapter
   that has `rope_theta` in host config.

4. **Round-trip stability**: `NativeModelConfig.to_dict() →
   from_dict()` preserves the new fields (`rope_mode`,
   `rope_theta`, `rope_scaling`, `partial_rotary_factor`).

### At-scale (M4, M4-runnable only)

5. **Gemma-2-2B forge KL drops measurably** from the 13.19 baseline.
   Target band: KL < 6.0 nats (~6× of `ln(V_gemma)` below the
   pre-fix value). This is a *necessary* condition for the fix; the
   exact post-fix number gets filled into
   `docs/flagship-gemma2-2b-demo.md` after the M4 run.

6. **No regression on GPT-2 forges**: existing GPT-2 reference smoke
   (host-wrapped on K=211 jbloom basis, archived in
   `2026-05-19-add-host-wrapped-forge-fallback/smoke-results.md`)
   SHALL produce the same numbers post-change. GPT-2 doesn't use
   `rope_mode`, so the field's presence is byte-neutral.

### Pre-implementation smoke (this proposal's gate)

A pre-impl prototype in `scripts/prototype_llama_rope.py` will
exercise gates (1), (2), (4) on Intel before the production code
lands — same cadence as `add-host-wrapped-forge-fallback` and
`add-sae-moe-forge`. If the prototype shows mechanical gates fail
for a reason I haven't anticipated, the proposal gets revised
before production code lands.

## Capabilities

### Modified Capabilities

- `architecture-adapters` — every Llama-family adapter
  (`LlamaAdapter`, `Gemma2Adapter`, `Qwen2Adapter`, `Qwen3Adapter`,
  `Qwen3MoEAdapter`) gains a documented positional-encoding
  requirement (RoPE-with-rope_theta-from-host). `build_native_config`
  populates `rope_theta` / `rope_scaling` / `partial_rotary_factor`
  from the host config. The `NativeModelConfig` field list and the
  `ForgeResult` schema gain the corresponding entries.

## Impact

- **Modified**:
  - `saeforge/adapters/llama.py` — `LlamaSelfAttention.forward` gains
    `apply_rotary_pos_emb` call; `LlamaTransformer.forward`
    precomputes `(cos, sin, position_ids)` once per forward;
    `LlamaAdapter.build_native_config` populates the three new
    `NativeModelConfig` fields.
  - `saeforge/adapters/qwen3.py` — `build_native_config` passes
    `partial_rotary_factor` through to the shared
    `LlamaAdapter.build_native_config` machinery.
  - `saeforge/adapters/gemma2.py` — no code change beyond the
    inherited fix; the per-family override needs to be confirmed
    not to skip the RoPE branch.
  - `saeforge/model.py` — `NativeModelConfig` gains
    `rope_mode: str = "standard"`, `rope_theta: float = 10000.0`,
    `rope_scaling: dict | None = None`,
    `partial_rotary_factor: float = 1.0`. `__post_init__`
    validates `rope_mode`. `from_dict` tolerates payloads missing
    the four new fields (backward-compat with serialised configs
    pre-fix).
  - `saeforge/forge.py` — `ForgeResult.positional_encoding`
    field; `_run_real_imperative` and `_run_synthetic_imperative`
    populate it.
  - `examples/forge_gemma2_2b.py` — surface `positional_encoding`
    in `run_summary.json`.
  - `docs/algorithm.md` — the §5 "host-attention" paragraph (lines
    205-207) loses the false positional-handling claim and gains
    the `ε_rope` accounting.
- **New modules**:
  - `saeforge/_positional/rope.py` — `apply_rotary_pos_emb(q, k,
    cos, sin, position_ids)` and the `compute_rope_cache(seq_len,
    head_dim, theta, partial_factor, device, dtype)` helper.
    Pure-torch; lazy-imports torch.
- **No breaking changes**: the `rope_mode` default
  (`"standard"`) reproduces the host's behaviour exactly; only
  callers who relied on the buggy no-RoPE forge (probably nobody)
  see different numbers. The `"none"` opt-out preserves the
  pre-fix behaviour for anyone who needs it.
- **Dependencies**: no new external dependencies. RoPE math is a
  ~20-line torch helper.

## Risks

- **At-scale Gemma run is M4-only**. Gate (5) is the headline
  outcome but can't be measured on Intel. Mitigation: gates (1)–(4)
  are Intel-runnable and validate the mechanical fix. The
  flagship-demo runbook (`docs/flagship-gemma2-2b-demo.md`) already
  has a slot for the post-fix number; the M4 owner fills it in.

- **`rope_scaling` types beyond default**. v1 raises clearly for
  `linear`, `dynamic`, `yarn`, `longrope`. Llama 3.1+, Qwen3-128K,
  and other long-context variants need follow-up
  (`add-rope-scaling-types`). Documented in the proposal's "Out of
  scope" section. Risk: some Gemma Scope releases may be 3.1+ and
  hit this. Mitigation: surfaces as a clean
  `NotImplementedError` in the forward, not a silent skip — caller
  can either pick a different model variant or contribute the
  scaling type to the follow-up.

- **Per-layer position cache cost**. RoPE precomputes `(cos, sin)`
  tables of shape `(seq_len, head_dim)` per forward. At Gemma-2-2B
  (`head_dim=256`, `seq_len=512`) that's ~1 MB of float32 per
  forward — negligible. The table is recomputed per forward rather
  than cached on the module to keep the change diff small; can be
  promoted to a buffer in a follow-up if benchmarks show overhead.

- **Implications for queued `add-host-wrapped-{llama,gemma2,…}`
  proposals**. Those follow-ups should NOT be written until this
  proposal lands and Gemma-2 forge KL has been re-measured. The
  `add-host-wrapped-forge-fallback` smoke gate (KL 89.9 → 15.4 on
  GPT-2 K=211) isolated the LayerNorm-pinv pathology cleanly
  because GPT-2 has working positional encoding. The same gate on
  Gemma would conflate two effects without RoPE landing first. If
  RoPE alone closes most of the headroom in Gemma's forge KL, the
  per-family host-wrapped rollout may not be needed at all —
  this proposal explicitly defers that decision to a post-impl
  re-evaluation. Documented in `design.md`.

## Out of scope, deliberately

- **`rope_scaling` types other than "default"**. Linear / dynamic /
  yarn / longrope scaling raise `NotImplementedError` from the
  forward with a pointer at `add-rope-scaling-types`. v1 supports
  the no-scaling regime (which is what Llama-3-base, Gemma-2-2B,
  and Qwen3-base ship with).
- **`partial_rotary_factor != 1.0` cases beyond Qwen3's defaults**.
  Qwen3 ships `partial_rotary_factor=1.0` in its base config; the
  parameter exists in the config for future variants. v1 honors
  the field but is only tested against the `1.0` default.
- **GPT-2 changes**. GPT-2's `wpe` projection is correct; this
  proposal does not touch it.
- **Whisper-encoder changes**. Sinusoidal positional embedding is
  applied via the frozen-copied conv stem (per the existing
  `forge-whisper-encoder` capability). No change.
- **Re-evaluation of `add-host-wrapped-{family}` follow-ups**.
  Documented as a *consequence* of this proposal in `design.md`
  but not in the change scope. After impl + re-measure, the queued
  per-family host-wrapped rollouts get reprioritized based on
  whether RoPE alone closes the gap.
- **Polygram cluster diagnostics improvements** (`redundancy_ratio`
  first-class, `n_clusters=0` loud warning,
  `recommend_feature_selection` helper). Filed separately against
  polygram 0.10 per reviewer's note. The sae-forge consumer side
  may surface these once they're available upstream
  (`add-polygram-cluster-diagnostics` follow-up — already on the
  queue from a prior session).
