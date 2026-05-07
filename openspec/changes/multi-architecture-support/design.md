## Context

`SubspaceProjector` and `NativeModel` are the two GPT-2-shaped components in sae-forge. They were correct for the v0 milestone (smoke target: GPT-2-small) but the projection algebra in `docs/algorithm.md` is architecture-agnostic — every linear map projects the same way. The blocker for Llama / Gemma is plumbing: HF parameter naming, SwiGLU's three matrices instead of GeLU's two, RMSNorm instead of LayerNorm, and Llama-3 GQA's `n_kv_heads < n_q_heads`.

The shipped `examples/forge_gemma2_2b.py` doesn't work end-to-end because:

1. `ForgePipeline.run` (`saeforge/forge.py:185`) hardcodes `transformers.GPT2LMHeadModel.from_pretrained(self.host_model_id)`. Passing `"google/gemma-2-2b"` triggers HF's permissive class-mismatch path that loads the Gemma weights into a GPT-2 *config*, silently warns, and emits a randomly-initialised model.
2. Even if the load were architecture-aware, `SubspaceProjector.project_module` (`saeforge/projector.py:127`) only knows GPT-2 parameter names.
3. Even if `project_module` were architecture-aware, `_build_torch_module` in `saeforge/model.py:80` only constructs a GPT-2-shaped target.

All three are fixable without touching polygram.

## Goals / Non-Goals

**Goals:**

- One pluggable mechanism for adding a new host architecture: an adapter class + one `register_adapter(...)` call. No future contributor adds a third `if isinstance` branch.
- GPT-2 path moves through the new mechanism without behavioural change. Existing tests stay green by construction.
- Llama-3 and Gemma-2 work end-to-end: `ForgePipeline.run("/output")` against a real (or tiny-synthetic) Llama / Gemma host produces a forged model whose every weight is set from a projected source weight (no `nn.Parameter`s with their default init).
- The shipped `examples/forge_gemma2_2b.py` script runs end-to-end with `--steps 0`. Smoke test in `tests/` exercises it with a `pytest.importorskip("transformers")` guard for the Gemma weights.
- Failure mode for an unregistered architecture is a clear `NotImplementedError` naming the type and the registered classes — no random-init fallback.

**Non-Goals:**

- **Pythia / GPT-NeoX**. Deferred — the parallel Q/K/V parameterisation it uses also needs a small upstream polygram addition, which is its own change.
- **Mistral / Qwen / Phi / etc.** Out of scope; the adapter pattern lets them be added without further refactor, but no adapter ships in this PR.
- **Replicating Gemma-2's logit soft-cap exactly** in the native module. Treated as `ε_nonlin` per `docs/algorithm.md` §5; fine-tuning corrects the drift. NativeModel surfaces a `final_logit_softcap: float | None` config field but applies it as a no-op when None and a `tanh(...) * cap` clamp on `lm_head` output when set; the projection itself is unaffected.
- **Replicating Gemma-2's alternating local/global attention** exactly. Sliding-window masks are an attention-mechanic detail orthogonal to the projection; v0.2 NativeModel uses the standard causal mask everywhere. Faithfulness drops on long-context tasks are accepted; tracked as future work.
- **Tied embeddings as a default**. Llama models often share `lm_head.weight` and `embed_tokens.weight`. NativeModel supports `tied_embeddings: bool` on the config; the Llama / Gemma adapters set it from the host's `config.tie_word_embeddings`. When tied, only one projected weight is set and the linear layer aliases it.
- **Per-layer streaming projection** (mentioned in the README hardware notes). Out of scope here; the adapter walk does the entire forward pass in numpy. Memory-conscious projection is a follow-up.

## Decisions

### Decision 1 — Adapter registry, not a giant dispatch function

```python
# saeforge/adapters/__init__.py
_REGISTRY: list[tuple[type, "ArchitectureAdapter"]] = []

def register_adapter(host_class, adapter):
    _REGISTRY.append((host_class, adapter))

def adapter_for(host_model):
    for cls, adapter in _REGISTRY:
        if isinstance(host_model, cls):
            return adapter
    raise NotImplementedError(
        f"No architecture adapter registered for {type(host_model).__name__}. "
        f"Registered: {[cls.__name__ for cls, _ in _REGISTRY]}. "
        f"To add a new architecture, register a saeforge.adapters.ArchitectureAdapter."
    )
```

The registry is populated at import time by the `gpt2` / `llama` / `gemma2` modules, each of which imports its HF class lazily. A user adding a fourth architecture writes one ~150-line module and calls `register_adapter` once.

**Why a list of tuples, not a dict by class?** HF subclasses (e.g. `Gemma2Model` extending `Gemma2PreTrainedModel`) need first-match-wins semantics; a dict-by-exact-type would miss subclasses. The list also lets us order specific classes before generic bases.

**Alternatives considered:**

- **Class hierarchy / inheritance** — adapters subclass a base; dispatch via `mro()`. Rejected: contributors would have to know which base to extend; the registry keeps the dispatch contract explicit.
- **Plugin entry points** — register via `[project.entry-points."saeforge.adapters"]`. Rejected as over-engineered; sae-forge's surface is small and adapters live in-tree.
- **Single-function dispatch on `model.config.model_type`** — would work for HF, but the `isinstance` form catches custom subclasses (e.g. wrapping a Gemma model in a peft `PeftModel` shouldn't break dispatch as long as the underlying class still satisfies the registered base).

### Decision 2 — `ArchitectureAdapter` ABC with three methods

```python
class ArchitectureAdapter(ABC):
    family: str  # "gpt2" | "llama" | "gemma2", set on NativeModelConfig

    @abstractmethod
    def walk(self, host, projector, *, attention_width: str) -> dict[str, np.ndarray]:
        """Project every relevant weight; return a flat dict keyed by the
        target NativeModel's parameter names."""

    @abstractmethod
    def build_native_config(
        self, host, n_features: int, *, attention_width: str
    ) -> NativeModelConfig:
        """Pull the host's per-block dimensions into a NativeModelConfig
        whose family field matches this adapter."""

    @abstractmethod
    def native_module_class(self):
        """Return the `nn.Module` subclass used to instantiate forged
        models for this family. Used by NativeModel.__init__."""
```

Three methods rather than one because they're called at different points in the pipeline and have orthogonal failure modes. `walk` is pure-numpy (no torch), `build_native_config` reads HF config fields, `native_module_class` returns a torch class — each gets its own lazy-import boundary.

### Decision 3 — `family` field on `NativeModelConfig`, no default

```python
@dataclass
class NativeModelConfig:
    family: str  # required, no default
    hidden_size: int
    qkv_inner_size: int
    ...
```

**Why no default?** The current shipped `NativeModelConfig` produces a GPT-2-shaped module. Adding a `family` default of `"gpt2"` would let downstream callers silently get a GPT-2 module for a Llama config. The `__post_init__` validation already does cross-field checks; adding family-vs-rest-of-config validation there is a one-time cost. The migration is one line per call site, all in-tree.

**Alternatives considered:**

- **Sentinel default** (`family: str = "gpt2"`) — rejected; preserves the silent footgun.
- **Three concrete config dataclasses** (`GPT2NativeConfig`, `LlamaNativeConfig`, `Gemma2NativeConfig`) — cleaner type-safety, but `NativeModel` then needs an `Any`-typed config or a `Union`. Rejected as not-yet-pulling-its-weight; revisit if family-specific knobs proliferate.

### Decision 4 — `_build_torch_module` dispatches; family-specific factories sit alongside

```python
def _build_torch_module(config: NativeModelConfig):
    if config.family == "gpt2":
        return _build_gpt2_module(config)
    if config.family in ("llama", "gemma2"):
        return _build_llama_family_module(config)
    raise ValueError(f"unknown family {config.family!r}")
```

Llama and Gemma-2 share enough structure (RMSNorm + SwiGLU + Q/K/V/O proj + GQA + no wpe + optional tied embeddings) that one `_build_llama_family_module` covers both. Gemma-2-specific quirks (final_logit_softcap, attn_logit_softcap, alternating attention) sit on the config and the module branches accordingly. If the divergences grow, the factory splits.

**Alternatives considered:**

- **One unified module** with conditional branches throughout — rejected; readability suffers fast.
- **Per-family files** (`saeforge/native/{gpt2,llama,gemma2}.py`) — rejected for now; adds three files of single-class-each. Folded into adapter modules instead: each adapter exposes its `native_module_class` and the module classes live next to the walk logic.

### Decision 5 — `AutoModelForCausalLM` for host loading; adapter dispatches by class

`ForgePipeline.run` switches from `GPT2LMHeadModel.from_pretrained(host_model_id)` to `AutoModelForCausalLM.from_pretrained(host_model_id)`. The returned object's class drives adapter dispatch via `adapter_for(host)`. This:

- Works correctly for any HF-supported architecture (Auto picks the right class from `config.json`'s `architectures`).
- Surfaces unregistered architectures at adapter-dispatch time (clear `NotImplementedError`), not at random-init-eval time.
- Removes the GPT-2-specific import from `forge.py` entirely.

**Alternative considered:** parsing `config.json` ourselves and instantiating the specific class. Rejected; `AutoModelForCausalLM` is the boring path and HF maintains the architectures-to-class mapping.

### Decision 6 — RMSNorm γ projects through `project_residual_aligned`; no β

RMSNorm is `x * γ / rms(x)`. There's no β, no mean-subtraction. γ is a residual-aligned vector, so the existing `project_residual_aligned` (which uses the encode pseudoinverse) works. The Llama / Gemma adapter walks emit `*.weight` for every RMSNorm layer (matching HF's `LlamaRMSNorm.weight` parameter name) and skip `*.bias`.

LayerNorm-vs-RMSNorm faithfulness is the same lossy fallback documented in `docs/algorithm.md` §5 / `subspace-projector` capability spec — γ projection is not equivariant under linear projection, so faithfulness drops. Tracked.

### Decision 7 — GQA via `n_kv_heads` config field; projection is per-matrix

Llama-3-style GQA has `q_proj` mapping to `n_q_heads * head_dim` and `k_proj`/`v_proj` mapping to `n_kv_heads * head_dim`. The projection algebra doesn't care about head shape — it acts on the residual-input axis only. The adapter walk:

- `q_proj.weight: (n_q_heads * head_dim, hidden_size)` → `D @ W` shape `(n_q_heads * head_dim, n_features)` — wait, that's transposed; HF's `Linear.weight` is `(out, in)`. Project as `W @ E` for the `(out, in) @ (in, n_features)` direction. Adapter unit-tests pin the shape contract.
- `k_proj`, `v_proj` same pattern, but with `n_kv_heads * head_dim` as the output dim.
- `o_proj.weight: (hidden_size, n_q_heads * head_dim)` → `D @ W` (residual-input-from-the-attention-output side); shape `(n_features, n_q_heads * head_dim)`.

`NativeModelConfig` carries both `n_heads` (= `n_q_heads`) and `n_kv_heads`. When `n_kv_heads == n_heads`, GQA collapses to MHA — no special-casing needed in the module factory.

### Decision 8 — Failure mode: clear NotImplementedError, no fallback

The current silent-random-init failure (passing a non-GPT-2 host) is replaced with a hard `NotImplementedError` listing the registered classes. No per-architecture warning suppression, no "try GPT-2 anyway and hope" mode. The user prompt is explicit: *"Don't add a backwards-compat shim for the silent-random-init behavior; it's a bug, fail loudly."*

## Risks / Trade-offs

- **[Risk] `AutoModelForCausalLM` triggers heavyweight imports for any host model.** → **Mitigation:** that's already what `from_pretrained` does end-to-end; AutoModel doesn't add overhead beyond the architecture-specific class load. The lazy-import in `ForgePipeline.run` is unchanged.
- **[Risk] Gemma-2 logit soft-cap mismatch hurts faithfulness.** → **Mitigation:** documented as `ε_nonlin`; if the drift is unacceptable in practice, a follow-up change adds it. The current acceptance bar (faithfulness need not be good per the prompt) is the test plan: every weight is non-random after projection.
- **[Risk] GQA shape miscount in the adapter walk produces silently wrong projected weights.** → **Mitigation:** the per-key shape audit in the new tests asserts the projected shape matches the target slot exactly. Mismatch raises in `from_projected_weights`.
- **[Risk] `family="gpt2"` migration burden on downstream callers.** → **Mitigation:** the in-tree callers are updated in this PR; downstream callers see a clear `TypeError: __init__() missing 1 required positional argument: 'family'`. The CHANGELOG notes the one-line fix.
- **[Risk] Tied embeddings handling on Llama / Gemma.** → **Mitigation:** `NativeModelConfig.tied_embeddings: bool = False`; the adapter sets it from `host.config.tie_word_embeddings`. The Llama family module factory aliases `lm_head.weight = transformer.embed_tokens.weight` post-init when tied. Tests cover both modes.
- **[Trade-off] Gemma-2 alternating local/global attention not replicated.** Sliding-window masks are an attention-mechanic detail; v0.2 uses standard causal. Long-context faithfulness will be worse on Gemma-2 than Llama-3. Acceptable for v0.2; tracked as future work.
- **[Trade-off] Single PR for Llama + Gemma-2.** They share enough structure that splitting them would mostly duplicate the adapter scaffolding. If implementation reveals more divergence than expected, splitting Gemma-2 into a follow-up is a clean cut (it imports the Llama adapter's helpers).

## Migration Plan

1. Land the adapter package + GPT-2 adapter (extracted from current code paths). All existing tests pass.
2. Add the `family` field to `NativeModelConfig`; update in-tree call sites; bump CHANGELOG.
3. Add the Llama adapter + tiny-synthetic Llama tests + native module factory.
4. Add the Gemma-2 adapter + tiny-synthetic Gemma-2 tests; share the Llama family module factory.
5. Switch `ForgePipeline.run` to `AutoModelForCausalLM`. Negative test exercising `BertModel` raises clean.
6. Wire `examples/forge_gemma2_2b.py` end-to-end with `--steps 0`; add the smoke test (skip-if-no-Gemma-token).
7. Add `examples/forge_synthetic_llama.py`. Update README hardware-tier table and "Integration with Polygram" section.

Each step is its own commit; reverting any individual step is safe (later steps don't quietly depend on earlier behaviour).

## Open Questions

- **Tied embeddings on the GPT-2 path**: GPT-2 doesn't tie by default in HF, but if a user passes a GPT-2 variant that does, the current code silently loads two copies. Out-of-scope here, but the new `tied_embeddings` config field could clean this up later.
- **Dtype handling on Gemma-2**: Gemma-2 was trained in bfloat16; loading at fp32 inflates memory 2×. The adapter walk converts to float64 numpy regardless (existing pattern); the native module's `_move(dtype="bfloat16")` happens after projection. Worth a docs note in the example header.
- **`final_logit_softcap` faithfulness**: if practical usage shows soft-cap drift dominates `ε_attn` + `ε_nonlin`, replicating the soft-cap in NativeModel is a 5-line addition. Defer until the eval shows the drift matters.
