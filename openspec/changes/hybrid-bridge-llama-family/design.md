# Design: hybrid-bridge-llama-family

## The minimal diff

This change is small enough to describe as a localized diff to one file
(`saeforge/adapters/llama.py`) plus mirror tests. The proposal lays out
the scope; this file pins the surface area that matters for reviewers.

## The edit, in concrete form

Today `LlamaTransformer` looks like this:

```python
class LlamaTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.rms_norm_eps if cfg.rms_norm_eps is not None else 1e-6
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=eps)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

After the change:

```python
class LlamaTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.rms_norm_eps if cfg.rms_norm_eps is not None else 1e-6
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=eps)
        # Bridges are inserted on the forward path between block 0 -> block 1
        # (the embed/mid boundary) and between block L-2 -> block L-1 (the
        # mid/lm-head boundary). Mirrors the GPT-2 implementation; same
        # _build_bridges helper shape, same emb_mid / mid_lm keys.
        self.bridges = self._build_bridges(cfg)

    @staticmethod
    def _build_bridges(cfg):
        if not getattr(cfg, "bridges", False):
            return None
        from saeforge.bridges import BridgeConfig, make_bridge

        bcfg = BridgeConfig(
            init=cfg.bridge_init,
            nonlin=cfg.bridge_nonlin,
            pre_layernorm=cfg.bridge_pre_layernorm,
            train=True,
        )
        return nn.ModuleDict(
            {
                "emb_mid": make_bridge(cfg.hidden_size, bcfg),
                "mid_lm": make_bridge(cfg.hidden_size, bcfg),
            }
        )

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.bridges is not None:
                if i == 0 and n >= 3:
                    x = self.bridges["emb_mid"](x)
                elif i == n - 2 and n >= 3:
                    x = self.bridges["mid_lm"](x)
        return self.norm(x)
```

That's the whole change to the factory. The `_build_bridges` helper is
deliberately a copy of the GPT-2 version, not a shared import, because
each native module is constructed in a separate factory closure and
sharing the helper would require lifting `BridgeConfig` / `make_bridge`
imports out of the closure — a refactor cost that buys nothing.

## Why mirror, not unify

A reasonable instinct is to lift `_build_bridges` into a shared helper
that both `ForgedGPT2.Transformer` and `LlamaTransformer` import. Two
reasons not to do that for v1:

1. **The closures are intentionally self-contained.** The factory
   pattern (`_get_forged_gpt2_class` / `_get_forged_llama_class`) caches
   the dynamically-built class on first call. Sharing helpers across
   closures means the lazy-torch-import structure becomes harder to
   reason about — both helpers already lazy-import inside themselves,
   and the cost is one extra copy of an 8-line function.
2. **The state-dict key prefix differs.** GPT-2 puts bridges under
   `transformer.bridges.*` (matching its `self.transformer` submodule
   name); Llama-family puts them under `model.bridges.*` (matching
   `self.model`). A shared helper wouldn't unify the key prefixes
   because those are determined by where the parent module attaches
   the ModuleDict. So even with a shared helper, save-time/load-time
   tests still have to know the family-specific prefix.

The deferred `forged-module-state-dict-normalization` follow-up can
unify both naming and the construction helper at once; trying to do
half of that here adds churn without finishing the job.

## State-dict key prefixes (recap)

| Family | Bridge state-dict prefix |
|---|---|
| `gpt2` | `transformer.bridges.emb_mid.*` / `transformer.bridges.mid_lm.*` |
| `llama` / `gemma2` / `qwen2` | `model.bridges.emb_mid.*` / `model.bridges.mid_lm.*` |

This asymmetry is honest — it reflects HF's own naming for each host —
and is preserved on save/load. Test fixtures and the
`save_pretrained`/`load_pretrained` round-trip don't care because the
state_dict is round-tripped wholesale.

## The `n >= 3` guard

Both factories guard `n >= 3` because:

- The embed region is layer `0` (exactly 1 block).
- The lm-head region is layer `n-1` (exactly 1 block).
- The mid region is `[1, n-2]`. For `n=2` this would be empty (block 0
  is embed, block 1 is lm-head). For `n=1` only one region exists at
  all.
- `HybridBasisBundle.__post_init__` already rejects `n_layer < 3`, so
  by the time the forged module is constructed the host satisfies
  `n_layer >= 3`. The runtime guard is belt-and-suspenders: if a future
  path bypasses the bundle's check (synthetic host construction with a
  manually-built config), the forward still no-ops the bridges instead
  of indexing out of bounds.

## Test architecture

Two new integration test files: `test_hybrid_bridge_llama.py` and
`test_hybrid_bridge_qwen2.py`. Each mirrors
`test_hybrid_bridge_gpt2.py` exactly:

- **T0 smoke** — construct hybrid pipeline against an in-memory untied
  host with 4 layers, build the forged module, assert bridges appear
  in `state_dict`, forward produces finite logits.
- **Round-trip** — `save_pretrained` → `load_pretrained` preserves
  bridge tensors bit-for-bit.
- **Tied-embedding refusal** — Llama with `tie_word_embeddings=True`
  raises the documented error.
- **Byte-equivalence when disabled** — `hybrid_bridge=False` on a
  Llama host produces the same `state_dict` as the pre-change path
  (no bridge keys present).
- **Zero-init inversion** (`Llama` only — Qwen2 redundant) — confirms
  the algebraic claim still holds on the Llama-family factory.

The four-test pattern (smoke / round-trip / refusal / disabled) plus
one inversion test gives ~9 tests per family — call it ~18 tests
total across the two files. Plus the conftest changes (bumping
`tiny_llama` to 4 layers, adding `tiny_qwen2_untied_4layer`) ship as
part of the same change.

## Conftest changes

The current `tiny_llama` fixture has `num_hidden_layers=2`, which is
fine for the existing single-basis tests but insufficient for hybrid
(`n_layer >= 3` required). Two options:

1. **Bump `tiny_llama` to 4 layers globally.** Every existing test
   using `tiny_llama` still passes because the architecture is
   structurally identical. Slight test-runtime increase (~milliseconds
   per fixture instantiation).
2. **Add `tiny_llama_4layer` as a parallel fixture.** No impact on
   existing tests; new tests opt in.

**Picked: option 1 (bump globally).** The existing 2-layer fixture
exists only because the early Llama tests didn't need depth; bumping
to 4 is cheap and removes the need to maintain two near-identical
fixtures. The tiny_llama_2layer renaming option from the proposal is
dropped to avoid touching every existing test import.

A symmetric `tiny_qwen2_untied_4layer` joins the conftest mirroring
the GPT-2 pattern.

## What happens if a downstream change forgets to wire bridges on a new family

The capability spec (`specs/hybrid-bridge-llama-family/spec.md`) pins
the *family coverage* explicitly. When a new architecture lands (e.g.
`qwen3-moe-support`), the capability requires the new family's native
module to apply the same bridge insertion contract. CI catches the
omission via the family integration test; the spec is the contract
that says "you need a family integration test for hybrid before
shipping a new family."

## Risks

### Risk: the L-2 boundary lands inside Gemma-2's wrapped block

Gemma-2 wraps the attn + MLP path with extra norms:

```
x = x + post_attention_layernorm(self_attn(input_layernorm(x)))
x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
```

The bridge is applied *after* the block as a whole returns — i.e.
between blocks `i` and `i+1`'s residual streams, which is the same
location as the bridge insertion point for Llama. The Gemma-2 family
tag does not change where the bridge sits relative to the residual
stream. Verified by inspection; pinned by the Gemma-2 integration test
to be added in this change (`test_hybrid_bridge_gemma2.py`).

Actually, let me revise — this proposal scopes Llama + Qwen2 only.
Gemma-2 integration tests are deferred to the same follow-up that
runs the M4 Gemma-2-2B reproduction (T3). The mechanism works for
Gemma-2 by inheritance (it shares the `LlamaTransformer.forward` path
with a family-tag branch inside the `LlamaBlock`), but pinning a test
on Intel without Gemma weights cached doesn't add signal. Adding the
Gemma-2 family test is a one-line follow-up; calling it out as
deferred here is the honest scoping.

### Risk: closure-cached class doesn't pick up bridges on second forge

`_get_forged_llama_class()` caches `_FORGED_LLAMA_CLASS` after the
first call. The cached class is constructed *with* the bridge logic
in its `__init__` — the bridges are gated on `cfg.bridges` per
construction, not per class. So caching is fine: every instantiation
re-evaluates `_build_bridges(cfg)`, and the bridge or no-bridge
decision happens at instance construction time.

This is exactly how GPT-2 works today, and the same caching pattern
applies. Confirmed by inspection.

### Risk: tests/fsm/test_topology.py covers any FSM changes

This change touches none of the FSM machinery (no transitions, no
states, no actions, no ctx fields). Topology drift CI passes
trivially.
