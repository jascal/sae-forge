# Add NanochatAdapter

Architecture adapter for the nanochat-style single-file GPT used by `karpathy/autoresearch` and its CPU/MPS-port sibling `lm-sae`. Adds a fourth substrate to the SAE benchmark tree (alongside bio-sae's ESM-2, econ-sae's TemporalWorldModel, sm-sae's cascade-host).

## Why

The three existing application repos (bio-/econ-/sm-sae) all vary *substrate* but hold *host architecture* fixed within each repo. `lm-sae` is the inverse: substrate stays fixed (the climbmix BPE token stream) while the host architecture varies along the `karpathy/autoresearch` agent's edit path. Forge-side capability evaluation across an architecture trajectory is a use case the existing adapters can't serve — none of them target the nanochat surface (rotary-only positions, separate q/k/v projections, value-residual ResFormer pattern, RMSNorm without learnable scale, per-layer scalar mixers, no positional embedding matrix). A NanochatAdapter is the prerequisite for any forge run against an lm-sae checkpoint.

## What Changes

### Scope

Add `saeforge/adapters/nanochat.py` implementing `ArchitectureAdapter` for the nanochat-family GPT. Register it in `saeforge.adapters.__init__` so `adapter_for(host)` dispatches when the host is a duck-typed match — see Design for the dispatch story (this is not an HF transformers class so the registry's `isinstance` path needs a sentinel).

Concretely:

- `family = "nanochat"`
- `walk(host, projector, *, attention_width)` — emits a projected weight dict keyed by parameter name. The walk semantics mirror the GPT-2 adapter except for the four nanochat-specific quirks (rotary buffers skipped, separate qkv, value embeds, per-layer scalars).
- `build_native_config(host, projector, *, attention_width)` — builds a `NativeModelConfig` from the host's `config: GPTConfig` dataclass.
- `native_module_class()` — returns a forged `nn.Module` that mirrors the host's forward graph in the SAE's feature basis. Implementation can subclass the lm-sae `GPT` class (lm-sae will need to surface its model classes via a stable import path; see Tasks).
- `default_faithfulness_target()` — `TokenCosineTarget` is the cleanest fixture-side default; same family as the ESM-2 adapter's choice.

### Non-scope

- **No sliding-window attention.** The lm-sae fork dropped the `SSSL` pattern for MPS portability. The adapter follows: all layers are full causal.
- **No `host_wrapped_module` initially.** That code path is an optimization for runs where the analyst wants to keep the host weights in float32 while the forge is in bfloat16; not needed for a CPU/MPS fixture.
- **No torch.compile path.** lm-sae has `USE_COMPILE=False` because MPS is flaky with fullgraph. The adapter follows.
- **No SAE-MoE / closed-form-expert-routing integration.** Those proposals (`add-sae-moe-forge`, `add-closed-form-expert-routing`) can layer on top later; the bring-up here is the minimal-viable adapter.

### Acceptance gate

A retained-mAUC number distinguishable from chance (>0.55 mean over the 8192 token-identity probes) on lm-sae's layer-1 residuals through a TopK SAE forged into the host. Absolute value doesn't matter for the bring-up; non-trivial existence is the gate. This mirrors how `forge-whisper-encoder` shipped: prove the walk and the forge produce a working forged module before tuning numbers.

## Coupling

- **lm-sae** must expose its `GPT` and `GPTConfig` classes via a stable import path. Today they live in `lm-sae/train.py` and are reachable after the `if __name__ == "__main__":` gate (see `lm-sae` commit 5ec50df+). The adapter should `from lm_sae.train import GPT, GPTConfig` after lm-sae is installable as `lm_sae` (today it's installed as `lm-sae` with top-level modules `train` / `prepare`; a small `[project.scripts]` or rename will be needed).
- **polygram** ≥ 0.15.0 (already pinned in sae-forge).
- **n-orca**: lm-sae's SAEs are built via `n_orca.sae.topk_sae` compiled to PyTorch. The forge consumes the SAE's `W_dec` exactly as bio/econ/sm-sae do — no n-orca dep needed in sae-forge itself.

## Status

Proposal only. Implementation deferred — this change documents the intent so anyone wiring lm-sae into the forge has a target. Smoke evidence that the fixture chain works end-to-end will land first in `lm-sae` (commits ahead), and this proposal will be re-scoped against that evidence before implementation begins.
