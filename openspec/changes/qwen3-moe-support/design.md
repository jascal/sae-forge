# Design: qwen3-moe-support

## What an HF Qwen3-MoE block looks like

For a real `Qwen3MoeForCausalLM`, each transformer block has the
standard Qwen3 structure (RMSNorm, GQA, q_norm/k_norm) but the MLP is
replaced by a `Qwen3MoeSparseMoeBlock`:

```
block.mlp                                        # Qwen3MoeSparseMoeBlock
  block.mlp.gate                                 # nn.Linear(hidden_size, num_experts, bias=False)
  block.mlp.experts                              # nn.ModuleList of num_experts entries
    block.mlp.experts.{i}.gate_proj              # nn.Linear(hidden_size, moe_intermediate_size, bias=False)
    block.mlp.experts.{i}.up_proj                # nn.Linear(hidden_size, moe_intermediate_size, bias=False)
    block.mlp.experts.{i}.down_proj              # nn.Linear(moe_intermediate_size, hidden_size, bias=False)
```

Key dimensions (Qwen3-30B-A3B-Base reference values, subject to actual
HF config):

- `num_experts = 128`
- `num_experts_per_tok = 8` (top-K)
- `hidden_size = 2048`
- `moe_intermediate_size = 768` (per-expert FF inner dim; *much smaller*
  than a dense Qwen3's `intermediate_size`)
- `norm_topk_prob = True`

## Walker contract for the MoE block

Each block emits, in addition to the inherited Qwen3 attention keys
(q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, two RMSNorms):

| Host key | Shape | Projection rule | Forged key |
|---|---|---|---|
| `model.layers.{i}.mlp.gate.weight` | `(num_experts, hidden_size)` | `project_residual_input` (residual reader) | `model.layers.{i}.mlp.gate.weight` shape `(num_experts, n_features)` |
| `model.layers.{i}.mlp.experts.{e}.gate_proj.weight` | `(moe_intermediate_size, hidden_size)` | `project_residual_input` (right axis is residual) | shape `(moe_intermediate_size, n_features)` |
| `model.layers.{i}.mlp.experts.{e}.up_proj.weight` | `(moe_intermediate_size, hidden_size)` | `project_residual_input` | shape `(moe_intermediate_size, n_features)` |
| `model.layers.{i}.mlp.experts.{e}.down_proj.weight` | `(hidden_size, moe_intermediate_size)` | `project_residual_output` (left axis is residual) | shape `(n_features, moe_intermediate_size)` |

Note that the **per-expert FF inner dim (`moe_intermediate_size`) is
preserved**, not projected. The basis only touches the residual-stream
axis. This is why the MoE block's expert MLPs project identically to a
dense Llama MLP — same rule applied to each of N experts.

The router (`mlp.gate.weight`) is residual-aligned on its input axis
(it reads the residual stream and outputs num_experts logits), so it
gets the same projection rule as `q_proj` / `k_proj` / `v_proj`.

## NativeModelConfig — four new fields

```python
@dataclass
class NativeModelConfig:
    # ... existing fields ...

    # MoE configuration. num_experts == 0 -> dense path (existing behavior
    # for every other family). num_experts > 0 -> Qwen3-MoE-style MLP.
    num_experts: int = 0
    num_experts_per_tok: int = 0     # top-K routing; required when num_experts > 0
    moe_intermediate_size: int = 0   # per-expert FF inner width
    norm_topk_prob: bool = True      # renormalize top-K probs after gate softmax
```

`__post_init__` validation:

```python
if self.num_experts > 0:
    if self.num_experts_per_tok <= 0:
        raise ValueError(...)
    if self.num_experts_per_tok > self.num_experts:
        raise ValueError(...)
    if self.moe_intermediate_size <= 0:
        raise ValueError(...)
```

Llama, Gemma-2, Qwen2, Qwen3-dense, GPT-2 all keep
`num_experts=0` (the default) and never see the MoE branch. Existing
tests pass unchanged.

## The forged MoE MLP

```python
class Qwen3MoEMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen3Expert(cfg) for _ in range(cfg.num_experts)
        ])

    def forward(self, x):
        # x: (B, T, hidden)
        B, T, H = x.shape
        x_flat = x.reshape(-1, H)               # (B*T, H)
        gate_logits = self.gate(x_flat)         # (B*T, num_experts)
        # HF Qwen3MoE applies softmax in float32 then takes top-K.
        weights = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)
        top_w, top_i = weights.topk(self.top_k, dim=-1)   # both (B*T, top_k)
        if self.norm_topk_prob:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w.to(x.dtype)

        out = torch.zeros_like(x_flat)
        # v1: naive expert loop. Correct but slow on MPS for large N.
        # The fused scatter-add path is tracked as moe-fused-dispatch.
        for e in range(self.num_experts):
            # Tokens whose top-K includes expert e
            mask = (top_i == e)                  # (B*T, top_k)
            if not mask.any():
                continue
            # Find which positions in top_k each routing event sits at
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_in = x_flat[token_idx]        # (n_tok_for_e, H)
            expert_out = self.experts[e](expert_in)  # (n_tok_for_e, H)
            expert_w = top_w[token_idx, slot_idx].unsqueeze(-1)  # (n_tok_for_e, 1)
            out.index_add_(0, token_idx, expert_w * expert_out)

        return out.view(B, T, H)


class Qwen3Expert(nn.Module):
    """A single SwiGLU expert. Identical to dense Qwen3's MLP but with
    moe_intermediate_size instead of intermediate_size."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.moe_intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

The naive `for e in range(num_experts)` loop is intentional for v1:

- Correctness over performance. The scatter-add fused path has subtle
  index/dtype interactions that a follow-up
  (`moe-fused-dispatch`) handles cleanly. Shipping correct-and-slow
  before fast-and-buggy is the right order.
- On NVIDIA the smoke runs in seconds even at 128 experts × 16 tokens
  because each per-expert call processes only the tokens routed to it
  (~`B*T*top_K/num_experts ≈ 1` token per expert per batch at small B).
- On MPS the same loop is slower (kernel launch overhead) but the M4
  validation surface only ever runs synthetic 4-expert configs, where
  the loop is fine.

## Routing math: verify against HF before shipping

The implementation must match `transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward`
exactly. Specifically:

1. **softmax-then-topk vs topk-then-softmax.** The proposal assumes
   *softmax-then-topk* (full softmax over all `num_experts` logits,
   then take top-K, then optionally renormalize). This matches
   Mixtral / Qwen3-MoE conventions. Verify before merging.
2. **float32 softmax dtype.** HF uses fp32 for the softmax to avoid
   underflow when num_experts is large. The proposed forward replicates
   this (`dtype=torch.float32` argument to `softmax`).
3. **Renormalization gate.** `norm_topk_prob=True` renormalizes the
   top-K weights to sum to 1 across the K kept experts. False leaves
   them at the raw softmax values (small bleed-out to unselected
   experts).

A correctness test (`test_qwen3_moe_routing_matches_host` in
`tests/integration/test_qwen3_moe_adapter.py`) compares the forged
MoE block's routing decisions to the host's on the same input. If the
top-K indices match and the renormalized weights match within
fp32 tolerance, the routing is faithful.

## Compression modes

The `moe_strategy` field on `ForgePipeline` selects between three modes:

### `preserve` (v1 default)

Pass-through: `forged_cfg.num_experts = host.config.num_experts` and
`forged_cfg.num_experts_per_tok = host.config.num_experts_per_tok`.
The walker projects every expert independently. Storage:
`num_experts × forged_expert_size` per layer. **This is the only mode
that preserves Qwen3-MoE behavior fidelity.** Use it unless you have
a specific storage reason to collapse.

### `collapse` (v1 supported, opt-in)

Storage-aggressive: average the host's expert weights into a single
dense MLP per layer; remove the router; treat the block as dense.
Forged config has `num_experts=0` and `intermediate_size=moe_intermediate_size`
(the per-expert size; not the host's dense `intermediate_size` which
the MoE host doesn't have anyway).

The averaging walker:

```python
def _collapse_experts(host_block) -> dict:
    """Average expert weights into a single dense MLP."""
    experts = host_block.mlp.experts
    n = len(experts)
    avg_gate = sum(e.gate_proj.weight for e in experts) / n
    avg_up = sum(e.up_proj.weight for e in experts) / n
    avg_down = sum(e.down_proj.weight for e in experts) / n
    return {
        "mlp.gate_proj.weight": avg_gate,
        "mlp.up_proj.weight": avg_up,
        "mlp.down_proj.weight": avg_down,
    }
```

Degraded behavior. Useful when storage is binding and approximate
fidelity is acceptable. Marked clearly in the documentation as
"experimental — produces a model that thinks like the average expert,
not like any specific expert."

### `top_n` (v1 placeholder; raises NotImplementedError)

Calibration-required: run the host on a corpus, log per-expert
activation frequency per layer, keep the top-N most-used experts,
renormalize the router. The calibration utility
(`scripts/calibrate_moe_experts.py`) is a separate change
(`moe-expert-calibration`).

v1 ships the enum value and the validation
(`ForgePipeline.__post_init__` requires `moe_keep_n > 0` when
`moe_strategy="top_n"`) but the action raises `NotImplementedError`
pointing at the follow-up. The contract is pinned so the follow-up
doesn't have to reopen the design.

## Hybrid-bridge compatibility

Bridges sit at the residual-stream boundary between block 0 and block 1
(emb_mid bridge) and between block L-2 and block L-1 (mid_lm bridge).
The MLP choice (dense SwiGLU vs MoE) is local to the block; the bridge
operates on the residual stream as it flows between blocks. MoE adds
nothing the bridge needs to know about.

The `hybrid-bridge-llama-family` family-coverage requirement specifies
that every host family routed through `build_llama_family_module` and
supported under `_SUPPORTED_FAMILIES` SHALL have a hybrid-bridge
integration test. This change satisfies that for Qwen3-MoE by adding
`tests/integration/test_hybrid_bridge_qwen3_moe.py` (synthetic small
MoE host; mirrors the Qwen3 dense file shape).

## The NVIDIA-tier smoke script

`scripts/smoke_qwen3_moe.py` ships in this PR. It is the only place
in the test surface where a *real* Qwen3-MoE host gets loaded and
forged. The Intel and CI surfaces cannot run it (transformers<4.50
on `[intel]`, no transformers in `[dev]`); M4 cannot run it
(insufficient memory for 60GB+ host); only NVIDIA ≥80GB validates it.

Script responsibilities:

1. Lazy-import + Qwen3MoE availability check. Exit code 2 + clear
   message if transformers <4.51.
2. CUDA availability check. Exit code 2 + clear message if no GPU.
3. Load host with `device_map="auto"` and `dtype=torch.bfloat16`.
4. Dispatch the adapter; confirm `family == "qwen3_moe"`.
5. Walker sanity: count `mlp.gate.weight` + `mlp.experts.{e}.gate_proj.weight`
   + `up_proj.weight` + `down_proj.weight` keys; confirm
   `num_experts × num_hidden_layers × 3 + num_hidden_layers` matches.
6. Build native config; confirm `num_experts`, `num_experts_per_tok`,
   `moe_intermediate_size` populated correctly.
7. Build forged module; confirm each block's `mlp` has a `gate` and a
   `experts` ModuleList of the right length.
8. Forward pass on a short prompt; confirm shape and finite logits.
9. **Optional** (when `--log-expert-utilization` is set): run the same
   prompt through both host and forged module, log routing decisions
   per layer, and compare top-K agreement rate. If routing matches
   host on every block within the top-K set (allowing for fp tolerance
   in the weights), the projection is faithful at the router level.

The script is bundled in this PR so a reviewer with NVIDIA access can
run it the moment the implementation PR (not this one) lands. For the
proposal PR itself, the script is best-effort — it will fail gracefully
on Intel and on Qwen3-MoE-unaware transformers, with messages pointing
at the implementation requirement.

## Risks (revisited from the roadmap review)

### Risk: routing collapse during fine-tune

The router weights get projected through the basis. If the projection
significantly shifts the residual distribution at the gate's input, the
"which experts fire" pattern may drift from the host's. Mitigation: the
smoke script's `--log-expert-utilization` mode is the diagnostic. If
top-K agreement is below ~80% on a representative prompt, routing has
collapsed and the forge needs investigation (either a different basis,
a higher `n_features`, or a routing-aware fine-tune objective).

### Risk: MPS performance on 128 small experts

Many small kernel launches is a known MPS weakness. The synthetic M4
tests use 4 experts, where the loop overhead is negligible. Real
Qwen3-30B-A3B on M4 is out of memory anyway, so MPS-specific MoE
perf isn't on the v1 critical path.

### Risk: Qwen3-30B-A3B-Base is gated

License acceptance + HF token may be required. The smoke script
checks for the standard HF auth flow and surfaces a helpful error
if the host load fails with a 401/403. Fallback host: the script
accepts `--host-model` so any future smaller Qwen3-MoE release (or
a community-mirrored variant) can substitute.

### Risk: SAE format heterogeneity

Qwen-official SAEs ship as `.pt` (pickle), not `.safetensors`. The
adapter doesn't load SAEs directly — that's
`FeatureBasis.from_polygram_checkpoint`'s job. The smoke script uses
a random basis to validate the *forge* part, not the SAE part. Real
Qwen3-MoE SAE integration is a downstream concern tracked separately.

### Risk: HF Qwen3MoE attribute names may drift

The walker uses `block.mlp.gate.weight` and
`block.mlp.experts.{i}.gate_proj.weight` style attribute paths. If a
future HF transformers release renames these, the walker breaks
silently (KeyError) rather than producing wrong output. Tight failure
mode; documented but not actively guarded against. The smoke script
catches it loudly on the NVIDIA tier before merge.

## Why not split this proposal further

A reasonable instinct is to split into "adapter + MLP class" then
"compression modes." Two reasons not to:

1. The compression-mode `top_n` already deferred to a follow-up
   (calibration utility); `preserve` is trivially "no special logic,
   just project everything"; `collapse` is one ~20-line averaging
   helper. The compression-mode surface in this PR is small.
2. The adapter + native module + compression-mode-`preserve` are
   tightly coupled — they share the new MoE fields on
   `NativeModelConfig`, the same forward path, and the same
   integration test fixture. Splitting forces double the OpenSpec
   overhead and double the review pass for what is fundamentally one
   coherent change.

If review finds it too large, the easiest split is "adapter + MLP +
preserve only" (this PR) vs "collapse + top_n placeholder" (a
follow-up). Hold that option in reserve.
