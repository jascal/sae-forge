# Fine-tune recipe

The v0.3 recipe replaces the v0.1 4-step smoke fine-tune with a real
LM cross-entropy training loop. This document covers the corpus
formats, the loop's design choices, and how to use it from
`ForgePipeline`.

## Quick reference

```python
from saeforge import ForgePipeline, FeatureBasis, SubspaceProjector

pipeline = ForgePipeline(
    basis=basis,
    projector=projector,
    host_model_id="gpt2",
    eval_prompts=["The mitochondrion is the powerhouse of the"],
    finetune_corpus="/path/to/local/corpus.txt",     # local file → no HF
    finetune_total_steps=1000,
    finetune_warmup_steps=100,
    finetune_peak_lr=5e-5,
    finetune_batch_size=8,
    finetune_seq_len=512,
    finetune_precision="bf16",                       # "fp32" / "bf16" / "fp16"
    finetune_grad_checkpoint=True,                   # enable for Gemma-2 on 24GB
    finetune_eval_every=100,
    finetune_save_every=250,
)
result = pipeline.run(output_dir="/tmp/forge-run/")
print(result.faithfulness_kl)
```

## Host distillation

`forge-finetune-recipe` v0.3 computes `faithfulness_kl(host, forged)`
as an **eval** metric. `add-host-distillation-finetune-loss` makes
it optionally a **training** signal too — `TrainingConfig` accepts
two new knobs:

| Knob | Default | Effect |
|---|---|---|
| `distill_alpha` | `1.0` | Loss = `α·CE(corpus) + (1-α)·τ²·KL(host ‖ forged)`. At `α=1.0` (default) the KD branch is skipped entirely (zero compute vs the pre-change path). |
| `distill_temperature` | `2.0` | Softmax temperature `τ` applied to both host and forged logits before computing KL. Standard Hinton-style scaling. |

The `ForgePipeline` exposes the same knobs as `finetune_distill_alpha`
and `finetune_distill_temperature`.

### When to enable it

- **Small custom corpora.** Pure LM-CE on a short corpus under-samples
  the host's behavior. Adding the KD term pulls the student toward
  the host's logit distribution on every batch, not just the corpus
  ground truth.
- **High-faithfulness target.** If `faithfulness_kl` is your real eval
  metric, optimizing it directly is more efficient than hoping CE
  recovers it.

### When NOT to enable it

- **Per-step compute matters.** A host forward at every training step
  roughly doubles wall-clock. Rough numbers on a 24 GB card:
  - GPT-2 small host: +30-50% wall time (host is small relative to
    optimizer state + student backward).
  - Llama-3-8B / Gemma-2-9B host: +80-120% wall time (host forward
    is the dominant cost; nearly a full doubling).
  - 70B-class hosts: not recommended without gradient checkpointing
    on the student and `bf16` autocast; expected +120-150% if the
    host fits at all.
  The `α=1.0` default avoids this cost entirely.
- **Eval set is your training corpus.** If the corpus IS what the
  host was trained on, CE already approximates the host's distribution
  — KD is mostly redundant.

### Recommended starting values

If you're enabling KD for the first time on a forge run:

- `distill_alpha = 0.5` — equal weighting of corpus CE and host KL.
- `distill_temperature = 2.0` — standard literature default.

Sweep `α ∈ {0.3, 0.5, 0.7}` and `τ ∈ {1.0, 2.0, 4.0}` if the
single-point default doesn't beat the pure-CE baseline on
`faithfulness_kl`.

### Requirements

- `host` must be non-None when `distill_alpha < 1.0`. `run_finetune`
  raises `ValueError` at the top of the function if you violate this
  — no batches are consumed before the check.
- Host and forged models must share the same tokenizer / vocab size
  (the `SubspaceProjector` already ensures this by construction;
  defensive shape check at the KL site).

## Swapping the faithfulness target

By default, `ForgePipeline` uses KL divergence between the host's and
the forged model's next-token distributions as the loop-gating
faithfulness signal (for LM hosts; per-frame cosine for
`whisper_encoder` hosts). The `pluggable-faithfulness` change makes
that signal pluggable: pass any object satisfying the
`saeforge.eval.faithfulness.FaithfulnessTarget` protocol to
`ForgePipeline(faithfulness=...)` and the FSM consults it instead.

Protocol shape:

```python
from typing import Any, Literal, Mapping, Protocol


class FaithfulnessTarget(Protocol):
    name: str                           # e.g. "kl", "cosine", "gt_alignment"
    better_when: Literal["higher", "lower"]

    def score(
        self,
        *,
        forged: Any,
        host: Any,
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        """Returns (score, perplexity_analog)."""
```

The second tuple element is the "perplexity analog" the FSM's
`perplexity < best_perplexity` progress check consumes. For
`better_when="lower"` targets (KL, MSE) the canonical analog is
`exp(score)`; for `better_when="higher"` targets (cosine,
GT-alignment) it is `1 - score` clamped at 0.

Built-in targets:

| Target | `name` | `better_when` | Default for |
|--------|--------|---------------|-------------|
| `saeforge.eval.targets.KLTarget` | `"kl"` | `"lower"` | LM hosts (gpt2 / llama / gemma2 / qwen2 / qwen3) |
| `saeforge.eval.targets.CosineTarget` | `"cosine"` | `"higher"` | `whisper_encoder` hosts |

A minimal custom target (full version in
`examples/forge_with_gt_alignment.py`):

```python
from saeforge.eval.faithfulness import FaithfulnessTarget


class GTAlignmentTarget:
    name = "gt_alignment"
    better_when = "higher"

    def __init__(self, labels):
        self._labels = labels

    def score(self, *, forged, host, ctx):
        # `host` is ignored — GT alignment doesn't need a teacher.
        features = forged.encode(ctx["_gt_alignment_inputs"])
        alignment = _cluster_alignment(features, self._labels)
        return float(alignment), float(1.0 - alignment)


pipeline = ForgePipeline(
    basis=basis,
    projector=projector,
    host_model_id="gpt2",
    faithfulness=GTAlignmentTarget(labels=cluster_ids),
    # …
)
result = pipeline.run(output_dir)
print(result.faithfulness, result.faithfulness_target_name)
```

Third-party targets SHOULD namespace their `ctx` keys with a
module-specific prefix (e.g. `_myorg_inputs`) to avoid clashes with
the built-in `_eval_*` keys.

`ForgePipeline(faithfulness=None)` (the default) is byte-identical to
the pre-change behaviour: the family-based default policy picks
`KLTarget` for LM hosts and `CosineTarget` for `whisper_encoder`. No
migration is required.

`ForgeResult.faithfulness_kl` is deprecated in favour of the generic
`ForgeResult.faithfulness` (with `ForgeResult.faithfulness_target_name`
naming the active scorer). Reads still work for one minor version and
emit `DeprecationWarning`; new code should use `.faithfulness` directly.

## Corpus formats

`build_iterator` (used internally when you pass `finetune_corpus`) accepts:

- **Local `.txt`** — one document per line. Most ergonomic for work
  / proprietary data flows.
- **Local `.jsonl`** — each line a JSON object with a `"text"` field.
  Common scrape format.
- **Local directory** — read recursively, picking up `.txt` and
  `.jsonl` files. Useful when corpus is sharded across many files.
- **HuggingFace dataset name** (e.g. `"HuggingFaceFW/fineweb-edu"`)
  — lazy-imports `datasets`. Requires the `[recipe]` extra.
- **Pre-tokenized iterable** — any iterable yielding
  `(batch_size, sequence_length)` int64 tensors. For users with their
  own tokenization pipeline.

Local-source paths trigger zero `datasets` imports — the recipe is
**offline-safe by construction** for proprietary work data.

## Local-only data handling

The recipe is local-only by spec:

- **No automatic uploads.** Forged checkpoints, loss curves, eval
  outputs all stay under `output_dir` / `save_dir`. Nothing pushes to
  HF Hub, W&B, comet, or any remote service.
- **No telemetry.** sae-forge has zero phone-home behaviour.
- **HF cache stays local.** Models downloaded from HuggingFace go to
  `~/.cache/huggingface/`. Nothing about the training corpus, the
  SAE, or the forged outputs is sent to HF.
- **Forged checkpoints are derivative works** of the training corpus
  under most copyright doctrines. If your corpus is restricted, treat
  the forged checkpoint with the same restrictions.

These guarantees are tested at the spec level — see
`tests/test_fsm_finetune_recipe.py::test_pretokenized_iterator_does_not_import_datasets`.

## Knob choices

The defaults are derived from Gemma-2's published recipe:

| Knob | Default | When to change |
|------|---------|----------------|
| `finetune_total_steps` | 1000 | Raise to 5k–10k for production runs |
| `finetune_warmup_steps` | 100 (10% of total) | Standard 5–10% rule |
| `finetune_peak_lr` | 5e-5 | 1e-4 for GPT-2-class hosts; 1e-5 for Gemma-2-9B+ |
| `finetune_batch_size` | 8 | Drop to 4 for 16GB VRAM; raise to 32 for 80GB+ |
| `finetune_seq_len` | 512 | 256 for memory-tight runs; 1024 for richer context |
| `finetune_precision` | `fp32` | `bf16` on M-series / modern CUDA; `fp16` on older NVIDIA |
| `finetune_grad_checkpoint` | False | Enable for Gemma-2-2B+ on 24GB VRAM |
| `finetune_eval_every` | 100 | Lower for shorter runs; raise to reduce overhead |
| `finetune_save_every` | 250 | Disk-bound; raise if checkpoint disk is slow |

## When to enable gradient checkpointing

`gradient_checkpointing=True` wraps each transformer block in
`torch.utils.checkpoint.checkpoint`, halving activation memory at the
cost of ~25% wall-clock recompute. Enable when:

- Forged model has ≥6 layers AND
- `batch_size × sequence_length` activations dominate over optimizer
  state.

For toy GPT-2 forges: not worth it. For Gemma-2-2B forges on 24GB:
roughly the difference between fitting and OOMing.

## bf16 vs fp16 vs fp32

- **`fp32`** (default): no autocast, no scaler. Maximum precision,
  highest memory.
- **`bf16`**: `torch.autocast(dtype=torch.bfloat16)`, no scaler.
  bf16's exponent range eliminates the need for loss scaling.
  **Recommended for M-series Macs and modern CUDA (Ampere+).**
- **`fp16`**: `torch.autocast` + `GradScaler`. Backward-compat for
  older NVIDIA cards (V100, T4). Requires CUDA — MPS doesn't support
  the GradScaler.

## Convergence heuristic

`TrainingResult.converged` is `True` when the trailing-100-step
loss EMA changes by less than 1% across two consecutive 100-step
windows. v0.3 records this in the result but **does not early-exit**
on it; the loop always runs `total_steps`. Future change
`forge-finetune-early-stop` can wire it as an exit signal once the
heuristic is empirically validated.

## OOM handling

The loop catches `torch.cuda.OutOfMemoryError` once and halves the
batch size on retry. Second OOM raises to the FSM error handler,
which transitions to `failed` state cleanly with a populated
`error_message`. On MPS, `cuda.OutOfMemoryError` doesn't fire — MPS
just kills the process; size your batches conservatively up-front
or watch Activity Monitor's GPU memory pressure.

## What's the relationship between projection and fine-tuning

The forged model is initialized via projection — every linear map in
the host transfers into the basis-width residual stream. Linear
components match the host exactly when the basis spans the residual.
The projection error sits in three places:

- `ε_rare` (rare features compressed away)
- `ε_attn` (softmax acting on projected Q/K)
- `ε_nonlin` (GeLU not commuting with projection)

Fine-tuning targets `ε_attn` and `ε_nonlin` directly. Because the
linear init is geometrically faithful, gradients descend a
much-better-conditioned loss surface than they would for a
randomly-initialized model — this is why the recipe uses a
*conservative* peak LR (5e-5) by default, not the larger LRs
appropriate for from-scratch training. We're correcting nonlinear
mismatches, not learning representations.

The full algebra is in [`docs/algorithm.md`](algorithm.md).
