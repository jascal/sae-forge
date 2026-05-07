# sae-forge

**Forge a Polygram-compressed SAE into a small, semantically-native transformer.**

sae-forge takes a [Polygram](https://github.com/jascal/polygram)-compressed
sparse autoencoder and projects a host model's weights into the SAE's
surviving feature basis, producing a small transformer whose residual
stream *is* the SAE feature space — interpretable by construction, not by
post-hoc probing.

It is the practical bridge between modern SAEs (Gemma Scope, Llama Scope)
and usable, narrow models. Because Polygram's scale-aware compression
([PR #34](https://github.com/jascal/polygram/pull/34)) preserves both
original and merged decoder magnitudes, the forged model inherits faithful
feature scales rather than a degenerated unit-norm basis.

## Status

Pre-alpha. v0 milestone: working end-to-end on GPT-2-small + a toy
compressed SAE, then Gemma-2-2B / 9B on a single 4090. New work is staged
through OpenSpec changes — see `openspec/changes/`.

## Install

```bash
pip install -e ".[dev,torch,polygram,orca]"   # editable install with test deps + torch + polygram + FSM
pytest                                         # run the suite
```

Optional extras: `[plot]` (matplotlib), `[notebook]` (jupyter +
matplotlib), `[torch]` (torch + transformers — required for `NativeModel`
construction, `SubspaceProjector` projection from a real source model, and
fine-tuning), `[polygram]` (the upstream compressed-SAE producer),
`[orca]` (`orca-runtime-python` for the v0.1 FSM orchestrator).

### Setting up `.venv`

sae-forge expects Python 3.10+ and is developed against an in-repo
virtualenv at `.venv/`. The standard bootstrap:

```bash
git clone git@github.com:jascal/sae-forge.git
cd sae-forge

# 1. Create the venv (use a 3.10+ interpreter; check with `python3 --version`)
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Upgrade pip inside the venv (avoids stale-resolver headaches with torch wheels)
python -m pip install --upgrade pip

# 4. Editable install with the extras you need
pip install -e ".[dev,torch,polygram,orca]"

# 5. Verify
pytest -q
python -c "import saeforge; print(saeforge.__version__)"
```

Deactivate with `deactivate` when you're done. The platform-specific
sections below assume an activated `.venv` and only differ in which
torch wheel gets pulled.

> **Intel Mac (x86_64) caveat — use Python 3.10/3.11 and the `[intel]`
> extra.** PyTorch's last x86_64 macOS wheels are torch 2.2.2, which only
> ship for CPython 3.8–3.11 *and* were built against numpy 1.x. That
> creates two failure modes:
>
> 1. **Wrong Python.** On 3.12+, `pip install -e ".[torch,…]"` fails with
>    `Could not find a version that satisfies the requirement torch>=2.2
>    … (from versions: none)`.
> 2. **numpy 2 ABI break.** With Python 3.10/3.11 + numpy 2, `import
>    torch` "succeeds" with a UserWarning but the C extensions are
>    disabled. transformers's `is_torch_available()` then returns False
>    and tests fail with the misleading `GPT2LMHeadModel requires the
>    PyTorch library but it was not found in your environment`.
>
> The `[intel]` extra is a drop-in replacement for `[torch]` that pins
> the compatible set (`torch==2.2.2`, `transformers>=4.46,<4.50`,
> `numpy<2`). Use it instead of `[torch]`:
>
> ```bash
> brew install python@3.11           # if not already installed
> deactivate 2>/dev/null
> rm -rf .venv
> python3.11 -m venv .venv
> source .venv/bin/activate
> python -m pip install --upgrade pip
> pip install -e ".[dev,intel,polygram,orca]"
> ```
>
> Apple Silicon and Linux/CUDA hosts are unaffected — they should keep
> using `[torch]`, which tracks current wheels for 3.10–3.13.

### Running on Apple Silicon (M-series)

sae-forge runs natively on M-series Macs with MPS (Apple's GPU
backend). arm64 hosts get current torch wheels (2.4+) for CPython
3.10–3.13, so any interpreter in that range works — MPS support is
mature, bf16 paths work, op coverage is high, and unified memory
eliminates host-device transfer overhead.

```bash
git clone git@github.com:jascal/sae-forge.git
cd sae-forge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch,polygram,orca]"

# Smoke check (synthetic basis on real gpt2, ~12s):
python examples/forge_gpt2_real.py /tmp/sae-forge-test

# Real-SAE forge with MPS (~20s on M4):
python examples/forge_gpt2_real_sae.py /tmp/sae-forge-real-sae 32 mps
```

Tier guidance for the workloads sae-forge currently ships:

| Mac configuration            | What's comfortable                              |
|------------------------------|-------------------------------------------------|
| 16GB unified                 | GPT-2 family, real-SAE forge + smoke fine-tune  |
| 24GB unified (M4 / M3 Pro)   | Gemma-2-2B forge + serious fine-tune (planned)  |
| 36GB+ unified (M3/M4 Max)    | Gemma-2-2B comfortable, Gemma-2-9B forward-only |
| 64GB+ unified (Max/Ultra)    | Gemma-2-9B forge + fine-tune territory          |

**Intel Mac (x86_64) is supported but constrained.** PyTorch dropped
x86_64 macOS wheels after 2.2.2, and 2.2.2 only ships for CPython
3.8–3.11 — so Intel Macs must pin the venv to Python 3.10 or 3.11
(see the Intel-Mac caveat under [Setting up `.venv`](#setting-up-venv))
and stay on the 2.2.2 line, missing recent MPS improvements.

### Running on Linux + CUDA (NVIDIA)

sae-forge has no CUDA-specific code; it picks up `device="cuda"` like
any torch program. The `[torch]` extra installs whichever torch wheel
matches the host (CUDA-enabled if CUDA libs are present, CPU-only
otherwise). Standard install:

```bash
git clone https://github.com/jascal/sae-forge.git
cd sae-forge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch,polygram,orca]"

# Real-SAE forge with CUDA:
python examples/forge_gpt2_real_sae.py /tmp/sae-forge-real-sae 32 cuda
```

If you need a specific CUDA build (e.g. CUDA 12.1 wheels for a system
with older drivers), install torch from the PyTorch index *before* the
sae-forge editable install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,polygram,orca]"  # omit [torch] to keep your torch
```

Tier guidance for the workloads sae-forge can run today and the ones
the v0.3 milestone unlocks:

| GPU configuration              | What's comfortable today          | With v0.3 forge-finetune-recipe |
|--------------------------------|-----------------------------------|---------------------------------|
| Single 24GB (RTX 3090/4090)    | GPT-2-family + smoke fine-tune    | Gemma-2-2B forge + 1k-step ft   |
| Single 40GB (A100-40)          | Gemma-2-2B comfortable            | Gemma-2-9B forward-only forge   |
| Single 80GB (A100-80, H100)    | Gemma-2-9B forge + smoke ft       | Gemma-2-9B forge + 1k-step ft   |
| 2×24GB or 2×48GB               | Same as single-card; v0 doesn't  | 8B-class with model parallel    |
|                                | implement model parallelism yet   | (would need a separate change)  |

Notes for first-run on a fresh CUDA host:

- **Gemma / Llama license acceptance**: Google's Gemma checkpoints
  and Meta's Llama checkpoints on HuggingFace are gated. Run
  `huggingface-cli login` with a token from
  https://huggingface.co/settings/tokens, then visit each model's HF
  page and click "Agree and access" once.
- **Disk**: keep at least 50GB free under `~/.cache/huggingface/`
  if you plan to compare across SAE layers — Gemma Scope's full
  release for one model is ~100GB across all layers, but a single
  layer is ~3GB.
- **CUDA driver version**: torch 2.4+ wheels assume CUDA ≥11.8. If
  you're stuck on an older driver, pin torch to a matching wheel via
  the `--index-url` trick above.
- **v0 doesn't yet do model parallelism.** Single-GPU is the only
  supported layout in v0.1; multi-GPU lands as a separate
  `forge-multi-gpu` change once there's a workload that actually
  needs it.

## Layout

```
saeforge/         — Python package
openspec/         — spec-driven change proposals + capability specs
tests/            — pytest suite + fixtures (small synthetic SAEs)
examples/         — scripts + notebooks (GPT-2 toy forge, Gemma-2 forge, domain adaptation)
docs/             — design notes, research write-ups, README screenshots
```

## How it works

A Polygram-compressed SAE checkpoint exposes a feature basis: a set of
*kept* decoder rows `W_dec[kept_ids]` whose magnitudes have been preserved
through scale-aware merging, plus the original-scale norms used to forge
faithful weights. sae-forge:

1. Loads the basis (`FeatureBasis`).
2. Projects the host model's weight matrices into and out of that basis
   (`SubspaceProjector`).
3. Assembles a small transformer whose residual width equals the number of
   kept features (`NativeModel`).
4. Optionally fine-tunes against a faithfulness target on the original
   model's outputs (`ForgePipeline`).

The four components are independent — you can stop after `FeatureBasis`
to inspect the geometry, after `SubspaceProjector` to ship the projected
weights to your own training stack, or run the full `ForgePipeline` for
the turn-key path.

> **Mathematical foundation.** The full projection algebra (notation,
> projection rules, error model, theoretical guarantees, and the v0
> implementation notes flagging where the shipped code diverges from
> the canonical spec) lives in [`docs/algorithm.md`](docs/algorithm.md).
> Read it before changing the projector or proposing a v1 architecture.

> **Fine-tune recipe (v0.3).** The training loop (cosine LR + warmup,
> gradient clipping, optional gradient checkpointing, optional bf16/
> fp16 autocast, periodic eval, periodic saves, structured loss
> tracking) lives in [`docs/finetune-recipe.md`](docs/finetune-recipe.md).
> Local-corpus-first, offline-safe by spec — designed for proprietary
> data flows where nothing should leak to remote services. The
> headline demo is [`examples/forge_gemma2_2b.py`](examples/forge_gemma2_2b.py).

## Quickstart

```python
from saeforge import FeatureBasis, ForgePipeline, NativeModel, SubspaceProjector

basis = FeatureBasis.from_polygram_checkpoint("sae.compressed.safetensors")
print(basis.n_features, basis.d_model)        # kept-feature count, host width
print(basis.merged_norms.mean())              # scale-aware merged decoder norm

projector = SubspaceProjector(basis, scale_boost=1.0)
model = NativeModel.from_host(
    host_model_id="gpt2",
    projector=projector,
)

forge = ForgePipeline(
    basis=basis,
    projector=projector,
    model=model,
    eval_prompts=eval_prompts,
)
result = forge.run(output_dir="forged/")
print(result.faithfulness_kl, result.n_params)
```

### CLI

The `sae-forge` console script wraps the pipeline. Match the polygram CLI
style — verbs first, file paths positional:

```bash
sae-forge forge sae.compressed.safetensors \
    --host-model gpt2 \
    --output-dir forged/ \
    --eval-prompts prompts.jsonl

sae-forge inspect sae.compressed.safetensors --report basis_report.md
sae-forge --version
```

`sae-forge inspect` is the no-torch triage command: it loads the basis,
prints kept-id count, decoder-norm distribution, scale-compression ratio
(from Polygram's `CompressionReport`), and a quick rank estimate of the
basis — useful for deciding whether a given compression is worth forging
against.

## Hardware notes

- **GPT-2-small** (toy / smoke target): forging runs comfortably on CPU.
  v0 integration tests use a synthetic 64-feature SAE and a randomly
  initialized host model.
- **Gemma-2-2B**: forging fits in a single 4090 (24 GB) under fp16, with
  the `SubspaceProjector` operating on a per-layer streaming basis so the
  full host weight tree is never resident.
- **Gemma-2-9B**: same per-layer streaming path; tested on a single 4090
  with bf16 host weights and fp32 projection math. Fine-tuning the forged
  model is the bottleneck, not the projection step.

## Components

### `FeatureBasis`

Loads a Polygram compressed checkpoint (`.safetensors` + companion
`compression_report.json`) and exposes:

- `kept_ids: np.ndarray[int]` — surviving feature indices in original-SAE
  ordering,
- `W_dec: np.ndarray[float]` — kept decoder rows at original scale,
- `merged_norms: np.ndarray[float]` — per-feature decoder norms after
  Polygram's scale-aware merge (or originals when no merge happened),
- `scale_compression_ratio: float` — Polygram's roll-up scale stat,
- `pseudoinverse() -> np.ndarray` — cached `(W_dec.T)†` for the projector.

Pure-numpy. The `[torch]` extra is **not** required for inspection.

### `SubspaceProjector`

Performs the weight projection math:

- `embed: (V, d_model) -> (V, n_features)` via `W_embed @ pinv(W_dec.T)`,
- `qkv: (d_model, 3·d_head·n_heads) -> (n_features, 3·d_head·n_heads)`
  per attention block,
- `mlp_in: (d_model, d_ff) -> (n_features, d_ff)`,
- `mlp_out: (d_ff, d_model) -> (d_ff, n_features)`,
- `unembed: (d_model, V) -> (n_features, V)`.

The optional `scale_boost` knob compensates for under-coverage when the
basis spans less than the host residual stream — defaults to `1.0` (no
boost). See `docs/research/scale-boost-design.md` for the rationale (TBD).

### `NativeModel`

A lightweight HF-compatible small transformer skeleton whose
`hidden_size` equals `basis.n_features`. v0 supports decoder-only blocks
matching the host architecture's attention + MLP shapes. Wraps a minimal
in-tree implementation (no dependency on a specific HF model class
beyond `transformers.PreTrainedModel` for tokenizer round-trip).

### `ForgePipeline`

Orchestrates the full flow: basis load → projection → native model
construction → optional fine-tune → faithfulness eval. Emits a
`ForgeResult` with the projected model, faithfulness KL against the host
on a held-out prompt set, parameter count, and a structured artifact tree
under `output_dir/`.

## Examples

- `examples/forge_gpt2_toy.py` — toy 64-feature SAE → forged GPT-2-small
  variant. Smoke target, CPU-friendly.
- `examples/forge_gemma2_2b.py` — single-4090 Gemma-2-2B forge.
- `examples/domain_adaptation.py` — restrict the basis to a domain
  subset of features and forge a narrow specialist.

## Integration with Polygram

sae-forge is a downstream consumer of Polygram, not a fork. The contract:

- **Input**: a `.safetensors` file produced by `polygram compress` (or
  `polygram compress-epoch`) plus its companion `compression_report.json`.
- **Required Polygram version**: `>=0.1.0`, the polygram-tuning-config
  release that ships the typed config dataclasses sae-forge plumbs
  through (`CompressionConfig`, `EpochCompressionConfig`, `RegrowConfig`,
  `ValidationConfig`).
- **What sae-forge does not do**: it does not run validation, does not
  pick clusters, does not zero or merge — those are Polygram's job. It
  consumes the artifact and projects.

If you want to build a custom compression upstream (a different rep
selector, a non-Polygram SAE format), hand-roll a dict matching
`FeatureBasis`'s fields and call `FeatureBasis(**fields)` directly — the
loader is one entry point among several.

### Polygram tuning passthrough

`ForgePipeline` exposes three typed polygram-tuning fields:

| Field | Type | Drives |
|---|---|---|
| `compression` | `polygram.CompressionConfig` | `polygram.Compressor` (strategy / rep_selection / merge_mode / confirmer) |
| `epoch_compression` | `polygram.EpochCompressionConfig` | `polygram.EpochCompressor` (coverage_target, cosine_threshold, max_iterations, embedded `ValidationConfig`) |
| `regrow` | `polygram.RegrowConfig` | `polygram.Regrower.from_compression_report` (model_name, layer, strategy, prompts, seed) |

When `regrow_count > 0`, `regrow=RegrowConfig(model_name=..., layer=...)`
is **required** (`__post_init__` raises otherwise). The pre-change
`layer=10` / `model_name="gpt2"` ctx fallbacks were removed in 0.1.0
because they silently bound regrowth to GPT-2.

Configs round-trip through the FSM context as JSON-friendly dicts —
`ForgePipeline._build_context` calls `cfg.to_dict()` on each non-None
field, and the polygram-driven actions (`compress_with_polygram`,
`perform_regrowth`) reconstitute via `<Config>.from_dict(ctx[key])`
before calling polygram. This keeps the orca-runtime trace tooling
JSON-trivially-serialisable while end-to-end-typed at the Python API
boundary.

#### Loading from YAML/JSON

`ForgePipeline.from_dict(data)` accepts a flat mapping where the
`compression` / `epoch_compression` / `regrow` keys are nested dicts;
unknown top-level keys emit a `UserWarning` and are dropped (matching
polygram's forward-compat policy). One-shot YAML configs become a
two-line bootstrap:

```python
import yaml
from saeforge import ForgePipeline

with open("forge_config.yaml") as f:
    pipeline = ForgePipeline.from_dict(yaml.safe_load(f))
```

See `docs/forge_config_example.yaml` for an end-to-end example.

#### CLI flags

The five high-frequency knobs are reachable from the CLI:

```bash
sae-forge forge ckpt.safetensors --host-model gpt2 --output-dir out/ \
  --coverage-target 0.6 \
  --cosine-threshold 0.30 \
  --max-compress-iterations 2 \
  --regrow-count 2 --regrow-layer 4 --regrow-strategy residual_kmeans
```

Long-tail tuning (jaccard threshold, min_both_fire, etc.) lives behind
`ForgePipeline.from_dict` — pass a YAML/JSON config there.

## Development

```bash
pip install -e ".[dev,torch,polygram]"
pytest
ruff check saeforge tests examples
```

## License

Apache-2.0.
