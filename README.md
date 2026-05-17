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

Pre-alpha. Working end-to-end on GPT-2-small + a toy compressed SAE.
Recent landed work:

- **multi-architecture-support** (0.2.0): Llama-3 and Gemma-2 host
  families now project through the same pipeline as GPT-2 via the
  ``saeforge.adapters`` registry. Pythia / GPT-NeoX deferred.
- **forge-finetune-recipe**: cosine-LR + warmup, gradient clipping,
  optional gradient checkpointing, optional bf16/fp16 autocast,
  periodic eval/save, structured loss tracking. See
  [`docs/finetune-recipe.md`](docs/finetune-recipe.md).
- **forge-continual-learning-loop**: three-loop continual-learning
  topology (stream / refine / basis), protected-feature compression
  (structural EWC), replay buffer for fine-tune. All knobs default to
  values that recover the single-shard pipeline byte-identically. See
  [`docs/advanced-fsm-options.md`](docs/advanced-fsm-options.md).
- **forge-whisper-encoder**: encoder-only Whisper forging — the first
  non-causal-LM architecture in the registry. New
  `WhisperEncoderAdapter`, `ForgedWhisperEncoder` native module (with
  a frozen-copied conv stem and a `basis_encode` buffer at the
  d → f boundary), `cosine_faithfulness` eval, family-aware
  `evaluate_faithfulness` dispatch. LM byte-equivalence net stays
  green. See [`docs/audio-forge.md`](docs/audio-forge.md).

New work is staged through OpenSpec changes — see ``openspec/changes/``.

## Install

```bash
pip install -e ".[dev,torch,polygram,orca]"   # editable install with test deps + torch + polygram + FSM
pytest                                         # run the suite
```

Optional extras: `[plot]` (matplotlib), `[notebook]` (jupyter +
matplotlib), `[torch]` (torch + transformers — required for `NativeModel`
construction, `SubspaceProjector` projection from a real source model, and
fine-tuning), `[polygram]` (the upstream compressed-SAE producer),
`[orca]` (`orca-runtime-python` for the FSM orchestrator that drives the
forge pipeline + the continual-learning extensions).

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
| 24GB unified (M4 / M3 Pro)   | Gemma-2-2B / Llama-3-8B forge + serious fine-tune |
| 36GB+ unified (M3/M4 Max)    | Gemma-2-2B comfortable, Gemma-2-9B forward-only |
| 64GB+ unified (Max/Ultra)    | Gemma-2-9B forge + fine-tune territory          |

Supported host families (post-multi-architecture-support):
**GPT-2 family**, **Llama-3** (Llama-2 also works via the same
adapter), and **Gemma-2**. Pythia / GPT-NeoX is deferred and will
need a small upstream polygram addition for parallel Q/K/V — track
on the [issue list](https://github.com/jascal/sae-forge/issues).

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

> **Fine-tune recipe.** The training loop (cosine LR + warmup,
> gradient clipping, optional gradient checkpointing, optional bf16/
> fp16 autocast, periodic eval, periodic saves, structured loss
> tracking) lives in [`docs/finetune-recipe.md`](docs/finetune-recipe.md).
> Local-corpus-first, offline-safe by spec — designed for proprietary
> data flows where nothing should leak to remote services. The
> headline demo is [`examples/forge_gemma2_2b.py`](examples/forge_gemma2_2b.py).

> **Continual-learning loop.** The single-shard pipeline above is the
> default. The continual-learning extension adds three nested loops on
> top of the same FSM — *stream* (per shard), *refine* (per-shard
> convergence), *basis* (compress↔regrow refinement) — plus
> protected-feature compression (structural EWC) and a replay buffer
> for fine-tune. All opt-in behind defaults that recover the
> single-shard pipeline byte-identically. See
> [`docs/advanced-fsm-options.md`](docs/advanced-fsm-options.md) for
> the full knob reference, the decision tree for choosing a
> `task_trigger`, and worked recipes per pattern.

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

### Continual learning

Opt in by setting any of `n_tasks > 1`, `inner_refine_passes > 1`,
`protect_top_k > 0`, or `replay_ratio > 0`. Defaults are
single-shard, byte-identical with the snippet above.

```python
from saeforge import ForgePipeline
from saeforge.training import LabeledTaskStream

forge = ForgePipeline(
    basis=basis,
    projector=projector,
    orchestrator="fsm",
    # Stream loop: five labeled task shards
    n_tasks=5,
    task_trigger="labeled",
    task_stream=LabeledTaskStream([shard1, shard2, shard3, shard4, shard5]),
    # Basis loop: one extra compress↔regrow refinement pass per shard
    inner_refine_passes=2,
    regrow_count=32,
    # Structural EWC: pin the top-32 highest-magnitude features per shard
    protect_top_k=32,
    # Replay: 25% of fine-tune batches drawn from past tasks, stratified
    replay_ratio=0.25,
    replay_buffer_size=1024,
    replay_policy="per_task",
)
result = forge.run(output_dir="forged/")
```

The full knob reference, decision tree for choosing a `task_trigger`,
and worked recipes per pattern live in
[`docs/advanced-fsm-options.md`](docs/advanced-fsm-options.md).

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

#### Pareto sweep (Axis 4)

`sae-forge sweep-pareto` forges across the per-K SAE checkpoints
produced by `polygram compress --pareto --pareto-materialize`,
optionally spanning multiple labelled encodings. It is the
load-bearing primitive for Axis 4 of polygram's rung-viability
methodology — end-to-end downstream confirmation that the
compression-coverage lift visible in EpochCompressor cashes out in
forged-model KL space.

There are two ways to run an Axis-4 sweep:

##### One-tool workflow (recommended): `--auto-materialise`

`sae-forge sweep-pareto --auto-materialise` collapses polygram-side
compression and the per-K forge sweep into a single invocation, with
the validation-vs-eval-prompts leakage firewall as a first-class API
constraint.

**Pre-flight first**: before paying validator cost, dry-run with
`--plan-only` to inspect what would happen — per-encoding cache
status, SHA-256 fingerprints of the SAE and validation prompts, the
target K list, and an estimated validator-forward count:

```bash
sae-forge sweep-pareto --auto-materialise --plan-only \
    --encoding mps:mps_sae.safetensors \
    --host-model gpt2 --layer 8 \
    --pareto 8,16,24,32 \
    --validation-prompts data/validation.jsonl \
    --eval-prompts data/eval.jsonl \
    --output-dir runs/axis4/
```

Output (cold cache):

```
sweep-pareto --plan-only: per-encoding plan
  label=mps
    cache_status=MISS (cold)
    sae_sha256=4f3a...
    validation_prompts_sha256=1b9c...
    targets=[8, 16, 24, 32]
    encoding_class=MPSRung1
    encoding_kwargs={}
    validator_forward_count_estimate=2400
```

If everything looks right, drop `--plan-only` and run for real:

```bash
sae-forge sweep-pareto \
    --auto-materialise \
    --encoding mps:mps_sae.safetensors \
    --encoding rung4:rung4_sae.safetensors \
    --encoding-class mps:MPSRung1 \
    --encoding-class rung4:Rung4 \
    --host-model gpt2 --layer 8 \
    --pareto 8,16,24,32 \
    --validation-prompts data/validation.jsonl \
    --eval-prompts data/eval.jsonl \
    --validation-threshold 0.95 \
    --rep-selection kl_attribution \
    --output-dir runs/axis4/
```

This runs polygram's `BehaviouralValidator → Compressor.plan_pareto →
apply` per encoding (artifacts cached under
`runs/axis4/_materialised/<label>/`), then forges each materialised
K and emits `frontier.jsonl` with the four diagnostics fields PLUS
three provenance fields (`validation_threshold`, `encoding_class`,
`validation_eval_overlap`).

**Leakage firewall**: `--validation-prompts` and `--eval-prompts`
MUST resolve to distinct file paths by default. The CLI refuses
same-path resolution at parse time; override via
`--allow-validation-eval-overlap` if you accept the methodological
compromise (surfaces as `validation_eval_overlap=true` in every
frontier row so analysis can flag it). This separation is the
*reason* the auto-materialise flow exists — collapsing prompt sets
would invite the validator to gate features against the same
corpus that later scores faithfulness.

**For SAEs with >8 features**: MPSRung1's default cap is 8. Use
`--encoding-class LABEL:HEA_Rung2 --encoding-qubits LABEL:N` (cap
= 2^N) for larger feature counts:

```bash
sae-forge sweep-pareto --auto-materialise \
    --encoding rung4:rung4_sae.safetensors \
    --encoding-class rung4:HEA_Rung2 \
    --encoding-qubits rung4:5 \
    ...
```

**Pre-flight check before paying validator cost**: `--plan-only`
prints per-encoding cache status (`HIT` / `MISS` with diffing
fields), SHA-256 fingerprints, target K list, and a
validator-forward-count estimate, then exits 0 without running
anything. Mutually exclusive with `--frontier-only`.

**Escape hatch**: `--force-rematerialise` bypasses the cache when
you've manually edited polygram-side state the cache doesn't
fingerprint (rare).

##### Two-tool workflow (manual control)

When you need polygram-side knobs the auto-materialise CLI doesn't
expose (`min_firing_rate`, `min_both_fire`, custom `confirmer`,
exotic encoding kwargs), drop down to the two-tool flow. **Step 1
(polygram, cheap-then-expensive):**

```bash
# Plan + materialise N SAEs per encoding. Pareto planning is
# O(one validator pass) per encoding amortised across all K.
polygram compress --sae-checkpoint mps_sae.safetensors \
    --validation-report mps_report.json \
    --pareto 200,500,1000,2000 --pareto-materialize \
    --out runs/mps/

polygram compress --sae-checkpoint rung4_sae.safetensors \
    --validation-report rung4_report.json \
    --pareto 200,500,1000,2000 --pareto-materialize \
    --out runs/rung4/
```

**Step 2 (sae-forge, the actual sweep):**

```bash
sae-forge sweep-pareto \
    --encoding mps:runs/mps/pareto \
    --encoding rung4:runs/rung4/pareto \
    --host-model gpt2 \
    --output-dir runs/axis4/ \
    --eval-prompts data/eval.jsonl
```

This writes `runs/axis4/frontier.jsonl` (one row per `(encoding, K)`)
and per-forge directories under `runs/axis4/<label>/k_{K}/`. The
JSONL row schema is in
`openspec/specs/pareto-sweep/spec.md`; the key fields are
`encoding_label`, `target_n_features_kept`, `n_features_kept_actual`,
`faithfulness_kl`, `perplexity`, `final_fine_tune_loss`. Filter on
`error_message is None` before reading metric fields.

The sweep is **resumable** (rerun the same command after a crash —
completed rows are skipped) and **per-row failure-isolated** (one
bad K records `error_message` and the sweep continues). It exits
non-zero if any row errored, with `frontier.jsonl` still written.

For cheap exploratory triage before committing forge compute, add
`--frontier-only` — it emits a JSONL with only the manifest-derived
columns (no forge calls). Pipe through `jq` to find candidate K
values:

```bash
sae-forge sweep-pareto --encoding mps:runs/mps/pareto \
    --host-model gpt2 --output-dir runs/triage/ --frontier-only

jq -r 'select(.error_message == null) |
    [.encoding_label, .target_n_features_kept, .n_features_kept_actual]
    | @tsv' runs/triage/frontier.jsonl | sort -t$'\t' -k2 -n
```

For large hosts (Gemma-2-2B / 8B-tier), split sweeps by encoding
into separate processes rather than packing many `--encoding` flags
into one invocation — every row inside a single sweep loads the
host + per-K forged model into the same process, and transient
state accumulates across rows.

##### Forge-quality diagnostics

Every sweep row carries four diagnostic fields telling you whether
the row's KL is worth reading at all:

- `host_d_model` — host transformer's residual stream width
  (`AutoConfig.hidden_size`)
- `basis_rank` — numerical rank of the kept-features `W_dec`
- `quality_ratio` — `basis_rank / host_d_model`
- `quality_tier` — one of `saturated` / `good` / `undersized` /
  `degenerate` (heuristic thresholds: 1.0 / 0.5 / 0.0625)

The recommended frontier-triage workflow is to filter on
`quality_tier` *before* reading `faithfulness_kl`:

```bash
jq -r 'select(.quality_tier == "good" or .quality_tier == "saturated") |
    [.encoding_label, .target_n_features_kept, .quality_tier, .faithfulness_kl]
    | @tsv' runs/axis4/frontier.jsonl | sort -t$'\t' -k2 -n
```

When the smallest K's basis falls into the `undersized` or
`degenerate` tier for any encoding, the sweep prints a stderr
advisory before doing any forge work and suggests a higher K floor.
For strict refusal (exit non-zero before any forge cost), add
`--quality-floor 0.5` — sweeps only proceed if every row would be
at least in the `good` tier. `--quality-tier-thresholds
saturated:1.0,good:0.5,undersized:0.0625` overrides the heuristic
boundaries for callers running specific research.

The wording note in the advisory body matters: `degenerate`
describes the **rank ratio**, not the validity of the run.
Exploratory low-rank smokes remain valid for impl validation; the
advisory is informational, not a refusal by default.

**Custom hosts**: `host_d_model` is resolved automatically from
`AutoConfig.from_pretrained(host_model_id).hidden_size`. For hosts
whose config doesn't expose `hidden_size` canonically (Whisper
encoder, encoder-decoder architectures, non-transformer hosts), the
resolution returns `None` and diagnostics fall back gracefully —
all four row fields stay `None` and no advisory prints. If you know
the residual width for your host, the Python API accepts
`host_d_model_override=N` on `ForgePipeline.sweep_pareto(...)` to
short-circuit the AutoConfig lookup and force diagnostics on.

### Inspect

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
boost). Two resolution modes:

- **Literal float**: `SubspaceProjector(basis, scale_boost=0.25)`.
  Reproducible, no calibration overhead. Use when you've characterised
  the basis and know what you want.
- **`"auto"`**: `min(1.0, d_model/n_features)`. Basis-shape-aware
  fallback that defends against the over-complete blow-up footgun on
  random-Gaussian bases. Under-corrects on polygram-compressed bases
  (see `openspec/changes/fix-scale-boost-calibration/design.md`).

An earlier draft of `fix-scale-boost-calibration` added a
`scale_boost="calibrate"` auto-picker. The 2026-05-16 smoke gate
falsified the mechanism (three successive proxies for forge KL all
picked the wrong value) and the mode was dropped. The change shipped
as forge-magnitude diagnostics instead: `--magnitude-diagnostics
tokens:N` (or `prompts:PATH`) populates `logit_std_ratio` and
`top1_anomalous` on every row, and `--rank-monotonicity-check`
prints a post-sweep advisory if `faithfulness_kl` is non-monotone in
K. Together they help diagnose WHY a sweep produced poor forge KL
without claiming to fix it.

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
