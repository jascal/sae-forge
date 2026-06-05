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
- **sae-moe-forge**: `forge_to_moe(basis)` projects a polygram-compressed
  SAE into a routed mixture-of-experts (`ForgedMoE`) whose per-token
  decode cost scales as `k_experts / n_experts` of the flat SAE. v1 is
  inference-only with zero new parameters: each expert is a deterministic
  slice of the SAE decoder (`sub_dictionary`) and routing wraps polygram's
  summed-activation heuristic (`polygram_heuristic`). Faithfulness is
  free on clusterable bases and advisory on isotropic ones (a reported
  `coherence_diagnostic`). See [`docs/moe-forge.md`](docs/moe-forge.md).

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

##### Polygram concept-structure diagnostics

`quality_ratio` answers "can this basis span the host residual
stream." It does not answer "how many distinct concepts does the
dictionary actually encode?" — and two SAEs with identical
`basis_rank` can have wildly different *concept concentration* (one
could encode 6 clean concepts plus 40 redundant copies, the other 40
distinct concepts with no redundancy). The forge consequences are
completely different. Every sweep row therefore carries four
polygram-side concept-structure diagnostic fields, populated from the
`compression_report.json` that polygram drops next to each
compressed SAE:

- `polygram_n_clusters` — number of distinct concept clusters
  polygram's compressor identified in the dictionary
- `polygram_n_zeroed` — number of dictionary slots polygram zeroed
  as redundant during compression
- `polygram_redundancy_ratio` — `n_zeroed / (n_clusters + n_zeroed)`;
  the single number to colour a frontier plot by to surface "concept
  concentration"
- `polygram_encoding_capacity` — the encoding's cap (Rung3=16,
  Rung4=32, Rung5=128, HEA_Rung2(n)=2ⁿ), resolved from the encoding
  label

These metrics are as polygram reports them — see the polygram docs
for the definitional details of how clusters are formed under each
compressor strategy. High redundancy ≈ concentrated concepts; the
econ-sae Phase 7.2 supervised vs unsupervised contrast at Rung5
cap=128 produced 6 clusters / 88 zeroed (69% redundancy) vs 7
clusters / 62 zeroed (48% redundancy) for the same substrate.

To filter the frontier on concept structure:

```bash
jq 'select(.polygram_n_clusters != null) | {enc: .encoding_label,
    k: .n_features_kept_actual, clusters: .polygram_n_clusters,
    redundancy: .polygram_redundancy_ratio, kl: .faithfulness_kl}' \
    runs/axis4/frontier.jsonl
```

**Cluster-count saturation sweep.** Cluster count grew 2 → 3 → 6 on
the econ-sae supervised SAE across Rung3 → Rung4 → Rung5 and then
saturated at 6 — i.e., bumping capacity past 128 didn't find more
concepts. The pre-flight advisory surfaces this signal: when the
largest-K SAE in any encoding reports `polygram_n_clusters ==
polygram_encoding_capacity`, the advisory appends a one-line note
suggesting the next encoding rung (Rung5 → `HEA_Rung2(n_qubits=8)`,
etc.). The note is informational only; `--quality-floor` continues
to react to `quality_ratio` only. The recipe is the existing
multi-encoding flag:

```bash
sae-forge sweep-pareto \
    --encoding rung3:runs/rung3 \
    --encoding rung4:runs/rung4 \
    --encoding rung5:runs/rung5 \
    --host-model gpt2 --output-dir runs/capacity-sweep/
```

When `polygram_encoding_capacity` is `None` (unknown encoding label
that doesn't parse to Rung3/4/5/HEA_Rung2), the saturation check is
skipped and the row's capacity field stays `None` — no false
positives. When the compression report is missing (sweeping against
a non-polygram-compressed SAE), all four polygram fields are
populated with `None` and the sweep proceeds normally.

### Capability-aware forge tuning

`sae-forge sweep-capability` + `sae-forge recommend` answer the
question **"does the forged model retain the downstream task?"** — in
contrast to `sweep-pareto`'s cosine / KL faithfulness metrics, which
ask "are the forged hidden states numerically close to host?".
Bio-sae's empirical investigation found those two Pareto frontiers
disagree by up to 16× on optimal width
([openspec/changes/add-downstream-capability-target](openspec/changes/add-downstream-capability-target)).

Workflow:

```bash
# 1. Describe the dataset (encoder = a trained SAE; labels = GT binary matrix).
cat > bio-residue.yaml <<'YAML'
encoder_checkpoint: runs/uniref50_n5000/pooled_w1024_k64/sae.pt
sequences_path:    data/uniref50_sample__n5000_seed0.parquet
labels_path:       data/bio_bundle_uniref50.safetensors
feed:              pooled
tokenizer_id:      facebook/esm2_t6_8M_UR50D
aggregator:        pool_then_encode
min_prevalence:    10
sae_variant:       topk
sae_k:             64
YAML

# 2. Sweep — Pareto over (encoding × width × scale_boost) in retained-AUC space.
sae-forge sweep-capability \
    --dataset-config bio-residue.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --widths 16,64,128,256,512,1024 \
    --scale-boosts 1.0,auto \
    --output-dir runs/capability_sweep/

# 3. Recommend the smallest config meeting a retention target.
sae-forge recommend \
    --frontier runs/capability_sweep/frontier.jsonl \
    --target retained-mauc>=0.85 \
    --target gap-p95<=0.08
```

The `recommend` predicate parser accepts kebab-case
(`retained-mauc`) or snake_case (`retained_mauc_vs_host`); multiple
`--target` flags AND together; `--json` emits the picked row as
machine-readable JSON.

**Host-extraction cache.** First sweep cell populates a content-
addressed safetensors cache under `<output-dir>/host_cache/`; all
subsequent cells (and re-runs with the same inputs) skip the host
forward entirely. Opt-out via `--no-host-cache` for non-deterministic
hosts or scarce disk.

**Frontier schema.** `frontier.jsonl` rows are
`saeforge.ParetoFrontierRow` (the same dataclass `sweep-pareto`
emits) with optional capability fields populated:
`host_baseline_mauc`, `forge_mauc`, `retained_mauc_vs_host`,
`gap_median` / `gap_p25` / `gap_p75` / `gap_p95`,
`n_features_gap_above_0_1`, `n_features_negative_gap`,
`capability_aggregator`, `capability_min_prevalence`.
Pre-change frontier files load unchanged (back-compat); rows lacking
these fields default them to `None`.

### Progressive capability sweep — smallest n robust to data scale

`sae-forge sweep-capability-progressive` answers a stronger question
than the single-shot sweep above: **what's the smallest n that's
stable across data scales?**

Single-shot `sweep-capability` reports the argmax retained_mauc on
whatever eval sample you fed it. Bio-sae's residue-feed empirical
work showed that argmax position drifts with data scale: n=16 at 10
proteins → n=48 at 100 proteins, both at retained_mauc ≈ 1.03. The
PEAK value is data-scale-stable; *which* small basis is the argmax
isn't. A user running single-shot at low data picks a noise-driven
argmax; a user running at higher data picks a different noise-driven
argmax. Neither is robust.

The progressive wrapper runs the sweep at increasing protein counts,
identifies the **plateau** of widths within `plateau_tolerance` of
the peak, and **converges only when the smallest plateau-member
stops shifting** across stages. The recommendation contract becomes:

> Smallest `target_n_features_kept` whose retained_mauc is stable
> across the last K stages of data scaling.

This is **Occam's razor applied to forge basis selection**: among
widths that explain the labels equally well across data scales, pick
the smallest. The
[openspec proposal](openspec/changes/add-progressive-capability-sweep/proposal.md)
walks through the empirical motivation + the connection to classical
model selection (BIC / AIC / MDL).

Workflow (same YAML config as `sweep-capability`):

```bash
sae-forge sweep-capability-progressive \
    --dataset-config bio-residue.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --candidate-widths 4,8,16,32,64,128,256,512,1024 \
    --schedule 10,50,200 \
    --convergence-n-stages 2 \
    --output-dir runs/progressive_residue/

sae-forge recommend \
    --frontier runs/progressive_residue/frontier.jsonl \
    --target retained-mauc>=0.95
```

Schedule shape: comma-separated protein counts per stage, monotone
non-decreasing. Cumulative subsampling means stage K+1's protein set
is a strict superset of stage K's, so the host-extraction cache
survives across stages.

**Convergence-aware `recommend`.** When `sae-forge recommend` is
invoked against a progressive frontier (any row carrying a `stage`
field), it reads the companion `progressive_summary.json` for the
`recommendation.converged` flag. If `False`, the subcommand
**refuses to emit a recommendation** with a rich diagnostic naming:

- The recommended n + retained_mauc.
- The list of shifted stages drawn from `convergence_trajectory`.
- The on-disk rationale string.
- **Four informed opt-outs**: `--accept-unconverged`, longer schedule,
  looser `plateau_tolerance`, `convergence_n_stages=1`.

Single-shot frontiers (no `stage` field) bypass the check entirely
— back-compat with v0.8.x.

**Two opt-in "less-strict" modes** that are NOT
`--accept-unconverged`:

- `--convergence-n-stages 1`: looser data-scale check. Asks "did
  the last stage shift?", doesn't require K-in-a-row stability.
  The right tool for spread regimes whose peaks are stable but whose
  plateau-argmins shift subtly.
- Single-element schedule (e.g. `--schedule 200`): degenerate single-
  shot via the progressive reporting surface. Emits a frontier with
  one stage; `converged=True` by definition. "I want the progressive
  outputs but not the strictness."

**Empirical reference points** (bio-sae's
`bio-sae/runs/forge/` measurements against `facebook/esm2_t6_8M_UR50D`):

| fixture | feed | schedule | host_mauc | rec_n | retained_mauc | converged | wall time CPU |
|---|---|---|---|---|---|---|---|
| `uniref50_small/residue` | residue | [10, 50, 100] | 0.946 | 48 | 1.04 | ✓ in 3 stages | ~45 s |
| `uniref50_n5000/pooled_w1024_k64` | pooled | [200, 500] | 0.765 | 256 | 0.92 | ✗ (argmin shift n=384→n=256) | ~5 min |
| same fixture | pooled | [200] | 0.765 | 256 | 0.93 | ✓ (single-shot via progressive surface) | ~2 min |
| same fixture | pooled | [1000, 5000] | 0.765→0.795 | 256 (stable) | 0.92→0.90 (drops) | ✗ (retained_mauc variance > 0.005) | ~45 min |

The pooled regime's failure to converge under default strictness is
the expected outcome BUT for two distinct reasons at different data
scales:

- **At small protein counts ([200, 500])**: the plateau's argmin
  position shifts (n=384→n=256) because the plateau membership
  contracts as the AUC estimate tightens with more data.
- **At larger protein counts ([1000, 5000])**: the argmin position
  is stable (n=256 across both stages) but `retained_mauc` itself
  drifts because **the host's AUC grows faster than the forge's**
  as more discriminating labels surface. Writeup §3.2 measured 0.93
  at n=500 proteins; the same fixture under the wrapper drops to
  0.90 at n=5000. The "uniform tax" framing held at the writeup's
  measurement scale but widens at 10× the data.

`convergence_n_stages=1` is the documented opt-out for both shapes.
The deeper question — why the forge's discriminative power
doesn't track the host's as data scale grows — is a substrate-
specific follow-up; see
`bio-sae/docs/forge-capability-bottleneck.md` §4 for the structural-
tax-on-spread-regimes characterisation.

### Multi-encoding capability sweep — compare basis choices in one run

Single-encoding sweeps (above) commit to ONE basis encoding (e.g.
the raw row-norm slice or a partition-aware slice with
`partition_block_ids`). The multi-encoding sweep compares MULTIPLE
encodings in a single sweep call, producing per-encoding
recommendations and a cross-encoding winner pick.

```bash
sae-forge sweep-capability-progressive \
    --dataset-config bio-pooled.yaml \
    --host facebook/esm2_t6_8M_UR50D \
    --encoding raw_slice:runs/.../sae.pt \
    --encoding partition_q4:runs/.../sae_partition_q4.pt \
    --encoding partition_q8:runs/.../sae_partition_q8.pt \
    --candidate-widths 16,64,128,256,384,512,768,1024 \
    --schedule 1000,5000 \
    --output-dir runs/multi_encoding/

sae-forge recommend \
    --frontier runs/multi_encoding/frontier.jsonl \
    --target retained-mauc>=0.90
```

The output emits the picked encoding + width AND a per-encoding
ranking table:

```
recommended config: encoding=partition_q4, target_n_features_kept=128
  retained_mauc_vs_host: 0.9096

Per-encoding ranking (over 6 survivors after predicate filtering)
  Ranking: smallest target_n_features_kept WINS; ties broken by CLI --encoding flag order.
  rank  encoding             n  retained_mauc  converged
  1     partition_q8        64         0.9004        False
  2     partition_q4       128         0.9096        False
  3     raw_slice          256         0.8975        False
```

**Empirical reference points** (bio-sae's pooled fixture at n=5000;
slice-4 acceptance gate):

| encoding | rec_n | retained_mauc | factor vs raw_slice |
|---|---|---|---|
| raw_slice | n=256 | 0.8975 | 1× (baseline) |
| **partition_q4** (winner) | n=128 | 0.9096 | 2× fewer parameters |
| partition_q8 | n=64 | 0.9004 | 4× fewer parameters |

The architecture's claim is **Pareto-shift, not level-lift**:
encodings achieve comparable retained_mauc at meaningfully fewer
parameters. On this substrate, partition_q4 won by lowest
trajectory variance (the cross-encoding tiebreaker fires when no
encoding converged at default strictness — see `bio-sae/docs/forge-capability-bottleneck.md` §5.6).

**Dry-run cost projection** before committing to a multi-encoding
sweep at scale:

```bash
sae-forge sweep-capability-progressive ... --dry-run --dollars-per-gpu-hr 3.0
```

Counts cells (K encodings × N widths × S scale_boosts × T stages),
benchmarks ONE cell, projects total wall time + optional cost.
Exits 0 without running. ~instant; use before a multi-encoding
sweep at production scale.

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

### `saeforge.isf` — concise interpretability via routing

A large SAE is a **substrate, not a dictionary.** `saeforge.isf` builds a
small, faithful interpretability model by *routing* each concept to the
specialist that reads it best, instead of pruning one monolithic SAE — see
[`docs/concise-via-routing.md`](docs/concise-via-routing.md) for the thesis,
the **salience heuristic** (a rule of thumb), and the cross-fixture validation protocol.

```python
from saeforge import recipe_auc_matrix, ensemble_route, salience_headroom

A = recipe_auc_matrix([r.encode(X) for r in recipes], Y)   # (R, V) per-label AUC
route = ensemble_route(A, [r.name for r in recipes], host=0)
route["ensemble_lift"]   # > 0 ⇔ the routed ensemble beats every single recipe
route["retained"]        # ensemble mAUC / host mAUC
salience_headroom(A[0])  # 1 − host_auc — where a specialist will pay off
```

`Recipe` is anything with `name` + `encode(X) -> (N, d)`: a raw host, a
supervised specialist, a Polygram-tier slice. Validated on bio-sae (6/6
synthetic motifs, +0.105 tier lift) and econ-sae (ensemble 0.812, lift
concentrated on the low-salience regime/conjunctive tiers).

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

> **⚠️ SAE hook point vs `--layer`: mind the `resid_pre`/`resid_post`
> off-by-one.** sae-forge and Polygram both interpret `layer=N` as the
> **input** to transformer block N — i.e. `blocks.N.hook_resid_pre`
> (Polygram registers a `forward_pre_hook` on `layers[N]`; sae-forge's
> calibration reads `hidden_states[N]`, the same point). Match it to where
> your SAE was *trained*:
>
> - SAE trained on **`blocks.N.hook_resid_pre`** → use `--layer N`.
> - SAE trained on **`blocks.N.hook_resid_post`** → use `--layer N+1` (a
>   block's `resid_post` *is* the next block's `resid_pre`).
>
> Getting this wrong is **silent** — the forge still runs, but the basis is
> measured a different block from the SAE's activations and faithfulness
> degrades (empirically ~2× worse KL on a 24-layer host). Published SAEs
> vary: `jbloom/GPT2-Small-SAEs-Reformatted` is `resid_pre` (use
> `--layer N`), while `chanind/sae-qwen2-0.5b-res` is `resid_post` (use
> `--layer N+1`). Check the SAE's `cfg.json` `hook_name`. sae-forge emits a
> `UserWarning` (via `saeforge.utils.sae_layer.check_sae_layer_alignment`)
> when it can read the hook point and the layer looks off, but it never
> auto-corrects — an intentional probe of a different layer stays valid.

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
