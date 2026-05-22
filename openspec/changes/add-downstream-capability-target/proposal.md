# Downstream Capability Target — per-dataset forge-vs-host capability eval

Add a `DownstreamCapabilityTarget` (a new `FaithfulnessTarget`) and a `sweep_pareto_capability` wrapper that score the forge against a host's downstream task — a labeled dataset, a task encoder, and the AUC of the encoder's latents vs the labels — instead of against the host's residual stream numerically.

## Why

The bundled faithfulness targets (`KLTarget`, `CosineTarget`, `TokenCosineTarget`) answer the question *"are the forged hidden states numerically close to the host's?"* That question goes negative under any usable compression — bio-sae's first ESM-2 forge smoke ran cosine = **−0.535 at n_features=16** (5 % of d_model=320), cosine = **+0.095 at n=256** (80 % of d_model). By cosine alone, the forge is useless.

But cosine is the wrong question for downstream-task users. A bio-sae user wants to know: *"does the forged ESM-2 retain the biological features bio-sae's SAE has already learned to discriminate?"* When bio-sae actually measured that — same forge, different metric, decode the forged residuals through the bio-sae SAE and score per-feature AUC against GO/Pfam/EC labels — the picture inverted completely. See [`bio-sae/docs/forge-capability-bottleneck.md`](https://github.com/jascal/bio-sae/blob/main/docs/forge-capability-bottleneck.md) for the full writeup; key data points:

| substrate | forge n_features | retained mAUC | retained cov95 | cosine said |
|---|---|---|---|---|
| categorical residue SAE (W_dec norms concentrated) | 16 (5 %) | **103.2 %** | 90.0 % | "useless" (cosine −0.60) |
| categorical residue SAE | 192 (60 %) | 91.5 % | **0.0 %** | (cliff collapse) |
| hierarchical pooled SAE (W_dec norms spread) | 512 (160 %, over-complete) | **93.2 %** | 16.2 % | best cosine point but plateau |
| hierarchical pooled SAE | 16 | 86.4 % | 7.4 % | (mostly preserved) |

Two distinct regimes, two different bottleneck profiles — and **both** are dramatically misled by the cosine probe:

- **Concentrated substrate.** Features cluster near AUC=1.0 on host. Forge introduces a small but monotonically-growing AUC drop; cov95 cliff-collapses 90 % → 40 % → 0 % between n=128–192 because features cross the AUC=0.95 bar in a phase transition. The smallest basis (n=16) is *optimal* — fewer features = less low-norm noise = sharper signal.
- **Spread substrate.** Features spread across AUC ∈ [0.5, 0.95] on host. Forge introduces a uniform ~5–8 % absolute mAUC tax — no cliff, but biology never fully recovers. Optimal width is mid-rank (n=512); above and below degrade.

A cosine-driven Pareto sweep picks the wrong forge for both substrates. The right metric is the downstream task itself.

## What

### 1. `DownstreamCapabilityTarget` — new built-in `FaithfulnessTarget`

A `FaithfulnessTarget` that scores per-feature × per-label AUC through a downstream task encoder:

```python
from saeforge.eval.targets import DownstreamCapabilityTarget

target = DownstreamCapabilityTarget(
    encoder=biosae_sae,           # any callable d_model -> latent_width
    labels=Y_protein,             # (N_items, V) binary GT label matrix
    aggregator="pool_then_encode", # or "encode_then_pool" or callable
    min_prevalence=10,            # drop singleton-inflated features (optional)
    decode_via_basis=True,        # forged → decode via basis.W_dec → encoder
)

result = pipeline.run(output_dir)  # uses target as the faithfulness signal
```

Pipeline inside `score()`:

```
sequences -> forged ESM-2 -> forged hidden states (n_features) ->
decode via basis.W_dec -> (d_model) ->
encoder -> task latents (latent_width) ->
aggregator (pool then encode | encode then pool) ->
AUC vs labels (per feature × per label, best-over-features per label, mean)
```

Mirrors `GroundTruthTarget`'s `(score, perplexity_analog)` return convention with `better_when="higher"`.

### 2. `sweep_pareto_capability` — Pareto in retained-AUC space

A wrapper over the existing `sweep_pareto` that uses `DownstreamCapabilityTarget` as the metric and surfaces retained-AUC fields on `ParetoFrontierRow`:

```python
from saeforge import sweep_pareto_capability

frontier = sweep_pareto_capability(
    sae_checkpoint=Path("sae.safetensors"),
    host_model_id="facebook/esm2_t6_8M_UR50D",
    dataset=CapabilityDataset(
        sequences=protein_seqs,
        labels=Y_protein,
        encoder=biosae_sae,
        tokenizer_id="facebook/esm2_t6_8M_UR50D",
        aggregator="pool_then_encode",
        min_prevalence=10,
    ),
    widths=[16, 64, 128, 256, 512, 1024],
    encodings=["Rung5(n_amp_qubits=2)"],
    scale_boosts=[1.0, "auto"],
    output_dir=Path("runs/capability_sweep/"),
)
```

Each row carries the existing forge-quality fields **plus**:

- `host_baseline_mauc`, `host_baseline_cov95`
- `forge_mauc`, `forge_cov95`
- `retained_mauc_vs_host`, `retained_cov95_vs_host`
- `gap_median`, `gap_p25`, `gap_p75`, `gap_p95` (per-feature host - forge AUC)
- `n_features_gap_above_0_1` (count of features with > 0.1 AUC loss)
- `n_features_negative_gap` (count where forge *outperforms* host — the bio-sae "denoise" pattern)

### 3. `CapabilityDataset` dataclass + simple "bio-sae://" URI loader (optional v1.1)

```python
@dataclass(frozen=True)
class CapabilityDataset:
    sequences: list[str]
    labels: np.ndarray              # (N_items, V), binary
    encoder: Callable               # d_model -> latent_width
    tokenizer_id: str
    aggregator: str | Callable = "pool_then_encode"
    min_prevalence: int = 0
    decode_via_basis: bool = True
```

`CapabilityDataset.from_bio_sae(run_dir, bundle_path, sequences_path, feed="pooled")` constructs the dataset from a bio-sae bundle. The contract is documented; sm-sae and econ-sae provide their own `from_sm_sae` / `from_econ_sae` constructors in their respective fixture repos.

A URI scheme (`bio-sae://uniref50_n5000_pooled`) is **out of scope for v1** — explicit constructors are clearer.

### 4. CLI

```bash
sae-forge sweep capability \
    --sae path/to/sae.safetensors \
    --host facebook/esm2_t6_8M_UR50D \
    --dataset-config bio-sae-dataset.yaml \
    --widths 16,64,128,256,512,1024 \
    --scale-boosts 1.0,auto \
    --encodings Rung5:n_amp_qubits=2 \
    --output runs/capability_sweep/
```

with a follow-up `sae-forge recommend --frontier frontier.jsonl --target retained-mauc>=0.95` that picks the smallest-parameter forge meeting a retention target.

### 5. Example: bio-sae's n=5000 pooled SAE (concrete)

Given bio-sae's bundled fixture, the end-to-end flow:

```yaml
# bio-sae-dataset.yaml
encoder_checkpoint: runs/uniref50_n5000/pooled_w1024_k64/sae.pt
sequences_path:    data/uniref50_sample__n5000_seed0.parquet
labels_path:       data/bio_bundle_uniref50.safetensors
labels_key:        labels_protein_Y
tokenizer_id:      facebook/esm2_t6_8M_UR50D
aggregator:        pool_then_encode
min_prevalence:    10
sae_variant:       topk
sae_k:             64
```

```bash
sae-forge sweep capability \
    --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \
    --host facebook/esm2_t6_8M_UR50D \
    --dataset-config bio-sae-dataset.yaml \
    --widths 16,64,128,256,512,1024 \
    --scale-boosts 1.0,auto \
    --encodings Rung5:n_amp_qubits=2 \
    --output runs/capability_sweep/uniref50_n5000_pooled/

sae-forge recommend \
    --frontier runs/capability_sweep/uniref50_n5000_pooled/frontier.jsonl \
    --target retained-mauc>=0.90
```

Expected `recommend` output (per the falsifiable acceptance gate
below):

```
target_n_features_kept: 512
encoding_label:         Rung5(n_amp_qubits=2)
scale_boost:            auto
host_baseline_mauc:     0.857
forge_mauc:             0.799
retained_mauc_vs_host:  0.932
forge_cov95:            0.028
retained_cov95_vs_host: 0.162
gap_median:             0.052
gap_p95:                0.159
n_params_forged:        ~2.0M
```

For the concentrated-substrate fixture
(`runs/uniref50_small/residue`), the same sweep should pick n=16
with retained_mauc ≈ 1.03 — both predictions are pre-measured by
bio-sae's manual scripts and pinned in §"Falsifiable acceptance gate".

## How (sketch)

- `saeforge/eval/targets/downstream_capability.py` — new file. Implements the `FaithfulnessTarget` protocol. Lazy-imports torch via `require_extra`. Calls the encoder on decoded forged states; computes per-feature × per-label AUC via the Mann-Whitney rank-sum identity (same vectorised matmul as `GroundTruthTarget` and `biosae.sae.evaluation`).
- `saeforge/datasets/capability.py` — new file. `CapabilityDataset` dataclass + `from_bio_sae(...)` constructor (bundle parsing pulled from bio-sae's `forge_capability_eval.py`).
- `saeforge/sweep.py` — extend `ParetoFrontierRow` with the new optional fields (all default `None`; serialisation back-compat with v0.7's row schema). Add `sweep_pareto_capability(...)` as a thin wrapper that constructs the target + drives the existing `sweep_pareto`.
- `saeforge/cli.py` — `sae-forge sweep capability` and `sae-forge recommend` subcommands.

## Falsifiable acceptance gate

The sweep on bio-sae's two fixtures recommends the **substrate-correct** configs that bio-sae's manual eval already identified:

| fixture | predicted optimal | falsified if … |
|---|---|---|
| `runs/uniref50_small/residue` (concentrated) | n=16, retained_mauc ≥ 1.00 | sweep recommends n ≥ 128 OR retained_mauc < 0.95 |
| `runs/uniref50_n5000/pooled_w1024_k64` (spread) | n=512, retained_mauc ≈ 0.93 | sweep recommends n=16 OR retained_mauc > 0.99 (would mean we lost the tax) |

Both predictions are tight to bio-sae's pre-existing measurements (`bio-sae/runs/forge/capability_eval_smoke/`, `bio-sae/runs/forge/capability_pooled_n500*/`); a clean re-run via this new target must reproduce them within 1 % retained-mAUC.

## What this does NOT solve

- The **fundamental forge tax** on spread substrates (~9 % uniform mAUC drop, robust to scale_boost / width / pool-order) is structural — caused by layer-norm non-commutation with non-orthonormal projection + TopK rank-shuffling in the downstream encoder. This proposal makes the tax *legible* per dataset; *eliminating* the tax requires separate algebra work (orthonormalise the basis, smoother encoder, RMSNorm-vs-LN substitution, …) and is out of scope.
- This is a **capability-aware metric**, not a capability-aware *forge algorithm*. The forge itself doesn't see the labels. Future work could feed labels into the projection (a "supervised forge"), but that's a different proposal.

## Related

- bio-sae writeup (motivation + data): `bio-sae/docs/forge-capability-bottleneck.md`
- bio-sae prototype (scripts): `bio-sae/scripts/forge_capability_eval.py`, `forge_collapse_diagnostic.py`, `forge_pool_after_encode.py`
- existing primitives this composes: `saeforge.ForgePipeline`, `saeforge.sweep_pareto`, `saeforge.eval.targets.GroundTruthTarget` (the basis-coord cousin of this proposal; `DownstreamCapabilityTarget` adds the encoder + decode step)
- joint substrate-feedback precedent: `bio-sae/docs/polygram-feedback-2026-05-20.md` (three repos surfacing one primitive's wrong-question failure)
