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
pip install -e ".[dev,torch,polygram]"   # editable install with test, torch, and polygram bridge
pytest                                    # run the suite
```

Optional extras: `[plot]` (matplotlib), `[notebook]` (jupyter +
matplotlib), `[torch]` (torch + transformers — required for `NativeModel`
construction, `SubspaceProjector` projection from a real source model, and
fine-tuning), `[polygram]` (the upstream compressed-SAE producer; pinned
at `>=0.4` for scale-aware merged norms).

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
- **Required Polygram version**: `>=0.4`, the first release with
  scale-aware merged norms in `CompressionReport.scale_compression_ratio`
  and per-cluster `merged_norm`.
- **What sae-forge does not do**: it does not run validation, does not
  pick clusters, does not zero or merge — those are Polygram's job. It
  consumes the artifact and projects.

If you want to build a custom compression upstream (a different rep
selector, a non-Polygram SAE format), hand-roll a dict matching
`FeatureBasis`'s fields and call `FeatureBasis(**fields)` directly — the
loader is one entry point among several.

## Development

```bash
pip install -e ".[dev,torch,polygram]"
pytest
ruff check saeforge tests examples
```

## License

Apache-2.0.
