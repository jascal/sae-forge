# Audio architecture forge: Whisper encoder

> **Status:** v0.4 introduces encoder-only audio support behind
> `family == "whisper_encoder"`. The decoder forge is tracked as a
> follow-up change (`forge-whisper-decoder`) and not started.
> Every existing LM forge flow (gpt2 / llama / gemma2 / qwen2 /
> qwen3) is byte-equivalent to v0.3 — the audio path is purely
> additive.

This document covers how to forge a Whisper-encoder native model
from a polygram-compressed Whisper SAE. See `docs/algorithm.md` §5
for the projection-algebra foundation; this is the audio-architecture
specialization.

## When to use it

Use the Whisper-encoder forge when:

- Your SAE was trained on Whisper encoder residuals (e.g.
  `cherrvak/topkautoencoder_baseline`, `cherrvak/large_v1_block_16_audioset_topk_16`,
  or one of the other audio SAEs in
  [polygram's five-SAE panel](https://github.com/<owner>/polygram/blob/main/docs/research/sae-geometry-regimes.md)).
- You want a forged model with the host encoder's input/output
  contract: mel-spectrogram in `(batch, 80, n_frames)` → per-frame
  encoder states in `(batch, n_frames // 2, n_features)`.
- The downstream consumer is an encoder-state probe, classifier, or
  feature extractor — not a transcription system. (Whisper decoder
  forge is a separate change.)

Do not use it when the SAE was trained on the decoder side, on
cross-attention residuals, or on raw audio waveforms — none of
those paths are supported in v0.4.

## Recommended polygram-side profile

Compress your Whisper SAE under polygram's `"uniform-sphere"`
geometric profile, not the default `"clustered"`. Polygram 0.2.0
introduced the named-profile concept after a five-SAE smoke probe
established that audio SAEs sit on a near-uniform-sphere projection
geometry where the v0.1 Pearson `tier_preservation` and k=2 binary
β-spread give selection-driven noise rather than fidelity signal.
The full rationale is in
[polygram's `sae-geometry-regimes.md`](https://github.com/<owner>/polygram/blob/main/docs/research/sae-geometry-regimes.md);
the short version:

| Predictor | Threshold | Whisper-tiny enc.b2 | Whisper-large-v1 enc.b16 |
|---|---|---|---|
| `n_features` | ≥ ~16K | 6,144 (under) | 20,480 (over) |
| `d_model` | ≥ ~1K | 384 (under) | 1,280 (over) |
| `width × d_model` | ≥ regime threshold | borderline | inside the regime |

Whisper-large-v1 enc.b16 lands solidly inside the uniform-sphere
regime; Whisper-tiny enc.b2 sits at the edge but shares the same
cosine signature. Both are calibrated for the `uniform-sphere`
profile in polygram 0.2.0+.

```python
from polygram import Compressor
from polygram.sae_import import from_sae_lens

# Recommended audio-SAE setting:
dictionary, report = from_sae_lens(
    records,
    feature_ids=[...],
    profile="uniform-sphere",
)
```

sae-forge does not consume the profile name — it's a polygram-side
compression knob, not a forge-side configuration. The forge reads
only the on-disk `W_dec` matrix + compression report; the
`profile` field round-trips in the report's metadata for audit
purposes but doesn't drive any forge behavior. The recommendation
is purely upstream: compress audio SAEs with `uniform-sphere`,
then forge them with sae-forge as you would any other compressed
SAE.

## Pipeline shape

The forge pipeline for Whisper encoder differs from the LM path in
three places, all hidden behind the family dispatch:

1. **Adapter walk** — `WhisperEncoderAdapter` projects every weight
   whose input or output touches the residual stream; the conv stem
   and positional embeddings are frozen-copied (counted as ε_conv
   per `algorithm.md` §5). The walk emits 18 keys per block plus a
   top-level `basis_encode` buffer that carries the d → f projection
   matrix for the conv-stem → first-block boundary.
2. **Eval signal** — `evaluate_faithfulness` dispatches to
   `saeforge.audio_eval.cosine_faithfulness` instead of
   `_kl_from_input_ids`. Faithfulness is per-frame cosine
   similarity in basis space (`[0, 1]`, higher = better);
   `min_faithfulness` is the minimum cosine threshold (positive
   convention, not the LM-path's KL-negation convention).
3. **Pre-capture fast path** — set
   `ForgePipeline.eval_encoder_states` to pre-captured host
   encoder states once outside the FSM and the action skips the
   host forward inside every refine step. The audio-side analog of
   pre-tokenised `eval_input_ids`.

## ε_conv and the d → f bridge

Whisper's conv1 and conv2 (the mel stem) and `embed_positions`
(sinusoidal positional embeddings stored as a learned tensor) are
not residual-aligned — their kernels operate on a 1D spatial
structure (the time axis), not the residual stream. Polygram's
projection algebra doesn't define a projection for these; the forge
copies them bit-for-bit from the host. This is the **ε_conv** term
in the algorithm doc.

The forged encoder's transformer blocks still expect inputs in the
SAE basis (width `n_features`), so the forge applies an explicit
d → f bridge at the conv-stem → first-block boundary:

```
mel (B, 80, 3000)
    → conv1 (frozen-copied)    → (B, d_model, 3000)
    → conv2 (frozen-copied)    → (B, d_model, 1500)
    → permute + pos embed      → (B, 1500, d_model)
    → @ basis_encode buffer    → (B, 1500, n_features)
    → N transformer blocks     → (B, 1500, n_features)
    → final layer_norm         → (B, 1500, n_features)
```

The `basis_encode` buffer is `projector.basis.pseudoinverse() *
scale_boost` — the matrix form of `SubspaceProjector.encode`. It's
registered as a non-parameter buffer (state-dict-resident, but not
visible to `named_parameters()`) so save/load preserves it and the
no-randomly-initialised-weights invariant treats it correctly.

## Example: end-to-end synthetic forge

`examples/forge_whisper_synthetic.py` runs the full pipeline on a
tiny synthetic Whisper without any HF download or `.wav` file:

```sh
python examples/forge_whisper_synthetic.py /tmp/forged_whisper \
    --d-model 64 --encoder-layers 2 --n-features 32
```

It builds a hand-rolled `WhisperModel`, projects through a random
`FeatureBasis`, assembles the forged encoder, and computes cosine
faithfulness on a synthetic sine-sweep mel from
`saeforge.audio_data.synthetic_mel_features`.

On random weights and a random basis, cosine ≈ 0.0 is expected —
ε from LayerNorm projection compounds across blocks and the
forged output decorrelates from the basis-projected host output.
The example demonstrates the pipeline shape, not eventual eval
quality. A real polygram-compressed Whisper SAE — where the basis
is trained to align with the host's natural feature directions —
gives a much higher score.

## Limitations (v0.4)

- **Encoder only.** Whisper decoder forge is tracked as
  `forge-whisper-decoder` and not started.
- **No real audio loaders.** Use pre-extracted mel features. The
  `[audio]` extra (`librosa >= 0.10`) is planned for a future
  change; for now, `saeforge.audio_data.synthetic_mel_features`
  covers test + smoke needs and any production user can pre-extract
  features via `WhisperFeatureExtractor` outside sae-forge.
- **8-feature rung-1 MPS cap is upstream.** Polygram's rung-1 MPS
  encoding caps a `Dictionary` at 8 features. A real TopK Whisper
  SAE (k=16–50) is forged on a hand-selected 8-feature subset; the
  forge runs end-to-end at any `n_features`, but the polygram side
  enforces the cap. Multi-Dictionary stitching is tracked
  separately on the polygram side.
- **`uniform-sphere` is provisional.** Polygram 0.2.0 ships the
  profile as "provisional pending behavioural validation" — the
  Phase-2 behavioural probe on Whisper SAEs hasn't run. The forge
  works against any `uniform-sphere`-compressed basis; the
  *meaningfulness* of the resulting cosine score on real audio is
  pending that validation.
