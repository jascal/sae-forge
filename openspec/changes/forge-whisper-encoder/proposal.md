## Why

sae-forge currently forges *causal language models* — GPT-2, Llama,
Gemma-2. The shipped pipeline assumes:

1. Token IDs as input (an embedding table is the entry point).
2. A vocabulary-shaped output head (`lm_head` produces logits over
   `vocab_size`).
3. Per-token KL faithfulness as the eval signal
   (`evaluate_faithfulness` calls `_kl_from_input_ids`).

Polygram 0.2.0 (merged in PR #12) shipped the `uniform-sphere`
geometric profile, validated against a five-SAE panel that includes
**Whisper** SAEs (audio encoder). The compressed audio SAEs *load*
into sae-forge today via `FeatureBasis.from_polygram_checkpoint` —
the basis loader is architecture-agnostic — but there is no path to
actually forge a model from one. Three gaps stand in the way:

1. **No audio adapter.** `saeforge/adapters/{gpt2,llama,gemma2}.py`
   walk decoder-only LM weights. Whisper's encoder uses a different
   shape — Conv1d stem, sinusoidal positional embeddings, post-norm
   blocks with self-attention only (no cross-attention in the
   encoder), and a final layer norm with no `lm_head`.
2. **`NativeModelConfig` is LM-shaped.** The `vocab_size` field is
   required and `_SUPPORTED_FAMILIES = {"gpt2", "llama", "gemma2"}`.
3. **`evaluate_faithfulness` assumes token IDs.** Audio encoders
   produce per-frame hidden states, not logits over a vocabulary —
   KL-divergence is the wrong signal.

This change adds Whisper-encoder support: an adapter, an encoder-shaped
native module family, and a parallel cosine-based eval. **Decoder is
deliberately out of scope** — full Whisper (encoder + cross-attention
decoder + real audio data loaders) deserves its own change. Encoder-only
is the minimum-viable path to validate that sae-forge can forge a
non-causal-LM architecture, and matches the natural unit of an SAE
trained on encoder residuals (which is what the polygram-side Whisper
SAEs target).

## What Changes

### Scope

Forge a Whisper-encoder native model from a polygram-compressed SAE
attached to one of the encoder's residual streams. The forged model
has the same input/output contract as the host encoder
(mel-spectrogram → per-frame hidden states), with all attention/MLP
weights projected into the SAE feature basis.

### New artifacts

- **`saeforge/adapters/whisper.py`** — `WhisperEncoderAdapter`. Walks
  `WhisperEncoder.layers[*]` (self-attn `q_proj`/`k_proj`/`v_proj`/
  `out_proj` and MLP `fc1`/`fc2`), `embed_positions`, `layer_norm`.
  Skips the `conv1`/`conv2` mel stem entirely (counted as ε_conv per
  `docs/algorithm.md` §5; tracked as a follow-up). Registers itself
  at import time for `transformers.WhisperForConditionalGeneration`
  (extracts `.model.encoder` internally) and `transformers.WhisperModel`.
- **`saeforge/model.py` extensions** — `family = "whisper_encoder"`
  added to `_SUPPORTED_FAMILIES`. New `output_kind: str = "logits" |
  "encoder_states"` field on `NativeModelConfig`; defaults to
  `"logits"` for byte-identical LM behavior. `vocab_size` becomes
  optional (allowed when `output_kind == "encoder_states"`).
- **`ForgedWhisperEncoder`** native module — torch `nn.Module`
  matching Whisper's encoder layout: 2-conv mel stem (frozen,
  copied from host), sinusoidal positional embeddings, N transformer
  blocks with pre-LN self-attention + GELU MLP, final `layer_norm`.
  No decoder, no cross-attention, no `lm_head`.
- **`saeforge/audio_eval.py`** — `cosine_faithfulness(forged_states,
  host_states)` returning a `[0, 1]` scalar. Per-frame cosine
  similarity averaged over the time axis. Mirrors the `_kl_from_input_ids`
  signature so the FSM eval action can dispatch by family.
- **`saeforge/audio_data.py`** — minimal mel-spectrogram fixture loader
  for tests. Pure-numpy synthesis (sine sweep + noise; no real audio
  files), tokenized via `WhisperFeatureExtractor` lazy-imported
  through the `[audio]` extra. Tests stub this with a fixed-shape
  random tensor.
- **`tests/test_whisper_encoder_adapter.py`** — walker shape audit on
  a tiny synthetic Whisper, four-norm/single-norm contract, no-randomly-
  initialized-weights invariant, GQA negative test (Whisper is MHA).
- **`tests/test_audio_eval.py`** — `cosine_faithfulness` returns
  `1.0` for identical states, `0.0` for orthogonal, monotone
  decreasing as noise is added.
- **`examples/forge_whisper_synthetic.py`** — end-to-end synthetic
  Whisper forge with no HF download or real audio.

### Modified artifacts

- **`saeforge/actions/__init__.py`** — `evaluate_faithfulness`
  dispatches on `ctx["_native_model"].config.family`. LM families
  go through `_kl_from_input_ids` unchanged. `whisper_encoder` goes
  through `cosine_faithfulness` against pre-captured host encoder
  states (`ctx["_eval_encoder_states"]`).
- **`saeforge/forge.py`** — `_build_fsm_ctx` accepts an
  `eval_encoder_states` kwarg parallel to `eval_input_ids`.
  `ForgePipeline` gains an `eval_audio_features: torch.Tensor | None`
  field for the audio-side input.
- **`saeforge/adapters/__init__.py`** — registers the new
  `WhisperEncoderAdapter` for both `WhisperForConditionalGeneration`
  (extracts `.model.encoder` in `walk`) and `WhisperModel` (uses
  `.encoder` directly).

### CLI surface

- **`sae-forge inspect`** is unchanged — already works on audio SAEs
  today (verified during PR #12 testing).
- **`sae-forge forge`** gains a `--audio-features-path` flag that
  loads pre-extracted mel features (`.pt` file) for the eval pass.
  Tokenizer-driven flags (`--eval-prompts`) are mutually exclusive
  with `--audio-features-path`; selecting one auto-selects the
  matching family.

### Optional dependency

- **New `[audio]` extra** pinning
  `transformers>=4.46` (already present transitively) and
  `librosa>=0.10` for the optional real-audio path. The pure-numpy
  synthetic fixture path does not require it. Mirrors the
  `[recipe]` extra contract for `datasets`.

## Capabilities

### Modified Capabilities

- **`architecture-adapters`** — extends the registry with a fourth
  family (`whisper_encoder`). The dispatcher contract is unchanged;
  only the registered set grows. Registry tests are extended to
  cover the new class registration.

### New Capabilities

- **`whisper-encoder-eval`** — the cosine-similarity faithfulness
  evaluator that `evaluate_faithfulness` dispatches to when
  `family == "whisper_encoder"`. Distinct from the existing per-token
  KL path because the signals are not commensurable.

### Out of Scope (deferred)

- **Whisper decoder** — cross-attention to encoder states, vocab
  output, beam search. Tracked as a separate follow-up change
  `forge-whisper-decoder`. Forging the encoder alone is useful in
  its own right (encoder-state probes, downstream classifiers).
- **Real audio data loaders** — mel-spectrogram extraction from
  `.wav`/`.flac`. The synthetic fixture path covers test needs;
  CLI users supplying pre-extracted features cover production needs.
- **Conv1d mel stem projection** — Whisper's `conv1` (80 mel bins
  → `d_model`) and `conv2` (downsample to 1500 frames) are *outside*
  the SAE's residual stream. The forged encoder copies them
  unchanged from the host. Their projection is non-trivial (1D
  spatial structure) and is tracked as ε_conv per
  `docs/algorithm.md` §5.
- **Multilingual / multitask tokens** — `<|startoftranscript|>` etc.
  are decoder concerns; encoder is task-agnostic.

## Impact

- **No breaking changes.** Existing LM forge flows are untouched —
  defaults preserve byte-identical v0.3.0 behavior. The `output_kind`
  field defaults to `"logits"`, `_SUPPORTED_FAMILIES` is extended
  not narrowed, and `vocab_size` only becomes optional behind
  `output_kind == "encoder_states"`.
- **New files only** in `saeforge/adapters/whisper.py`,
  `saeforge/audio_eval.py`, `saeforge/audio_data.py`,
  `tests/test_whisper_encoder_adapter.py`,
  `tests/test_audio_eval.py`,
  `examples/forge_whisper_synthetic.py`.
- **`pyproject.toml`** gains the `[audio]` extra; the `[all]` extra
  includes it.
- **`AGENTS.md`** gains an "Audio architecture support" subsection
  documenting the encoder-only scope and the LM-vs-encoder eval
  dispatch.
- **`README.md`** Status section gets a `forge-whisper-encoder`
  bullet alongside the other landed openspec changes.
- **Test surface** grows by ~25 tests (12 in
  `test_whisper_encoder_adapter.py`, 8 in `test_audio_eval.py`,
  3 fixture-related in `tests/conftest.py`, plus the existing
  `test_examples_smoke.py` gains a Whisper smoke).

## Sequencing

This change ships as a single PR. The natural follow-up
(`forge-whisper-decoder`) is filed but not started until this lands
and a real Whisper-encoder forge has been validated end-to-end on
a polygram-compressed Whisper SAE.
