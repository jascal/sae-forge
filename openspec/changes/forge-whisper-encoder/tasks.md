## 1. NativeModelConfig extension

- [x] 1.1 Add `output_kind: str = "logits"` field to `NativeModelConfig` (saeforge/model.py); default preserves byte-equivalent v0.3 LM behavior
- [x] 1.2 Make `vocab_size` default to `0` (was required); the `__post_init__` invariant `vocab_size > 0` is gated by `output_kind == "logits"`
- [x] 1.3 Add `whisper_encoder` to `_SUPPORTED_FAMILIES`; assert that `family == "whisper_encoder"` implies `output_kind == "encoder_states"` and `vocab_size == 0`
- [x] 1.4 Update `to_dict` / `from_dict` to round-trip the new fields — handled automatically by the dataclass `asdict` + `cls(**payload)` pattern; pre-change configs without `output_kind` deserialise with the default `"logits"`
- [x] 1.5 Add `_build_torch_module` dispatch branch for `whisper_encoder` → `saeforge.adapters.whisper.build_whisper_encoder_module(config)`
- [x] 1.6 Tests: NativeModelConfig invalid-combination matrix (vocab=0+logits, vocab>0+encoder_states, whisper_encoder+vocab>0) — landed as `TestNativeModelConfigOutputKind` in `tests/test_architecture_adapters.py`

## 2. WhisperEncoderAdapter

- [x] 2.1 New module `saeforge/adapters/whisper.py` exposing `WhisperEncoderAdapter` (subclass of `ArchitectureAdapter`)
- [x] 2.2 `family = "whisper_encoder"`; `walk(host, projector)` produces a dict of every encoder weight (per design.md §"Native module shape" projected list)
- [x] 2.3 `_extract_encoder(host)` handles both `WhisperForConditionalGeneration` (`.model.encoder`) and `WhisperModel` (`.encoder`)
- [x] 2.4 Frozen-copy path for `conv1`, `conv2`, `embed_positions` (no projector call); module docstring carries the ε_conv accounting and the known limitation for real-audio use ("the unprojected conv stem feeds non-basis-aligned features into the first encoder block; bounded but not zero error in the forged encoder's outputs vs the host")
- [x] 2.5 `build_native_config(host, n_features)` reads `host.config.encoder_layers / encoder_attention_heads / d_model / encoder_ffn_dim`; produces `NativeModelConfig(family="whisper_encoder", output_kind="encoder_states", vocab_size=0, ...)`
- [x] 2.6 `native_module_class()` returns `ForgedWhisperEncoder` (lazy torch import)
- [x] 2.7 `grad_checkpoint_targets(module)` returns `(module.layers, module.embed_positions.weight)` so the recipe path works on Whisper
- [x] 2.8 Register adapter at module import time for both `WhisperForConditionalGeneration` and `WhisperModel`
- [x] 2.9 Tests in `tests/test_whisper_encoder_adapter.py`: walker shape audit (every key + shape), MHA invariant (encoder is not GQA — `n_kv_heads == num_heads`), frozen-copy check (conv1/conv2/embed_positions match host bit-for-bit), no-randomly-initialized-weights invariant, registry dispatch test (both Whisper classes resolve to the same adapter)

## 3. ForgedWhisperEncoder torch module

NOTE: §3.6 (`from_projected_weights`) and §3.7 (`save_pretrained` /
`load_pretrained`) are satisfied by `NativeModel.from_projected_weights`
and `NativeModel.{save,load}_pretrained` respectively — that is the
existing GPT-2 / Llama precedent (those families do not ship per-class
overrides either). The d → f bridge between the frozen-copied conv
stem and the f-wide transformer blocks is carried by a non-parameter
`basis_encode` buffer materialised from
`projector.basis.pseudoinverse() * scale_boost`; the adapter walk
emits it as a top-level key and the forged module registers it as a
state-dict-resident buffer so save/load round-trips it.

- [x] 3.1 New `nn.Module` `ForgedWhisperEncoder` in `saeforge/adapters/whisper.py` (or `saeforge/whisper_encoder.py` — pick by repo convention; the Llama family lives in adapters/, follow that)
- [x] 3.2 Block layout: `pre_layernorm → self_attn → residual → pre_layernorm → mlp → residual` (matches HF Whisper post-LN-ish pre-LN-via-LN-before-each-sublayer)
- [x] 3.3 Self-attn: separate `q_proj` / `k_proj` / `v_proj` / `out_proj` Linear layers (Whisper uses Linear, not GPT-2's fused Conv1d); MHA only (no GQA)
- [x] 3.4 MLP: `fc1 → gelu → fc2`; activation is GELU, not SiLU (Whisper diverges from Llama here)
- [x] 3.5 `forward(input_features)` matches HF's encoder forward signature: `(batch, n_mels, n_frames)` → `(batch, n_frames // 2, hidden_size)` after the conv stem
- [x] 3.6 `from_projected_weights(config, weights_dict)` classmethod loads the projected state dict via `load_state_dict(strict=True)` — handled via `NativeModel.from_projected_weights` (GPT-2 / Llama precedent — no per-class override)
- [x] 3.7 `save_pretrained(dir)` / `load_pretrained(dir)` mirror the existing GPT-2/Llama save patterns; persist a `config.json` derived from `config.to_dict()` and a `model.safetensors` from `state_dict()` — handled via `NativeModel.{save,load}_pretrained` (same precedent)
- [x] 3.8 Tests in `tests/test_whisper_encoder_module.py`: forward-shape sanity, save/load round-trip, conv-stem-frozen invariant (forging does not change `conv1.weight` etc.), `basis_encode` buffer invariants

## 4. Audio eval

- [x] 4.1 New module `saeforge/audio_eval.py` exposing `cosine_faithfulness(forged, host, audio_features, *, device="cpu") -> float`
- [x] 4.2 Computes per-frame cosine similarity between forged encoder output and pre-captured host encoder output; averages across the batch and time axis. Host states are projected through the forged module's `basis_encode` buffer so both vectors live in the same SAE-basis space.
- [x] 4.3 Returns a Python `float` in `[0.0, 1.0]` — negative cosines clamp to 0.0; the upper end is clipped at 1.0 to absorb fp32 noise. Zero-norm forged or host states map to 0.0 rather than NaN.
- [x] 4.4 Tests in `tests/test_audio_eval.py`: identical states → 1.0; zero-forged / zero-host / anti-correlated → 0.0; monotone decreasing under additive Gaussian noise; batch axis correctly averaged as the mean of per-example cosines; fp16 + bf16 input round-trip; plus an end-to-end smoke through the real `ForgedWhisperEncoder` + `WhisperModel` host
- [x] 4.5 Document the metric choice in a module-level docstring (cribbed from design.md §"Eval dispatch")

## 5. evaluate_faithfulness dispatch

- [ ] 5.1 In `saeforge/actions/__init__.py`, modify `evaluate_faithfulness` to dispatch on `forged.config.family`
- [ ] 5.2 LM families (`"gpt2" | "llama" | "gemma2"`) use the existing `_kl_from_input_ids` path verbatim — no behavior change for v0.3 LM forges
- [ ] 5.3 `"whisper_encoder"` reads `ctx["_eval_audio_features"]` and `ctx["_eval_encoder_states"]`, calls `cosine_faithfulness`, writes the result to the existing `faithfulness` ctx field
- [ ] 5.4 The `should_continue` predicate logic is unchanged: `min_faithfulness` semantic is reinterpreted as "minimum cosine similarity" for the encoder family — document this in the action's docstring
- [ ] 5.5 Tests in `tests/test_evaluate_faithfulness_dispatch.py`: LM family routes to KL path (mock `_kl_from_input_ids`, assert called); whisper_encoder routes to cosine (mock `cosine_faithfulness`, assert called); unknown family raises a clear error

## 6. ForgePipeline + FSM context wiring

- [ ] 6.1 Add `eval_audio_features: torch.Tensor | None = None` and `eval_encoder_states: torch.Tensor | None = None` to `ForgePipeline`
- [ ] 6.2 `_build_fsm_ctx` populates `ctx["_eval_audio_features"]` and `ctx["_eval_encoder_states"]` from the new fields when set
- [ ] 6.3 Construction-time validation: `eval_audio_features` requires the host to be a Whisper class (`adapter_for(host).family == "whisper_encoder"`); `eval_prompts` is mutually exclusive with `eval_audio_features`
- [ ] 6.4 `run_synthetic` accepts the new audio-side kwargs alongside the existing text kwargs

## 7. Audio data fixture

- [ ] 7.1 New module `saeforge/audio_data.py` exposing `synthetic_mel_features(seed, batch=1, n_mels=80, n_frames=3000)` — pure-numpy mel-spectrogram synthesis (sine sweep + noise)
- [ ] 7.2 `tests/conftest.py` gains a `tiny_synthetic_whisper` fixture (39M-class WhisperConfig with d_model=64, encoder_layers=2)
- [ ] 7.3 Tests confirm the fixture produces a valid `WhisperModel` with `.encoder` accessible

## 8. CLI

- [ ] 8.1 `sae-forge forge --audio-features-path FILE.pt` flag; loads a torch tensor of shape `(batch, n_mels, n_frames)`
- [ ] 8.2 Mutually exclusive with `--eval-prompts` (argparse-level)
- [ ] 8.3 Tests in `tests/test_cli.py` for the new flag's argparse contract

## 9. Examples

- [ ] 9.1 New `examples/forge_whisper_synthetic.py` — full synthetic Whisper forge end-to-end (no HF download, no real audio)
- [ ] 9.2 Add an `examples/test_examples_smoke.py` entry that runs the synthetic Whisper example with a 30s wall-clock budget

## 10. CI / extras

- [ ] 10.1 `pyproject.toml`: add `[audio]` extra pinning `librosa>=0.10` (optional — only the real-audio path needs it; the synthetic fixture path is pure-numpy)
- [ ] 10.2 `[audio]` is included in the `[all]` extra
- [ ] 10.3 Update CI matrix to install `[dev,intel,polygram,orca,audio]` so the Whisper smoke runs

## 11. Polygram coordination

- [ ] 11.1 Verify a polygram-compressed Whisper SAE (one of the five-SAE panel checkpoints) loads through `FeatureBasis.from_polygram_checkpoint` end-to-end. The Llama-Scope inspect verified the bf16 path; this confirms the same for Whisper specifically
- [ ] 11.2 Note in `docs/advanced-fsm-options.md` (or a new `docs/audio-forge.md`) that audio SAE compressions tagged with `profile=uniform-sphere` are the recommended polygram-side setting; sae-forge does not consume the profile but flags the recommendation for users

## 12. Documentation

- [ ] 12.1 New `docs/audio-forge.md` — user-facing reference for forging audio encoders. Covers: when to use it, how to pre-extract mel features, recommended polygram profile, the cosine eval semantics, the conv-stem ε_conv accounting
- [ ] 12.2 Update `AGENTS.md` "Adapter contract" subsection to mention the encoder-only audio scope and the LM-vs-encoder eval dispatch
- [ ] 12.3 Update `README.md` Status section bullet for `forge-whisper-encoder`
- [ ] 12.4 Document the new `NativeModelConfig.output_kind` and the `vocab_size = 0` semantics in `docs/algorithm.md` §"Native model shape" (or equivalent section). The math foundation doc is the canonical reference for how the projection algebra sees the native model — adding a non-LM family is the kind of change that needs to land there, not just the user-facing audio doc
- [ ] 12.5 Add a `## [Unreleased]` section to `CHANGELOG.md` (the repo currently lands entries at archive time per the file's preamble; this change introduces the Unreleased convention so the implementation PR has a place to drop notes incrementally). The Unreleased entry SHALL list the four artefacts this change introduces (adapter, native module, eval, audio-forge doc) so reviewers can see the surface area at a glance before the release-bump commit lands

## 13. Tests (overall)

- [ ] 13.1 The byte-equivalence safety net `test_imperative_and_fsm_byte_equivalent` continues to pass unchanged (LM family default behavior preserved)
- [ ] 13.2 Total new test count target: ~25 tests across `test_whisper_encoder_adapter.py` (12), `test_whisper_encoder_module.py` (5), `test_audio_eval.py` (8), and `test_evaluate_faithfulness_dispatch.py` (3)
- [ ] 13.3 No existing tests modified except where the dispatch branch in `evaluate_faithfulness` requires updating a stub return value (LM-path stubs unchanged)

## 14. OpenSpec scaffolding

- [x] 14.1 `openspec/changes/forge-whisper-encoder/proposal.md`
- [x] 14.2 `openspec/changes/forge-whisper-encoder/design.md`
- [x] 14.3 `openspec/changes/forge-whisper-encoder/tasks.md` (this file)
- [x] 14.4 `openspec/changes/forge-whisper-encoder/specs/architecture-adapters/spec.md` (delta — adds `whisper_encoder` family)
- [x] 14.5 `openspec/changes/forge-whisper-encoder/specs/whisper-encoder-eval/spec.md` (new capability — cosine faithfulness)

## 15. Deferred follow-ups (out of scope for this change)

- [ ] 15.1 **`forge-whisper-decoder`** — full Whisper decoder forge with cross-attention to encoder states. Separate change
- [ ] 15.2 **Real audio data loaders** — `.wav` / `.flac` ingestion via `librosa`. The synthetic-mel and pre-extracted-features paths cover test + production needs in v0.4
- [ ] 15.3 **Conv stem projection** — Whisper's `conv1` / `conv2` are frozen-copied today (ε_conv per `docs/algorithm.md` §5). A research follow-up
- [ ] 15.4 **Whisper-large validation** — adapter is shape-agnostic; tested on tiny in this change. Production validation against a real `openai/whisper-tiny` SAE is a follow-up smoke
- [ ] 15.5 **Compressed-basis cosine eval** — the v0.4 cosine_faithfulness projects host states *into* the SAE basis via the same projector used to build the forged model, then compares forged states to projected host states in basis-space. A future variant could compare in the *compressed* basis (post-Polygram-`Compressor` index space) to disentangle compression error from forge error. Out of scope here; tracked because Grok flagged it as a worthwhile follow-up
