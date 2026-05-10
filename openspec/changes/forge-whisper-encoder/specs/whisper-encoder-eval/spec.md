# whisper-encoder-eval Specification

## Purpose

The `whisper-encoder-eval` capability defines the cosine-similarity
faithfulness evaluator that `evaluate_faithfulness` dispatches to
when the forged native model's `family == "whisper_encoder"`. It is
distinct from the existing per-token KL evaluator
(`_kl_from_input_ids`) because audio encoder states are real-valued
per-frame vectors, not distributions over a vocabulary — KL is the
wrong signal.

## Requirements

### Requirement: cosine_faithfulness returns per-frame averaged similarity

`saeforge.audio_eval.cosine_faithfulness(forged, host, audio_features,
*, device="cpu") -> float` SHALL:

1. Run the host encoder on `audio_features` to produce host states
   of shape `(batch, n_frames, d_model)`. The host's encoder is
   `host.encoder` (for `WhisperModel`) or `host.model.encoder` (for
   `WhisperForConditionalGeneration`).
2. Run the forged encoder on `audio_features` to produce forged
   states of shape `(batch, n_frames, n_features)`. The forged
   states live in the SAE's feature basis; the host states live in
   the original `d_model` basis. To make them comparable the host
   states SHALL be projected through the same basis used to build the
   forged model — pulled from `forged.config` indirectly via the
   pinned basis on disk, or via a `basis` argument the action passes
   in.
3. Compute per-frame cosine similarity between the matched-basis
   host states and the forged states.
4. Average across the batch and time axes; clamp negative values to
   `0.0` (the FSM `min_faithfulness` predicate assumes non-negative
   faithfulness).
5. Return a Python `float` in `[0.0, 1.0]`.

The function SHALL accept `forged` and `host` as torch nn.Module
instances and `audio_features` as a torch tensor; bf16 / fp16
inputs SHALL be cast to fp32 internally before similarity
computation.

#### Scenario: identical states return 1.0

- **GIVEN** a `forged` model whose projected weights are
  byte-identical to the host's basis-projected weights (built via
  the synthetic-Whisper fixture with the projector run twice with
  the same seed)
- **WHEN** `cosine_faithfulness(forged, host, audio_features)` is
  called
- **THEN** the returned float is `1.0` to within
  `atol=1e-6` (allowing for fp32 accumulation noise)

#### Scenario: orthogonal forged states return 0.0

- **GIVEN** a `forged` model whose projected weights have been
  zeroed out (so the forward produces all-zero states)
- **WHEN** `cosine_faithfulness(forged, host, audio_features)` is
  called
- **THEN** the returned float is `0.0` (because the cosine of any
  vector with the zero vector is undefined; the implementation
  SHALL return `0.0` for zero-norm states rather than NaN)

#### Scenario: noise monotonically degrades similarity

- **GIVEN** a `forged` model identical to the host's basis-projected
  forge
- **AND** three noise levels `σ ∈ {0.0, 0.1, 0.5}` injected into
  `audio_features` before the forward pass
- **WHEN** `cosine_faithfulness` is computed at each noise level
- **THEN** the resulting floats are strictly monotonically decreasing
  (`s_0 >= s_1 >= s_2`)

### Requirement: evaluate_faithfulness dispatches by family

`saeforge.actions.evaluate_faithfulness` SHALL inspect
`forged.config.family` and dispatch:

- `"gpt2" | "llama" | "gemma2"`: existing `_kl_from_input_ids` path.
  Reads `ctx["_eval_input_ids"]`. Writes the resulting KL to
  `ctx["faithfulness"]`. Behavior SHALL be byte-identical to the v0.3
  implementation.
- `"whisper_encoder"`: cosine path. Reads `ctx["_eval_audio_features"]`.
  Calls `cosine_faithfulness(forged, host, audio_features)`. Writes
  the resulting cosine to `ctx["faithfulness"]`.
- Unknown family: raises `ValueError` whose message names the
  offending family and the supported set.

The single `ctx["faithfulness"]` field SHALL carry the resulting
scalar in both cases. Downstream consumers (the `should_continue`
predicate, the FSM transition log, `forge_result.json` summary) SHALL
treat the value uniformly. The semantic of `min_faithfulness`
SHALL be reinterpreted per-family:

- LM families: `min_faithfulness` is a maximum allowed KL
  (the existing predicate `kl <= min_faithfulness * -1` semantics).
- `whisper_encoder`: `min_faithfulness` is a minimum required cosine
  similarity (the new `cosine >= min_faithfulness` semantics).

The dispatch SHALL be local to the action — no FSM topology change.

#### Scenario: GPT-2 host routes to KL evaluator

- **GIVEN** a `ForgePipeline` with a GPT-2 host and a forged GPT-2
  native model
- **WHEN** the FSM reaches `evaluate_faithfulness`
- **THEN** the action calls `_kl_from_input_ids` (verified via mock)
- **AND** does NOT call `cosine_faithfulness`

#### Scenario: Whisper encoder host routes to cosine evaluator

- **GIVEN** a `ForgePipeline` with a Whisper host and a forged
  Whisper-encoder native model, with `eval_audio_features` set
- **WHEN** the FSM reaches `evaluate_faithfulness`
- **THEN** the action calls `cosine_faithfulness` (verified via mock)
- **AND** does NOT call `_kl_from_input_ids`

#### Scenario: unknown family raises actionable error

- **GIVEN** a forged model whose `config.family == "fictional"`
- **WHEN** the FSM reaches `evaluate_faithfulness`
- **THEN** `ValueError` is raised whose message contains
  `"fictional"` and the list of supported families

### Requirement: Pre-captured host states are an optimization, not a contract

The action MAY consume a pre-captured `ctx["_eval_encoder_states"]`
tensor instead of running the host encoder forward inline, when the
caller has chosen to pre-capture them outside the FSM (e.g. to skip
loading the host model into the action). When
`_eval_encoder_states` is present, the action SHALL use it directly
and SHALL NOT call `host.encoder(audio_features)` — the host encoder
forward is the costly step we are skipping.

When `_eval_encoder_states` is absent, the action SHALL run the host
forward via `host.encoder(audio_features)` (or
`host.model.encoder(...)` for `WhisperForConditionalGeneration`)
inline. Both paths SHALL produce the same `faithfulness` scalar to
within fp32 accumulation noise.

#### Scenario: pre-captured states bypass host forward

- **GIVEN** a ctx with both `_eval_audio_features` and
  `_eval_encoder_states` set
- **WHEN** `evaluate_faithfulness` runs (host encoder forward mocked)
- **THEN** `host.encoder` is NOT called
- **AND** the returned faithfulness equals the value computed from
  `_eval_encoder_states`

#### Scenario: missing pre-capture triggers inline host forward

- **GIVEN** a ctx with `_eval_audio_features` set but
  `_eval_encoder_states` absent
- **WHEN** `evaluate_faithfulness` runs (host loaded normally)
- **THEN** `host.encoder` is called exactly once
- **AND** the returned faithfulness is non-negative
