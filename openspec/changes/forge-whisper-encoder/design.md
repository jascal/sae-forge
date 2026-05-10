# Design: forge-whisper-encoder

## Architecture choice: encoder-only

Whisper has two stages: an audio encoder (mel-spectrogram → encoder
states) and a causal decoder (encoder states + tokens → next-token
logits). They share `d_model` but differ in everything else:

| Aspect | Encoder | Decoder |
|---|---|---|
| Input | Mel features (80, 3000) | Token IDs |
| Pos. encoding | Sinusoidal, learned scale | Learned absolute |
| Self-attn | Yes | Yes (causal mask) |
| Cross-attn | No | **Yes (to encoder states)** |
| Output head | Final `layer_norm`; states pass to decoder | `proj_out` (vocab logits) |
| Faithfulness signal | Encoder-state similarity | Logit KL |

Forging both at once means walking two distinct shapes, projecting
through cross-attention, and supporting a dual eval. That's a 2-3×
larger change. **Encoder-only** is the natural unit because:

1. The polygram-side Whisper SAEs in the validation panel target
   *encoder* residuals — that's where the interesting interpretability
   work is.
2. Encoder-only forge has a meaningful standalone use case: feed the
   forged encoder states into your own classifier or probe.
3. Shape-wise it's the simplest non-causal-LM architecture sae-forge
   can support — minimal new walking surface, no cross-attention.

## Native module shape

`ForgedWhisperEncoder` mirrors `WhisperEncoder` from transformers:

```
mel_features (B, 80, 3000)
  → conv1 (frozen, copied from host) → (B, d_model, 3000)
  → conv2 (frozen, copied from host) → (B, d_model, 1500)
  → permute → (B, 1500, d_model)
  → + sinusoidal_pos_embedding(1500, d_model)
  → N × WhisperEncoderLayer:
      pre_layernorm → self_attn (q/k/v/o) → residual
      pre_layernorm → mlp (fc1 → gelu → fc2) → residual
  → final layer_norm
  → encoder_states (B, 1500, d_model)
```

Forging projects every weight whose **input or output side touches
the residual stream** into the SAE's `n_features`-dim basis:

- `self_attn.q_proj.weight: (d_model, d_model) → (n_features, d_model)`
- `self_attn.k_proj.weight: (d_model, d_model) → (n_features, d_model)`
- `self_attn.v_proj.weight: (d_model, d_model) → (n_features, d_model)`
- `self_attn.out_proj.weight: (d_model, d_model) → (d_model, n_features)`
- `fc1.weight: (intermediate_size, d_model) → (intermediate_size, n_features)`
- `fc2.weight: (d_model, intermediate_size) → (n_features, intermediate_size)`
- `layer_norm.weight: (d_model,) → (n_features,)`
- `layer_norm.bias: (d_model,) → (n_features,)`
- The two pre-attention/pre-MLP `LayerNorm`s per block: same as `layer_norm` above

**Not projected** (frozen, copied from host):
- `conv1.weight: (d_model, 80, 3)` — 1D conv on mel input. Spatial
  structure breaks the residual-stream projection rule. Counted as
  ε_conv per `docs/algorithm.md` §5.
- `conv2.weight: (d_model, d_model, 3)` — same.
- `embed_positions.weight: (1500, d_model)` — sinusoidal in HF, but
  stored as a tensor; copy unchanged. Projecting positional encodings
  is a separate research question.

The frozen-copy path is precedented: `SubspaceProjector` already
exists for projecting weights *into* the basis, so weights it doesn't
project go through unchanged via the adapter's `walk` method, which
copies them straight into the output dict.

## `NativeModelConfig` extensions

Two changes, both backward-compatible:

```python
@dataclass
class NativeModelConfig:
    family: str
    hidden_size: int
    qkv_inner_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int = 0           # was required; default 0 means "no lm_head"
    output_kind: str = "logits"   # NEW: "logits" | "encoder_states"
    # ... unchanged fields below
```

Validation rules in `__post_init__`:

- `output_kind` must be `"logits"` or `"encoder_states"`.
- `output_kind == "logits"` requires `vocab_size > 0` (preserves the
  v0.3 contract for LM families).
- `output_kind == "encoder_states"` requires `vocab_size == 0` (no
  vocab head; an encoder has none) and `family == "whisper_encoder"`
  (no other encoder family ships in this change).
- `family == "whisper_encoder"` implies `output_kind ==
  "encoder_states"` — enforced as an invariant, not a default.

This keeps the existing LM contract byte-identical: callers who don't
set `output_kind` get `"logits"` and the existing `vocab_size > 0`
requirement.

## Eval dispatch

The current `evaluate_faithfulness` action computes per-token KL via
`_kl_from_input_ids`. For Whisper encoder states there's no
distribution to compute KL against — they're real-valued vectors per
audio frame. Two natural alternatives:

| Metric | Pros | Cons |
|---|---|---|
| Per-frame cosine similarity | `[0, 1]`, scale-invariant, matches "directional" SAE bases | Insensitive to magnitude |
| Per-frame L2 / MSE | Magnitude-aware, geometrically intuitive | Scale-dependent, hard to threshold |

Cosine wins for our case because:

1. The SAE basis itself is direction-aligned (compress + decoder
   norms preserve angular structure; magnitude lives in the
   per-feature scale factors).
2. `[0, 1]` lets us reuse the `min_faithfulness` threshold semantics
   from the LM path with a small reinterpretation (`>= 0.95` is the
   conventional "good forge" cutoff).
3. Audio encoder activations have wide dynamic range (mel-input scale
   varies); cosine normalizes that out.

The dispatch lives in `evaluate_faithfulness`:

```python
forged = ctx.get("_native_model")
host = ctx.get("_host_model")
family = forged.config.family if forged is not None else None

if family == "whisper_encoder":
    audio_features = ctx.get("_eval_audio_features")
    score = cosine_faithfulness(forged, host, audio_features, device=...)
else:
    eval_input_ids = ctx.get("_eval_input_ids")
    score = _kl_from_input_ids(forged, host, eval_input_ids, device=...)
```

The single `faithfulness` ctx field carries the resulting scalar in
both cases. Downstream consumers (logging, FSM `should_continue`
predicates) treat it uniformly. The test
`test_imperative_and_fsm_byte_equivalent` continues to use
`_kl_from_input_ids` unchanged because its host is GPT-2.

## Why pre-captured eval states, not online

`_kl_from_input_ids` runs the host model inside the eval action.
For Whisper that means a 39M-param encoder forward per eval — trivial
for occasional use, not free. The audio-side trade-off is different:
the eval input is mel features (already preprocessed), not raw audio,
and the host states are deterministic given the input. Pre-capturing
host states once outside the FSM and passing them in via
`_eval_encoder_states` is faster and side-steps any host-model
loading concerns inside the eval. The same pattern exists today for
`_eval_input_ids` (pre-tokenized).

```python
# Caller does this once before run_machine:
host_states = host_encoder(eval_audio_features).last_hidden_state
ctx["_eval_audio_features"] = eval_audio_features
ctx["_eval_encoder_states"] = host_states
```

## Adapter: walk + register

The `WhisperEncoderAdapter` follows the
`saeforge/adapters/llama.py` pattern. The dispatch tweak: the
adapter registers for **two** HF classes,
`WhisperForConditionalGeneration` and `WhisperModel`, both extracting
the encoder via attribute lookup (`.model.encoder` and `.encoder`
respectively). The walk is identical regardless of which class
loaded.

```python
class WhisperEncoderAdapter(ArchitectureAdapter):
    family = "whisper_encoder"

    def walk(self, host, projector, *, attention_width="host"):
        encoder = self._extract_encoder(host)
        weights = {}
        for i, block in enumerate(encoder.layers):
            weights[f"layers.{i}.self_attn.q_proj.weight"] = projector.qkv(
                _to_numpy(block.self_attn.q_proj.weight)
            )
            # ... k_proj, v_proj, out_proj, fc1, fc2, norms
        # frozen weights pass through:
        weights["conv1.weight"] = _to_numpy(encoder.conv1.weight)
        weights["conv1.bias"] = _to_numpy(encoder.conv1.bias)
        weights["conv2.weight"] = _to_numpy(encoder.conv2.weight)
        weights["conv2.bias"] = _to_numpy(encoder.conv2.bias)
        weights["embed_positions.weight"] = _to_numpy(encoder.embed_positions.weight)
        weights["layer_norm.weight"] = projector.norm(_to_numpy(encoder.layer_norm.weight))
        weights["layer_norm.bias"] = projector.norm(_to_numpy(encoder.layer_norm.bias))
        return weights

    def _extract_encoder(self, host):
        if hasattr(host, "encoder") and not hasattr(host, "model"):
            return host.encoder  # WhisperModel
        return host.model.encoder  # WhisperForConditionalGeneration
```

## Synthetic fixture (no HF download)

Tests need a tiny Whisper encoder. The pattern from
`tests/conftest.py:tiny_synthetic_llama` extends naturally:

```python
@pytest.fixture
def tiny_synthetic_whisper():
    """39M Whisper-tiny → 64-dim, 2-layer, MHA-only, no decoder."""
    from transformers import WhisperConfig, WhisperModel
    config = WhisperConfig(
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_layers=1,  # required by config validation; never used
        decoder_attention_heads=1,
        decoder_ffn_dim=8,
        vocab_size=51865,  # standard, but encoder ignores it
        num_mel_bins=80,
        max_source_positions=1500,
    )
    return WhisperModel(config).eval()
```

This stays under 1MB and runs in <1s on CPU.

## What we are *not* doing in this change

- **Whisper decoder** — cross-attention forge is a separate change
  with its own openspec proposal. Out of scope.
- **Real audio data ingestion** — `librosa`/`soundfile` for `.wav`
  loaders. CLI accepts pre-extracted features only.
- **Multilingual / multitask language tokens** — decoder concern.
- **Streaming inference** — Whisper's 30s chunk model is preserved
  in the forged encoder; no chunking changes.
- **Conv stem projection** — counted as ε_conv per
  `docs/algorithm.md` §5; conv weights are frozen-copied.
- **Quantum-aware path** — `quantum_aware=True` is unchanged; it
  remains a Polygram-side `confirmer` selector, irrelevant to
  whether the host is GPT-2 or Whisper.

## Migration discipline

The byte-equivalence safety net stays the gold standard. The new
fields (`output_kind`, `vocab_size = 0` allowed) default to LM
behavior, so:

- `test_imperative_and_fsm_byte_equivalent` passes unchanged.
- All existing `tests/test_*` files are untouched. The only
  modification to existing source is `evaluate_faithfulness`'s
  dispatch — and its LM branch is the v0.3 code verbatim.

## Open questions

- **Conv stem real eval** — when the forged encoder runs end-to-end
  on real audio, the frozen conv stem's outputs still feed into the
  basis projection. Does ε_conv hurt or help? Tracked as a research
  follow-up after a real Whisper SAE forge runs.
- **`embed_positions` projection** — Whisper's positions are
  sinusoidal in spirit but stored as a learned tensor. Projecting
  them into the feature basis vs. copying unchanged is a research
  question — copying matches the v0.1 GPT-2 `wpe` precedent (which
  is also copied unchanged in the projector).
- **Whisper-large vs tiny** — d_model varies (384 / 512 / 768 /
  1024 / 1280). The adapter is shape-agnostic; testing on tiny is
  sufficient for v0.4. Production validation lives in a follow-up.
