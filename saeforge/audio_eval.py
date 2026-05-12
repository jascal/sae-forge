"""Cosine-similarity faithfulness evaluator for forged Whisper encoders.

The faithfulness signal for an LM forge is per-token KL divergence
between the forged model's vocab-shaped logits and the host's
(:func:`saeforge.forge._kl_from_input_ids`). Audio encoders don't
produce a distribution over a vocabulary — they emit real-valued
per-frame hidden states — so KL is the wrong signal. The natural
alternative is per-frame cosine similarity between the forged
encoder's output and the host encoder's output, projected into the
same SAE basis so the two vectors live in the same space.

The forged module already carries the d → f projection as a non-
parameter ``basis_encode`` buffer (set by the adapter walk from
``projector.basis.pseudoinverse() * scale_boost``). Reusing that
buffer here means the metric uses the same encode matrix the forge
itself was built with — no separate basis argument required, and no
risk of basis drift between the forge and the eval.

The returned scalar is in ``[0.0, 1.0]``:

- ``1.0`` — forged encoder output matches the basis-projected host
  output exactly (within fp32 noise).
- ``0.0`` — uncorrelated outputs, or zero-norm states on either side
  (the rare degenerate case is mapped to 0 rather than NaN so the
  FSM ``min_faithfulness`` predicate behaves consistently).

This bound matches the LM path's ``[0, 1]`` faithfulness convention,
so the FSM ``min_faithfulness`` threshold logic (in
:mod:`saeforge.actions`) carries over directly — only the metric
itself dispatches by family. Negative cosines (anti-correlated
outputs, possible in principle) are clamped to ``0.0`` since the FSM
``should_continue`` predicate assumes non-negative faithfulness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saeforge.utils.lazy import require_extra

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from saeforge.model import NativeModel


def cosine_faithfulness(
    forged: "NativeModel",
    host: Any,
    audio_features: Any,
    *,
    precomputed_host_states: Any | None = None,
    device: str = "cpu",
) -> float:
    """Per-frame cosine similarity between forged and host encoder states.

    Parameters
    ----------
    forged:
        A :class:`saeforge.model.NativeModel` whose ``config.family ==
        "whisper_encoder"``. Its ``torch_module`` exposes a
        ``basis_encode`` buffer that doubles as the d → f projection
        used to put host states into basis space.
    host:
        Either a ``transformers.WhisperModel`` or a
        ``transformers.WhisperForConditionalGeneration``. The encoder
        is extracted via :class:`WhisperEncoderAdapter`'s helper so
        both classes work identically. May be ``None`` when
        ``precomputed_host_states`` is supplied — the helper never
        looks at ``host`` in that case.
    audio_features:
        A ``torch.Tensor`` of shape ``(batch, n_mels, n_frames)``
        — the mel-spectrogram input both encoders consume. fp16 /
        bf16 inputs are cast to fp32 internally before the similarity
        computation; the forward pass through each encoder runs at
        the encoder's own dtype.
    precomputed_host_states:
        Optional pre-captured host encoder states of shape
        ``(batch, n_frames, d_model)``. When supplied, the helper
        skips the host encoder forward entirely — the caller is
        responsible for guaranteeing these are the output of running
        the host's encoder on ``audio_features``. This is the audio-
        side analog of pre-tokenised ``_eval_input_ids`` and unblocks
        the FSM's pre-capture fast path on big Whisper hosts.
    device:
        Torch device string; both models and the input are moved
        there before the forward pass. Default ``"cpu"``.

    Returns
    -------
    float
        Scalar in ``[0.0, 1.0]``. ``1.0`` indicates the forged
        encoder's output matches the basis-projected host output
        exactly (within fp32 noise); ``0.0`` indicates uncorrelated
        outputs or zero-norm states on either side.
    """
    torch = require_extra("torch", "torch")

    forged_module = forged.torch_module.to(device).eval()
    audio_features = audio_features.to(device)

    with torch.no_grad():
        forged_states = forged_module(audio_features)
        if precomputed_host_states is not None:
            host_states = precomputed_host_states.to(device)
        else:
            from saeforge.adapters.whisper import WhisperEncoderAdapter

            host = host.to(device).eval()
            host_encoder = WhisperEncoderAdapter._extract_encoder(host)
            host_out = host_encoder(audio_features)
            # WhisperEncoder.forward returns BaseModelOutput when called
            # outside of the parent model; fall back to indexing for the
            # tuple-style return.
            host_states = (
                host_out.last_hidden_state
                if hasattr(host_out, "last_hidden_state")
                else host_out[0]
            )

    # Cast both to fp32 for the similarity math. bf16 / fp16 inputs
    # round-trip through here without overflow on small encoders.
    forged_states = forged_states.float()
    host_states_d = host_states.float()

    # Project host states into the SAE basis using the same encode
    # matrix the forge was built with. basis_encode is shape (d, f) so
    # right-multiplying a (..., d) tensor produces (..., f).
    basis_encode = forged_module.basis_encode.float()
    host_states_f = host_states_d @ basis_encode

    # Per-frame cosine similarity. Manual computation so zero-norm
    # frames map to 0 rather than NaN.
    dot = (forged_states * host_states_f).sum(dim=-1)
    forged_norm = forged_states.norm(dim=-1)
    host_norm = host_states_f.norm(dim=-1)
    denom = forged_norm * host_norm
    valid = denom > 0
    cos = torch.zeros_like(dot)
    cos[valid] = dot[valid] / denom[valid]

    # Mean over batch + time; clamp negatives to 0 per the FSM
    # min_faithfulness >= 0 contract; clip the upper end at 1.0 to
    # absorb fp32 noise above unity.
    score = float(cos.mean().item())
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score
