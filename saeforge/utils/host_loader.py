"""Host-architecture-aware ``from_pretrained`` dispatch.

The bundled adapters cover both causal-LM hosts (GPT-2 / Llama /
Gemma-2 / Qwen-2/3 / Qwen-3-MoE) and encoder-only hosts (Whisper-
encoder, ESM-2). ``AutoModelForCausalLM.from_pretrained`` works for
the first group but silently fails for masked-LM checkpoints
(``EsmForMaskedLM``) and audio encoders. This helper inspects the HF
config's ``model_type`` and picks the matching AutoModel class so the
real-host forge paths work for every supported family.

The mapping is intentionally explicit (not "guess by looking at
modelcard tags") so future families get added at a single, auditable
site rather than scattered across the forge codebase.
"""

from __future__ import annotations

from typing import Any


def load_host_for_forge(host_model_id: str, *, torch_dtype: Any = None) -> Any:
    """Load an HF host model with the right AutoModel class for forging.

    Dispatch strategy: try ``AutoModelForCausalLM`` first (the historical
    default that covers GPT-2 / Llama / Gemma-2 / Qwen). When that raises
    because the config is incompatible (e.g. ``EsmConfig`` is a masked-LM
    architecture and ``AutoModelForCausalLM`` rejects it), fall back to
    ``AutoModelForMaskedLM``. This keeps every prior test that mocks
    ``AutoModelForCausalLM.from_pretrained`` byte-equivalent — the mock
    intercepts the first call and the fallback never runs.

    Returns the loaded model in ``.eval()`` mode. The dtype kwarg is
    forwarded as ``torch_dtype=`` on the underlying ``from_pretrained``
    call when set; the caller is responsible for resolving the string
    dtype name (e.g. ``"float32"``) to a ``torch.dtype`` before calling.
    """
    from saeforge.utils.lazy import require_extra

    transformers = require_extra("transformers", "torch")

    kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            host_model_id, **kwargs
        ).eval()
    except ValueError as exc:
        # ``AutoModelForCausalLM`` raises ValueError when the config is
        # not in the causal-LM mapping. The masked-LM families (ESM-2)
        # land here. Other ValueErrors (corrupt checkpoint, malformed
        # config) need to propagate, but matching on the substring
        # "Unrecognized configuration class" disambiguates cleanly: that
        # phrase is the canonical AutoModel-rejection message across
        # transformers versions 4.x.
        if "Unrecognized configuration class" not in str(exc):
            raise
        return transformers.AutoModelForMaskedLM.from_pretrained(
            host_model_id, **kwargs
        ).eval()
