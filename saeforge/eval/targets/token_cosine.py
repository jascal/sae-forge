"""TokenCosineTarget — per-token cosine similarity for token-input encoders.

The token analog of :class:`CosineTarget` (which targets Whisper-style
audio encoders that take mel features). For host architectures whose
forward consumes token IDs and emits hidden states — ESM-2 is the
canonical example, with bio-sae as the first downstream consumer.

Implements the :class:`saeforge.eval.faithfulness.FaithfulnessTarget`
protocol with ``better_when="higher"``. Reads two ctx keys:

- ``_eval_input_ids`` (required): tokenised input. Shape
  ``(batch, seq_len)``. Forge pipelines populate this from
  ``eval_prompts`` via the host's tokenizer.
- ``device``: torch device string, defaults to ``"cpu"``.

For ESM-2 the host is :class:`EsmModel` (or :class:`EsmForMaskedLM`'s
``host.esm`` submodule); we call ``host(input_ids).last_hidden_state``
and compare to the forged model's output. The scoring strips ``CLS``
(position 0) and ``EOS`` (position -1) so the comparison is over
real-residue positions only — matches how
:class:`biosae.proteins.esm_extract.EsmExtractor` consumes the encoder.
"""

from __future__ import annotations

from typing import Any, Mapping


class TokenCosineTarget:
    """Per-token cosine-similarity faithfulness scorer for token-input encoders."""

    name = "token_cosine"
    better_when = "higher"

    def score(
        self,
        *,
        forged: Any,
        host: Any,
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        from saeforge.utils.lazy import require_extra

        torch = require_extra("torch", "torch")

        try:
            input_ids = ctx["_eval_input_ids"]
        except KeyError as exc:  # pragma: no cover — explicit message
            raise KeyError(
                "TokenCosineTarget.score requires ctx['_eval_input_ids']. "
                "Populate it on the pipeline via eval_prompts (which gets "
                "tokenised by the host tokenizer)."
            ) from exc
        if input_ids is None:
            raise KeyError(
                "TokenCosineTarget.score requires ctx['_eval_input_ids'] "
                "to be non-None"
            )

        device = ctx.get("device", "cpu")
        input_ids = input_ids.to(device)

        forged_module = (
            forged.torch_module if hasattr(forged, "torch_module") else forged
        )
        forged_module.eval()

        with torch.no_grad():
            # Encoder-state output: EsmModel exposes
            # ``.last_hidden_state`` on its forward result.
            # EsmForMaskedLM wraps EsmModel and returns
            # ``MaskedLMOutput`` (logits + optional hidden_states); we
            # unwrap by calling the inner encoder (``host.esm``) when
            # present so the call signature is the same in both cases.
            encoder_host = host.esm if hasattr(host, "esm") else host
            host_device = host.device if hasattr(host, "device") else "cpu"
            host_out = encoder_host(input_ids=input_ids.to(host_device))
            host_hidden = _extract_last_hidden_state(host_out)
            forged_hidden = forged_module(input_ids)

        # Strip CLS (position 0) and EOS (last position) — ESM-2's
        # bookkeeping tokens. Bio-sae's EsmExtractor does the same so the
        # cosine matches what real downstream consumers see.
        #
        # Align onto the forged/eval device, not just dtype: the host model
        # may stay on CPU while the forged module runs on GPU (the default
        # for ``ForgePipeline(device="cuda", …)`` driving an encoder host),
        # so a dtype-only cast would leave ``host_hidden`` on CPU and the
        # downstream ``@ basis_encode`` / cosine would mix devices.
        host_hidden = host_hidden[:, 1:-1, :].to(
            device=forged_hidden.device, dtype=forged_hidden.dtype
        )
        forged_hidden = forged_hidden[:, 1:-1, :]

        # Under-complete basis case: forged residual width
        # (``n_features``) is smaller than the host's d_model. Project
        # the host states into the basis using the forged module's
        # ``basis_encode`` buffer (``pinv(W_dec) * scale_boost``) so the
        # cosine compares same-shape vectors. Matches Whisper's
        # ``cosine_faithfulness`` shape-handling.
        if host_hidden.shape != forged_hidden.shape:
            if not hasattr(forged_module, "basis_encode"):
                raise RuntimeError(
                    f"TokenCosineTarget: host hidden state shape "
                    f"{tuple(host_hidden.shape)} does not match forged "
                    f"{tuple(forged_hidden.shape)} and the forged "
                    f"module has no ``basis_encode`` buffer to project "
                    f"between them. Forge a family that emits a "
                    f"``basis_encode`` buffer in its walk (currently "
                    f"esm2 / whisper_encoder)."
                )
            # Defensively pin device too — keeps the matmul single-device
            # even if the host-hidden device path changes later.
            basis_encode = forged_module.basis_encode.to(
                device=forged_hidden.device, dtype=forged_hidden.dtype
            )
            host_hidden = host_hidden @ basis_encode

        # Per-residue cosine, then mean.
        host_flat = host_hidden.reshape(-1, host_hidden.shape[-1])
        forged_flat = forged_hidden.reshape(-1, forged_hidden.shape[-1])
        eps = 1e-8
        host_norm = host_flat / (host_flat.norm(dim=-1, keepdim=True) + eps)
        forged_norm = forged_flat / (forged_flat.norm(dim=-1, keepdim=True) + eps)
        cosine = (host_norm * forged_norm).sum(dim=-1).mean().item()

        # FSM's perplexity-analog: monotonically decreasing in cosine,
        # positive-real. Mirror the convention in CosineTarget.
        perplexity = max(0.0, 1.0 - cosine)
        return float(cosine), float(perplexity)


def _extract_last_hidden_state(host_output) -> Any:
    """Pull the last hidden state out of an HF model output.

    ``EsmModel`` and ``EsmForMaskedLM`` return objects with the same
    ``.last_hidden_state`` attribute on the underlying BaseModelOutput.
    Tuple-style returns from older HF versions are also tolerated.
    """
    if hasattr(host_output, "last_hidden_state"):
        return host_output.last_hidden_state
    if hasattr(host_output, "hidden_states") and host_output.hidden_states:
        return host_output.hidden_states[-1]
    if isinstance(host_output, tuple) and len(host_output) > 0:
        return host_output[0]
    raise RuntimeError(
        f"TokenCosineTarget: host returned an output without a "
        f"recognisable last_hidden_state. Got "
        f"{type(host_output).__name__}."
    )
