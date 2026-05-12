"""Whisper-encoder architecture adapter.

Walks a HuggingFace ``WhisperForConditionalGeneration`` or
``WhisperModel`` host's *encoder* into a projected weight dict keyed
by the same parameter names the matching native module declares.
The decoder is out of scope for this change — encoder-only forge is
the minimum-viable path for the polygram-side audio SAEs in the
five-SAE validation panel.

Projection algebra (HF ``Linear.weight`` stores as ``(out, in)``):

- ``self_attn.{q,k,v}_proj.weight (d_model, d_model)`` →
  ``project_residual_output`` (acts on the in-axis, which is the
  residual) → ``(d_model, n_features)``. ``q_proj`` / ``v_proj``
  have bias in head-space (``(d_model,)``) — passed through
  unprojected. HF Whisper's ``k_proj`` has no bias.
- ``self_attn.out_proj.weight (d_model, d_model)`` →
  ``project_residual_input`` (acts on the first axis, which writes
  the residual) → ``(n_features, d_model)``. ``out_proj.bias`` is
  added to the residual stream — projected via
  ``project_residual_aligned`` to ``(n_features,)``.
- ``fc1.weight (intermediate_size, d_model)`` →
  ``project_residual_output`` → ``(intermediate_size, n_features)``;
  bias in inner-space passed through.
- ``fc2.weight (d_model, intermediate_size)`` →
  ``project_residual_input`` → ``(n_features, intermediate_size)``;
  bias in residual space projected.
- ``self_attn_layer_norm.{weight,bias}``,
  ``final_layer_norm.{weight,bias}``, and the encoder-final
  ``layer_norm.{weight,bias}``: residual-aligned vectors of shape
  ``(d_model,)`` → ``project_residual_aligned`` → ``(n_features,)``.

Frozen-copied (NOT projected — counted as ε_conv per
``docs/algorithm.md`` §5):

- ``conv1.weight (d_model, n_mels, 3)``, ``conv1.bias (d_model,)``
- ``conv2.weight (d_model, d_model, 3)``, ``conv2.bias (d_model,)``
- ``embed_positions.weight (max_source_positions, d_model)``

The unprojected conv stem feeds non-basis-aligned features into the
first encoder block; this introduces a bounded but non-zero error
in the forged encoder's outputs versus the host. Projecting the
1D-spatial conv kernels is a separate research question and is
deferred. See the ``forge-whisper-encoder`` design doc for the full
rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saeforge.adapters.base import ArchitectureAdapter, to_numpy
from saeforge.utils.lazy import require_extra

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from saeforge.model import NativeModelConfig
    from saeforge.projector import SubspaceProjector


class WhisperEncoderAdapter(ArchitectureAdapter):
    """Adapter for HF :class:`transformers.WhisperForConditionalGeneration`
    and :class:`transformers.WhisperModel`.

    The same instance handles both host classes; ``_extract_encoder``
    pulls ``host.model.encoder`` (WhisperForConditionalGeneration) or
    ``host.encoder`` (WhisperModel). ``attention_width='feature_native'``
    is not supported in v0.4 — Whisper's MHA layout is residual-only.
    """

    family = "whisper_encoder"

    def walk(
        self,
        host: Any,
        projector: "SubspaceProjector",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        if attention_width != "host":
            raise NotImplementedError(
                f"{type(self).__name__} only supports attention_width='host'"
                f" in v0.4; got {attention_width!r}."
            )

        encoder = self._extract_encoder(host)
        out: dict[str, np.ndarray] = {}

        # Frozen-copy path: conv stem + sinusoidal positions are NOT
        # residual-aligned and pass through the adapter unchanged. The
        # forged module copies them bit-for-bit into the corresponding
        # parameter slots.
        out["conv1.weight"] = to_numpy(encoder.conv1.weight)
        out["conv1.bias"] = to_numpy(encoder.conv1.bias)
        out["conv2.weight"] = to_numpy(encoder.conv2.weight)
        out["conv2.bias"] = to_numpy(encoder.conv2.bias)
        out["embed_positions.weight"] = to_numpy(encoder.embed_positions.weight)

        for i, block in enumerate(encoder.layers):
            prefix = f"layers.{i}"

            out[f"{prefix}.self_attn_layer_norm.weight"] = (
                projector.project_residual_aligned(
                    to_numpy(block.self_attn_layer_norm.weight)
                )
            )
            out[f"{prefix}.self_attn_layer_norm.bias"] = (
                projector.project_residual_aligned(
                    to_numpy(block.self_attn_layer_norm.bias)
                )
            )

            # Q/K/V: HF Linear weight (out, in); in-axis = residual.
            # q_proj and v_proj have biases in head-space (d_model,);
            # k_proj has no bias (HF Whisper convention).
            out[f"{prefix}.self_attn.q_proj.weight"] = (
                projector.project_residual_output(
                    to_numpy(block.self_attn.q_proj.weight)
                )
            )
            out[f"{prefix}.self_attn.q_proj.bias"] = to_numpy(
                block.self_attn.q_proj.bias
            )
            out[f"{prefix}.self_attn.k_proj.weight"] = (
                projector.project_residual_output(
                    to_numpy(block.self_attn.k_proj.weight)
                )
            )
            out[f"{prefix}.self_attn.v_proj.weight"] = (
                projector.project_residual_output(
                    to_numpy(block.self_attn.v_proj.weight)
                )
            )
            out[f"{prefix}.self_attn.v_proj.bias"] = to_numpy(
                block.self_attn.v_proj.bias
            )

            # out_proj writes the residual: project the out-axis (first axis).
            # Bias is residual-aligned.
            out[f"{prefix}.self_attn.out_proj.weight"] = (
                projector.project_residual_input(
                    to_numpy(block.self_attn.out_proj.weight)
                )
            )
            out[f"{prefix}.self_attn.out_proj.bias"] = (
                projector.project_residual_aligned(
                    to_numpy(block.self_attn.out_proj.bias)
                )
            )

            out[f"{prefix}.final_layer_norm.weight"] = (
                projector.project_residual_aligned(
                    to_numpy(block.final_layer_norm.weight)
                )
            )
            out[f"{prefix}.final_layer_norm.bias"] = (
                projector.project_residual_aligned(
                    to_numpy(block.final_layer_norm.bias)
                )
            )

            # MLP: fc1 reads residual (in-axis projected), fc2 writes residual
            # (out-axis projected). fc1.bias lives in inner/ffn space.
            out[f"{prefix}.fc1.weight"] = projector.project_residual_output(
                to_numpy(block.fc1.weight)
            )
            out[f"{prefix}.fc1.bias"] = to_numpy(block.fc1.bias)
            out[f"{prefix}.fc2.weight"] = projector.project_residual_input(
                to_numpy(block.fc2.weight)
            )
            out[f"{prefix}.fc2.bias"] = projector.project_residual_aligned(
                to_numpy(block.fc2.bias)
            )

        out["layer_norm.weight"] = projector.project_residual_aligned(
            to_numpy(encoder.layer_norm.weight)
        )
        out["layer_norm.bias"] = projector.project_residual_aligned(
            to_numpy(encoder.layer_norm.bias)
        )

        return out

    def build_native_config(
        self,
        host: Any,
        n_features: int,
        *,
        attention_width: str = "host",
    ) -> "NativeModelConfig":
        from saeforge.model import NativeModelConfig

        cfg = host.config
        d_model = cfg.d_model
        num_heads = cfg.encoder_attention_heads
        head_dim = d_model // num_heads
        return NativeModelConfig(
            family=self.family,
            hidden_size=n_features,
            qkv_inner_size=d_model,
            num_layers=cfg.encoder_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=cfg.encoder_ffn_dim,
            vocab_size=0,
            output_kind="encoder_states",
            max_position_embeddings=cfg.max_source_positions,
            activation="gelu",
            attention_width=attention_width,
            # Whisper encoder is MHA, not GQA. Locked here as an invariant —
            # if a future Whisper variant ships GQA, this assertion will
            # surface it loudly.
            n_kv_heads=num_heads,
            layer_norm_epsilon=1e-5,
        )

    def native_module_class(self) -> type:
        return _get_forged_whisper_encoder_class()

    def grad_checkpoint_targets(self, module):
        # ForgedWhisperEncoder.layers is the iterable of transformer blocks;
        # embed_positions.weight is the input-side parameter that needs
        # requires_grad=True so the checkpointed graph has a leaf input.
        # The conv stem is intentionally excluded — those weights are
        # frozen-copied from the host (ε_conv accounting).
        return module.layers, module.embed_positions.weight

    @staticmethod
    def _extract_encoder(host: Any):
        """Return the encoder submodule for either Whisper host class.

        - :class:`WhisperForConditionalGeneration`: ``host.model.encoder``
        - :class:`WhisperModel`: ``host.encoder``

        The walk is identical regardless of which class loaded; this
        helper isolates the attribute-lookup dispatch.
        """
        if hasattr(host, "encoder") and not hasattr(host, "model"):
            return host.encoder
        return host.model.encoder


# ---------------------------------------------------------------------------
# Native module factory (stub) — full forward pass tracked as §3 follow-up.
# ---------------------------------------------------------------------------


_FORGED_WHISPER_ENCODER_CLASS = None


def build_whisper_encoder_module(config: "NativeModelConfig"):
    """Construct a Whisper encoder native module. Lazy-imports torch."""
    cls = _get_forged_whisper_encoder_class()
    return cls(config)


def _get_forged_whisper_encoder_class():
    """Return the ForgedWhisperEncoder class (lazy torch import).

    v0.4 ships the parameter skeleton: every slot the adapter's walk
    emits has a matching ``nn.Parameter`` so ``load_state_dict`` and
    the no-randomly-initialised-weights invariant work. The forward
    pass is deferred to the §3 follow-up — calling ``.forward()`` on a
    ForgedWhisperEncoder raises ``NotImplementedError`` naming the
    deferred task. This keeps the adapter merge unblocked while the
    eval-side wiring lands in a separate commit.
    """
    global _FORGED_WHISPER_ENCODER_CLASS
    if _FORGED_WHISPER_ENCODER_CLASS is not None:
        return _FORGED_WHISPER_ENCODER_CLASS

    torch = require_extra("torch", "torch")
    import torch.nn as nn

    class WhisperEncoderSelfAttention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d_head = cfg.num_heads * cfg.head_dim
            # HF Whisper's k_proj has no bias; q_proj / v_proj / out_proj do.
            self.q_proj = nn.Linear(cfg.hidden_size, d_head, bias=True)
            self.k_proj = nn.Linear(cfg.hidden_size, d_head, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_size, d_head, bias=True)
            self.out_proj = nn.Linear(d_head, cfg.hidden_size, bias=True)
            self.num_heads = cfg.num_heads
            self.head_dim = cfg.head_dim

        def forward(self, x):  # pragma: no cover — §3 follow-up
            raise NotImplementedError(
                "ForgedWhisperEncoder.forward is deferred to the §3 "
                "follow-up commit on forge-whisper-encoder. The adapter "
                "and parameter skeleton are in place; the forward pass "
                "(matching HF WhisperEncoder's mel → encoder_states "
                "contract) ships separately."
            )

    class WhisperEncoderLayer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            eps = cfg.layer_norm_epsilon
            self.self_attn_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=eps)
            self.self_attn = WhisperEncoderSelfAttention(cfg)
            self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=eps)
            self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
            self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

        def forward(self, x):  # pragma: no cover — §3 follow-up
            raise NotImplementedError(
                "ForgedWhisperEncoder.forward is deferred to the §3 "
                "follow-up commit."
            )

    class ForgedWhisperEncoder(nn.Module):
        """Forged Whisper encoder — parameter skeleton.

        Every parameter slot matches a key emitted by
        :meth:`WhisperEncoderAdapter.walk`. The conv stem and positional
        embedding are sized to host shapes (frozen-copied) so
        ``load_state_dict(strict=True)`` accepts the walk output.

        The forward pass is intentionally not implemented — see
        :func:`_get_forged_whisper_encoder_class` for the deferral
        rationale.
        """

        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            d_head = cfg.num_heads * cfg.head_dim
            # n_mels is fixed at 80 for every Whisper variant shipped
            # by openai/whisper-* (tiny through large-v3).
            n_mels = 80
            # conv1: (n_mels) → (d_head); kernel=3, padding=1.
            # conv2: (d_head) → (d_head); kernel=3, stride=2, padding=1.
            # Both convs are frozen-copied from the host so they keep
            # the host's d_model = d_head channel size, not n_features.
            self.conv1 = nn.Conv1d(n_mels, d_head, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(d_head, d_head, kernel_size=3, stride=2, padding=1)
            self.embed_positions = nn.Embedding(
                cfg.max_position_embeddings, d_head
            )
            self.layers = nn.ModuleList(
                [WhisperEncoderLayer(cfg) for _ in range(cfg.num_layers)]
            )
            self.layer_norm = nn.LayerNorm(
                cfg.hidden_size, eps=cfg.layer_norm_epsilon
            )

        def forward(self, input_features):  # pragma: no cover — §3 follow-up
            raise NotImplementedError(
                "ForgedWhisperEncoder.forward is deferred to the §3 "
                "follow-up commit on forge-whisper-encoder. Parameter "
                "loading via from_projected_weights works; calling the "
                "forward path requires the conv stem → block stack → "
                "final norm pipeline, tracked separately."
            )

    _FORGED_WHISPER_ENCODER_CLASS = ForgedWhisperEncoder
    return ForgedWhisperEncoder


# Register at module-import time. The HF classes are lazy-loaded so importing
# this module without [torch] doesn't crash the package.
try:
    from transformers import WhisperForConditionalGeneration, WhisperModel

    from saeforge.adapters import register_adapter

    _adapter = WhisperEncoderAdapter()
    register_adapter(WhisperForConditionalGeneration, _adapter)
    register_adapter(WhisperModel, _adapter)
except ImportError:  # pragma: no cover
    pass
