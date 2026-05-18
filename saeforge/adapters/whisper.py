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

d → f bridge: because the conv stem and positional embeddings stay
at ``d_model`` width while the transformer blocks operate at
``n_features`` width, the forged module needs a runtime projection
from ``d_model`` to ``n_features`` at the conv-stem→first-block
boundary. This matrix is the basis encode operator
(``projector.basis.pseudoinverse() * scale_boost``, shape
``(d_model, n_features)``) and is materialised as a non-parameter
``basis_encode`` buffer on :class:`ForgedWhisperEncoder`. The walk
emits a ``basis_encode`` key whose value is exactly that matrix; the
buffer is part of ``state_dict()`` so save/load round-trips preserve
it, but it does not appear in ``named_parameters()`` and so doesn't
participate in training-side concerns (gradient checkpointing,
weight-decay groups, the no-randomly-initialised-weights invariant).
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

    def default_faithfulness_target(self):
        """Whisper-encoder default: per-frame cosine.

        Overrides the LM-shape :class:`KLTarget` default with
        :class:`CosineTarget`. Matches the policy
        ``_default_target_for("whisper_encoder")`` enforced via the
        ``_LM_FAMILIES`` table before the world-model-protocol
        refactor.
        """
        from saeforge.eval.targets.cosine import CosineTarget

        return CosineTarget()

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

        # d → f bridge buffer: applied inside forward() at the conv-stem →
        # first-block boundary. Matches SubspaceProjector.encode exactly so
        # the forged encoder reproduces the basis-projected residual stream
        # that every downstream layer's projected weights expect.
        out["basis_encode"] = projector.basis.pseudoinverse() * projector.scale_boost

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

    Closes §3 of forge-whisper-encoder. The forged module reproduces
    HF Whisper's encoder forward path with two differences:

    1. The transformer blocks operate at residual width ``n_features``
       (the SAE basis size), not ``d_model``. The conv stem and
       positional embeddings stay at ``d_model`` (frozen-copied), and
       the d → f projection is applied via the ``basis_encode`` buffer
       at the conv-stem → first-block boundary.
    2. Dropout is omitted everywhere — the forge path is eval-only.

    There is no causal mask: Whisper's encoder self-attention is
    bidirectional (the conv stem produces a fixed-length 1500-frame
    output and every frame attends to every other).
    """
    global _FORGED_WHISPER_ENCODER_CLASS
    if _FORGED_WHISPER_ENCODER_CLASS is not None:
        return _FORGED_WHISPER_ENCODER_CLASS

    torch = require_extra("torch", "torch")
    import math

    import torch.nn as nn
    import torch.nn.functional as F

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

        def forward(self, x):
            # x: (B, T, hidden_size)
            B, T, _ = x.shape
            q = (
                self.q_proj(x)
                .view(B, T, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(x)
                .view(B, T, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(x)
                .view(B, T, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            # Whisper encoder is bidirectional — no causal mask.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            out = (
                (attn @ v)
                .transpose(1, 2)
                .contiguous()
                .view(B, T, self.num_heads * self.head_dim)
            )
            return self.out_proj(out)

    class WhisperEncoderLayer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            eps = cfg.layer_norm_epsilon
            self.self_attn_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=eps)
            self.self_attn = WhisperEncoderSelfAttention(cfg)
            self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=eps)
            self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
            self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

        def forward(self, x):
            # Pre-LN self-attention sublayer (matches HF Whisper).
            residual = x
            x = self.self_attn_layer_norm(x)
            x = self.self_attn(x)
            x = residual + x
            # Pre-LN FFN sublayer with GELU.
            residual = x
            x = self.final_layer_norm(x)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            x = residual + x
            return x

    class ForgedWhisperEncoder(nn.Module):
        """Forged Whisper encoder.

        Every parameter slot matches a key emitted by
        :meth:`WhisperEncoderAdapter.walk`; the ``basis_encode`` buffer
        carries the d → f projection plumbing applied at the conv-stem
        → first-block boundary. ``load_state_dict(strict=True)``
        accepts the walk output and the buffer in one pass.
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
            # d → f projection buffer. Populated by from_projected_weights
            # from the walk's ``basis_encode`` entry. The default-init
            # zero matrix collapses the encoder output to zero — it is
            # never the value used at runtime, but it lets the module
            # construct before weights are loaded.
            self.register_buffer(
                "basis_encode",
                torch.zeros(d_head, cfg.hidden_size),
            )

        def forward(self, input_features):
            """HF-Whisper-compatible encoder forward.

            ``input_features`` shape ``(B, n_mels=80, n_frames)`` →
            return ``(B, n_frames // 2, n_features)``. The factor-of-2
            downsample comes from ``conv2``'s ``stride=2``.
            """
            # Conv stem (frozen-copied; ε_conv per docs/algorithm.md §5).
            x = F.gelu(self.conv1(input_features))
            x = F.gelu(self.conv2(x))
            # (B, d_model, n_frames // 2) → (B, n_frames // 2, d_model).
            x = x.permute(0, 2, 1)
            T = x.size(1)
            if T > self.embed_positions.weight.size(0):
                raise ValueError(
                    f"ForgedWhisperEncoder.forward: input produces "
                    f"{T} frames after the conv stem but the positional "
                    f"embedding table has only "
                    f"{self.embed_positions.weight.size(0)} slots. Pad "
                    f"input mel features to max_source_positions * "
                    f"conv1.stride * conv2.stride."
                )
            x = x + self.embed_positions.weight[:T]
            # d → f projection at the residual-stream entry boundary.
            x = x @ self.basis_encode
            for layer in self.layers:
                x = layer(x)
            x = self.layer_norm(x)
            return x

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
