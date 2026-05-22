"""ESM-2 (protein language model) adapter.

Walks a HuggingFace ``EsmModel`` or ``EsmForMaskedLM`` host into a
projected weight dict keyed by the same parameter names the matching
native module declares. The encoder is the only target — the optional
``lm_head`` and ``contact_head`` are not projected. ESM-2 is encoder-
only with bidirectional attention and rotary positional embeddings;
that puts it between Whisper-encoder (bidirectional, encoder-only) and
Llama (RoPE) in adapter shape.

Projection algebra (HF ``Linear.weight`` stores as ``(out, in)``;
residual width ``d_model`` → forged width ``n_features = f``):

- ``embeddings.word_embeddings.weight (V, d)`` →
  ``project_embed`` → ``(V, f)``.
- ``encoder.layer.{i}.attention.LayerNorm.{weight,bias} (d,)`` →
  ``project_residual_aligned`` → ``(f,)``. Pre-LN on the attention
  sublayer's input (matches HF ``EsmAttention.forward``).
- ``encoder.layer.{i}.attention.self.{query,key,value}.weight
  (d, d)`` → ``project_residual_output`` (in-axis = residual) →
  ``(d, f)``.
- ``encoder.layer.{i}.attention.self.{query,key,value}.bias (d,)``
  → unprojected (head-space bias, not residual).
- ``encoder.layer.{i}.attention.output.dense.weight (d, d)`` →
  ``project_residual_input`` (out-axis = residual) → ``(f, d)``.
- ``encoder.layer.{i}.attention.output.dense.bias (d,)`` →
  ``project_residual_aligned`` → ``(f,)`` (added to the residual
  stream inside HF ``EsmSelfOutput``).
- ``encoder.layer.{i}.LayerNorm.{weight,bias} (d,)`` →
  ``project_residual_aligned`` → ``(f,)``. Pre-LN on the FFN
  sublayer's input.
- ``encoder.layer.{i}.intermediate.dense.weight
  (intermediate, d)`` → ``project_residual_output`` →
  ``(intermediate, f)``; bias unprojected (inner-space).
- ``encoder.layer.{i}.output.dense.weight (d, intermediate)`` →
  ``project_residual_input`` → ``(f, intermediate)``; bias
  ``project_residual_aligned`` → ``(f,)``.
- ``encoder.emb_layer_norm_after.{weight,bias} (d,)`` →
  ``project_residual_aligned`` → ``(f,)``.

Skipped (not in the walk):

- ``contact_head`` — diagnostic for residue-residue contacts, not on
  the residual stream.
- ``pooler`` — optional CLS-pooler, not used for SAE feature work.
- ``lm_head`` — MLM head. Forge defaults to ``output_kind=
  'encoder_states'`` (cosine faithfulness on per-residue hidden
  states); the MLM head can be projected separately by a follow-up
  if logit-space faithfulness is needed.
- ``embeddings.position_embeddings`` — ESM-2 uses
  ``position_embedding_type='rotary'`` so this Embedding table is
  absent in production checkpoints. Pre-RoPE ESM-1 ('absolute') is
  out of scope; the adapter raises if it sees that config.
- ``rotary_embeddings.inv_freq`` — recomputed deterministically in
  the forged attention from ``head_dim`` + ``rope_theta``.

The forward path mirrors HF ``modeling_esm.py`` (pre-LN attention,
pre-LN FFN, no causal mask, ESM-specific GELU). Validated against a
running ``EsmModel`` by ``tests/test_esm_adapter.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saeforge.adapters.base import ArchitectureAdapter, to_numpy
from saeforge.utils.lazy import require_extra

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from saeforge.eval.faithfulness import FaithfulnessTarget
    from saeforge.model import NativeModelConfig
    from saeforge.projector import SubspaceProjector


class Esm2Adapter(ArchitectureAdapter):
    """Adapter for HF :class:`transformers.EsmModel` and
    :class:`transformers.EsmForMaskedLM`.

    ESM-2 checkpoints (``facebook/esm2_t6_8M_UR50D`` through
    ``facebook/esm2_t48_15B_UR50D``) all use
    ``position_embedding_type='rotary'`` and ``emb_layer_norm_before=
    False`` — those are pinned invariants for the adapter.
    """

    family = "esm2"

    def default_faithfulness_target(self) -> "FaithfulnessTarget":
        """ESM-2 default: per-residue cosine on encoder hidden states.

        Matches the policy whisper_encoder uses for its encoder-only
        host (CosineTarget on audio features). ESM-2 is encoder-only
        for the SAE-feature use case, so cosine on hidden states is
        the right faithfulness signal. Users wanting logit-space KL
        on the MLM head can pass an explicit ``FaithfulnessTarget``
        via ``ForgePipeline(faithfulness=...)``.
        """
        from saeforge.eval.targets.token_cosine import TokenCosineTarget

        return TokenCosineTarget()

    def walk(
        self,
        host: Any,
        projector: "SubspaceProjector",
        *,
        attention_width: str = "host",
    ) -> dict[str, np.ndarray]:
        if attention_width != "host":
            raise NotImplementedError(
                f"{type(self).__name__} only supports attention_width="
                f"'host' in v1; got {attention_width!r}. The feature-"
                f"native attention path is GPT-2-only for now."
            )

        encoder_root = self._extract_encoder_root(host)
        self._assert_rotary_only(encoder_root)
        out: dict[str, np.ndarray] = {}

        out["embeddings.word_embeddings.weight"] = projector.project_embed(
            to_numpy(encoder_root.embeddings.word_embeddings.weight)
        )

        for i, block in enumerate(encoder_root.encoder.layer):
            prefix = f"encoder.layer.{i}"

            # Pre-LN on the attention sublayer's input (HF
            # ``EsmAttention.LayerNorm`` — applied before Q/K/V).
            out[f"{prefix}.attention.LayerNorm.weight"] = projector.project_residual_aligned(
                to_numpy(block.attention.LayerNorm.weight)
            )
            out[f"{prefix}.attention.LayerNorm.bias"] = projector.project_residual_aligned(
                to_numpy(block.attention.LayerNorm.bias)
            )

            # Q/K/V — HF Linear weight is (out, in); in = residual.
            # ESM-2 has biases on Q/K/V (unlike Llama; matches BERT-derivation).
            for qkv in ("query", "key", "value"):
                lin = getattr(block.attention.self, qkv)
                out[f"{prefix}.attention.self.{qkv}.weight"] = projector.project_residual_output(
                    to_numpy(lin.weight)
                )
                if lin.bias is not None:
                    out[f"{prefix}.attention.self.{qkv}.bias"] = to_numpy(lin.bias)

            # attention.output.dense writes the residual stream;
            # out-axis is residual.
            out[f"{prefix}.attention.output.dense.weight"] = projector.project_residual_input(
                to_numpy(block.attention.output.dense.weight)
            )
            out[f"{prefix}.attention.output.dense.bias"] = projector.project_residual_aligned(
                to_numpy(block.attention.output.dense.bias)
            )

            # Pre-LN on the FFN sublayer's input (HF ``EsmLayer.LayerNorm``).
            out[f"{prefix}.LayerNorm.weight"] = projector.project_residual_aligned(
                to_numpy(block.LayerNorm.weight)
            )
            out[f"{prefix}.LayerNorm.bias"] = projector.project_residual_aligned(
                to_numpy(block.LayerNorm.bias)
            )

            # FFN: intermediate reads residual (in-axis projected); output
            # writes residual (out-axis projected). intermediate.bias is in
            # inner-FFN space; output.bias is residual-aligned.
            out[f"{prefix}.intermediate.dense.weight"] = projector.project_residual_output(
                to_numpy(block.intermediate.dense.weight)
            )
            out[f"{prefix}.intermediate.dense.bias"] = to_numpy(
                block.intermediate.dense.bias
            )
            out[f"{prefix}.output.dense.weight"] = projector.project_residual_input(
                to_numpy(block.output.dense.weight)
            )
            out[f"{prefix}.output.dense.bias"] = projector.project_residual_aligned(
                to_numpy(block.output.dense.bias)
            )

        out["encoder.emb_layer_norm_after.weight"] = projector.project_residual_aligned(
            to_numpy(encoder_root.encoder.emb_layer_norm_after.weight)
        )
        out["encoder.emb_layer_norm_after.bias"] = projector.project_residual_aligned(
            to_numpy(encoder_root.encoder.emb_layer_norm_after.bias)
        )

        # d → f buffer for downstream cosine eval. Unlike Whisper-encoder
        # we don't use this inside the forged forward (word_embeddings
        # are projected directly, so the residual enters the encoder
        # already at n_features). The buffer exists so the
        # ``TokenCosineTarget`` can encode the host's d-dim hidden states
        # into the same basis as the forged output without re-deriving
        # ``pinv(W_dec)`` separately. Same matrix shape and population
        # contract as the Whisper adapter's ``basis_encode``.
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
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        return NativeModelConfig(
            family=self.family,
            hidden_size=n_features,
            qkv_inner_size=cfg.num_attention_heads * head_dim,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            head_dim=head_dim,
            intermediate_size=cfg.intermediate_size,
            vocab_size=cfg.vocab_size,
            output_kind="encoder_states",
            max_position_embeddings=cfg.max_position_embeddings,
            activation="gelu",
            attention_width=attention_width,
            n_kv_heads=cfg.num_attention_heads,  # ESM-2 is MHA, not GQA.
            layer_norm_epsilon=float(getattr(cfg, "layer_norm_eps", 1e-5)),
            # ESM-2 always uses theta=10000 (the RotaryEmbedding default in
            # HF modeling_esm.py:92). Pin here so the forged forward
            # reproduces the host's rotation phase exactly.
            rope_theta=10000.0,
        )

    def native_module_class(self) -> type:
        return _get_forged_esm2_class()

    def grad_checkpoint_targets(self, module):
        # ForgedEsm2: every block is at ``module.encoder.layer.{i}``;
        # the input-side parameter is the word-embeddings weight.
        return module.encoder.layer, module.embeddings.word_embeddings.weight

    @staticmethod
    def _extract_encoder_root(host: Any):
        """Return the submodule that owns ``embeddings`` + ``encoder``.

        - :class:`EsmModel` exposes them directly (``host.embeddings``,
          ``host.encoder``).
        - :class:`EsmForMaskedLM` wraps :class:`EsmModel` as
          ``host.esm``; the walk targets that inner module.
        """
        if hasattr(host, "esm") and hasattr(host.esm, "embeddings"):
            return host.esm
        return host

    @staticmethod
    def _assert_rotary_only(encoder_root: Any) -> None:
        """ESM-2 always uses position_embedding_type='rotary'. Pre-RoPE
        ESM-1 ('absolute') is out of scope; surface a clear error rather
        than projecting a non-existent position-embeddings table.
        """
        pe_type = getattr(
            encoder_root.config, "position_embedding_type", "absolute"
        )
        if pe_type != "rotary":
            raise NotImplementedError(
                f"Esm2Adapter only supports position_embedding_type='rotary' "
                f"(every facebook/esm2_t*_*_UR50D checkpoint). Got "
                f"{pe_type!r}. ESM-1 ('absolute') and the 'relative_key'/"
                f"'relative_key_query' variants are not in scope for v1."
            )


# ---------------------------------------------------------------------------
# Native module factory.
# ---------------------------------------------------------------------------


_FORGED_ESM2_CLASS = None


def build_esm2_module(config: "NativeModelConfig"):
    """Construct an ESM-2 native module. Lazy-imports torch."""
    cls = _get_forged_esm2_class()
    return cls(config)


def _get_forged_esm2_class():
    """Return the ForgedEsm2 class (lazy torch import).

    Mirrors HF ``modeling_esm.py`` (transformers 4.49+):

    - Bidirectional attention (no causal mask).
    - Rotary positional embeddings applied to Q and K after projection-
      and-reshape and before scaled dot-product (matches the ``rotary``
      arm of ``EsmSelfAttention.forward``).
    - Pre-LN inside each block: LayerNorm before Q/K/V; LayerNorm before
      the FFN. Residual adds happen AFTER the projection-back, with the
      pre-LayerNorm input as the residual (matches
      ``EsmSelfOutput.forward`` and ``EsmLayer.feed_forward_chunk``).
    - ESM-specific GELU: ``x * 0.5 * (1 + erf(x / sqrt(2)))`` — the
      original ESM repo's formulation. Bit-identical to
      ``F.gelu(approximate='none')`` in practice but pinned for
      reproducibility against the HF reference.
    - Final ``emb_layer_norm_after`` after all blocks.
    - Scaling: ESM scales the query (not the attention logits) by
      ``head_dim ** -0.5``; we replicate that to keep RoPE phases
      consistent with the host.
    """
    global _FORGED_ESM2_CLASS
    if _FORGED_ESM2_CLASS is not None:
        return _FORGED_ESM2_CLASS

    torch = require_extra("torch", "torch")
    import math

    import torch.nn as nn
    import torch.nn.functional as F

    def esm_gelu(x):
        """ESM's GELU. Matches modeling_esm.py:56-60."""
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    class Esm2Embeddings(nn.Module):
        """Word-embeddings only. ESM-2 sets ``emb_layer_norm_before=
        False`` and ``position_embedding_type='rotary'`` so there is
        no LayerNorm or position embedding here. ``token_dropout`` is
        a training-only path — in eval-only forge use it never fires
        unless the caller hand-feeds mask-token IDs.
        """

        def __init__(self, cfg):
            super().__init__()
            self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        def forward(self, input_ids):
            return self.word_embeddings(input_ids)

    class Esm2SelfAttention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.num_heads = cfg.num_heads
            self.head_dim = cfg.head_dim
            self.all_head_size = cfg.num_heads * cfg.head_dim
            # ESM-2 has biases on Q/K/V (BERT-derived). out_proj.dense
            # also has a bias (handled by Esm2SelfOutput).
            self.query = nn.Linear(cfg.hidden_size, self.all_head_size, bias=True)
            self.key = nn.Linear(cfg.hidden_size, self.all_head_size, bias=True)
            self.value = nn.Linear(cfg.hidden_size, self.all_head_size, bias=True)
            self.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))

        def forward(self, x):
            B, T, _ = x.shape
            q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            # ESM scales the query (not the logits) by 1/sqrt(head_dim).
            # See modeling_esm.py:359-363 — this is load-bearing for RoPE
            # phase correctness.
            q = q * (self.head_dim ** -0.5)

            from saeforge._positional.rope import (
                apply_rotary_pos_emb,
                compute_rope_cache,
            )

            cos, sin = compute_rope_cache(
                T, self.head_dim, theta=self.rope_theta, device=q.device, dtype=q.dtype
            )
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Bidirectional — no causal mask. Logits already pre-scaled
            # via the q multiplier above; the eager_attention_forward
            # path uses scaling=1.0 (see modeling_esm.py:334).
            scores = q @ k.transpose(-2, -1)
            # fp32 softmax for numerical stability (matches HF
            # eager_attention_forward).
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.all_head_size)
            return out

    class Esm2SelfOutput(nn.Module):
        """Maps attention output back to residual width and adds residual."""

        def __init__(self, cfg):
            super().__init__()
            self.dense = nn.Linear(
                cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=True
            )

        def forward(self, hidden_states, input_tensor):
            return self.dense(hidden_states) + input_tensor

    class Esm2Attention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.self = Esm2SelfAttention(cfg)
            self.output = Esm2SelfOutput(cfg)
            self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)

        def forward(self, x):
            x_ln = self.LayerNorm(x)
            attn_out = self.self(x_ln)
            return self.output(attn_out, x)

    class Esm2Intermediate(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)

        def forward(self, x):
            return esm_gelu(self.dense(x))

    class Esm2Output(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

        def forward(self, hidden_states, input_tensor):
            return self.dense(hidden_states) + input_tensor

    class Esm2Layer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.attention = Esm2Attention(cfg)
            self.intermediate = Esm2Intermediate(cfg)
            self.output = Esm2Output(cfg)
            self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)

        def forward(self, x):
            attn_out = self.attention(x)
            # FFN sublayer: pre-LN → intermediate(GELU) → output(+ residual).
            # Residual is attn_out (pre-LN input), matching HF's
            # feed_forward_chunk.
            x_ln = self.LayerNorm(attn_out)
            inter = self.intermediate(x_ln)
            return self.output(inter, attn_out)

    class Esm2Encoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layer = nn.ModuleList([Esm2Layer(cfg) for _ in range(cfg.num_layers)])
            self.emb_layer_norm_after = nn.LayerNorm(
                cfg.hidden_size, eps=cfg.layer_norm_epsilon
            )

        def forward(self, x):
            for layer in self.layer:
                x = layer(x)
            return self.emb_layer_norm_after(x)

    class ForgedEsm2(nn.Module):
        """Forged ESM-2 encoder.

        Every parameter slot matches a key emitted by
        :meth:`Esm2Adapter.walk`. ``load_state_dict(strict=True)``
        accepts the walk output in one pass.

        Forward: ``(B, T)`` of token IDs → ``(B, T, n_features)`` of
        residue-level hidden states.
        """

        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.embeddings = Esm2Embeddings(cfg)
            self.encoder = Esm2Encoder(cfg)
            # d → f projection buffer; populated by from_projected_weights
            # from the walk's ``basis_encode`` entry. Used only by the
            # cosine-faithfulness path (TokenCosineTarget) — the forge
            # itself does not consult this buffer at forward time
            # because the residual stream is already at n_features after
            # the projected ``word_embeddings`` lookup. The default-init
            # zero matrix collapses any unwise encode-via-buffer to
            # zero, surfacing a missing-walk-key bug visibly. Inferring
            # d_head from cfg keeps the buffer shape correct under any
            # forging configuration.
            d = cfg.num_heads * cfg.head_dim
            self.register_buffer(
                "basis_encode",
                torch.zeros(d, cfg.hidden_size),
            )

        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            x = self.encoder(x)
            return x

    _FORGED_ESM2_CLASS = ForgedEsm2
    return ForgedEsm2


# Register at module-import time. The HF classes are lazy-loaded so importing
# this module without ``[torch]`` doesn't crash the package.
try:
    from transformers import EsmForMaskedLM, EsmModel

    from saeforge.adapters import register_adapter

    _adapter = Esm2Adapter()
    register_adapter(EsmModel, _adapter)
    register_adapter(EsmForMaskedLM, _adapter)
except ImportError:  # pragma: no cover
    pass
