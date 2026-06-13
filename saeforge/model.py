"""NativeModel — small transformer whose residual width equals the feature-basis size.

Why an in-tree implementation: a v0 forged model has residual width ``n_features``
but inherits the host's attention internal width (``n_heads * head_dim``) and MLP
inner width. Those don't generally factor as ``n_features = n_heads * head_dim``,
so stock ``GPT2LMHeadModel`` config-driven shapes don't apply. The minimal nn.Module
below preserves the host's internal widths and projects only the residual-touching
edges of every block.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from saeforge.projector import SubspaceProjector
from saeforge.utils.lazy import require_extra


_SUPPORTED_OUTPUT_KINDS = ("logits", "encoder_states")

# Canonical bundled architecture families. Static so that
# ``NativeModelConfig.__post_init__`` accepts any of these names even
# when the corresponding adapter failed to register (e.g.
# ``transformers`` is not installed, or the installed version is too
# old for ``qwen3``/``qwen3_moe``). Runtime dispatch
# (``_build_torch_module``, ``_default_target_for``) still requires an
# actually-registered adapter and surfaces a clear error otherwise.
_SUPPORTED_FAMILIES = (
    "gpt2",
    "gpt_neox",
    "llama",
    "gemma2",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "whisper_encoder",
    "esm2",
)

# Families whose forged module returns encoder-state hidden vectors
# (output_kind="encoder_states") rather than vocab logits. The
# NativeModelConfig validator allows ``vocab_size`` to be informative
# (for token-embedding sizing) but rejects a logits head.
_ENCODER_STATES_FAMILIES = frozenset({"whisper_encoder", "esm2"})


def _supported_families() -> tuple[str, ...]:
    """Sorted union of bundled families and any third-party
    registrations. Third-party adapters that register before config
    construction (the usual pattern: register at module import) widen
    the accepted set; bundled families are accepted unconditionally so
    config construction works on a base install without
    ``transformers``."""
    from saeforge.adapters import registered_families

    return tuple(sorted(set(_SUPPORTED_FAMILIES) | registered_families()))


@dataclass
class NativeModelConfig:
    """Architecture knobs for a forged native model.

    ``hidden_size`` is fixed by the feature basis (``basis.n_features``).
    ``qkv_inner_size`` and ``intermediate_size`` are inherited from the host
    when ``attention_width == "host"``. When ``attention_width ==
    "feature_native"`` (v0.2 opt-in), ``qkv_inner_size`` MUST equal
    ``hidden_size`` and ``num_heads * head_dim == hidden_size``.

    ``family`` (required, no default) selects the native module shape:

    - ``"gpt2"`` — Conv1D matrices, GeLU, LayerNorm, ``wpe`` position
      embeddings, fused ``c_attn`` for Q/K/V.
    - ``"llama"`` — Linear matrices, SwiGLU MLP (gate/up/down), RMSNorm,
      no ``wpe``, separate ``q_proj``/``k_proj``/``v_proj``/``o_proj``
      with optional GQA (``n_kv_heads``), optional tied lm_head.
    - ``"gemma2"`` — Llama-shaped + four norms per block
      (``input_layernorm``, ``post_attention_layernorm``,
      ``pre_feedforward_layernorm``, ``post_feedforward_layernorm``)
      and optional logit soft-capping post-``lm_head``.
    - ``"whisper_encoder"`` — non-causal audio encoder. Conv1d mel stem
      (frozen-copied, not projected), sinusoidal positional embeddings,
      Linear MHA q/k/v/o, GELU MLP (fc1/fc2). No ``lm_head``;
      ``output_kind`` is ``"encoder_states"`` and ``vocab_size`` is 0.
    """

    family: str
    hidden_size: int
    qkv_inner_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int = 0
    # ``"logits"`` (default — every LM family) produces a vocab-shaped head.
    # ``"encoder_states"`` (whisper_encoder) returns per-frame hidden states
    # and forbids a vocab head. Cross-constraints enforced in ``__post_init__``.
    output_kind: str = "logits"
    max_position_embeddings: int = 1024
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu"
    attention_width: str = "host"
    # Llama / Gemma-2 fields. ``n_kv_heads=None`` collapses to MHA where
    # ``n_kv_heads == num_heads``. ``rms_norm_eps`` mirrors HF's
    # ``LlamaConfig.rms_norm_eps`` / ``Gemma2Config.rms_norm_eps``.
    n_kv_heads: int | None = None
    tied_embeddings: bool = False
    rms_norm_eps: float | None = None
    # Gemma-2-specific. Applied as ``tanh(x / cap) * cap`` post-``lm_head``
    # (``final_logit_softcap``) and post-attention scores
    # (``attn_logit_softcap``); ``None`` is a no-op.
    final_logit_softcap: float | None = None
    attn_logit_softcap: float | None = None
    # Qwen2-specific (Llama-shaped but with Q/K/V biases). False for Llama
    # and Gemma-2 (no biases on attention projections). The o_proj remains
    # bias-free across all families.
    qkv_bias: bool = False
    # Qwen3-specific. When True, the attention block applies RMSNorm(head_dim)
    # on Q and K per head between projection-and-reshape and the scaled dot-
    # product. Llama / Gemma-2 / Qwen2 default to False (no q_norm/k_norm).
    qk_norm: bool = False
    # Qwen3-MoE configuration. num_experts == 0 -> dense MLP path (existing
    # behavior for every other family). num_experts > 0 -> Mixtral-style
    # sparse routing: gate + N experts × SwiGLU, with top-K dispatch.
    # See ``saeforge.adapters.qwen3_moe`` and the qwen3-moe-support spec.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    # Hybrid-bridge knobs. When ``bridges=True``, the native module
    # constructs and registers two ``BridgeModule`` instances on the
    # forward path between the embed/mid and mid/lm-head regions.
    # See ``saeforge.bridges`` and the ``hybrid-bridge-forge`` capability spec.
    # v1 ships GPT-2 only; Llama/Gemma-2 honor ``bridges=False`` and ignore
    # the related knobs (will raise when ``bridges=True`` until T3 lands).
    bridges: bool = False
    bridge_init: str = "orthogonal"
    bridge_nonlin: str = "none"
    bridge_pre_layernorm: bool = True
    # Forward-mode dispatch. ``"auto"`` (default) selects
    # ``"native_in_basis"`` for good/saturated quality tiers and
    # ``"host_wrapped"`` for undersized/degenerate. The latter wraps
    # host's exact transformer blocks with decode/encode at every block
    # boundary — avoids the rank-dependent amplification documented in
    # add-host-wrapped-forge-fallback at the cost of host-equal compute.
    forward_mode: str = "auto"
    # Llama-family rotary positional embedding. ``rope_mode="standard"``
    # (default) applies RoPE to Q and K after the projection-and-reshape
    # in every Llama-family forged attention block; ``rope_mode="none"``
    # skips the rotation entirely and reproduces the pre-fix (buggy)
    # behaviour byte-identically. The "none" arm exists as a regression-
    # diff knob and emits a UserWarning when set on Llama-family configs.
    # GPT-2 / Whisper-encoder ignore these fields. See
    # openspec/changes/add-llama-family-rope/proposal.md.
    rope_mode: str = "standard"
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 1.0

    def __post_init__(self) -> None:
        # Validate against the union of bundled families and runtime-
        # registered third-party adapters. Bundled family names are
        # static facts about sae-forge — accepted even when their
        # adapter failed to register (e.g. base install without the
        # ``[torch]`` extra, where ``transformers`` is unavailable and
        # every adapter's ``register_adapter`` call is short-circuited
        # by the ImportError guard). Runtime dispatch
        # (``_build_torch_module``, ``_default_target_for``) still
        # requires an actually-registered adapter and surfaces a
        # different error.
        supported = _supported_families()
        if self.family not in supported:
            raise ValueError(
                f"family must be one of {supported}; "
                f"got {self.family!r}"
            )
        if self.output_kind not in _SUPPORTED_OUTPUT_KINDS:
            raise ValueError(
                f"output_kind must be one of {_SUPPORTED_OUTPUT_KINDS}; "
                f"got {self.output_kind!r}"
            )
        # Cross-constraints between family / output_kind / vocab_size.
        # Encoder-states families (whisper_encoder, esm2) emit per-token
        # hidden states instead of vocab logits; LM families require a
        # vocab head. Within encoder-states, whisper has no word
        # embeddings (conv-stem input) and pins vocab_size to 0; esm2
        # has a word-embedding table sized to the amino-acid vocab.
        if self.family == "whisper_encoder":
            if self.output_kind != "encoder_states":
                raise ValueError(
                    f"family='whisper_encoder' requires "
                    f"output_kind='encoder_states'; got {self.output_kind!r}"
                )
        if self.family == "esm2":
            if self.output_kind != "encoder_states":
                raise ValueError(
                    f"family='esm2' requires output_kind='encoder_states' "
                    f"in v1; got {self.output_kind!r}. Logit-space forging "
                    f"on the MLM head is a future extension."
                )
        if self.output_kind == "logits" and self.vocab_size <= 0:
            raise ValueError(
                f"output_kind='logits' requires vocab_size > 0; "
                f"got vocab_size={self.vocab_size}"
            )
        if self.output_kind == "encoder_states":
            if self.family not in _ENCODER_STATES_FAMILIES:
                raise ValueError(
                    f"output_kind='encoder_states' is only valid for "
                    f"families {sorted(_ENCODER_STATES_FAMILIES)!r}; "
                    f"got family={self.family!r}"
                )
            if self.family == "whisper_encoder" and self.vocab_size != 0:
                raise ValueError(
                    f"family='whisper_encoder' requires vocab_size == 0 "
                    f"(no word-embedding table — input is mel features); "
                    f"got vocab_size={self.vocab_size}"
                )
            if self.family == "esm2" and self.vocab_size <= 0:
                raise ValueError(
                    f"family='esm2' requires vocab_size > 0 (sizes the "
                    f"word-embedding table); got vocab_size={self.vocab_size}"
                )
        if self.attention_width not in ("host", "feature_native"):
            raise ValueError(
                f"attention_width must be 'host' or 'feature_native'; got {self.attention_width!r}"
            )
        if self.forward_mode not in ("auto", "native_in_basis", "host_wrapped"):
            raise ValueError(
                f"forward_mode must be one of "
                f"('auto', 'native_in_basis', 'host_wrapped'); "
                f"got {self.forward_mode!r}"
            )
        if self.rope_mode not in ("standard", "none"):
            raise ValueError(
                f"rope_mode must be one of ('standard', 'none'); "
                f"got {self.rope_mode!r}"
            )
        # Llama-family + rope_mode="none" reproduces the pre-fix buggy
        # behaviour byte-identically. Surface a warning so nobody ships
        # it accidentally. add-llama-family-rope/proposal.md documents
        # this as the regression-diff knob.
        _llama_family = ("llama", "gemma2", "qwen2", "qwen3", "qwen3_moe")
        if self.rope_mode == "none" and self.family in _llama_family:
            import warnings

            warnings.warn(
                f"NativeModelConfig: rope_mode='none' on family={self.family!r} "
                f"reproduces the pre-fix Llama-family no-RoPE behaviour from "
                f"before add-llama-family-rope. This is a regression-diff "
                f"knob — Llama-family forges should use rope_mode='standard' "
                f"(the default) in production. See "
                f"openspec/specs/architecture-adapters/spec.md for the "
                f"per-family positional-encoding contract; the smoke gate "
                f"and proposal context live under openspec/changes/"
                f"add-llama-family-rope/ until archived, then under "
                f"openspec/changes/archive/<date>-add-llama-family-rope/.",
                UserWarning,
                stacklevel=2,
            )
        if self.num_heads * self.head_dim != self.qkv_inner_size:
            raise ValueError(
                f"qkv_inner_size {self.qkv_inner_size} must equal "
                f"num_heads ({self.num_heads}) * head_dim ({self.head_dim})"
            )
        # MoE validation. num_experts > 0 requires the top-K and per-expert
        # FF width fields to be coherent. num_experts == 0 is the dense
        # default and ignores the other MoE fields.
        if self.num_experts > 0:
            if self.num_experts_per_tok <= 0:
                raise ValueError(
                    f"num_experts={self.num_experts} > 0 requires "
                    f"num_experts_per_tok > 0; got "
                    f"num_experts_per_tok={self.num_experts_per_tok}"
                )
            if self.num_experts_per_tok > self.num_experts:
                raise ValueError(
                    f"num_experts_per_tok ({self.num_experts_per_tok}) "
                    f"must be <= num_experts ({self.num_experts})"
                )
            if self.moe_intermediate_size <= 0:
                raise ValueError(
                    f"num_experts={self.num_experts} > 0 requires "
                    f"moe_intermediate_size > 0; got "
                    f"moe_intermediate_size={self.moe_intermediate_size}"
                )
        # GQA: default n_kv_heads = num_heads (collapses to MHA).
        if self.n_kv_heads is None:
            self.n_kv_heads = self.num_heads
        if self.n_kv_heads <= 0:
            raise ValueError(
                f"n_kv_heads must be > 0; got {self.n_kv_heads}"
            )
        if self.num_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.attention_width == "feature_native":
            if self.qkv_inner_size != self.hidden_size:
                raise ValueError(
                    f"feature-native attention requires qkv_inner_size "
                    f"({self.qkv_inner_size}) to equal hidden_size "
                    f"({self.hidden_size})"
                )
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    f"feature-native attention requires hidden_size "
                    f"({self.hidden_size}) to be divisible by num_heads "
                    f"({self.num_heads}); set num_heads to a divisor of "
                    f"hidden_size or pad the basis"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> NativeModelConfig:
        # Tolerate older serialised configs that pre-date fields added
        # in later versions. New optional fields land with explicit
        # defaults; unknown keys raise as before.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"NativeModelConfig.from_dict: unknown fields {sorted(unknown)}; "
                f"known fields: {sorted(known)}"
            )
        return cls(**payload)


def _build_torch_module(config: NativeModelConfig):
    """Construct the torch nn.Module skeleton. Lazy-imports torch.

    Dispatches on ``config.family`` via
    :func:`saeforge.adapters.adapter_for_family`; the adapter's
    :meth:`native_module_class` returns the family-specific
    ``nn.Module`` subclass which is then instantiated with
    ``config``. Before the world-model-protocol refactor this was an
    explicit ``if/elif`` family tree; the dispatch is now registry-
    backed so a new architecture is one adapter file plus a
    ``register_adapter`` call.
    """
    from saeforge.adapters import adapter_for_family

    cls = adapter_for_family(config.family).native_module_class()
    return cls(config)



class NativeModel:
    """Forged transformer with a feature-basis-width residual stream."""

    def __init__(self, config: NativeModelConfig):
        self.config = config
        self._module = _build_torch_module(config)
        # Set by from_host when constructing via adapter dispatch;
        # remains None for direct construction (from_projected_weights
        # path, or test fixtures). Consumers that care SHOULD set this
        # explicitly via the from_host kwarg.
        self.resolved_forward_mode: str | None = None

    @property
    def torch_module(self):
        return self._module

    def parameters(self):
        return self._module.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._module.parameters())

    def forward(self, input_ids):
        return self._module(input_ids)

    @classmethod
    def from_host(
        cls,
        host_model_id: str,
        projector: SubspaceProjector,
        *,
        dtype: str = "float32",
        device: str = "cpu",
        forward_mode: str = "auto",
    ) -> NativeModel:
        """Construct a native model by projecting ``host_model_id``'s weights through ``projector``.

        The host is loaded via ``AutoModelForCausalLM`` so the actual
        architecture (GPT-2, Llama, Gemma-2, …) drives adapter dispatch.
        Loading a non-GPT-2 host as ``GPT2LMHeadModel`` (the v0.1
        behaviour) silently produced randomly-initialised weights —
        that footgun is gone after multi-architecture-support.

        ``forward_mode`` accepts ``"auto"`` (the default, dispatches by
        basis quality tier), ``"native_in_basis"`` (the existing forward
        path), or ``"host_wrapped"`` (the under-complete-basis fallback
        that wraps host's exact transformer with decode/encode at every
        block boundary). See ``saeforge.forward_mode.resolve_forward_mode``
        for the dispatch contract and
        ``openspec/changes/add-host-wrapped-forge-fallback`` for the
        falsifiable acceptance gate.
        """
        # Surface the missing-[torch] message early — load_host_for_forge
        # also lazy-imports transformers but the helper's ImportError
        # comes from a deeper stack, so an upfront require_extra call
        # gives a cleaner failure for users without the extra.
        require_extra("transformers", "torch")

        from saeforge.adapters import adapter_for
        from saeforge.forward_mode import resolve_forward_mode
        from saeforge.utils.host_loader import load_host_for_forge

        host = load_host_for_forge(host_model_id)
        adapter = adapter_for(host)
        resolved = resolve_forward_mode(projector.basis, forward_mode)

        if resolved == "host_wrapped":
            # Bypass project_module — host weights stay unprojected; the
            # adapter constructs a wrapper module that holds a frozen
            # reference to the host and the basis matrices.
            torch_module = adapter.host_wrapped_module(
                host, projector.basis, scale_boost=projector.scale_boost
            )
            model = cls.__new__(cls)
            # Build a config so .config inspection works downstream; we
            # do NOT use it to build the module (host_wrapped_module
            # returned the live module already).
            config = adapter.build_native_config(host, projector.basis.n_features)
            config.forward_mode = "host_wrapped"
            model.config = config
            model._module = torch_module
            model.resolved_forward_mode = "host_wrapped"
            model._move(dtype=dtype, device=device)
            return model

        weights = projector.project_module(host)
        config = adapter.build_native_config(host, projector.basis.n_features)
        config.forward_mode = forward_mode if forward_mode != "auto" else "auto"
        model = cls.from_projected_weights(config, weights)
        model.resolved_forward_mode = "native_in_basis"
        model._move(dtype=dtype, device=device)
        return model

    @classmethod
    def from_projected_weights(
        cls,
        config: NativeModelConfig,
        weights: dict[str, np.ndarray],
    ) -> NativeModel:
        """Assemble a native model from a dict of pre-projected ``np.ndarray`` weights."""
        torch = require_extra("torch", "torch")
        model = cls(config)
        state = model._module.state_dict()
        for name, arr in weights.items():
            target = name
            # HF GPT2's lm_head linear stores weight as (vocab, hidden), matching our key
            if target not in state:
                raise KeyError(f"projected key {name!r} has no slot in NativeModel state_dict")
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(state[target].dtype)
            if tensor.shape != state[target].shape:
                raise ValueError(
                    f"shape mismatch for {target}: projected {tensor.shape}, "
                    f"expected {tuple(state[target].shape)}"
                )
            state[target] = tensor
        model._module.load_state_dict(state)
        return model

    def _move(self, dtype: str, device: str) -> None:
        torch = require_extra("torch", "torch")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}; choose from {list(dtype_map)}")
        self._module.to(dtype=dtype_map[dtype], device=device)

    def save_pretrained(self, output_dir: str | Path) -> None:
        from safetensors.torch import save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        # host_wrapped: the forged module holds an unprojected host model
        # plus W_dec/pinv buffers. Saving the full host duplicates state
        # already on disk in HF caches. Persist only what's *new* —
        # the basis matrices — so reloading rebuilds via from_host. Note
        # the basis itself is on disk at the polygram checkpoint passed
        # to the pipeline; we save buffers here for self-contained reload
        # by tools that only see the forge output directory.
        if self.resolved_forward_mode == "host_wrapped":
            buffers = {
                "W_dec": self._module.W_dec.contiguous(),
                "pinv": self._module.pinv.contiguous(),
            }
            save_file(buffers, str(output_dir / "host_wrapped_buffers.safetensors"))
            return
        state = {k: v.contiguous() for k, v in self._module.state_dict().items()}
        if self.config.tied_embeddings:
            # ``lm_head.weight`` aliases ``model.embed_tokens.weight`` post-
            # construction (see ForgedLlama.__init__). safetensors refuses
            # shared-storage tensors on save; drop the alias so the file
            # contains a single copy of the embedding. ``load_pretrained``
            # rebuilds the alias via the constructor.
            state.pop("lm_head.weight", None)
        save_file(state, str(output_dir / "model.safetensors"))

    @classmethod
    def load_pretrained(cls, input_dir: str | Path) -> NativeModel:
        from safetensors.torch import load_file

        input_dir = Path(input_dir)
        config = NativeModelConfig.from_dict(json.loads((input_dir / "config.json").read_text()))
        # The constructor already aliases lm_head.weight to embed_tokens.weight
        # when tied_embeddings is True (no explicit slot present in the
        # safetensors file in that case).
        model = cls(config)
        state = load_file(str(input_dir / "model.safetensors"))
        # When tied, the saved state_dict legitimately omits lm_head.weight.
        # load_state_dict's ``strict=True`` would raise on the missing slot;
        # relax it for tied models so the alias survives.
        strict = not config.tied_embeddings
        model._module.load_state_dict(state, strict=strict)
        return model


def _config_from_host(
    host_model, n_features: int, *, attention_width: str = "host"
) -> NativeModelConfig:
    """Pull the host's per-block dimensions into a ``NativeModelConfig``.

    Dispatches via :mod:`saeforge.adapters`; the adapter for the host's
    class produces the matching ``family``-stamped config. Kept as a
    module-level function for downstream imports; new code should
    prefer ``adapter_for(host).build_native_config(host, n_features)``
    directly.
    """
    from saeforge.adapters import adapter_for

    return adapter_for(host_model).build_native_config(
        host_model, n_features, attention_width=attention_width
    )


