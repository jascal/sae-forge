"""ForgePipeline — orchestrate basis load -> projection -> native model -> faithfulness eval."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.bridges import BridgeConfig
from saeforge.eval.faithfulness import FaithfulnessTarget
from saeforge.hybrid_basis import HybridBasisBundle
from saeforge.model import NativeModel, _config_from_host
from saeforge.projector import SubspaceProjector


def _torch_dtype(name: str):
    """Map our string dtype names to torch.dtype for ``from_pretrained``."""
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


class ForgeFailed(RuntimeError):
    """The FSM ended in a ``failed`` state and the run could not produce
    a valid forged model.

    Pre-fix, an action raising inside the FSM (e.g. ``AttributeError``
    from the GPT-2-only grad-checkpointing path running against a
    ForgedLlama) was swallowed into ``final_state: failed`` and
    returned to the caller as a ``ForgeResult`` with ``n_params=0`` and
    ``faithfulness=0.0`` — exit code 0 with no exception. Callers
    had no signal that anything went wrong. The FSM dispatch now
    raises this exception when the run ends in ``failed``.

    Attributes:
        error_message: The ``log_error`` action's recorded message.
        transitions_log: The full FSM transitions log for debugging.
        extras: The unfinished ``ForgeResult.extras`` dict (n_params,
            faithfulness, etc. as far as the FSM got).
    """

    def __init__(
        self,
        error_message: str,
        *,
        transitions_log: list,
        extras: dict,
    ) -> None:
        super().__init__(error_message)
        self.error_message = error_message
        self.transitions_log = transitions_log
        self.extras = extras


def _raise_if_failed(final: dict) -> None:
    """Raise :class:`ForgeFailed` when the FSM ended in ``failed``.

    Inspects the transitions log for the last action; if it's
    ``log_error``, surfaces the recorded ``error_message`` so silent
    KL=0.0 returns can't recur.
    """
    if _final_state_for_log(final) != "failed":
        return
    msg = final.get("error_message") or "FSM ended in failed state"
    raise ForgeFailed(
        msg,
        transitions_log=final.get("transitions_log", []),
        extras={
            "n_params": final.get("n_params", 0),
            "faithfulness": final.get("faithfulness"),
            "n_features": final.get("current_feature_count"),
        },
    )

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from polygram import (  # noqa: F401
        CompressionConfig,
        EpochCompressionConfig,
        RegrowConfig,
    )
    from saeforge.training.task_stream import TaskStream  # noqa: F401


_FAITHFULNESS_KL_DEPRECATION_MSG = (
    "ForgeResult.faithfulness_kl is deprecated in favour of the generic "
    "ForgeResult.faithfulness (with ForgeResult.faithfulness_target_name "
    "naming the active scorer). The property returns the value only when "
    "the active target is 'kl' and will be removed one minor version after "
    "the pluggable-faithfulness change lands."
)


@dataclass
class ForgeResult:
    """Structured output of a ``ForgePipeline.run`` call.

    ``faithfulness`` is the active target's score (KL by default for LM
    hosts, cosine for whisper_encoder, or a user-supplied scorer when
    ``ForgePipeline(faithfulness=...)`` is set). ``faithfulness_target_name``
    names the scorer so downstream consumers can match without re-deriving
    it from ``host_model_id``.

    ``faithfulness_kl`` is a deprecated alias kept for one minor version:
    it returns ``faithfulness`` when the active target is ``"kl"`` and
    ``None`` otherwise, emitting a :class:`DeprecationWarning` on read.
    Constructing with ``faithfulness_kl=`` is also accepted and forwards
    to ``faithfulness`` / ``faithfulness_target_name = "kl"`` (with the
    same warning).
    """

    model: NativeModel | None
    output_dir: Path
    n_params: int = 0
    faithfulness: float | None = None
    faithfulness_target_name: str | None = None
    extras: dict = field(default_factory=dict)

    def __init__(
        self,
        model: NativeModel | None,
        output_dir: Path,
        n_params: int = 0,
        faithfulness: float | None = None,
        faithfulness_target_name: str | None = None,
        extras: dict | None = None,
        *,
        faithfulness_kl: float | None = None,
    ) -> None:
        if faithfulness_kl is not None:
            warnings.warn(
                _FAITHFULNESS_KL_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            if faithfulness is None:
                faithfulness = faithfulness_kl
            if faithfulness_target_name is None:
                faithfulness_target_name = "kl"
        self.model = model
        self.output_dir = output_dir
        self.n_params = n_params
        self.faithfulness = faithfulness
        self.faithfulness_target_name = faithfulness_target_name
        self.extras = extras if extras is not None else {}

    @property
    def faithfulness_kl(self) -> float | None:
        """Deprecated. Use :attr:`faithfulness` and
        :attr:`faithfulness_target_name`.

        Returns :attr:`faithfulness` when the active target is ``"kl"``,
        otherwise ``None``. Always emits :class:`DeprecationWarning`.
        Removed one minor version after the pluggable-faithfulness
        change lands.
        """
        warnings.warn(
            _FAITHFULNESS_KL_DEPRECATION_MSG,
            DeprecationWarning,
            stacklevel=2,
        )
        if self.faithfulness_target_name == "kl":
            return self.faithfulness
        return None

    @faithfulness_kl.setter
    def faithfulness_kl(self, value: float | None) -> None:
        warnings.warn(
            _FAITHFULNESS_KL_DEPRECATION_MSG,
            DeprecationWarning,
            stacklevel=2,
        )
        self.faithfulness = value
        if value is not None and self.faithfulness_target_name is None:
            self.faithfulness_target_name = "kl"


@dataclass
class ForgePipeline:
    """End-to-end forging pipeline.

    Two orchestrators ship in v0.1:

    - ``imperative`` (default): straight-line load -> project -> assemble ->
      eval -> save. The v0 stable surface.
    - ``fsm``: orca-runtime-python FSM driving the same stages plus optional
      regrow / fine-tune passes. Requires the ``[orca]`` extra; topology is
      defined as a three-machine hierarchy under ``saeforge/machines/``
      (``stream.orca.md`` / ``refine.orca.md`` / ``basis.orca.md``).

    Both orchestrators MUST produce byte-identical forged weights for the
    same inputs and seeds — that equivalence is the migration safety net
    from v0.1 (imperative-default) to v0.2 (fsm-default).
    """

    basis: FeatureBasis
    projector: SubspaceProjector
    host_model_id: str | None = None
    eval_prompts: list[str] = field(default_factory=list)
    # v0.4 forge-whisper-encoder: audio-side eval input for the
    # whisper_encoder family. When set, evaluate_faithfulness dispatches
    # to cosine_faithfulness using these mel features and the forged
    # encoder. Mutually exclusive with eval_prompts (validation in
    # __post_init__) — a pipeline targets either text-LM or audio
    # faithfulness, not both. The torch.Tensor type is left as Any in
    # the annotation so the dataclass keeps working without torch
    # installed; the runtime contract is (batch, n_mels, n_frames).
    eval_audio_features: Any | None = None
    # Optional pre-captured host encoder states for the
    # whisper_encoder family. When set, evaluate_faithfulness skips
    # the host forward inside the FSM and uses these states directly
    # — the audio-side analog of pre-tokenised ``_eval_input_ids``.
    # Shape contract: (batch, n_frames, d_model). Caller is responsible
    # for ensuring this is the output of host.encoder(eval_audio_features).
    eval_encoder_states: Any | None = None
    dtype: str = "float32"
    device: str = "cpu"
    orchestrator: str = "imperative"
    iterations: int = 1
    regrow_count: int = 0
    quantum_aware: bool = False
    validation_report_path: str | None = None
    # Polygram tuning bundles (polygram>=0.1.0). Each is plumbed
    # end-to-end via FSM context — serialised on the way in
    # (``cfg.to_dict()`` → ``ctx[key]``) and reconstituted on the way
    # out (action: ``<Config>.from_dict(ctx[key])``). When a field is
    # ``None`` the corresponding polygram constructor is called without
    # ``config=``, so polygram's own defaults apply.
    #
    # ``compression``        → polygram.Compressor
    #     (strategy / rep_selection / merge_mode / confirmer)
    # ``epoch_compression``  → polygram.EpochCompressor
    #     (coverage_target, cosine_threshold, max_iterations,
    #      n_visits_per_feature, embedded ValidationConfig)
    # ``regrow``             → polygram.Regrower.from_compression_report
    #     (model_name, layer, strategy, prompts, seed, n_init, device)
    compression: "CompressionConfig | None" = None
    epoch_compression: "EpochCompressionConfig | None" = None
    regrow: "RegrowConfig | None" = None
    finetune_steps: int = 0
    finetune_lr: float = 1e-3
    attention_width: str = "host"

    # v0.3 forge-finetune-recipe knobs. Default to v0.1 behaviour (no recipe
    # path activated) when finetune_corpus is None; opt in by setting it.
    finetune_corpus: str | Path | None = None
    finetune_total_steps: int = 1000
    finetune_warmup_steps: int = 100
    finetune_peak_lr: float = 5e-5
    finetune_batch_size: int = 8
    finetune_seq_len: int = 512
    finetune_precision: str = "fp32"
    finetune_grad_checkpoint: bool = False
    finetune_eval_every: int = 100
    finetune_save_every: int = 250
    finetune_save_dir: Path | None = None
    finetune_log_every: int = 10
    # add-host-distillation-finetune-loss. Defaults (alpha=1.0,
    # tau=2.0) are byte-identical to the pre-change loss; opt into
    # KD by setting alpha < 1.0.
    finetune_distill_alpha: float = 1.0
    finetune_distill_temperature: float = 2.0

    # v0.4 forge-continual-learning-loop knobs. All default to values that
    # recover v0.1 single-shard byte-identical behavior.
    n_tasks: int = 1
    task_trigger: str = "labeled"
    token_budget_per_task: int = 0
    loss_delta_threshold: float = 0.0
    inner_refine_passes: int = 1
    protect_top_k: int = 0
    protect_score: str = "mean_act"
    activation_buffer_size: int = 4096
    replay_ratio: float = 0.0
    replay_buffer_size: int = 0
    replay_policy: str = "reservoir"
    task_stream: "TaskStream | None" = None  # forward ref; resolved lazily

    # v0.5 hybrid-bridge-forge knobs. All default to v0.4 single-basis behavior
    # (no extra bases, no bridges). See
    # ``openspec/changes/hybrid-bridge-forge`` for the full design.
    hybrid_bridge: bool = False
    basis_embed: FeatureBasis | None = None
    basis_lm_head: FeatureBasis | None = None
    bridge_config: BridgeConfig = field(default_factory=BridgeConfig)

    # v0.6 qwen3-moe-support knobs. Control how a Qwen3-MoE host is forged:
    # - "preserve" (default): per-expert projection; full fidelity
    # - "collapse": average experts into a single dense MLP per layer
    # - "top_n": v1 placeholder, raises NotImplementedError pointing at
    #   the moe-expert-calibration follow-up
    moe_strategy: str = "preserve"
    moe_keep_n: int = 0

    # v0.5 adaptive-regrow knobs. All default to values that recover the
    # v0.2 fixed-regrow behavior. The master toggle is ``adaptive_regrow``;
    # when False, the other three are inert (silently ignored). When True,
    # ``regrow_max > regrow_count`` and ``n_features_target > 0`` are
    # required (validated in ``__post_init__``).
    adaptive_regrow: bool = False
    regrow_max: int = 0
    n_features_target: int = 0
    regrow_damping: float = 0.5

    # pluggable-faithfulness. ``None`` keeps the v0.4 family-dispatch
    # default (KLTarget for LM hosts, CosineTarget for whisper_encoder),
    # which is byte-identical to v0.4. Set to a ``FaithfulnessTarget``
    # instance to override; the instance flows through the imperative
    # path and the FSM ctx (as ``_faithfulness_target``) identically.
    faithfulness: FaithfulnessTarget | None = None

    # ----------------------------------------------------------------
    # Construction-time validation + dict round-trip
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
        # ``regrow_count > 0`` requires an explicit RegrowConfig — the
        # pre-change ``layer=10`` / ``model_name="gpt2"`` ctx fallbacks
        # silently bound regrowth to GPT-2 and produced nonsense layer
        # indices on other architectures. Polygram-tuning-config dropped
        # the matching polygram-side defaults; mirroring that here at
        # construction time keeps the failure mode loud and local.
        if self.regrow_count > 0 and self.regrow is None:
            raise ValueError(
                f"ForgePipeline: regrow_count={self.regrow_count} > 0 "
                f"requires an explicit RegrowConfig. Either:\n"
                f"  - pass regrow=RegrowConfig(model_name=..., layer=...) "
                f"naming the host model's residual stream layer, OR\n"
                f"  - set regrow_count=0 to skip regrowth entirely.\n"
                f"The pre-change layer=10 / model_name=\"gpt2\" fallbacks "
                f"were removed in polygram 0.1.0 because they silently "
                f"bound regrowth to GPT-2."
            )
        # v0.4 forge-whisper-encoder: text and audio eval inputs are mutually
        # exclusive. The downstream evaluate_faithfulness action dispatches
        # on forged.config.family, so a pipeline carrying both would be
        # ambiguous about which signal drove the FSM's faithfulness gate.
        if self.eval_audio_features is not None and len(self.eval_prompts) > 0:
            raise ValueError(
                "ForgePipeline: eval_audio_features and eval_prompts are "
                "mutually exclusive — a pipeline targets either text-LM "
                "faithfulness (KL via eval_prompts) or audio faithfulness "
                "(cosine via eval_audio_features), not both. Set exactly "
                "one."
            )
        # Issue #27 / Bug 1: forge-whisper-encoder finetune incompatibility.
        # The fine_tune_model action passes module.parameters() to the
        # optimizer; on whisper_encoder the parameter set includes the
        # frozen-copied conv stem (conv1, conv2) and embed_positions — the
        # adapter walk copies them bit-for-bit from the host and they MUST
        # stay frozen to keep the ε_conv accounting honest. There's no
        # per-frame loss signal defined for an encoder forge either; the
        # fine-tune path is LM-only by construction.
        if self.eval_audio_features is not None and self.finetune_steps > 0:
            raise ValueError(
                "ForgePipeline: finetune_steps > 0 is not supported on "
                "whisper_encoder forges. The conv stem (conv1, conv2, "
                "embed_positions) is frozen-copied from the host and must "
                "stay frozen — fine-tuning would corrupt those weights and "
                "break the ε_conv accounting. Additionally, no per-frame "
                "loss signal is defined for the encoder forge yet. Set "
                "finetune_steps=0 or pair the pipeline with an LM host."
            )
        # Issue #27 / Bug 2: hybrid_bridge has no Whisper-encoder semantics.
        # The flag wires three bases (basis_embed, basis, basis_lm_head)
        # whose insertion points are the embed / mid / lm-head regions of
        # an LM residual stream. Whisper encoder has no lm_head, and
        # ForgedWhisperEncoder already carries its own d → f bridge in the
        # basis_encode buffer at the conv-stem → first-block boundary. A
        # hybrid_bridge=True whisper forge would either double-project
        # the residual or crash deep in the projection step.
        if self.eval_audio_features is not None and self.hybrid_bridge:
            raise ValueError(
                "ForgePipeline: hybrid_bridge=True is not supported on "
                "whisper_encoder forges. The forged encoder already carries "
                "a d → f projection in the basis_encode buffer at the "
                "conv-stem → first-block boundary; layering a second bridge "
                "on top would double-project the residual. There is also "
                "no lm_head on a Whisper encoder for basis_lm_head to "
                "attach to. Disable hybrid_bridge for whisper_encoder "
                "forges."
            )
        if self.task_trigger not in ("labeled", "token_budget", "loss_delta"):
            raise ValueError(
                f"ForgePipeline: task_trigger={self.task_trigger!r} must be one of "
                "'labeled' | 'token_budget' | 'loss_delta'"
            )
        if self.task_trigger == "token_budget" and self.n_tasks > 1 and self.token_budget_per_task <= 0:
            raise ValueError(
                "ForgePipeline: task_trigger='token_budget' requires "
                "token_budget_per_task > 0"
            )
        if self.task_trigger == "loss_delta" and self.n_tasks > 1 and self.loss_delta_threshold <= 0:
            raise ValueError(
                "ForgePipeline: task_trigger='loss_delta' requires "
                "loss_delta_threshold > 0"
            )
        if self.replay_ratio > 0 and self.replay_buffer_size <= 0:
            raise ValueError(
                "ForgePipeline: replay_ratio > 0 requires replay_buffer_size > 0"
            )
        if self.replay_policy == "per_task" and self.task_trigger != "labeled":
            raise ValueError(
                "ForgePipeline: replay_policy='per_task' requires "
                "task_trigger='labeled' (per-task slots need task boundaries)"
            )
        if self.moe_strategy not in ("preserve", "collapse", "top_n"):
            raise ValueError(
                f"ForgePipeline: moe_strategy={self.moe_strategy!r} must be one of "
                "'preserve' | 'collapse' | 'top_n'"
            )
        if self.moe_strategy == "top_n" and self.moe_keep_n <= 0:
            raise ValueError(
                f"ForgePipeline: moe_strategy='top_n' requires moe_keep_n > 0; "
                f"got moe_keep_n={self.moe_keep_n}"
            )
        if self.hybrid_bridge:
            if self.basis_embed is None or self.basis_lm_head is None:
                raise ValueError(
                    "ForgePipeline: hybrid_bridge=True requires both "
                    "basis_embed and basis_lm_head; got "
                    f"basis_embed={'set' if self.basis_embed is not None else 'None'}, "
                    f"basis_lm_head={'set' if self.basis_lm_head is not None else 'None'}"
                )
            n_features = self.basis.n_features
            d_model = self.basis.d_model
            if (
                self.basis_embed.n_features != n_features
                or self.basis_lm_head.n_features != n_features
            ):
                raise ValueError(
                    f"ForgePipeline: hybrid_bridge=True requires all three bases to share "
                    f"n_features; got basis_embed.n_features={self.basis_embed.n_features}, "
                    f"basis.n_features={n_features}, "
                    f"basis_lm_head.n_features={self.basis_lm_head.n_features}"
                )
            if (
                self.basis_embed.d_model != d_model
                or self.basis_lm_head.d_model != d_model
            ):
                raise ValueError(
                    f"ForgePipeline: hybrid_bridge=True requires all three bases to share "
                    f"d_model; got basis_embed.d_model={self.basis_embed.d_model}, "
                    f"basis.d_model={d_model}, "
                    f"basis_lm_head.d_model={self.basis_lm_head.d_model}"
                )
        if self.adaptive_regrow:
            if self.regrow_max <= self.regrow_count:
                raise ValueError(
                    "ForgePipeline: adaptive_regrow=True requires "
                    f"regrow_max > regrow_count (got regrow_max="
                    f"{self.regrow_max}, regrow_count={self.regrow_count}). "
                    "Pick a regrow_max that caps the largest per-cycle "
                    "growth you'll tolerate; regrow_count is the base "
                    "floor / fallback the controller honors when the "
                    "target is reached."
                )
            if self.n_features_target <= 0:
                raise ValueError(
                    "ForgePipeline: adaptive_regrow=True requires "
                    f"n_features_target > 0 (got {self.n_features_target}). "
                    "Pick the target basis size the controller should "
                    "grow toward."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ForgePipeline":
        """Build a :class:`ForgePipeline` from a JSON/YAML-shaped mapping.

        Pops ``compression`` / ``epoch_compression`` / ``regrow`` keys
        (when present and non-None) and feeds each through the matching
        polygram ``<Config>.from_dict``. Other keys are passed through
        as constructor kwargs. Unknown top-level keys emit a
        :class:`UserWarning` and are dropped — matching polygram's own
        forward-compatibility policy.
        """
        from polygram import (
            CompressionConfig,
            EpochCompressionConfig,
            RegrowConfig,
        )

        if not isinstance(data, Mapping):
            raise TypeError(
                f"ForgePipeline.from_dict expected a mapping, "
                f"got {type(data).__name__}"
            )

        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        nested = {
            "compression": CompressionConfig,
            "epoch_compression": EpochCompressionConfig,
            "regrow": RegrowConfig,
        }
        unknown: list[str] = []
        for key, value in data.items():
            if key not in known:
                unknown.append(key)
                continue
            if key in nested and isinstance(value, Mapping):
                kwargs[key] = nested[key].from_dict(value)
            else:
                kwargs[key] = value
        if unknown:
            # Single warning collecting every unknown key — easier to
            # spot in logs than one warning per dropped key.
            warnings.warn(
                f"ForgePipeline.from_dict: ignoring unknown key(s): "
                f"{sorted(unknown)!r}",
                UserWarning,
                stacklevel=2,
            )
        return cls(**kwargs)

    def run(
        self,
        output_dir: str | Path,
        *,
        finetune_iterator=None,
    ) -> ForgeResult:
        """Forge against a real HuggingFace host (``host_model_id`` set).

        Dispatches on ``self.orchestrator``:

        - ``"imperative"`` (default) — straight-line forge + faithfulness +
          save. Fine-tune fields are NOT honoured on this path; if
          ``finetune_corpus`` is set, a UserWarning surfaces so callers
          who expect the recipe to run can spot the mismatch and switch
          to ``orchestrator="fsm"``.
        - ``"fsm"`` — runs the full
          load → compress → optional regrow → project → fine-tune → eval
          loop via the orca-runtime FSM, mirroring ``run_synthetic`` but
          with ``AutoModelForCausalLM.from_pretrained`` + the host's
          tokenizer for the eval pre-tokenisation. Fine-tune fields
          (``finetune_corpus`` / ``finetune_total_steps`` / etc.) flow
          into the recipe action and produce the post-tune
          ``forged/`` checkpoint plus ``finetuned/`` directory.

        ``finetune_iterator`` (optional, FSM path only) — a pre-tokenised
        iterable yielding ``(batch_size, sequence_length)`` int64 tensors.
        Mirrors the ``run_synthetic`` argument; useful for callers that
        already own their tokenisation pipeline and want to skip the
        ``AutoTokenizer + datasets`` round-trip the recipe action would
        otherwise do via ``finetune_corpus``.

        v0.3 footgun fix: the recipe was wired into the FSM only; the
        v0.1 imperative ``run()`` silently dropped every ``finetune_*``
        field on the floor when called against a real host. The new
        ``"fsm"`` dispatch (and the imperative warning) closes that gap.
        """
        from saeforge.utils.lazy import require_extra

        require_extra("torch", "torch")
        require_extra("transformers", "torch")

        if self.host_model_id is None:
            raise ValueError("ForgePipeline.run requires a host_model_id")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.orchestrator == "fsm":
            return self._run_real_fsm(output_dir, finetune_iterator=finetune_iterator)
        if finetune_iterator is not None:
            import warnings

            warnings.warn(
                "ForgePipeline.run(finetune_iterator=...) only takes effect "
                "with orchestrator='fsm'; the imperative path ignores it.",
                UserWarning,
                stacklevel=2,
            )
        return self._run_real_imperative(output_dir)

    def _apply_moe_collapse(self, weights: dict, native_cfg) -> tuple[dict, Any]:
        """Average per-expert weights into dense MLP keys; downgrade config to dense.

        For each layer with MoE keys
        (``mlp.experts.{e}.{gate,up,down}_proj.weight`` ×
        ``num_experts``), produce a single averaged
        ``mlp.{gate,up,down}_proj.weight`` set. Drop the router
        (``mlp.gate.weight``). The forged config is downgraded to a
        dense Qwen3 (family="qwen3", num_experts=0,
        intermediate_size=moe_intermediate_size).

        Used only when ``moe_strategy="collapse"``. Documented in the
        capability spec as "experimental — produces a model that thinks
        like the average expert, not like any specific expert."
        """
        import re
        from dataclasses import replace

        if native_cfg.num_experts == 0:
            return weights, native_cfg  # already dense

        expert_pattern = re.compile(
            r"^(model\.layers\.\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$"
        )
        new_weights: dict = {}
        # Buckets keyed by (layer_prefix, kind) → list of tensors
        buckets: dict = {}
        for k, v in weights.items():
            m = expert_pattern.match(k)
            if m:
                layer_prefix, kind = m.groups()
                buckets.setdefault((layer_prefix, kind), []).append(v)
            elif ".mlp.gate.weight" in k:
                # Drop the router — collapsed model has no routing
                continue
            else:
                new_weights[k] = v

        for (layer_prefix, kind), tensors in buckets.items():
            avg = sum(tensors) / len(tensors)
            new_weights[f"{layer_prefix}.mlp.{kind}.weight"] = avg

        new_cfg = replace(
            native_cfg,
            family="qwen3",
            num_experts=0,
            num_experts_per_tok=0,
            intermediate_size=native_cfg.moe_intermediate_size,
            moe_intermediate_size=0,
        )
        return new_weights, new_cfg

    def sweep_pareto(
        self,
        encodings: list[tuple[str, "str | Path"]],
        output_dir: "str | Path",
        *,
        frontier_only: bool = False,
        quality_floor: float | None = None,
        quality_thresholds: Any = None,
        host_d_model_override: int | None = None,
        auto_materialise_specs: Any = None,
        validation_prompts: "str | Path | None" = None,
        validation_threshold: float = 0.7,
        validation_jaccard_threshold: float = 0.3,
        layer: int | None = None,
        targets: list[int] | None = None,
        score_field: str = "polygram_overlap",
        rep_selection: str = "scale_aware",
        assign_phase_knobs: bool = False,
        validation_eval_overlap: bool = False,
        force_rematerialise: bool = False,
        plan_only: bool = False,
        magnitude_diagnostics: "int | str | Path | None" = None,
        rank_monotonicity_check: bool = False,
        **forge_kwargs: Any,
    ) -> Path:
        """Forge across per-K materialised SAE checkpoints; emit a JSONL frontier.

        Delegates to :func:`saeforge.sweep.sweep_pareto`. See the
        ``pareto-sweep`` capability spec for the row contract, lifecycle
        states (success / frontier-only / row failure), and resumability
        semantics.

        Each row hot-swaps ``self.basis`` and ``self.projector`` for the
        duration of one ``self.run`` call, then restores them — so the host
        model, eval config, and fine-tune knobs persist across rows while
        the SAE varies per K.

        Forge-quality diagnostics: ``quality_floor`` and
        ``quality_thresholds`` are forwarded to the sweep driver.
        ``host_d_model_override`` short-circuits the
        ``transformers.AutoConfig`` lookup for hosts whose config doesn't
        expose ``hidden_size`` canonically. See
        :mod:`saeforge.forge_quality`.
        """
        from saeforge.sweep import sweep_pareto as _sweep_pareto

        normalized = [(label, Path(path)) for label, path in encodings]
        validation_prompts_path = (
            Path(validation_prompts) if validation_prompts is not None else None
        )
        return _sweep_pareto(
            self,
            encodings=normalized,
            output_dir=Path(output_dir),
            frontier_only=frontier_only,
            quality_floor=quality_floor,
            quality_thresholds=quality_thresholds,
            host_d_model_override=host_d_model_override,
            auto_materialise_specs=auto_materialise_specs,
            validation_prompts=validation_prompts_path,
            validation_threshold=validation_threshold,
            validation_jaccard_threshold=validation_jaccard_threshold,
            layer=layer,
            targets=targets,
            score_field=score_field,
            rep_selection=rep_selection,
            assign_phase_knobs=assign_phase_knobs,
            validation_eval_overlap=validation_eval_overlap,
            force_rematerialise=force_rematerialise,
            plan_only=plan_only,
            magnitude_diagnostics=magnitude_diagnostics,
            rank_monotonicity_check=rank_monotonicity_check,
            **forge_kwargs,
        )

    def _build_hybrid_bundle(self, host) -> "HybridBasisBundle | None":
        """Return a hybrid bundle when enabled, else None. Enforces tied-embedding refusal.

        Tied embeddings (GPT-2 default, Llama-family) make the embed and lm-head
        bases algebraically constrained — the same weight matrix would have to
        project through two different feature spaces. v1 refuses; the principled
        fix is tracked as ``hybrid-bridge-tied-embeddings``.
        """
        if not self.hybrid_bridge:
            return None
        if getattr(host.config, "tie_word_embeddings", False):
            raise ValueError(
                "ForgePipeline: hybrid_bridge=True is incompatible with hosts "
                "where config.tie_word_embeddings=True. Embed and lm_head would "
                "have to project the same matrix through two different feature "
                "spaces. Either disable hybrid_bridge or reload the host with "
                "tie_word_embeddings=False. Tracked as follow-up "
                "`hybrid-bridge-tied-embeddings`."
            )
        n_layer = getattr(host.config, "n_layer", None) or getattr(
            host.config, "num_hidden_layers", None
        )
        if n_layer is None:
            raise ValueError(
                f"ForgePipeline: cannot derive n_layer from {type(host).__name__}.config "
                "(no n_layer or num_hidden_layers attribute)"
            )
        return HybridBasisBundle(
            basis_embed=self.basis_embed,
            basis_mid=self.basis,
            basis_lm_head=self.basis_lm_head,
            n_layer=int(n_layer),
        )

    def _score_faithfulness_imperative(
        self, model: NativeModel, host: Any, *, transformers: Any
    ) -> tuple[float | None, str | None]:
        """Real-host imperative scoring path.

        When ``self.faithfulness is None`` and ``eval_prompts`` is set,
        delegate to the legacy :func:`faithfulness_kl` call exactly as
        v0.4 did — this is what the byte-identity test pins. When a
        target is set, call it directly with an FSM-shaped ctx so it
        sees the same inputs the FSM action would pass.
        """
        if not self.eval_prompts and self.faithfulness is None:
            return None, None
        if self.faithfulness is None:
            from saeforge.eval.faithfulness import faithfulness_kl

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.host_model_id)
            score = faithfulness_kl(
                model,
                host,
                self.eval_prompts,
                tokenizer=tokenizer,
                device=self.device,
            )
            return float(score), "kl"
        ctx = self._build_imperative_score_ctx(transformers=transformers)
        score, _ = self.faithfulness.score(forged=model, host=host, ctx=ctx)
        return float(score), self.faithfulness.name

    def _score_faithfulness_synthetic(
        self,
        model: NativeModel,
        host_model: Any,
        eval_input_ids: Any,
    ) -> tuple[float | None, str | None]:
        """Synthetic-host imperative scoring path.

        Mirrors ``_score_faithfulness_imperative`` but uses the already-
        tokenised ``eval_input_ids`` argument (the synthetic path skips
        the AutoTokenizer step). When no target is set and no eval ids
        are present, return ``(None, None)`` — matches v0.4.
        """
        if self.faithfulness is None and eval_input_ids is None:
            return None, None
        if self.faithfulness is None:
            score = _kl_from_input_ids(model, host_model, eval_input_ids, device=self.device)
            return float(score), "kl"
        ctx = {
            "_eval_input_ids": eval_input_ids,
            "_eval_audio_features": self.eval_audio_features,
            "_eval_encoder_states": self.eval_encoder_states,
            "device": self.device,
        }
        score, _ = self.faithfulness.score(forged=model, host=host_model, ctx=ctx)
        return float(score), self.faithfulness.name

    def _build_imperative_score_ctx(self, *, transformers: Any) -> dict:
        """Construct the ctx dict a custom target sees on the imperative
        real-host path. Pre-tokenises ``eval_prompts`` so the target's
        ``ctx["_eval_input_ids"]`` is populated identically to the FSM
        path.
        """
        ctx: dict[str, Any] = {
            "device": self.device,
            "_eval_audio_features": self.eval_audio_features,
            "_eval_encoder_states": self.eval_encoder_states,
        }
        if self.eval_prompts:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.host_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer(
                list(self.eval_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            ctx["_eval_input_ids"] = encoded["input_ids"]
        return ctx

    def _resolve_target_name(self, model: NativeModel | None) -> str | None:
        """Best-effort target-name resolution for result metadata.

        When ``self.faithfulness`` is set, use its ``name``. Otherwise,
        ask the family-dispatch policy what the default would have been
        — mirrors what the action layer wrote into the FSM log entry.
        Returns ``None`` if the model isn't available or the family is
        unrecognised (e.g. a failed FSM run that never built a model).
        """
        if self.faithfulness is not None:
            return self.faithfulness.name
        from saeforge.eval.targets import _default_target_for

        family = (
            model.config.family
            if model is not None and hasattr(model, "config")
            else None
        )
        try:
            return _default_target_for(family).name
        except ValueError:
            return None

    def _run_real_imperative(self, output_dir: Path) -> ForgeResult:
        """Pre-recipe forge path: load → project → eval → save. No fine-tune."""
        import warnings

        from saeforge.utils.lazy import require_extra

        transformers = require_extra("transformers", "torch")

        # The recipe action only runs on the FSM path. Surface a warning
        # so callers who set finetune_corpus on the imperative path see
        # immediately that their fine-tune was skipped — pre-fix this
        # was a silent no-op.
        if self.finetune_corpus is not None or self.finetune_total_steps not in (
            0, 1000,  # 1000 is the dataclass default; treat both as "unset"
        ):
            warnings.warn(
                "ForgePipeline.run with orchestrator='imperative' (the "
                "default) does not run the fine-tune recipe. The "
                f"finetune_corpus={self.finetune_corpus!r} and "
                f"finetune_total_steps={self.finetune_total_steps} fields "
                "are being ignored on this path. Pass "
                "orchestrator='fsm' to enable the recipe end-to-end.",
                UserWarning,
                stacklevel=3,
            )

        # Load via AutoModelForCausalLM so the host's actual architecture
        # (GPT-2 / Llama / Gemma-2) drives adapter dispatch. The pre-multi-
        # arch v0.1 path called ``GPT2LMHeadModel.from_pretrained`` against
        # any host_model_id, which silently produced a randomly-initialised
        # GPT-2 for non-GPT-2 inputs (e.g. ``google/gemma-2-2b``).
        host = transformers.AutoModelForCausalLM.from_pretrained(
            self.host_model_id, torch_dtype=_torch_dtype(self.dtype)
        ).eval()
        bundle = self._build_hybrid_bundle(host)
        weights = self.projector.project_module(
            host, attention_width=self.attention_width, hybrid=bundle
        )
        config = _config_from_host(
            host, self.basis.n_features, attention_width=self.attention_width
        )
        if bundle is not None:
            config.bridges = True
            config.bridge_init = self.bridge_config.init
            config.bridge_nonlin = self.bridge_config.nonlin
            config.bridge_pre_layernorm = self.bridge_config.pre_layernorm
        # MoE strategy dispatch (only relevant when host is qwen3_moe; for
        # every other family config.num_experts is 0 and these branches no-op).
        if self.moe_strategy == "top_n":
            raise NotImplementedError(
                "ForgePipeline: moe_strategy='top_n' is a v1 placeholder. "
                "It requires a per-expert activation-frequency calibration "
                "utility tracked as the moe-expert-calibration follow-up. "
                "Use moe_strategy='preserve' (default, full fidelity) or "
                "moe_strategy='collapse' (averages experts into a dense MLP) "
                "in v1."
            )
        if self.moe_strategy == "collapse" and config.num_experts > 0:
            weights, config = self._apply_moe_collapse(weights, config)
        model = NativeModel.from_projected_weights(config, weights)
        model._move(dtype=self.dtype, device=self.device)

        faithfulness, target_name = self._score_faithfulness_imperative(
            model, host, transformers=transformers
        )

        forged_dir = output_dir / "forged"
        model.save_pretrained(forged_dir)

        n_params = model.num_parameters()
        result_meta = {
            "host_model_id": self.host_model_id,
            "n_params": n_params,
            "faithfulness": faithfulness,
            "faithfulness_target_name": target_name,
            "faithfulness_kl": faithfulness if target_name == "kl" else None,
            "n_features": self.basis.n_features,
            "scale_compression_ratio": self.basis.scale_compression_ratio,
        }
        (output_dir / "forge_result.json").write_text(json.dumps(result_meta, indent=2))

        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=n_params,
            faithfulness=faithfulness,
            faithfulness_target_name=target_name,
            extras=result_meta,
        )

    def _run_real_fsm(
        self, output_dir: Path, *, finetune_iterator=None
    ) -> ForgeResult:
        """Real-host FSM dispatch — mirrors ``_run_synthetic_fsm`` with
        ``AutoModelForCausalLM.from_pretrained`` and host-tokenizer-driven
        eval pre-tokenisation. Honours every ``finetune_*`` field via the
        recipe action.
        """
        from saeforge.orchestrator import run_machine
        from saeforge.utils.lazy import require_extra

        transformers = require_extra("transformers", "torch")

        # Persist the basis to disk so the FSM's load action picks it up.
        sae_checkpoint = output_dir / "synth_basis.safetensors"
        _write_basis_as_checkpoint(self.basis, sae_checkpoint)

        host = transformers.AutoModelForCausalLM.from_pretrained(
            self.host_model_id, torch_dtype=_torch_dtype(self.dtype)
        ).eval()

        # Pre-tokenise eval_prompts with the host's tokenizer so the FSM
        # ``evaluate_faithfulness`` action can compute KL via the existing
        # ``_kl_from_input_ids`` path (no tokenizer round-trip inside the FSM).
        eval_input_ids = None
        if self.eval_prompts:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.host_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer(
                list(self.eval_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            eval_input_ids = encoded["input_ids"]

        ctx = self._build_fsm_ctx(
            sae_checkpoint=sae_checkpoint,
            output_dir=output_dir,
            host_model=host,
            eval_input_ids=eval_input_ids,
            eval_audio_features=self.eval_audio_features,
            eval_encoder_states=self.eval_encoder_states,
            finetune_input_ids=None,
            finetune_iterator=finetune_iterator,
            host_model_id=self.host_model_id,
        )
        final = run_machine(ctx)
        # Surface FSM failures as exceptions instead of returning a
        # ForgeResult with n_params=0 / KL=0.0. See ForgeFailed.
        _raise_if_failed(final)
        model = final.get("_native_model")

        faithfulness = (
            final.get("faithfulness") if eval_input_ids is not None else None
        )
        target_name = self._resolve_target_name(model)
        result_meta = {
            "host_model_id": self.host_model_id,
            "n_params": final.get("n_params", 0),
            "faithfulness": faithfulness,
            "faithfulness_target_name": target_name,
            "faithfulness_kl": faithfulness if target_name == "kl" else None,
            "n_features": final.get("current_feature_count"),
            "scale_compression_ratio": self.basis.scale_compression_ratio,
            "transitions_log": final.get("transitions_log", []),
            "final_state": _final_state_for_log(final),
        }
        (output_dir / "forge_result.json").write_text(json.dumps(result_meta, indent=2))

        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=final.get("n_params", 0),
            faithfulness=faithfulness,
            faithfulness_target_name=target_name,
            extras=result_meta,
        )

    def run_synthetic(
        self,
        host_model,
        output_dir: str | Path,
        eval_input_ids=None,
        sae_checkpoint: str | Path | None = None,
        finetune_input_ids=None,
        finetune_iterator=None,
    ) -> ForgeResult:
        """Run the pipeline against an already-loaded host model.

        This skips the ``from_pretrained`` step — handy for tests and the
        toy example, where we build a tiny GPT-2 in memory rather than
        pulling the canonical 124M-param checkpoint.

        ``finetune_iterator`` is a pre-tokenized iterable yielding
        ``(batch_size, sequence_length)`` int64 tensors. When supplied (or
        when ``finetune_corpus`` is set on the pipeline), the FSM
        ``fine_tune_model`` action delegates to ``run_finetune``. Without
        either, the action either runs the v0.1 4-step smoke loop (when
        ``finetune_input_ids`` is supplied) or passes through entirely.
        """
        if self.orchestrator == "fsm":
            return self._run_synthetic_fsm(
                host_model, output_dir, eval_input_ids, sae_checkpoint,
                finetune_input_ids, finetune_iterator,
            )
        return self._run_synthetic_imperative(host_model, output_dir, eval_input_ids)

    def _run_synthetic_imperative(
        self,
        host_model,
        output_dir: str | Path,
        eval_input_ids=None,
    ) -> ForgeResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        weights = self.projector.project_module(host_model, attention_width=self.attention_width)
        config = _config_from_host(
            host_model, self.basis.n_features, attention_width=self.attention_width
        )
        model = NativeModel.from_projected_weights(config, weights)
        model._move(dtype=self.dtype, device=self.device)

        faithfulness, target_name = self._score_faithfulness_synthetic(
            model, host_model, eval_input_ids
        )

        forged_dir = output_dir / "forged"
        model.save_pretrained(forged_dir)
        n_params = model.num_parameters()
        result_meta = {
            "host_model_id": "<in-memory>",
            "n_params": n_params,
            "faithfulness": faithfulness,
            "faithfulness_target_name": target_name,
            "faithfulness_kl": faithfulness if target_name == "kl" else None,
            "n_features": self.basis.n_features,
            "scale_compression_ratio": self.basis.scale_compression_ratio,
            "attention_width": self.attention_width,
        }
        (output_dir / "forge_result.json").write_text(json.dumps(result_meta, indent=2))
        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=n_params,
            faithfulness=faithfulness,
            faithfulness_target_name=target_name,
            extras=result_meta,
        )

    def _run_synthetic_fsm(
        self,
        host_model,
        output_dir: str | Path,
        eval_input_ids,
        sae_checkpoint: str | Path | None,
        finetune_input_ids=None,
        finetune_iterator=None,
    ) -> ForgeResult:
        from saeforge.orchestrator import run_machine

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if sae_checkpoint is None:
            # The FSM expects an on-disk checkpoint to load. Synthesize one
            # from the current basis so run_synthetic stays hermetic.
            sae_checkpoint = output_dir / "synth_basis.safetensors"
            _write_basis_as_checkpoint(self.basis, sae_checkpoint)

        ctx = self._build_fsm_ctx(
            sae_checkpoint=sae_checkpoint,
            output_dir=output_dir,
            host_model=host_model,
            eval_input_ids=eval_input_ids,
            eval_audio_features=self.eval_audio_features,
            eval_encoder_states=self.eval_encoder_states,
            finetune_input_ids=finetune_input_ids,
            finetune_iterator=finetune_iterator,
            host_model_id="<in-memory>",
        )
        final = run_machine(ctx)
        # Surface FSM failures as exceptions instead of returning a
        # ForgeResult with n_params=0 / KL=0.0. See ForgeFailed.
        _raise_if_failed(final)
        model = final.get("_native_model")
        faithfulness = (
            final.get("faithfulness") if eval_input_ids is not None else None
        )
        target_name = self._resolve_target_name(model)
        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=final.get("n_params", 0),
            faithfulness=faithfulness,
            faithfulness_target_name=target_name,
            extras={
                "host_model_id": "<in-memory>",
                "n_params": final.get("n_params", 0),
                "faithfulness": faithfulness,
                "faithfulness_target_name": target_name,
                "faithfulness_kl": faithfulness if target_name == "kl" else None,
                "n_features": final.get("current_feature_count"),
                "transitions_log": final.get("transitions_log", []),
                "final_state": _final_state_for_log(final),
            },
        )

    def _build_fsm_ctx(
        self,
        *,
        sae_checkpoint: str | Path,
        output_dir: Path,
        host_model,
        eval_input_ids,
        finetune_input_ids,
        finetune_iterator,
        host_model_id: str,
        eval_audio_features=None,
        eval_encoder_states=None,
    ) -> dict:
        """Shared FSM-context builder for both ``_run_real_fsm`` and
        ``_run_synthetic_fsm`` — keeps the polygram-tuning + finetune-recipe
        ctx fields in lock-step across the two entry points.
        """
        return {
            "sae_checkpoint": str(sae_checkpoint),
            "host_model_id": host_model_id,
            "output_dir": str(output_dir),
            "iterations": self.iterations,
            "regrow_count": self.regrow_count,
            "current_iter": 0,
            "current_sae_path": "",
            "compressed_sae_path": "",
            "regrown_sae_path": "",
            "current_feature_count": 0,
            "projected_weights_path": "",
            "finetuned_model_path": "",
            "faithfulness": 0.0,
            "min_faithfulness": 0.0,
            "perplexity": 1e6,
            "best_perplexity": 1e6,
            "final_model_path": "",
            "error_message": "",
            "quantum_aware": self.quantum_aware,
            "n_params": 0,
            "transitions_log": [],
            "device": self.device,
            "_host_model": host_model,
            "_eval_input_ids": eval_input_ids,
            "_faithfulness_target": self.faithfulness,
            # v0.4 forge-whisper-encoder: audio-side eval input. None when
            # the host is an LM family; populated when a Whisper host is
            # being forged. ``_eval_encoder_states`` is an optional pre-
            # capture fast path — when present, the action uses these
            # states directly and skips the host forward.
            "_eval_audio_features": eval_audio_features,
            "_eval_encoder_states": eval_encoder_states,
            "_finetune_input_ids": finetune_input_ids,
            "_finetune_iterator": finetune_iterator,
            "validation_report_path": self.validation_report_path,
            # Polygram tuning bundles — serialised as JSON-friendly dicts
            # so the orca-runtime trace tooling stays unchanged. Absent
            # keys (rather than ``None`` values) signal "use polygram's
            # own defaults" to the action layer.
            **(
                {"compression": self.compression.to_dict()}
                if self.compression is not None
                else {}
            ),
            **(
                {"epoch_compression": self.epoch_compression.to_dict()}
                if self.epoch_compression is not None
                else {}
            ),
            **(
                {"regrow": self.regrow.to_dict()}
                if self.regrow is not None
                else {}
            ),
            "finetune_steps": self.finetune_steps,
            "finetune_lr": self.finetune_lr,
            "attention_width": self.attention_width,
            "finetune_corpus": str(self.finetune_corpus) if self.finetune_corpus else None,
            "finetune_total_steps": self.finetune_total_steps,
            "finetune_warmup_steps": self.finetune_warmup_steps,
            "finetune_peak_lr": self.finetune_peak_lr,
            "finetune_batch_size": self.finetune_batch_size,
            "finetune_seq_len": self.finetune_seq_len,
            "finetune_precision": self.finetune_precision,
            "finetune_grad_checkpoint": self.finetune_grad_checkpoint,
            "finetune_eval_every": self.finetune_eval_every,
            "finetune_save_every": self.finetune_save_every,
            "finetune_save_dir": str(self.finetune_save_dir) if self.finetune_save_dir else None,
            "finetune_log_every": self.finetune_log_every,
            "finetune_distill_alpha": self.finetune_distill_alpha,
            "finetune_distill_temperature": self.finetune_distill_temperature,
            # Adaptive-regrow knobs. Always written so the action layer
            # can read them with ``ctx.get(...)``; ``adaptive_regrow=False``
            # makes the other three inert.
            "adaptive_regrow": self.adaptive_regrow,
            "regrow_max": self.regrow_max,
            "n_features_target": self.n_features_target,
            "regrow_damping": self.regrow_damping,
            **self._build_continual_ctx(),
        }

    def _build_continual_ctx(self) -> dict:
        """Continual-learning fields + replay/task-stream side effects.

        Defaults preserve v0.1 byte-equivalence: when every continual
        knob is at its default, no replay buffer is created and no
        task stream is registered, so the FSM context is functionally
        identical to v0.1 (modulo the always-present scalar fields).
        """
        from saeforge.training import ReplayBuffer
        from saeforge.training import task_stream as ts_module

        ctx: dict = {
            "n_tasks": self.n_tasks,
            "task_idx": 0,
            "task_trigger": self.task_trigger,
            "token_budget_per_task": self.token_budget_per_task,
            "tokens_seen_in_task": 0,
            "loss_delta_threshold": self.loss_delta_threshold,
            "recent_eval_losses": [],
            "advance_stream": False,
            "inner_refine_passes": self.inner_refine_passes,
            "inner_refine_idx": 0,
            "protect_top_k": self.protect_top_k,
            "protect_score": self.protect_score,
            "protected_features": [],
            "activation_buffer_size": self.activation_buffer_size,
            "feature_usage": [],
            "replay_ratio": self.replay_ratio,
            "replay_policy": self.replay_policy,
            "replay_buffer_size": self.replay_buffer_size,
            "task_iterator_id": "",
        }

        if self.replay_buffer_size > 0:
            buf = ReplayBuffer(size=self.replay_buffer_size, policy=self.replay_policy)
            if self.replay_policy == "per_task":
                buf.configure_per_task(self.n_tasks)
            ctx["_replay_buffer"] = buf

        if self.task_stream is not None:
            ctx["task_iterator_id"] = ts_module.register(self.task_stream)

        return ctx


def _write_basis_as_checkpoint(basis: FeatureBasis, path: str | Path) -> None:
    """Persist a basis as a no-compression safetensors checkpoint for the FSM loader.

    Preserves the basis dtype (typically float64) so the round-trip through
    ``from_polygram_checkpoint`` is byte-exact — required for the
    imperative/FSM byte-equivalence safety net.
    """
    from safetensors.numpy import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({"W_dec": np.ascontiguousarray(basis.W_dec)}, str(path))


def _final_state_for_log(ctx: dict) -> str:
    log = ctx.get("transitions_log", [])
    last = log[-1]["action"] if log else None
    return "done" if last == "save_final_model" else "failed" if last == "log_error" else "unknown"


def _kl_from_input_ids(forged_model, host_model, input_ids, *, device: str = "cpu") -> float:
    """KL helper that bypasses tokenization — feeds pre-tokenized ids straight in."""
    from saeforge.utils.lazy import require_extra

    torch = require_extra("torch", "torch")
    F = torch.nn.functional

    forged_module = forged_model.torch_module.to(device).eval()
    host_module = host_model.to(device).eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        forged_logits = forged_module(input_ids)
        host_out = host_module(input_ids=input_ids)
        host_logits = host_out.logits if hasattr(host_out, "logits") else host_out[0]

    log_q = F.log_softmax(forged_logits, dim=-1)
    log_p = F.log_softmax(host_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())
