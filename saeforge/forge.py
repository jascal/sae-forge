"""ForgePipeline — orchestrate basis load -> projection -> native model -> faithfulness eval."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.model import NativeModel, _config_from_host
from saeforge.projector import SubspaceProjector


class ForgeFailed(RuntimeError):
    """The FSM ended in a ``failed`` state and the run could not produce
    a valid forged model.

    Pre-fix, an action raising inside the FSM (e.g. ``AttributeError``
    from the GPT-2-only grad-checkpointing path running against a
    ForgedLlama) was swallowed into ``final_state: failed`` and
    returned to the caller as a ``ForgeResult`` with ``n_params=0`` and
    ``faithfulness_kl=0.0`` — exit code 0 with no exception. Callers
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
            "faithfulness_kl": final.get("faithfulness"),
            "n_features": final.get("current_feature_count"),
        },
    )

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from polygram import (  # noqa: F401
        CompressionConfig,
        EpochCompressionConfig,
        RegrowConfig,
    )


@dataclass
class ForgeResult:
    """Structured output of a ``ForgePipeline.run`` call."""

    model: NativeModel | None
    output_dir: Path
    n_params: int = 0
    faithfulness_kl: float | None = None
    extras: dict = field(default_factory=dict)


@dataclass
class ForgePipeline:
    """End-to-end forging pipeline.

    Two orchestrators ship in v0.1:

    - ``imperative`` (default): straight-line load -> project -> assemble ->
      eval -> save. The v0 stable surface.
    - ``fsm``: orca-runtime-python FSM driving the same stages plus optional
      regrow / fine-tune passes. Requires the ``[orca]`` extra; topology is
      defined in ``saeforge/machines/sae_forge.orca.md``.

    Both orchestrators MUST produce byte-identical forged weights for the
    same inputs and seeds — that equivalence is the migration safety net
    from v0.1 (imperative-default) to v0.2 (fsm-default).
    """

    basis: FeatureBasis
    projector: SubspaceProjector
    host_model_id: str | None = None
    eval_prompts: list[str] = field(default_factory=list)
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
        host = transformers.AutoModelForCausalLM.from_pretrained(self.host_model_id).eval()
        weights = self.projector.project_module(host, attention_width=self.attention_width)
        config = _config_from_host(
            host, self.basis.n_features, attention_width=self.attention_width
        )
        model = NativeModel.from_projected_weights(config, weights)
        model._move(dtype=self.dtype, device=self.device)

        faithfulness = None
        if self.eval_prompts:
            from saeforge.eval.faithfulness import faithfulness_kl

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.host_model_id)
            faithfulness = faithfulness_kl(
                model,
                host,
                self.eval_prompts,
                tokenizer=tokenizer,
                device=self.device,
            )

        forged_dir = output_dir / "forged"
        model.save_pretrained(forged_dir)

        n_params = model.num_parameters()
        result_meta = {
            "host_model_id": self.host_model_id,
            "n_params": n_params,
            "faithfulness_kl": faithfulness,
            "n_features": self.basis.n_features,
            "scale_compression_ratio": self.basis.scale_compression_ratio,
        }
        (output_dir / "forge_result.json").write_text(json.dumps(result_meta, indent=2))

        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=n_params,
            faithfulness_kl=faithfulness,
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

        host = transformers.AutoModelForCausalLM.from_pretrained(self.host_model_id).eval()

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
        result_meta = {
            "host_model_id": self.host_model_id,
            "n_params": final.get("n_params", 0),
            "faithfulness_kl": faithfulness,
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
            faithfulness_kl=faithfulness,
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

        faithfulness = None
        if eval_input_ids is not None:
            faithfulness = _kl_from_input_ids(model, host_model, eval_input_ids, device=self.device)

        forged_dir = output_dir / "forged"
        model.save_pretrained(forged_dir)
        n_params = model.num_parameters()
        result_meta = {
            "host_model_id": "<in-memory>",
            "n_params": n_params,
            "faithfulness_kl": faithfulness,
            "n_features": self.basis.n_features,
            "scale_compression_ratio": self.basis.scale_compression_ratio,
            "attention_width": self.attention_width,
        }
        (output_dir / "forge_result.json").write_text(json.dumps(result_meta, indent=2))
        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=n_params,
            faithfulness_kl=faithfulness,
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
            finetune_input_ids=finetune_input_ids,
            finetune_iterator=finetune_iterator,
            host_model_id="<in-memory>",
        )
        final = run_machine(ctx)
        # Surface FSM failures as exceptions instead of returning a
        # ForgeResult with n_params=0 / KL=0.0. See ForgeFailed.
        _raise_if_failed(final)
        model = final.get("_native_model")
        return ForgeResult(
            model=model,
            output_dir=output_dir,
            n_params=final.get("n_params", 0),
            faithfulness_kl=final.get("faithfulness") if eval_input_ids is not None else None,
            extras={
                "host_model_id": "<in-memory>",
                "n_params": final.get("n_params", 0),
                "faithfulness_kl": final.get("faithfulness"),
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
        }


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
