"""ForgePipeline — orchestrate basis load -> projection -> native model -> faithfulness eval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.model import NativeModel, _config_from_host
from saeforge.projector import SubspaceProjector


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
    compression_strategy: str = "merge"
    rep_selection: str = "scale_aware"
    finetune_steps: int = 0
    finetune_lr: float = 1e-3

    def run(self, output_dir: str | Path) -> ForgeResult:
        from saeforge.utils.lazy import require_extra

        require_extra("torch", "torch")
        transformers = require_extra("transformers", "torch")

        if self.host_model_id is None:
            raise ValueError("ForgePipeline.run requires a host_model_id")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        host = transformers.GPT2LMHeadModel.from_pretrained(self.host_model_id).eval()
        weights = self.projector.project_module(host)
        config = _config_from_host(host, self.basis.n_features)
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

    def run_synthetic(
        self,
        host_model,
        output_dir: str | Path,
        eval_input_ids=None,
        sae_checkpoint: str | Path | None = None,
        finetune_input_ids=None,
    ) -> ForgeResult:
        """Run the pipeline against an already-loaded host model.

        This skips the ``from_pretrained`` step — handy for tests and the
        toy example, where we build a tiny GPT-2 in memory rather than
        pulling the canonical 124M-param checkpoint.
        """
        if self.orchestrator == "fsm":
            return self._run_synthetic_fsm(
                host_model, output_dir, eval_input_ids, sae_checkpoint, finetune_input_ids
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

        weights = self.projector.project_module(host_model)
        config = _config_from_host(host_model, self.basis.n_features)
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
    ) -> ForgeResult:
        from saeforge.orchestrator import run_machine

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if sae_checkpoint is None:
            # The FSM expects an on-disk checkpoint to load. Synthesize one
            # from the current basis so run_synthetic stays hermetic.
            sae_checkpoint = output_dir / "synth_basis.safetensors"
            _write_basis_as_checkpoint(self.basis, sae_checkpoint)

        ctx = {
            "sae_checkpoint": str(sae_checkpoint),
            "host_model_id": "<in-memory>",
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
            "validation_report_path": self.validation_report_path,
            "compression_strategy": self.compression_strategy,
            "rep_selection": self.rep_selection,
            "finetune_steps": self.finetune_steps,
            "finetune_lr": self.finetune_lr,
        }
        final = run_machine(ctx)
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
