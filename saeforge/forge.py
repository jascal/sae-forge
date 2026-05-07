"""ForgePipeline — orchestrate basis load -> projection -> native model -> faithfulness eval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    """End-to-end imperative forging pipeline.

    Stages, in order:

    1. ``stage_load_basis`` — return ``self.basis`` (already loaded by caller).
    2. ``stage_project_host`` — load host via ``host_model_id`` and project
       weights through ``self.projector``.
    3. ``stage_assemble`` — build the ``NativeModel`` from projected weights.
    4. ``stage_eval`` — measure faithfulness KL on ``eval_prompts`` (skipped
       when the list is empty).
    5. ``stage_save`` — write the forged model to ``output_dir``.
    """

    basis: FeatureBasis
    projector: SubspaceProjector
    host_model_id: str | None = None
    eval_prompts: list[str] = field(default_factory=list)
    dtype: str = "float32"
    device: str = "cpu"

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
    ) -> ForgeResult:
        """Run the pipeline against an already-loaded host model.

        This skips the ``from_pretrained`` step — handy for tests and the
        toy example, where we build a tiny GPT-2 in memory rather than
        pulling the canonical 124M-param checkpoint.
        """
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
