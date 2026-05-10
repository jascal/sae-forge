"""sae-forge console script — verbs first, file paths positional, matches polygram style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from saeforge import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sae-forge",
        description="Forge a Polygram-compressed SAE into a small native transformer.",
    )
    parser.add_argument("--version", action="version", version=f"sae-forge {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    forge = sub.add_parser("forge", help="Run the full forge pipeline.")
    forge.add_argument("checkpoint", help="Polygram-compressed .safetensors checkpoint.")
    forge.add_argument("--host-model", required=True, help="HuggingFace host model id.")
    forge.add_argument("--output-dir", required=True, help="Where to write the forged model.")
    forge.add_argument(
        "--eval-prompts",
        help="JSONL file of prompts for the faithfulness eval; optional in v0.",
    )
    forge.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    forge.add_argument("--device", default="cpu")
    forge.add_argument(
        "--feature-native-attention",
        action="store_true",
        help="opt in to v0.2 feature-native attention; default is host-inherited internal width",
    )

    # Polygram tuning — high-frequency knobs only. Long-tail tuning lives
    # behind ForgePipeline.from_dict (see README) for one-shot YAML configs.
    forge.add_argument(
        "--coverage-target",
        type=float,
        help="EpochCompressor.coverage_target (0, 1]; "
        "polygram-tuning-config default is 0.5",
    )
    forge.add_argument(
        "--cosine-threshold",
        type=float,
        help="EpochCompressor.cosine_threshold [-1, 1]; "
        "polygram-tuning-config default is 0.30",
    )
    forge.add_argument(
        "--max-compress-iterations",
        type=int,
        help="EpochCompressor.max_iterations; "
        "polygram-tuning-config default is 1 (iterative preset)",
    )
    forge.add_argument(
        "--regrow-count",
        type=int,
        default=0,
        help="number of regrow passes per outer-loop iteration "
        "(default: 0, no regrow). When > 0, --regrow-layer is required.",
    )
    forge.add_argument(
        "--regrow-layer",
        type=int,
        help="transformer layer (>= 0) whose residual stream feeds the "
        "regrower. Required when --regrow-count > 0; the polygram-side "
        "GPT-2-specific layer=10 default was removed in 0.1.0.",
    )
    forge.add_argument(
        "--regrow-strategy",
        type=str,
        default=None,
        help="RegrowConfig.strategy (default: residual_kmeans).",
    )

    inspect = sub.add_parser(
        "inspect",
        help="Triage a compressed checkpoint without torch — basis stats only.",
    )
    inspect_target = inspect.add_mutually_exclusive_group()
    inspect_target.add_argument("checkpoint", nargs="?", help="Polygram-compressed .safetensors checkpoint.")
    inspect_target.add_argument(
        "--fsm-diagram",
        action="store_true",
        help=(
            "Emit the auto-generated Mermaid diagram of the forge FSM "
            "hierarchy to stdout. Mutually exclusive with the checkpoint "
            "positional argument."
        ),
    )
    inspect.add_argument("--report", help="Write a markdown summary to this path.")

    return parser


def _cmd_forge(args: argparse.Namespace) -> int:
    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    # Build the polygram tuning bundles from the high-frequency CLI flags.
    # Long-tail tuning (jaccard_threshold, min_both_fire, …) lives behind
    # ForgePipeline.from_dict — feed it a YAML/JSON config there.
    epoch_compression = None
    epoch_kwargs = {}
    if args.coverage_target is not None:
        epoch_kwargs["coverage_target"] = args.coverage_target
    if args.cosine_threshold is not None:
        epoch_kwargs["cosine_threshold"] = args.cosine_threshold
    if args.max_compress_iterations is not None:
        epoch_kwargs["max_iterations"] = args.max_compress_iterations
    if epoch_kwargs:
        from polygram import EpochCompressionConfig

        epoch_compression = EpochCompressionConfig(**epoch_kwargs)

    regrow = None
    if args.regrow_count > 0:
        if args.regrow_layer is None:
            print(
                "sae-forge forge: --regrow-count > 0 requires --regrow-layer "
                "(no host-specific default after polygram 0.1.0).",
                file=sys.stderr,
            )
            return 2
        from polygram import RegrowConfig

        regrow_kwargs = {"model_name": args.host_model, "layer": args.regrow_layer}
        if args.regrow_strategy is not None:
            regrow_kwargs["strategy"] = args.regrow_strategy
        regrow = RegrowConfig(**regrow_kwargs)

    basis = FeatureBasis.from_polygram_checkpoint(args.checkpoint)
    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=args.host_model,
        dtype=args.dtype,
        device=args.device,
        attention_width="feature_native" if args.feature_native_attention else "host",
        epoch_compression=epoch_compression,
        regrow=regrow,
        regrow_count=args.regrow_count,
    )
    result = pipeline.run(args.output_dir)
    print(f"forged: {result.output_dir} ({result.n_params} params)")
    if result.faithfulness_kl is not None:
        print(f"faithfulness KL: {result.faithfulness_kl:.4f}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    if getattr(args, "fsm_diagram", False):
        from saeforge.machines.visualize import to_mermaid
        from saeforge.orchestrator import load_machine_hierarchy

        print(to_mermaid(load_machine_hierarchy()), end="")
        return 0

    if not args.checkpoint:
        print("error: pass either a checkpoint path or --fsm-diagram", file=sys.stderr)
        return 2

    from saeforge import FeatureBasis

    basis = FeatureBasis.from_polygram_checkpoint(args.checkpoint)
    summary = basis.to_summary()
    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).write_text(_render_inspect_markdown(args.checkpoint, summary))
    return 0


def _render_inspect_markdown(checkpoint: str, summary: dict) -> str:
    lines = [
        f"# sae-forge inspect — {checkpoint}",
        "",
        f"- kept features: **{summary['n_features']}**",
        f"- host residual width (d_model): **{summary['d_model']}**",
        f"- scale_compression_ratio: **{summary['scale_compression_ratio']:.4f}**",
        f"- merged norm mean / std: {summary['merged_norm_mean']:.4f} / {summary['merged_norm_std']:.4f}",
        f"- original norm mean: {summary['original_norm_mean']:.4f}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "forge":
        return _cmd_forge(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
