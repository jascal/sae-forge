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

    inspect = sub.add_parser(
        "inspect",
        help="Triage a compressed checkpoint without torch — basis stats only.",
    )
    inspect.add_argument("checkpoint", help="Polygram-compressed .safetensors checkpoint.")
    inspect.add_argument("--report", help="Write a markdown summary to this path.")

    return parser


def _cmd_forge(args: argparse.Namespace) -> int:
    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    basis = FeatureBasis.from_polygram_checkpoint(args.checkpoint)
    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=args.host_model,
        dtype=args.dtype,
        device=args.device,
        attention_width="feature_native" if args.feature_native_attention else "host",
    )
    result = pipeline.run(args.output_dir)
    print(f"forged: {result.output_dir} ({result.n_params} params)")
    if result.faithfulness_kl is not None:
        print(f"faithfulness KL: {result.faithfulness_kl:.4f}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
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
