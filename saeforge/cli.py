"""sae-forge console script — verbs first, file paths positional, matches polygram style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from saeforge import __version__

if TYPE_CHECKING:
    from saeforge.forge_quality import QualityThresholds  # noqa: F401


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
    # Eval signal selection. --eval-prompts (text-LM faithfulness via KL)
    # and --audio-features-path (audio faithfulness via per-frame cosine)
    # are mutually exclusive — a forge run targets one signal, not both.
    eval_group = forge.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval-prompts",
        help="JSONL file of prompts for the faithfulness eval; optional in v0.",
    )
    eval_group.add_argument(
        "--audio-features-path",
        help=(
            "Path to a torch.save'd tensor of mel features "
            "(shape: batch, n_mels=80, n_frames=3000) for the audio "
            "faithfulness eval on a Whisper-encoder forge. Loads via "
            "torch.load and passes to ForgePipeline.eval_audio_features. "
            "Mutually exclusive with --eval-prompts."
        ),
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
    # Hybrid-bridge-forge knobs. See openspec/specs/hybrid-bridge-forge.
    forge.add_argument(
        "--hybrid-bridge",
        action="store_true",
        help=(
            "Opt-in three-basis (embed/mid/lm_head) forge with learnable bridges. "
            "Requires --basis-embed and --basis-lm-head. v1 refuses tied-embedding hosts."
        ),
    )
    forge.add_argument(
        "--basis-embed",
        type=str,
        default=None,
        help="Path to the embed-anchored compressed SAE checkpoint. Required with --hybrid-bridge.",
    )
    forge.add_argument(
        "--basis-lm-head",
        type=str,
        default=None,
        help="Path to the lm-head-anchored compressed SAE checkpoint. Required with --hybrid-bridge.",
    )
    forge.add_argument(
        "--bridge-init",
        type=str,
        default="orthogonal",
        choices=("orthogonal", "identity", "zero"),
        help="BridgeModule linear-weight init (default: orthogonal).",
    )
    forge.add_argument(
        "--bridge-nonlin",
        type=str,
        default="none",
        choices=("none", "relu", "gelu"),
        help="BridgeModule activation (default: none — linear bridge).",
    )
    forge.add_argument(
        "--bridge-no-pre-ln",
        action="store_true",
        help="Disable the pre-LayerNorm in BridgeModule (default: enabled).",
    )

    sweep = sub.add_parser(
        "sweep-pareto",
        help=(
            "Forge across per-K materialised SAE checkpoints (Pareto sweep). "
            "Consumes `polygram compress --pareto --pareto-materialize` "
            "output; emits a JSONL frontier."
        ),
    )
    sweep.add_argument(
        "--encoding",
        action="append",
        required=True,
        metavar="LABEL:PATH",
        help=(
            "Repeatable. LABEL is a free-form name (e.g. mps, rung4); PATH "
            "is either a .safetensors file or a directory containing "
            "k_<K>.safetensors files (and optionally pareto.json). "
            "Pass --encoding once per encoding to sweep multiple "
            "encodings on the same coordinate system."
        ),
    )
    sweep.add_argument("--host-model", required=True, help="HuggingFace host model id.")
    sweep.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Sweep output root. frontier.jsonl is written here; per-row "
            "forge outputs land under <output-dir>/<label>/k_<K>/."
        ),
    )
    sweep.add_argument(
        "--eval-prompts",
        help="JSONL file of prompts for the faithfulness eval; optional.",
    )
    sweep.add_argument(
        "--frontier-only",
        action="store_true",
        help=(
            "Skip forge runs; emit a JSONL with manifest-derived columns "
            "only (target_n_features_kept, n_features_kept_actual, "
            "pareto_reached_target). Cheap exploratory mode."
        ),
    )
    sweep.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument(
        "--feature-native-attention",
        action="store_true",
        help="opt in to v0.2 feature-native attention; default is host-inherited internal width",
    )
    sweep.add_argument(
        "--max-encoding-warning",
        type=int,
        default=2,
        help=(
            "When --encoding is passed more than this many times, emit a "
            "stderr advisory about GPU memory pressure (large sweeps "
            "should split by encoding into separate processes). "
            "Default: 2."
        ),
    )
    sweep.add_argument(
        "--quality-floor",
        type=float,
        default=None,
        metavar="RATIO",
        help=(
            "Refuse the sweep if any encoding's smallest-K quality_ratio "
            "(basis_rank / host_d_model) falls below this float in [0, 1]. "
            "Without this flag, an advisory is printed but the sweep "
            "proceeds (default behaviour). Suggested usage: 0.5 for "
            "'I only want sweeps where every row is at least in the "
            "good tier'. Note: 'degenerate' describes the rank ratio, "
            "not the validity of the run — exploratory low-rank smokes "
            "remain valid."
        ),
    )
    sweep.add_argument(
        "--quality-tier-thresholds",
        type=str,
        default=None,
        metavar="STR",
        help=(
            "Override the default quality-tier boundaries. Format: "
            "'saturated:VAL,good:VAL,undersized:VAL'. All three names "
            "required; ordering constraint: saturated > good > "
            "undersized >= 0. Example: --quality-tier-thresholds "
            "saturated:1.0,good:0.5,undersized:0.0625 (defaults)."
        ),
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


def _parse_eval_prompts(path: Path) -> list[str]:
    """Parse a ``--eval-prompts`` file into a list of prompt strings.

    Closes #26. Three input shapes are supported in a single pass —
    each non-empty line is tried in this order:

    1. ``{"prompt": "Hello"}`` — JSON object with a ``"prompt"`` string
       field. Other dict shapes raise ``ValueError`` naming the
       expected field.
    2. ``"Hello world"`` — bare JSON string. ``json.loads`` returns
       the unquoted string.
    3. ``Hello world`` — non-JSON raw line. The ``json.JSONDecodeError``
       is caught and the line itself is used.

    Booleans, numbers, lists, and other JSON shapes raise
    ``ValueError``. The first-shape-wins ordering means a line that
    happens to be valid JSON is treated as JSON first; users who want
    raw-line semantics on JSON-looking lines should escape them
    explicitly (e.g. by writing ``"foo"`` as ``\"foo\"`` — though that
    would round-trip to the same string anyway).
    """
    prompts: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            # Shape 3: non-JSON raw line. Use the (stripped) line directly.
            prompts.append(line)
            continue
        if isinstance(parsed, str):
            # Shape 2: bare JSON string.
            prompts.append(parsed)
        elif isinstance(parsed, dict):
            # Shape 1: dict shorthand with required "prompt" field.
            if "prompt" not in parsed:
                raise ValueError(
                    f"--eval-prompts: dict entries must have a 'prompt' "
                    f"key with a string value; got keys "
                    f"{sorted(parsed.keys())!r} on line {raw!r}"
                )
            value = parsed["prompt"]
            if not isinstance(value, str):
                raise ValueError(
                    f"--eval-prompts: dict 'prompt' field must be a string; "
                    f"got {type(value).__name__} on line {raw!r}"
                )
            prompts.append(value)
        else:
            raise ValueError(
                f"--eval-prompts: entries must be a JSON string, a dict "
                f"with a 'prompt' field, or a raw text line; got JSON "
                f"{type(parsed).__name__} on line {raw!r}"
            )
    return prompts


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

    # v0.4 forge-whisper-encoder: load pre-extracted mel features for the
    # audio faithfulness path. torch is lazy-imported so the non-audio
    # CLI path keeps working without the [torch] extra exercised here.
    eval_prompts = (
        _parse_eval_prompts(Path(args.eval_prompts))
        if args.eval_prompts
        else []
    )

    eval_audio_features = None
    if args.audio_features_path is not None:
        try:
            import torch
        except ImportError:
            print(
                "sae-forge forge: --audio-features-path requires the "
                "[torch] extra (or [intel] on x86_64 macOS).",
                file=sys.stderr,
            )
            return 2
        eval_audio_features = torch.load(
            args.audio_features_path, map_location="cpu"
        )

    # Hybrid-bridge wiring. The mutually-required check is here (not argparse-level)
    # so we can produce a clear actionable message naming both missing flags at once.
    hybrid_kwargs = {}
    if args.hybrid_bridge:
        missing = []
        if args.basis_embed is None:
            missing.append("--basis-embed")
        if args.basis_lm_head is None:
            missing.append("--basis-lm-head")
        if missing:
            print(
                f"sae-forge forge: --hybrid-bridge requires {' and '.join(missing)}.",
                file=sys.stderr,
            )
            return 2
        from saeforge.bridges import BridgeConfig

        hybrid_kwargs = dict(
            hybrid_bridge=True,
            basis_embed=FeatureBasis.from_polygram_checkpoint(args.basis_embed),
            basis_lm_head=FeatureBasis.from_polygram_checkpoint(args.basis_lm_head),
            bridge_config=BridgeConfig(
                init=args.bridge_init,
                nonlin=args.bridge_nonlin,
                pre_layernorm=not args.bridge_no_pre_ln,
                train=True,
            ),
        )

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
        eval_prompts=eval_prompts,
        eval_audio_features=eval_audio_features,
        **hybrid_kwargs,
    )
    result = pipeline.run(args.output_dir)
    print(f"forged: {result.output_dir} ({result.n_params} params)")
    if result.faithfulness_kl is not None:
        print(f"faithfulness KL: {result.faithfulness_kl:.4f}")
    return 0


def _parse_quality_tier_thresholds(raw: str) -> "QualityThresholds":
    """Parse ``saturated:VAL,good:VAL,undersized:VAL`` into a QualityThresholds.

    All three names required; ordering constraint enforced by
    QualityThresholds.__post_init__. Raises ``ValueError`` with a clear
    message + corrected example on any malformation.
    """
    from saeforge.forge_quality import QualityThresholds

    expected_names = {"saturated", "good", "undersized"}
    parts: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--quality-tier-thresholds: malformed entry {chunk!r} "
                f"(expected 'name:value'). Format: "
                f"'saturated:VAL,good:VAL,undersized:VAL'. "
                f"Example: --quality-tier-thresholds "
                f"saturated:1.0,good:0.5,undersized:0.0625"
            )
        name, _, value_str = chunk.partition(":")
        name = name.strip()
        if name not in expected_names:
            raise ValueError(
                f"--quality-tier-thresholds: unknown name {name!r}; "
                f"required names are saturated, good, undersized. "
                f"Example: --quality-tier-thresholds "
                f"saturated:1.0,good:0.5,undersized:0.0625"
            )
        try:
            parts[name] = float(value_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"--quality-tier-thresholds: cannot parse {value_str!r} as "
                f"float for name {name!r}: {exc}. Example: "
                f"--quality-tier-thresholds "
                f"saturated:1.0,good:0.5,undersized:0.0625"
            ) from None

    missing = expected_names - parts.keys()
    if missing:
        raise ValueError(
            f"--quality-tier-thresholds: missing required name(s) "
            f"{sorted(missing)}; all three (saturated, good, undersized) "
            f"must be present. Ordering constraint: saturated > good > "
            f"undersized >= 0. Example: --quality-tier-thresholds "
            f"saturated:1.0,good:0.5,undersized:0.0625"
        )

    # Lets QualityThresholds.__post_init__ enforce the ordering invariant
    # and raise a focused error if violated.
    return QualityThresholds(
        saturated=parts["saturated"],
        good=parts["good"],
        undersized=parts["undersized"],
    )


def _parse_encoding_specs(raw: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``--encoding LABEL:PATH`` into normalized tuples.

    Splits on the FIRST ``:`` so paths containing colons (Windows-style
    drives, URIs) work. Raises ``ValueError`` on malformed specs — the CLI
    handler converts that to a non-zero exit.
    """
    out: list[tuple[str, Path]] = []
    for spec in raw:
        if ":" not in spec:
            raise ValueError(
                f"--encoding spec must be LABEL:PATH (no colon found): {spec!r}"
            )
        label, _, path = spec.partition(":")
        if not label:
            raise ValueError(f"--encoding spec has empty label: {spec!r}")
        if not path:
            raise ValueError(f"--encoding spec has empty path: {spec!r}")
        out.append((label, Path(path)))
    return out


def _cmd_sweep_pareto(args: argparse.Namespace) -> int:
    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector
    from saeforge.sweep import _enumerate_checkpoints

    try:
        encodings = _parse_encoding_specs(args.encoding)
    except ValueError as exc:
        print(f"sae-forge sweep-pareto: {exc}", file=sys.stderr)
        return 2

    # Forge-quality argument parsing.
    if args.quality_floor is not None:
        if not (0.0 <= args.quality_floor <= 1.0):
            print(
                f"sae-forge sweep-pareto: --quality-floor must be in [0, 1]; "
                f"got {args.quality_floor}",
                file=sys.stderr,
            )
            return 2

    quality_thresholds = None
    if args.quality_tier_thresholds is not None:
        try:
            quality_thresholds = _parse_quality_tier_thresholds(args.quality_tier_thresholds)
        except ValueError as exc:
            print(f"sae-forge sweep-pareto: {exc}", file=sys.stderr)
            return 2

    if len(encodings) > args.max_encoding_warning:
        print(
            f"sae-forge sweep-pareto: warning — {len(encodings)} encodings in one "
            f"process; large hosts may hit GPU memory limits. Consider splitting "
            f"into one process per --encoding (see design.md Risks).",
            file=sys.stderr,
        )

    # Bootstrap: use the first encoding's first checkpoint as the basis the
    # ForgePipeline is constructed with. The sweep driver hot-swaps basis +
    # projector per row, so this is purely a construction-time placeholder.
    try:
        first_checkpoints = _enumerate_checkpoints(encodings[0][1])
    except (FileNotFoundError, ValueError) as exc:
        print(f"sae-forge sweep-pareto: {exc}", file=sys.stderr)
        return 2

    bootstrap_ckpt = first_checkpoints[0][1]
    basis = FeatureBasis.from_polygram_checkpoint(bootstrap_ckpt)
    projector = SubspaceProjector(basis)

    eval_prompts = (
        _parse_eval_prompts(Path(args.eval_prompts))
        if args.eval_prompts
        else []
    )

    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=args.host_model,
        dtype=args.dtype,
        device=args.device,
        attention_width="feature_native" if args.feature_native_attention else "host",
        eval_prompts=eval_prompts,
    )

    try:
        frontier_path = pipeline.sweep_pareto(
            encodings=encodings,
            output_dir=Path(args.output_dir),
            frontier_only=args.frontier_only,
            quality_floor=args.quality_floor,
            quality_thresholds=quality_thresholds,
        )
    except RuntimeError as exc:
        # At-end failure: rows are still written to frontier.jsonl.
        print(f"sae-forge sweep-pareto: {exc}", file=sys.stderr)
        # Find the frontier.jsonl path the driver wrote even when raising.
        print(str(Path(args.output_dir) / "frontier.jsonl"))
        return 1

    print(str(frontier_path))
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
    if args.command == "sweep-pareto":
        return _cmd_sweep_pareto(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
