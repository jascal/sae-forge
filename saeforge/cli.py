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
    # Qwen3-MoE compression strategy. See openspec/specs/qwen3-moe-support.
    forge.add_argument(
        "--moe-strategy",
        type=str,
        default="preserve",
        choices=("preserve", "collapse", "top_n"),
        help=(
            "Compression strategy when forging a Qwen3-MoE host. "
            "'preserve' (default) keeps per-expert structure with full "
            "fidelity. 'collapse' averages experts into a single dense MLP "
            "per layer (storage-aggressive, experimental). 'top_n' is a v1 "
            "placeholder that raises NotImplementedError; needs the "
            "moe-expert-calibration follow-up."
        ),
    )
    forge.add_argument(
        "--moe-keep-n",
        type=int,
        default=0,
        help="Required with --moe-strategy=top_n. Number of most-used experts to keep per layer.",
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

    # ---------------------------------------------------------------
    # Auto-materialise mode (`add-auto-materialise-sweep` capability).
    # ---------------------------------------------------------------
    sweep.add_argument(
        "--auto-materialise",
        action="store_true",
        help=(
            "Opt-in: bundle polygram's BehaviouralValidator + "
            "Compressor.plan_pareto + apply into the same invocation. "
            "Flips --encoding LABEL:PATH semantic so PATH is a single "
            "uncompressed SAE checkpoint (not a directory of "
            "k_<K>.safetensors). Required flags under this mode: "
            "--validation-prompts, --pareto, --layer."
        ),
    )
    sweep.add_argument(
        "--validation-prompts",
        type=str,
        default=None,
        help=(
            "[--auto-materialise only] JSONL/text file of prompts fed to "
            "polygram's BehaviouralValidator. Distinct from --eval-prompts "
            "by default (refused if paths resolve identically — see "
            "--allow-validation-eval-overlap)."
        ),
    )
    sweep.add_argument(
        "--pareto",
        type=str,
        default=None,
        metavar="K1,K2,...",
        help=(
            "[--auto-materialise only] Comma-separated target K list for "
            "Compressor.plan_pareto (e.g. '8,16,24,32')."
        ),
    )
    sweep.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "[--auto-materialise only] Transformer layer whose residual "
            "stream the validator hooks (e.g. 8 for GPT-2 layer-8 SAEs)."
        ),
    )
    sweep.add_argument(
        "--validation-threshold",
        type=float,
        default=None,
        help=(
            "[--auto-materialise only] polygram_overlap_threshold for the "
            "BehaviouralValidator gate. Default: 0.7 (polygram's "
            "calibration). Try 0.95 for a tighter gate on small-prompt "
            "sweeps where 0.7 over-confirms."
        ),
    )
    sweep.add_argument(
        "--validation-jaccard-threshold",
        type=float,
        default=None,
        help=(
            "[--auto-materialise only] jaccard_threshold for the validator "
            "gate. Default: 0.3."
        ),
    )
    sweep.add_argument(
        "--score-field",
        type=str,
        default=None,
        choices=("polygram_overlap", "jaccard", "decoder_overlap"),
        help=(
            "[--auto-materialise only] CompressionConfig.score_field — "
            "Pareto sort axis. Default: polygram_overlap."
        ),
    )
    sweep.add_argument(
        "--rep-selection",
        type=str,
        default=None,
        choices=("n_fires", "scale_aware", "kl_attribution"),
        help=(
            "[--auto-materialise only] CompressionConfig.rep_selection. "
            "Default: scale_aware. Use kl_attribution for "
            "behavioural-ablation-based rep selection (polygram >=0.5.0); "
            "see polygram's recon-aware-rep-selection capability for "
            "when to prefer it."
        ),
    )
    sweep.add_argument(
        "--assign-phase-knobs",
        action="store_true",
        help=(
            "[--auto-materialise only] Pass assign_phase_knobs=True to "
            "polygram's from_sae_lens (polygram >=0.6.0). Un-dormants "
            "MPS-substrate α (PC2) and φ (PC3) per-feature from decoder "
            "PCA for MPSRung1 / Rung3 / Rung4. Structural no-op for "
            "HEA_Rung2. Default off (byte-identical to polygram 0.5.0 "
            "behaviour). Flips the cache key — expect one MISS the "
            "first time you set or clear this flag."
        ),
    )
    sweep.add_argument(
        "--encoding-class",
        action="append",
        default=None,
        metavar="LABEL:CLASS",
        help=(
            "[--auto-materialise only, repeatable] Map an encoding label "
            "to a polygram encoding class. Supported: MPSRung1 (default; "
            "cap=8), Rung3 (cap=16), Rung4 (cap=32), HEA_Rung2 (cap=2^N "
            "via --encoding-qubits). For N>8 sliced SAEs, use "
            "'--encoding-class LABEL:HEA_Rung2 --encoding-qubits LABEL:N'."
        ),
    )
    sweep.add_argument(
        "--encoding-qubits",
        action="append",
        default=None,
        metavar="LABEL:N",
        help=(
            "[--auto-materialise only, repeatable, HEA_Rung2 only] "
            "n_qubits for the named encoding. cap=2^N. When omitted, "
            "the encoding is constructed with polygram's default "
            "(n_qubits=3, cap=8) — usually too small for stride-sampled "
            "SAEs; pass --encoding-qubits LABEL:5 (cap=32) or higher "
            "to match your sliced feature count."
        ),
    )
    sweep.add_argument(
        "--allow-validation-eval-overlap",
        action="store_true",
        help=(
            "[--auto-materialise only] Override the same-path refusal "
            "between --validation-prompts and --eval-prompts. Surfaces "
            "as validation_eval_overlap=True in every frontier row so "
            "downstream analysis flags the methodological compromise."
        ),
    )
    sweep.add_argument(
        "--force-rematerialise",
        action="store_true",
        help=(
            "[--auto-materialise only] Bypass the materialisation cache; "
            "re-run validator + plan_pareto + apply for every encoding "
            "regardless of cached state. Existing files overwritten in "
            "place."
        ),
    )
    sweep.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "[--auto-materialise only] Print per-encoding cache status "
            "(HIT / MISS-with-diff-fields), target K list, SHA-256 "
            "fingerprints, and validator-forward-count estimate to "
            "stderr; exit 0 without invoking validator, Compressor, or "
            "forge. Mutually exclusive with --frontier-only."
        ),
    )

    # Adaptive-regrow knobs. ``--adaptive-regrow`` is the master toggle;
    # without it the other three are inert. With it, ``--regrow-max``
    # and ``--n-features-target`` are mutually required (checked at
    # argparse level for a fast CLI failure before any model is loaded).
    forge.add_argument(
        "--adaptive-regrow",
        action="store_true",
        help=(
            "Opt in to the adaptive-regrow controller. Requires "
            "--regrow-max and --n-features-target. Defaults to off; "
            "with --adaptive-regrow off the v0.2 fixed-regrow path is "
            "byte-identical."
        ),
    )
    forge.add_argument(
        "--regrow-max",
        type=int,
        default=0,
        help=(
            "Upper bound on per-cycle effective_regrow_count when "
            "--adaptive-regrow is set. Must exceed --regrow-count."
        ),
    )
    forge.add_argument(
        "--n-features-target",
        type=int,
        default=0,
        help=(
            "Target basis size the adaptive controller grows toward. "
            "Required when --adaptive-regrow is set."
        ),
    )
    forge.add_argument(
        "--regrow-damping",
        type=float,
        default=0.5,
        help=(
            "Damping factor for the adaptive controller (default 0.5, "
            "range [0.0, 1.0]). 1.0 jumps straight to target; lower "
            "values grow asymptotically across basis-loop cycles."
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

    # Adaptive-regrow argparse-level cross-flag validation. ``argparse``
    # itself can't express "A requires B AND C" cleanly, so the check
    # lives here — runs before any basis load so the CLI fails fast.
    if args.adaptive_regrow and (
        args.regrow_max <= 0 or args.n_features_target <= 0
    ):
        print(
            "sae-forge forge: --adaptive-regrow requires both "
            "--regrow-max and --n-features-target (each > 0).",
            file=sys.stderr,
        )
        return 2

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
        moe_strategy=args.moe_strategy,
        moe_keep_n=args.moe_keep_n,
        adaptive_regrow=args.adaptive_regrow,
        regrow_max=args.regrow_max,
        n_features_target=args.n_features_target,
        regrow_damping=args.regrow_damping,
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


def _parse_label_value_specs(
    raw: list[str], *, flag_name: str
) -> dict[str, str]:
    """Parse repeated ``LABEL:VALUE`` flag strings into a dict.

    Used by ``--encoding-class`` (LABEL:CLASS) and ``--encoding-qubits``
    (LABEL:N). Splits on the first colon; rejects empty labels/values
    with an actionable error message naming the flag.
    """
    result: dict[str, str] = {}
    for entry in raw:
        if ":" not in entry:
            raise ValueError(
                f"{flag_name}: malformed entry {entry!r} (expected "
                f"'LABEL:VALUE')"
            )
        label, _, value = entry.partition(":")
        label = label.strip()
        value = value.strip()
        if not label:
            raise ValueError(f"{flag_name}: empty label in {entry!r}")
        if not value:
            raise ValueError(f"{flag_name}: empty value in {entry!r}")
        result[label] = value
    return result


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

    # --------- Auto-materialise argument validation + parsing ---------
    auto_materialise = bool(args.auto_materialise)

    # Refuse auto-materialise-only flags when --auto-materialise is absent.
    auto_only_flags_set: list[str] = []
    if not auto_materialise:
        if args.validation_prompts is not None:
            auto_only_flags_set.append("--validation-prompts")
        if args.pareto is not None:
            auto_only_flags_set.append("--pareto")
        if args.layer is not None:
            auto_only_flags_set.append("--layer")
        if args.validation_threshold is not None:
            auto_only_flags_set.append("--validation-threshold")
        if args.validation_jaccard_threshold is not None:
            auto_only_flags_set.append("--validation-jaccard-threshold")
        if args.score_field is not None:
            auto_only_flags_set.append("--score-field")
        if args.rep_selection is not None:
            auto_only_flags_set.append("--rep-selection")
        if args.assign_phase_knobs:
            auto_only_flags_set.append("--assign-phase-knobs")
        if args.encoding_class:
            auto_only_flags_set.append("--encoding-class")
        if args.encoding_qubits:
            auto_only_flags_set.append("--encoding-qubits")
        if args.allow_validation_eval_overlap:
            auto_only_flags_set.append("--allow-validation-eval-overlap")
        if args.force_rematerialise:
            auto_only_flags_set.append("--force-rematerialise")
        if args.plan_only:
            auto_only_flags_set.append("--plan-only")
        if auto_only_flags_set:
            print(
                f"sae-forge sweep-pareto: {', '.join(auto_only_flags_set)} "
                f"require --auto-materialise. Validator-tuning flags are "
                f"only valid in auto-materialise mode; for pre-materialised "
                f"sweeps, tune thresholds via 'polygram compress --pareto'.",
                file=sys.stderr,
            )
            return 2

    # Mutually exclusive: --plan-only and --frontier-only.
    if args.plan_only and args.frontier_only:
        print(
            "sae-forge sweep-pareto: --plan-only and --frontier-only are "
            "mutually exclusive (different lifecycle stages).",
            file=sys.stderr,
        )
        return 2

    # Required flags under --auto-materialise.
    auto_materialise_specs = None
    targets: list[int] | None = None
    validation_prompts_path: Path | None = None
    validation_eval_overlap = False

    if auto_materialise:
        missing: list[str] = []
        if args.validation_prompts is None:
            missing.append("--validation-prompts")
        if args.pareto is None:
            missing.append("--pareto")
        if args.layer is None:
            missing.append("--layer")
        if missing:
            print(
                f"sae-forge sweep-pareto: --auto-materialise requires "
                f"{', '.join(missing)}",
                file=sys.stderr,
            )
            return 2

        validation_prompts_path = Path(args.validation_prompts).resolve()
        if not validation_prompts_path.is_file():
            print(
                f"sae-forge sweep-pareto: --validation-prompts file not found: "
                f"{validation_prompts_path}",
                file=sys.stderr,
            )
            return 2

        # Refuse same-path resolution between validation and eval prompts
        # unless --allow-validation-eval-overlap is set.
        if args.eval_prompts is not None:
            eval_resolved = Path(args.eval_prompts).resolve()
            if eval_resolved == validation_prompts_path:
                if not args.allow_validation_eval_overlap:
                    print(
                        "sae-forge sweep-pareto: --validation-prompts and "
                        "--eval-prompts resolve to the same path. This is a "
                        "methodological leakage risk: the validator's gate "
                        "decisions would be tuned against the same prompts "
                        "that score post-forge KL. Use distinct files, or "
                        "pass --allow-validation-eval-overlap to confirm "
                        "you understand the risk (surfaces as "
                        "validation_eval_overlap=true in every row).",
                        file=sys.stderr,
                    )
                    return 2
                validation_eval_overlap = True

        # Parse --pareto K1,K2,...
        try:
            targets = [int(k.strip()) for k in args.pareto.split(",") if k.strip()]
        except ValueError:
            print(
                f"sae-forge sweep-pareto: --pareto expected comma-separated "
                f"integers, got {args.pareto!r}",
                file=sys.stderr,
            )
            return 2
        if not targets:
            print(
                "sae-forge sweep-pareto: --pareto must contain at least one K",
                file=sys.stderr,
            )
            return 2

        # Each --encoding LABEL:PATH must be a single .safetensors file
        # (not a directory — mixed mode is disallowed).
        for label, enc_path in encodings:
            if enc_path.is_dir():
                print(
                    f"sae-forge sweep-pareto: --auto-materialise expects "
                    f"--encoding LABEL:PATH where PATH is a single "
                    f".safetensors file; got directory {enc_path} for "
                    f"label={label!r}. Mixed mode (auto + pre-materialised) "
                    f"is disallowed; pick one mode per invocation.",
                    file=sys.stderr,
                )
                return 2

        # Build per-label encoding-class + encoding-kwargs maps.
        try:
            class_map = _parse_label_value_specs(
                args.encoding_class or [], flag_name="--encoding-class"
            )
            qubits_map = _parse_label_value_specs(
                args.encoding_qubits or [], flag_name="--encoding-qubits"
            )
        except ValueError as exc:
            print(f"sae-forge sweep-pareto: {exc}", file=sys.stderr)
            return 2

        from saeforge.auto_materialise import (
            AutoMaterialiseSpec,
            _ENCODING_CLASS_REGISTRY,
        )

        auto_materialise_specs = []
        for label, enc_path in encodings:
            enc_class = class_map.get(label, "MPSRung1")
            if enc_class not in _ENCODING_CLASS_REGISTRY:
                print(
                    f"sae-forge sweep-pareto: --encoding-class "
                    f"{label}:{enc_class} is not a supported class; "
                    f"supported: {sorted(_ENCODING_CLASS_REGISTRY)}",
                    file=sys.stderr,
                )
                return 2
            enc_kwargs: dict[str, object] = {}
            if label in qubits_map:
                try:
                    enc_kwargs["n_qubits"] = int(qubits_map[label])
                except ValueError:
                    print(
                        f"sae-forge sweep-pareto: --encoding-qubits "
                        f"{label}:{qubits_map[label]} must be an integer",
                        file=sys.stderr,
                    )
                    return 2
            # HEA_Rung2 requires `depth` (no polygram default). Default to
            # depth=2, the standard HEA depth; expose --encoding-depth in a
            # future PR if real callers need to tune it.
            if enc_class == "HEA_Rung2":
                enc_kwargs.setdefault("depth", 2)
            auto_materialise_specs.append(
                AutoMaterialiseSpec(
                    label=label,
                    sae_checkpoint=enc_path.resolve(),
                    encoding_class=enc_class,
                    encoding_kwargs=enc_kwargs,
                )
            )

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
    # Under --auto-materialise, the encoding paths are uncompressed SAEs;
    # FeatureBasis.from_polygram_checkpoint still works on them.
    if auto_materialise:
        bootstrap_ckpt = encodings[0][1].resolve()
    else:
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

    sweep_kwargs: dict[str, object] = dict(
        encodings=encodings,
        output_dir=Path(args.output_dir),
        frontier_only=args.frontier_only,
        quality_floor=args.quality_floor,
        quality_thresholds=quality_thresholds,
    )
    if auto_materialise:
        sweep_kwargs.update(
            auto_materialise_specs=auto_materialise_specs,
            validation_prompts=validation_prompts_path,
            validation_threshold=(
                args.validation_threshold if args.validation_threshold is not None else 0.7
            ),
            validation_jaccard_threshold=(
                args.validation_jaccard_threshold
                if args.validation_jaccard_threshold is not None else 0.3
            ),
            layer=args.layer,
            targets=targets,
            score_field=args.score_field or "polygram_overlap",
            rep_selection=args.rep_selection or "scale_aware",
            assign_phase_knobs=bool(args.assign_phase_knobs),
            validation_eval_overlap=validation_eval_overlap,
            force_rematerialise=args.force_rematerialise,
            plan_only=args.plan_only,
        )

    try:
        frontier_path = pipeline.sweep_pareto(**sweep_kwargs)
    except RuntimeError as exc:
        # At-end failure: rows are still written to frontier.jsonl.
        print(f"sae-forge sweep-pareto: {exc}", file=sys.stderr)
        # Find the frontier.jsonl path the driver wrote even when raising.
        print(str(Path(args.output_dir) / "frontier.jsonl"))
        return 1

    if args.plan_only:
        # No frontier path printed under --plan-only; stderr already
        # contains the per-encoding blocks.
        return 0

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
