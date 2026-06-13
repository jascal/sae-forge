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


def _parse_composition_heads(spec: str):
    """Map the --composition-heads CLI string to a ForgePipeline value.

    'prev-token' / 'duplicate-token' / 'all' pass through as strings; a comma list of 'L.H' tokens
    (e.g. '4.11,2.2') becomes an explicit [(layer, head), ...] writer list.
    """
    if spec in ("prev-token", "duplicate-token", "all"):
        return spec
    heads = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "." not in tok:
            raise argparse.ArgumentTypeError(
                f"--composition-heads: explicit writer {tok!r} must be 'L.H' (e.g. '4.11'); "
                "or use a preset ('prev-token' / 'duplicate-token') or 'all'."
            )
        L, h = tok.split(".", 1)
        heads.append((int(L), int(h)))
    if not heads:
        raise argparse.ArgumentTypeError(
            f"--composition-heads={spec!r} parsed to no writer heads; "
            "use 'L.H,...', a preset, or 'all'."
        )
    return heads


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
        "regrower, interpreted as resid_pre of that block (a "
        "blocks.N.hook_resid_post SAE needs layer = N+1). Required when "
        "--regrow-count > 0; the polygram-side GPT-2-specific layer=10 "
        "default was removed in 0.1.0.",
    )
    forge.add_argument(
        "--regrow-strategy",
        type=str,
        default=None,
        help="RegrowConfig.strategy (default: residual_kmeans).",
    )
    forge.add_argument(
        "--regrow-n-init",
        type=int,
        default=None,
        help=(
            "RegrowConfig.n_init (polygram default: 4). sm-sae recommends "
            "8+ for LLM-scale SAEs. Implicitly set by --llm-scale unless "
            "explicitly passed."
        ),
    )
    # sm-sae LLM-scale preset. Bumps a small set of provisional defaults
    # that the sm-sae fixture page recommends for LLM-scale (thousands of
    # features) SAEs. Each individual flag still wins if explicitly set.
    forge.add_argument(
        "--llm-scale",
        action="store_true",
        help=(
            "Apply sm-sae provisional LLM-scale defaults: cosine_threshold=0.85, "
            "regrow.n_init=8. Explicit flag values still win. The sm-sae "
            "page also recommends save_intermediate_reports=True, but that "
            "knob isn't plumbed through ForgePipeline yet — out of scope "
            "for this flag. See https://jascal.github.io/sm-sae/ for the "
            "full recommendation table."
        ),
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
    # two-basis-forge knobs. See openspec/specs/composition-subspace-preserve.
    forge.add_argument(
        "--composition-preserve",
        action="store_true",
        help=(
            "Opt-in: preserve the host attention QK/OV read+write geometry (U_C) verbatim "
            "inside the projection so the forged circuits stay faithful (default off)."
        ),
    )
    forge.add_argument(
        "--composition-rank",
        type=int,
        default=None,
        help="Per-side rank cap for U_C (default: energy-knee).",
    )
    forge.add_argument(
        "--composition-heads",
        type=str,
        default="prev-token",
        help=(
            "Writer-head selector for U_C: a behavioral preset ('prev-token' (default) / "
            "'duplicate-token') detected on the eval corpus, a comma-separated 'L.H' list of explicit "
            "(layer, head) writers, or 'all' (legacy reader-geometry, weaker). "
            "Examples: 'prev-token' | '4.11,2.2' | 'all'."
        ),
    )
    forge.add_argument(
        "--composition-mode",
        type=str,
        default="writer-output",
        choices=("writer-output", "reader-geometry"),
        help=(
            "How U_C is built. 'writer-output' (default, validated): the circuit writer heads' "
            "OV-output row space. 'reader-geometry' (legacy/ablation): the aggregate per-layer "
            "QK/OV read+write geometry — does NOT protect circuits."
        ),
    )
    forge.add_argument(
        "--assertion-preserve",
        action="store_true",
        help="Opt-in: keep the top --assertion-k sharpest basis atoms verbatim (recovers cov95).",
    )
    forge.add_argument(
        "--assertion-k",
        type=int,
        default=0,
        help="Number of sharp atoms to preserve verbatim when --assertion-preserve is set.",
    )
    forge.add_argument(
        "--circuit-faithfulness",
        action="store_true",
        help="Emit the two-basis preserved-dimension budget + U_C∩basis overlap report.",
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
    # add-host-wrapped-forge-fallback. Default 'auto' dispatches by basis
    # quality tier: good/saturated → native_in_basis (existing path);
    # undersized/degenerate → host_wrapped (wraps host's exact transformer
    # with decode/encode at every block boundary). Forces with explicit
    # values for regression / debug. See
    # openspec/changes/add-host-wrapped-forge-fallback/proposal.md.
    forge.add_argument(
        "--forward-mode",
        type=str,
        default="auto",
        choices=("auto", "native_in_basis", "host_wrapped"),
        help=(
            "Forward implementation for the forged model. 'auto' (default) "
            "picks 'native_in_basis' for good/saturated basis quality and "
            "'host_wrapped' for undersized/degenerate. 'host_wrapped' is "
            "GPT-2 only in v1 and inference-only (finetune raises). See "
            "openspec/changes/add-host-wrapped-forge-fallback."
        ),
    )

    sweep = sub.add_parser(
        "sweep-pareto",
        help=(
            "Forge across per-K materialised SAE checkpoints (Pareto sweep). "
            "Consumes `polygram compress --pareto --pareto-materialize` "
            "output; emits a JSONL frontier. Each row carries forge-quality "
            "diagnostics (basis_rank, quality_ratio, quality_tier) AND "
            "polygram concept-structure diagnostics (polygram_n_clusters, "
            "polygram_redundancy_ratio, polygram_encoding_capacity) when "
            "the per-encoding compression report is available — see the "
            "Pareto-sweep section of the README for jq recipes."
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
    sweep.add_argument(
        "--magnitude-diagnostics",
        type=str,
        default=None,
        metavar="VALUE",
        help=(
            "Opt-in forge-magnitude diagnostics "
            "(fix-scale-boost-calibration). Accepts 'tokens:N' to use "
            "the built-in token-capped corpus (N tokens, default 1024) "
            "or 'prompts:PATH' to load JSONL prompts from PATH. When "
            "set, each row's logit_std_ratio (forged-logit std vs "
            "host-logit std on the calibration corpus) and "
            "top1_anomalous (mode top-1 prediction in the curated "
            "SolidGoldMagikarp-family set) fields are populated. "
            "Requires --layer and a host model id resolvable from "
            "--host-model. These are post-mortem diagnostics — they "
            "don't change scale_boost, just surface why a forge KL "
            "might be poor."
        ),
    )
    sweep.add_argument(
        "--rank-monotonicity-check",
        action="store_true",
        help=(
            "After the sweep completes, verify that within each "
            "encoding label, faithfulness_kl is non-increasing in "
            "n_features_kept_actual up to a 0.1-nat tolerance. "
            "Violations print a stderr advisory listing the offending "
            "(label, K_low, K_high, KL_low, KL_high) tuples — advisory "
            "only, no refusal. Useful for catching the documented "
            "blow-up pattern at default scale_boost=1.0."
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
            "stream the validator hooks. Interpreted as resid_pre of that "
            "block: a blocks.N.hook_resid_pre SAE uses --layer N; a "
            "blocks.N.hook_resid_post SAE uses --layer N+1. A mismatch only "
            "warns (faithfulness silently degrades)."
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
        "--assign-amp-knobs",
        action="store_true",
        help=(
            "[--auto-materialise only] Pass assign_amp_knobs=True to "
            "polygram's from_sae_lens (polygram >=0.6.0). Un-dormants "
            "MPS-substrate amp_knobs (PC4+) per-feature from decoder PCA "
            "for MPSRung1 / Rung3 / Rung4. Structural no-op for "
            "HEA_Rung2. Default on (recommended for MPS-substrate SAEs; "
            "see polygram v0.6.0 release notes). Flips the cache key — "
            "expect one MISS the first time you set or clear this flag."
        ),
    )
    sweep.add_argument(
        "--learn-axis-assignment",
        action="store_true",
        help=(
            "[--auto-materialise only] Pass learn_axis_assignment=True to "
            "polygram's from_sae_lens (polygram >=0.8.0). Replaces the "
            "hardcoded PC2→α / PC3→φ / PC4..→amp_knobs permutation with "
            "a greedy axis-to-knob search that maximises decoder-Gram "
            "Spearman. On synthetic SAEs the prototype lifts Spearman "
            "by ~3× while improving gram conditioning ~10 decades; "
            "real-SAE replication is the gate before flipping to default. "
            "Default off. HEA_Rung2 falls back to the hardcoded helpers. "
            "Flips the cache key — expect one MISS the first time you "
            "set or clear this flag. See polygram's "
            "docs/research/learned-axis-assignment.md."
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
            "cap=8), Rung3 (cap=16), Rung4 (cap=32), Rung5 (cap=8·2^k "
            "via --encoding-amp-qubits), HEA_Rung2 (cap=2^N via "
            "--encoding-qubits). For Rung5 sweeps, use "
            "'--encoding-class LABEL:Rung5 --encoding-amp-qubits LABEL:K'."
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
        "--encoding-amp-qubits",
        action="append",
        default=None,
        metavar="LABEL:K",
        help=(
            "[--auto-materialise only, repeatable, Rung5 only] "
            "n_amp_qubits for the named encoding. Per-feature Hilbert "
            "dim becomes 8·2^k. Required when --encoding-class LABEL:Rung5 "
            "is set (Rung5 has no default amp-width). Polygram caps k at "
            "RUNG5_MAX_N_AMP_QUBITS=16 (cap=524288)."
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

    # ------------------------------------------------------------------
    # sweep-capability — Pareto sweep on downstream-task retention.
    # Added by add-downstream-capability-target. Mirrors sweep-pareto's
    # contract but uses DownstreamCapabilityTarget as the metric.
    # ------------------------------------------------------------------
    cap = sub.add_parser(
        "sweep-capability",
        help=(
            "Capability-aware Pareto sweep. Scores forged models by "
            "per-feature × per-label AUC through a downstream task "
            "encoder (typically a trained SAE), not by cosine / KL "
            "faithfulness. Bio-sae's empirical study showed those two "
            "Pareto frontiers disagree by up to 16× on optimal width "
            "(see openspec/changes/add-downstream-capability-target)."
        ),
    )
    cap.add_argument(
        "--dataset-config",
        required=True,
        metavar="PATH",
        help=(
            "YAML config describing the capability dataset. Required "
            "keys: encoder_checkpoint, sequences_path, labels_path. "
            "Optional: feed (pooled|residue), tokenizer_id, "
            "aggregator (pool_then_encode|encode_then_pool), "
            "min_prevalence, sae_variant (topk|jumprelu|l1), sae_k. "
            "See openspec/changes/add-downstream-capability-target/"
            "proposal.md §5 for a complete example."
        ),
    )
    cap.add_argument(
        "--host",
        required=True,
        help="HuggingFace host model id (or local path).",
    )
    cap.add_argument(
        "--widths",
        required=True,
        help="Comma-separated basis widths to sweep (e.g. '16,64,128,256').",
    )
    cap.add_argument(
        "--scale-boosts",
        default="1.0,auto",
        help=(
            "Comma-separated scale_boost values. Floats or the literal "
            "'auto'. Default: '1.0,auto'."
        ),
    )
    cap.add_argument(
        "--encodings",
        default="raw_slice",
        help=(
            "Legacy informational encoding labels (comma-separated). "
            "Use --encoding LABEL:PATH (repeatable) for the v0.10+ "
            "multi-encoding mode that compares different polygram "
            "encodings / partition shadows in one sweep call. Default: "
            "'raw_slice' (single informational label)."
        ),
    )
    cap.add_argument(
        "--encoding",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help=(
            "Repeatable. LABEL is a free-form name (e.g. raw_slice, "
            "partition_q4, mps_rung1_x16); PATH is an SAE checkpoint "
            "(.pt or .safetensors). Pass --encoding once per encoding "
            "to COMPARE multiple encodings in one sweep call. Mirrors "
            "sweep-pareto's --encoding flag. When provided, supersedes "
            "the YAML config's encoder_checkpoint (with a stderr "
            "warning). Tiebreaker for the recommend output: CLI flag "
            "order is the user-supplied priority."
        ),
    )
    cap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Count expected cells + benchmark ONE cell + project total "
            "wall-time + optional cost (via --dollars-per-gpu-hr). "
            "Exits 0 without running the full sweep. ~30 seconds; use "
            "before committing to a multi-encoding sweep at scale. "
            "Spec'd in add-multi-encoding-capability-sweep/specs/"
            "pareto-sweep/spec.md."
        ),
    )
    cap.add_argument(
        "--dollars-per-gpu-hr",
        type=float, default=None,
        help=(
            "Optional cost rate for --dry-run projection. Populates the "
            "estimated_cost_usd column in the projection table. "
            "Informational; not enforced."
        ),
    )
    cap.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Sweep output root. frontier.jsonl is written here; the "
            "host-extraction cache lands under <output-dir>/host_cache/."
        ),
    )
    cap.add_argument(
        "--no-host-cache",
        action="store_true",
        help=(
            "Disable host-extraction caching across sweep cells. Use "
            "when the host model is non-deterministic or when disk is "
            "scarce."
        ),
    )
    cap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Truncate input sequences to this length (default: 512).",
    )
    cap.add_argument(
        "--device", default="cpu",
        help="torch device for forge + extraction (default: 'cpu').",
    )
    cap.add_argument(
        "--train-encoder",
        action="store_true",
        help=(
            "Fit a capability-trained encoder per cell (the 'supervised forge', "
            "add-capability-trained-encoder) instead of using only the Frobenius pinv(W_dec). "
            "Each cell reports retained_mauc_trained ALONGSIDE the always-computed pinv baseline, "
            "both on a HELD-OUT split (the gate); the trained E is saved as a <cell>.encoder.npy "
            "sidecar. A trained row is only better when its held-out delta_heldout clears the "
            "pinv baseline (see `recommend --trained-margin`); a tie is a valid descriptive pass."
        ),
    )
    cap.add_argument(
        "--basis-order",
        choices=["row_norm", "readout_aligned"],
        default="row_norm",
        help=(
            "How the width slice ranks kept features. 'row_norm' (default) = L2 decoder norm. "
            "'readout_aligned' = alignment with the host's readout (decode-decision) geometry, "
            "sourced from the host unembed (load_host_unembed). NOTE: encoder-only hosts "
            "(ESM-2, Whisper) have no vocabulary unembed — 'readout_aligned' then RAISES unless "
            "--readout-fallback is given (it never silently reverts to row_norm)."
        ),
    )
    cap.add_argument(
        "--readout-fallback",
        choices=["downstream_decode"],
        default=None,
        help=(
            "Only with --basis-order readout_aligned on an encoder-only host with no unembed: "
            "'downstream_decode' orders by the SAE's own decode geometry instead of raising "
            "(emits a one-shot warning). Explicit opt-in; default raises."
        ),
    )
    cap.add_argument(
        "--train-steps", type=int, default=300,
        help="Steps for --train-encoder per cell (default: 300; early-stops on held-out plateau).",
    )
    cap.add_argument(
        "--train-objective",
        choices=["proxy", "full_forge"],
        default="proxy",
        help=(
            "Objective for --train-encoder (add-full-forge-encoder-training). 'proxy' (default) fits E on "
            "the cheap activation path host_encoder((x@E)@W_dec)~host_encoder(x). 'full_forge' fits E through "
            "the FULL differentiable forge (E applied to the host weights -> forged forward) — the objective "
            "that matches the eval metric — esm2-only in v1, far heavier, pooled feed."
        ),
    )

    # ------------------------------------------------------------------
    # recommend — pick the smallest-parameter row meeting a predicate.
    # ------------------------------------------------------------------
    rec = sub.add_parser(
        "recommend",
        help=(
            "Recommend a forge config from a sweep frontier. Filters by "
            "predicate(s) on ParetoFrontierRow fields, returns the row "
            "minimising n_params_forged (fallback: "
            "target_n_features_kept)."
        ),
    )
    rec.add_argument(
        "--frontier",
        required=True,
        metavar="PATH",
        help="Path to a frontier.jsonl produced by sweep-pareto or sweep-capability.",
    )
    rec.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="EXPR",
        help=(
            "Predicate over ParetoFrontierRow fields. Format: "
            "FIELD<OP>VALUE where OP ∈ {>=, <=, ==, <, >}. Field "
            "names accept kebab-case or snake_case (retained-mauc, "
            "gap_p95, etc.). Repeat for AND-combined predicates: "
            "--target retained-mauc>=0.9 --target gap-p95<=0.05."
        ),
    )
    rec.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit the picked row as JSON instead of a tabular summary.",
    )
    rec.add_argument(
        "--accept-unconverged",
        action="store_true",
        help=(
            "Applies ONLY to progressive frontiers (those carrying the "
            "'stage' field on rows, emitted by "
            "sweep-capability-progressive). Single-shot frontiers from "
            "sweep-pareto / sweep-capability are always accepted; this "
            "flag is a no-op for them. For progressive frontiers: "
            "accept a recommendation that didn't converge across the "
            "configured stage budget. Default: refuse with a diagnostic "
            "explaining which stage's argmin-plateau-member shifted. "
            "Use only when you've separately verified the recommendation "
            "is appropriate for your workflow."
        ),
    )
    rec.add_argument(
        "--trained-margin",
        type=float,
        default=0.02,
        metavar="DELTA",
        help=(
            "Held-out retained-mAUC margin a capability-trained row "
            "(--train-encoder sweep) must clear over its pinv baseline to be "
            "preferred over the plain (pinv) forge. A trained row is chosen only "
            "when delta_heldout > DELTA AND overfit_flag is False; otherwise the "
            "simpler pinv forge wins (ties default to pinv). Default: 0.02."
        ),
    )

    # ------------------------------------------------------------------
    # sweep-capability-progressive — multi-stage capability sweep.
    # Added by add-progressive-capability-sweep. Returns a STABLE
    # recommendation (smallest n robust to data scale), not argmax-on-
    # one-sample.
    # ------------------------------------------------------------------
    prog = sub.add_parser(
        "sweep-capability-progressive",
        help=(
            "Multi-stage capability sweep. Progressively grows protein "
            "count + narrows active widths until the recommendation "
            "STOPS SHIFTING. Returns the smallest target_n_features_kept "
            "stable across the last K stages — the Pareto-optimal point "
            "on (capability, parameter-cost). See "
            "openspec/changes/add-progressive-capability-sweep/proposal."
            "md for the empirical motivation (bio-sae's n=10 -> n=100 "
            "argmax-drift surfaced this design)."
        ),
    )
    prog.add_argument(
        "--dataset-config", required=True, metavar="PATH",
        help=(
            "YAML config (same schema as sweep-capability). Required "
            "keys: encoder_checkpoint, sequences_path, labels_path. "
            "Optional: feed (pooled|residue), tokenizer_id, aggregator, "
            "min_prevalence, sae_variant, sae_k."
        ),
    )
    prog.add_argument(
        "--host", required=True,
        help="HuggingFace host model id (or local path).",
    )
    prog.add_argument(
        "--candidate-widths", required=True,
        help=(
            "Comma-separated basis widths to consider. The progressive "
            "wrapper only PRUNES + EXPANDS-TO-NEIGHBOURS within this "
            "list; it does not invent widths. E.g. "
            "'4,8,16,32,64,128,256,512,1024'."
        ),
    )
    prog.add_argument(
        "--schedule", required=True,
        help=(
            "Comma-separated protein counts per stage (monotone non-"
            "decreasing). E.g. '10,50,200,1000' is the bio-sae-"
            "calibrated default. Single element '200' degenerates to "
            "single-shot at 200 proteins."
        ),
    )
    prog.add_argument(
        "--scale-boosts", default="1.0",
        help="Comma-separated scale_boost values (default: '1.0').",
    )
    prog.add_argument(
        "--encodings", default="raw_slice",
        help=(
            "Legacy informational encoding labels (comma-separated). "
            "Use --encoding LABEL:PATH (repeatable) for the v0.10+ "
            "multi-encoding mode. Default: 'raw_slice'."
        ),
    )
    prog.add_argument(
        "--encoding",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help=(
            "Repeatable. LABEL is a free-form name; PATH is an SAE "
            "checkpoint. Pass --encoding once per encoding to COMPARE "
            "multiple encodings across data scales in one progressive "
            "sweep call. Mirrors sweep-pareto's --encoding flag. When "
            "provided, supersedes the YAML config's encoder_checkpoint "
            "(with stderr warning). Tiebreaker for winner pick: CLI "
            "flag order is the user-supplied priority."
        ),
    )
    prog.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Project total wall-time without running the full sweep. "
            "Counts cells across (encodings × widths × scale_boosts × "
            "stages), benchmarks ONE cell, multiplies. ~30 seconds. "
            "Useful before committing to a multi-encoding multi-stage "
            "sweep at scale."
        ),
    )
    prog.add_argument(
        "--dollars-per-gpu-hr",
        type=float, default=None,
        help=(
            "Optional cost rate for --dry-run projection. Populates "
            "estimated_cost_usd in the projection table."
        ),
    )
    prog.add_argument(
        "--retained-mauc-tolerance", type=float, default=0.005,
        help=(
            "Caps the max-pairwise-difference in retained_mauc across "
            "the trailing convergence_n_stages stages (default: 0.005)."
        ),
    )
    prog.add_argument(
        "--plateau-tolerance", type=float, default=0.01,
        help=(
            "Defines a band around the peak retained_mauc as 'tied for "
            "first' (default: 0.01 = 1%% AUC). Loosen for flat plateaus."
        ),
    )
    prog.add_argument(
        "--min-plateau-widths", type=int, default=3,
        help=(
            "Floor on plateau-size; widens effective tolerance when the "
            "natural plateau is too narrow (default: 3)."
        ),
    )
    prog.add_argument(
        "--convergence-n-stages", type=int, default=2,
        help=(
            "Number of consecutive stable stages required for "
            "convergence (default: 2; recommended production: 2 or 3). "
            "=1 is an explicit looser opt-out, not a recommended "
            "default."
        ),
    )
    prog.add_argument(
        "--output-dir", required=True,
        help=(
            "Where frontier.jsonl + progressive_summary.json land. "
            "Per-stage forge outputs under <output-dir>/stage_<K>/."
        ),
    )
    prog.add_argument(
        "--no-host-cache", action="store_true",
        help="Disable host-extraction cache across sweep cells.",
    )
    prog.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Truncate input sequences (default: 512).",
    )
    prog.add_argument(
        "--device", default="cpu",
        help="torch device for forge + extraction (default: 'cpu').",
    )

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

    # sm-sae --llm-scale preset: apply provisional LLM-scale defaults that
    # individual flags can still override. Mutates args in place before
    # the polygram tuning bundles get built below.
    if args.llm_scale:
        if args.cosine_threshold is None:
            args.cosine_threshold = 0.85
        if args.regrow_n_init is None:
            args.regrow_n_init = 8

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
        if args.regrow_n_init is not None:
            regrow_kwargs["n_init"] = args.regrow_n_init
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

    two_basis_kwargs: dict = {}
    if args.composition_preserve or args.assertion_preserve:
        heads = _parse_composition_heads(args.composition_heads)
        two_basis_kwargs = dict(
            composition_preserve=args.composition_preserve,
            assertion_preserve=args.assertion_preserve,
            composition_rank=args.composition_rank,
            composition_heads=heads,
            composition_mode=args.composition_mode,
            assertion_k=args.assertion_k,
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
        forward_mode=args.forward_mode,
        adaptive_regrow=args.adaptive_regrow,
        regrow_max=args.regrow_max,
        n_features_target=args.n_features_target,
        regrow_damping=args.regrow_damping,
        eval_prompts=eval_prompts,
        eval_audio_features=eval_audio_features,
        **hybrid_kwargs,
        **two_basis_kwargs,
    )
    result = pipeline.run(args.output_dir)
    print(f"forged: {result.output_dir} ({result.n_params} params)")
    resolved = getattr(result, "resolved_forward_mode", None)
    if resolved is None:
        # Pipeline-result schema may not surface this directly; pull from
        # the in-memory model if available.
        resolved = getattr(pipeline, "_last_resolved_forward_mode", None)
    if resolved is not None:
        print(f"forward_mode: {resolved}")
    if result.faithfulness is not None:
        target = result.faithfulness_target_name or "faithfulness"
        print(f"{target}: {result.faithfulness:.4f}")
    if args.circuit_faithfulness:
        rep = getattr(pipeline, "_last_augmented_report", None)
        if rep is None:
            print("circuit-faithfulness: no two-basis preserve active")
        else:
            print(f"circuit-faithfulness (d_model={rep['d_model']}):")
            for ell, lr in sorted(rep["layers"].items()):
                print(
                    f"  layer {ell}: preserved_dim={lr['preserved_dim']} "
                    f"({lr['preserved_fraction']:.1%})  "
                    f"U_C∩basis={lr['U_C_overlap_with_basis']:.2f}"
                )
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
        # --layer is also required by --magnitude-diagnostics (the
        # calibration corpus is hooked at this residual-stream layer).
        # Allow it when diagnostics are in use even outside --auto-materialise.
        if args.layer is not None and args.magnitude_diagnostics is None:
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
        if args.assign_amp_knobs:
            auto_only_flags_set.append("--assign-amp-knobs")
        if args.learn_axis_assignment:
            auto_only_flags_set.append("--learn-axis-assignment")
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
            amp_qubits_map = _parse_label_value_specs(
                args.encoding_amp_qubits or [],
                flag_name="--encoding-amp-qubits",
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
            if label in amp_qubits_map:
                try:
                    enc_kwargs["n_amp_qubits"] = int(amp_qubits_map[label])
                except ValueError:
                    print(
                        f"sae-forge sweep-pareto: --encoding-amp-qubits "
                        f"{label}:{amp_qubits_map[label]} must be an integer",
                        file=sys.stderr,
                    )
                    return 2
            # HEA_Rung2 requires `depth` (no polygram default). Default to
            # depth=2, the standard HEA depth; expose --encoding-depth in a
            # future PR if real callers need to tune it.
            if enc_class == "HEA_Rung2":
                enc_kwargs.setdefault("depth", 2)
            # Rung5 requires `n_amp_qubits` (no polygram default).
            # When --encoding-amp-qubits was not supplied for a Rung5
            # encoding label, refuse rather than silently picking a
            # value — k materially changes the per-feature Hilbert
            # dim (8·2^k) so the user must state intent explicitly.
            if enc_class == "Rung5" and "n_amp_qubits" not in enc_kwargs:
                print(
                    f"sae-forge sweep-pareto: --encoding-class "
                    f"{label}:Rung5 requires --encoding-amp-qubits "
                    f"{label}:K (Rung5 has no default amp-width).",
                    file=sys.stderr,
                )
                return 2
            auto_materialise_specs.append(
                AutoMaterialiseSpec(
                    label=label,
                    sae_checkpoint=enc_path.resolve(),
                    encoding_class=enc_class,
                    encoding_kwargs=enc_kwargs,
                )
            )

        # --learn-axis-assignment is a silent no-op on HEA_Rung2:
        # polygram's LearnedKnobAssignment.assign falls back to
        # ClusteredKnobAssignment when isinstance(encoding, HEA_Rung2)
        # (polygram/geometry/learned_axis_assignment.py — "known v1
        # limitation"). The fallback is logged at info level only, so a
        # sweep with --learn-axis-assignment + HEA_Rung2 produces a
        # cache-key MISS and re-materialises but yields a Dictionary
        # bit-identical to the OFF arm. Refuse rather than let a user
        # burn the compute and infer a null effect.
        if args.learn_axis_assignment:
            hea_labels = [
                spec.label
                for spec in auto_materialise_specs
                if spec.encoding_class == "HEA_Rung2"
            ]
            if hea_labels:
                print(
                    f"sae-forge sweep-pareto: --learn-axis-assignment is a "
                    f"no-op with HEA_Rung2 encodings (polygram's "
                    f"LearnedKnobAssignment falls back to "
                    f"ClusteredKnobAssignment for HEA — known v1 "
                    f"limitation). Offending labels: "
                    f"{', '.join(hea_labels)}. Use an MPS-substrate "
                    f"encoding (MPSRung1 cap=8, Rung3 cap=16, Rung4 "
                    f"cap=32, Rung5 cap=8*2^k) for these labels, or "
                    f"drop --learn-axis-assignment.",
                    file=sys.stderr,
                )
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

    # Parse --magnitude-diagnostics spec ("tokens:N" → int, "prompts:PATH" → Path).
    magnitude_diagnostics_arg: "int | Path | None" = None
    if args.magnitude_diagnostics is not None:
        raw = str(args.magnitude_diagnostics).strip()
        if ":" not in raw:
            print(
                f"sae-forge sweep-pareto: --magnitude-diagnostics expected "
                f"'tokens:N' or 'prompts:PATH'; got {raw!r}",
                file=sys.stderr,
            )
            return 2
        kind, value = raw.split(":", 1)
        kind = kind.strip().lower()
        value = value.strip()
        if kind == "tokens":
            try:
                n_tokens = int(value)
            except ValueError:
                print(
                    f"sae-forge sweep-pareto: --magnitude-diagnostics "
                    f"tokens: expected integer; got {value!r}",
                    file=sys.stderr,
                )
                return 2
            if n_tokens < 1:
                print(
                    f"sae-forge sweep-pareto: --magnitude-diagnostics "
                    f"tokens: must be >= 1; got {n_tokens}",
                    file=sys.stderr,
                )
                return 2
            magnitude_diagnostics_arg = n_tokens
        elif kind == "prompts":
            prompts_path = Path(value)
            if not prompts_path.is_file():
                print(
                    f"sae-forge sweep-pareto: --magnitude-diagnostics "
                    f"prompts: file not found: {prompts_path}",
                    file=sys.stderr,
                )
                return 2
            magnitude_diagnostics_arg = prompts_path
        else:
            print(
                f"sae-forge sweep-pareto: --magnitude-diagnostics kind "
                f"must be 'tokens' or 'prompts'; got {kind!r}",
                file=sys.stderr,
            )
            return 2
        if args.layer is None:
            print(
                "sae-forge sweep-pareto: --magnitude-diagnostics requires "
                "--layer (the residual-stream hook layer must match the "
                "SAE's training layer).",
                file=sys.stderr,
            )
            return 2

    sweep_kwargs: dict[str, object] = dict(
        encodings=encodings,
        output_dir=Path(args.output_dir),
        frontier_only=args.frontier_only,
        quality_floor=args.quality_floor,
        quality_thresholds=quality_thresholds,
        magnitude_diagnostics=magnitude_diagnostics_arg,
        rank_monotonicity_check=bool(args.rank_monotonicity_check),
    )
    # --layer is forwarded under --auto-materialise OR --magnitude-diagnostics.
    # The latter needs it to know which residual-stream layer to hook.
    if magnitude_diagnostics_arg is not None and not auto_materialise:
        sweep_kwargs["layer"] = args.layer
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
            assign_amp_knobs=bool(args.assign_amp_knobs),
            learn_axis_assignment=bool(args.learn_axis_assignment),
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

    # Surface the `__synthesised_keys__` safetensors-header metadata
    # from `_write_basis_as_checkpoint` (full-sae-keys-in-synth-basis).
    # When the checkpoint was written from a synth basis that lacked
    # real W_enc / b_enc / b_dec, the synthesised list shows which
    # tensors are placeholder rather than real encoder weights.
    summary["synthesised_keys"] = _read_synthesised_keys(args.checkpoint)

    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).write_text(_render_inspect_markdown(args.checkpoint, summary))
    return 0


def _read_synthesised_keys(checkpoint_path: str) -> list[str]:
    """Read the ``__synthesised_keys__`` metadata field from a
    safetensors checkpoint. Returns ``[]`` when absent or empty."""
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="numpy") as f:
        md = f.metadata() or {}
    raw = md.get("__synthesised_keys__", "")
    if not raw:
        return []
    return [k for k in raw.split(",") if k]


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
    synth = summary.get("synthesised_keys") or []
    if synth:
        lines.extend([
            "## Synthesised keys",
            "",
            f"This checkpoint was written from a synth basis lacking the "
            f"corresponding real SAE weights; the following tensors are "
            f"**placeholders** (W_enc=W_dec.T, biases=zeros) rather than "
            f"real encoder weights: **{', '.join(synth)}**.",
            "",
        ])
    return "\n".join(lines)


def _validate_encoding_specs(
    specs: list[tuple[str, "Path | str"]], *, kind: str,
) -> int:
    """Early validation of multi-encoding CLI specs.

    Checks duplicate labels + path existence BEFORE the dispatch
    function pays any YAML / dataset / forge cost. Returns 0 on
    success, 2 on failure (CLI config error exit code).

    Per PR #94 review: catch typos early; "all provided encoding
    paths exist (or at least are readable)".
    """
    # Duplicate labels: the wrapper also checks but a CLI-level
    # message with the specific kwarg name is clearer for users.
    seen: dict[str, "Path | str"] = {}
    for label, path in specs:
        if label in seen:
            print(
                f"sae-forge {kind}: duplicate --encoding label "
                f"{label!r} (first at {seen[label]}, second at {path}). "
                f"Encoding labels SHALL be unique.",
                file=sys.stderr,
            )
            return 2
        seen[label] = path
    # Path existence.
    for label, path in specs:
        path_obj = Path(path) if not isinstance(path, Path) else path
        if not path_obj.exists():
            print(
                f"sae-forge {kind}: --encoding {label!r} path not found: "
                f"{path_obj}. Verify the SAE checkpoint exists and is "
                f"readable.",
                file=sys.stderr,
            )
            return 2
    return 0


def _emit_dry_run_projection(
    *,
    kind: str,
    encodings: list[tuple[str, "Path | str"]],
    widths: list[int],
    scale_boosts: list,
    schedule: list[int],
    dollars_per_gpu_hr: float | None,
    output_dir: str,
) -> int:
    """Emit a structured wall-time projection table without running
    the full sweep.

    Per add-multi-encoding-capability-sweep/specs/pareto-sweep/spec.md
    "sae-forge sweep-capability --dry-run cost projection":

    1. Count expected cells (K × N × S × T).
    2. Benchmark would be ONE cell at the smallest stage × first
       encoding × first width × first scale_boost. v1 of this dry-run
       does NOT run the benchmark — it emits the cell count + a
       static "expect ~12 sec/cell on CPU at d_model=320" estimate
       (a reasonable upper bound for the bio-sae fixture; users
       benchmarking on their substrate calibrate from their own runs).
       Future enhancement: actual one-cell benchmark.
    3. Project wall time + cost.

    Returns 0 (always exits cleanly under dry-run).
    """
    K = len(encodings)
    N = len(widths)
    S = len(scale_boosts) if scale_boosts else 1
    T = len(schedule)
    total_cells = K * N * S * T
    # Static per-cell estimate. The bio-sae pooled fixture (n=5000
    # proteins, d_model=320) takes ~30-50 sec/cell on CPU; smaller
    # fixtures proportionally faster. We use 12 sec as a conservative
    # mid-point that works for the synthetic-fixture smoke tests and
    # is documented as "ballpark only".
    per_cell_seconds = 12.0
    projected_seconds = per_cell_seconds * total_cells
    # Apply host-cache amortisation: first-stage host extraction runs
    # once across all encodings; subsequent stages reuse the cache.
    # Rough heuristic: 60% of first-stage cells pay host extraction;
    # 40% read from cache.
    cache_factor = 1.0 - 0.15 * max(0, T - 1)
    projected_seconds_warm = max(60.0, projected_seconds * cache_factor)

    def _fmt_time(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f} sec"
        if secs < 3600:
            return f"~{secs / 60:.1f} min"
        return f"~{secs / 3600:.1f} hours"

    print(f"sae-forge {kind} --dry-run projection:")
    print(f"  NOTE: this is a CONSERVATIVE static estimate "
          f"({per_cell_seconds:.0f} sec/cell on CPU at d_model=320). "
          f"For tighter calibration, run ONE real cell and divide by "
          f"the projected cell count.")
    print(f"  encodings (K):             {K} ({[label for label, _ in encodings]!r})")
    print(f"  widths (N):                {N} ({widths!r})")
    print(f"  scale_boosts (S):          {S}")
    print(f"  stages (T):                {T} ({schedule!r})")
    print(f"  total cells:               {total_cells}")
    print(f"  per-cell estimate:         {per_cell_seconds:.1f} sec "
          f"(ballpark for CPU on d_model=320; calibrate via one real cell)")
    print(f"  projected wall (cold):     {_fmt_time(projected_seconds)}")
    print(f"  projected wall (warm host cache): {_fmt_time(projected_seconds_warm)}")
    if dollars_per_gpu_hr is not None:
        gpu_hours_cold = projected_seconds / 3600.0
        gpu_hours_warm = projected_seconds_warm / 3600.0
        cost_cold = gpu_hours_cold * dollars_per_gpu_hr
        cost_warm = gpu_hours_warm * dollars_per_gpu_hr
        print(f"  projected cost @ ${dollars_per_gpu_hr:.2f}/GPU-hr:")
        print(f"    cold cache: ${cost_cold:.2f}")
        print(f"    warm cache: ${cost_warm:.2f}")
    print(f"  output_dir (would be):     {output_dir}")
    print()
    print("Pass without --dry-run to run the sweep.")
    return 0


def _cmd_sweep_capability(args: argparse.Namespace) -> int:
    """Drive ``sweep_pareto_capability`` from a YAML dataset-config.

    Parses the YAML at ``--dataset-config``, constructs a
    ``CapabilityDataset`` via ``from_bio_sae`` (the only fixture
    loader v1 ships; other domains add their own constructors),
    then invokes the sweep wrapper. Output: ``frontier.jsonl`` under
    ``--output-dir``.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "sae-forge sweep-capability requires PyYAML to parse the "
            "dataset config. Install with `pip install pyyaml` or pull "
            "it in via a sae-forge extra that depends on it."
        ) from exc

    from saeforge.datasets import CapabilityDataset
    from saeforge.sweep_capability import sweep_pareto_capability

    cfg_path = Path(args.dataset_config)
    if not cfg_path.exists():
        print(f"sae-forge sweep-capability: dataset config not found: {cfg_path}",
              file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text())
    _required = {"encoder_checkpoint", "sequences_path", "labels_path"}
    missing = _required - set(cfg)
    if missing:
        print(f"sae-forge sweep-capability: dataset config missing required keys: "
              f"{sorted(missing)}", file=sys.stderr)
        return 2

    encoder_checkpoint = Path(cfg["encoder_checkpoint"])
    # from_bio_sae takes run_dir (the directory containing sae.pt), not the file path.
    run_dir = encoder_checkpoint.parent

    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir,
        bundle_path=cfg["labels_path"],
        sequences_path=cfg["sequences_path"],
        feed=cfg.get("feed", "pooled"),
        n_proteins=cfg.get("n_proteins"),
        max_seq_len=int(cfg.get("max_seq_len", args.max_seq_len)),
        tokenizer_id=cfg.get("tokenizer_id", "facebook/esm2_t6_8M_UR50D"),
        aggregator=cfg.get("aggregator", "pool_then_encode"),
        min_prevalence=int(cfg.get("min_prevalence", 0)),
        sae_variant=cfg.get("sae_variant", "topk"),
        sae_k=int(cfg.get("sae_k", 64)),
    )

    widths = [int(w.strip()) for w in args.widths.split(",") if w.strip()]
    scale_boosts: list[float | str] = []
    for token in args.scale_boosts.split(","):
        t = token.strip()
        if not t:
            continue
        if t == "auto":
            scale_boosts.append("auto")
        else:
            scale_boosts.append(float(t))

    # Multi-encoding mode: --encoding LABEL:PATH (repeatable). When
    # provided, supersedes the YAML config's encoder_checkpoint with
    # a stderr warning. Legacy --encodings string list is informational
    # and runs through the legacy single-encoding path.
    multi_encoding_specs: list[tuple[str, Path]] = []
    if args.encoding:
        try:
            multi_encoding_specs = _parse_encoding_specs(args.encoding)
        except ValueError as exc:
            print(f"sae-forge sweep-capability: {exc}", file=sys.stderr)
            return 2
        # Early validation: encoding paths SHALL exist + duplicate
        # labels are rejected before any dataset / YAML / forge cost.
        if (rc := _validate_encoding_specs(
            multi_encoding_specs, kind="sweep-capability",
        )) != 0:
            return rc
        print(
            f"sae-forge sweep-capability: --encoding flag(s) provided; "
            f"YAML config's encoder_checkpoint ({encoder_checkpoint}) "
            f"will be ignored. Multi-encoding mode active with "
            f"{len(multi_encoding_specs)} encodings: "
            f"{[label for label, _ in multi_encoding_specs]!r}.",
            file=sys.stderr,
        )
    legacy_encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]

    if args.dry_run:
        return _emit_dry_run_projection(
            kind="sweep-capability",
            encodings=multi_encoding_specs or [("raw_slice", encoder_checkpoint)],
            widths=widths,
            scale_boosts=scale_boosts,
            schedule=[len(dataset.sequences)],  # single-shot ≡ one stage
            dollars_per_gpu_hr=args.dollars_per_gpu_hr,
            output_dir=args.output_dir,
        )

    # Capability-trained encoder + readout-aligned ordering flags (add-capability-trained-encoder, task 4).
    train_kwargs = dict(
        train_encoder=args.train_encoder,
        train_objective=args.train_objective,
        basis_order=args.basis_order,
        readout_fallback=args.readout_fallback,
        train_steps=args.train_steps,
    )
    if multi_encoding_specs:
        rows = sweep_pareto_capability(
            encodings=multi_encoding_specs,
            host_model_id=args.host,
            dataset=dataset,
            widths=widths,
            scale_boosts=scale_boosts,
            output_dir=args.output_dir,
            cache_host=(not args.no_host_cache),
            max_seq_len=args.max_seq_len,
            device=args.device,
            **train_kwargs,
        )
    else:
        rows = sweep_pareto_capability(
            sae_checkpoint=encoder_checkpoint,
            host_model_id=args.host,
            dataset=dataset,
            widths=widths,
            encodings=legacy_encodings,
            scale_boosts=scale_boosts,
            output_dir=args.output_dir,
            cache_host=(not args.no_host_cache),
            max_seq_len=args.max_seq_len,
            device=args.device,
            **train_kwargs,
        )
    errors = sum(1 for r in rows if r.error_message is not None)
    encoding_summary = (
        f"; encodings: {[label for label, _ in multi_encoding_specs]!r}"
        if multi_encoding_specs else ""
    )
    print(f"sae-forge sweep-capability: {len(rows)} cells; "
          f"errors: {errors}; frontier: "
          f"{Path(args.output_dir) / 'frontier.jsonl'}{encoding_summary}")
    return 1 if errors == len(rows) else 0


def _cmd_sweep_capability_progressive(args: argparse.Namespace) -> int:
    """Drive ``sweep_pareto_capability_progressive`` from a YAML
    dataset-config + CLI knobs.

    Exit codes:
      0  recommendation converged; trustworthy for production.
      1  schedule exhausted without convergence; recommendation
         emitted with converged=False (caller decides whether to ship
         via ``sae-forge recommend --accept-unconverged``).
      2  config error (missing required flag, bad YAML, schedule not
         monotone, etc.).
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "sae-forge sweep-capability-progressive requires PyYAML to "
            "parse the dataset config. Install with `pip install pyyaml`."
        ) from exc

    from saeforge import sweep_pareto_capability_progressive
    from saeforge.datasets import CapabilityDataset

    cfg_path = Path(args.dataset_config)
    if not cfg_path.exists():
        print(f"sae-forge sweep-capability-progressive: dataset config "
              f"not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text())
    _required = {"encoder_checkpoint", "sequences_path", "labels_path"}
    missing = _required - set(cfg)
    if missing:
        print(f"sae-forge sweep-capability-progressive: dataset config "
              f"missing required keys: {sorted(missing)}",
              file=sys.stderr)
        return 2

    encoder_checkpoint = Path(cfg["encoder_checkpoint"])
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=encoder_checkpoint.parent,
        bundle_path=cfg["labels_path"],
        sequences_path=cfg["sequences_path"],
        feed=cfg.get("feed", "pooled"),
        n_proteins=cfg.get("n_proteins"),
        max_seq_len=int(cfg.get("max_seq_len", args.max_seq_len)),
        tokenizer_id=cfg.get("tokenizer_id", "facebook/esm2_t6_8M_UR50D"),
        aggregator=cfg.get("aggregator", "pool_then_encode"),
        min_prevalence=int(cfg.get("min_prevalence", 0)),
        sae_variant=cfg.get("sae_variant", "topk"),
        sae_k=int(cfg.get("sae_k", 64)),
    )

    candidate_widths = [
        int(w.strip()) for w in args.candidate_widths.split(",") if w.strip()
    ]
    schedule = [
        int(n.strip()) for n in args.schedule.split(",") if n.strip()
    ]
    scale_boosts: list[float | str] = []
    for token in args.scale_boosts.split(","):
        t = token.strip()
        if not t:
            continue
        scale_boosts.append("auto" if t == "auto" else float(t))
    # Multi-encoding mode: --encoding LABEL:PATH (repeatable).
    multi_encoding_specs: list[tuple[str, Path]] = []
    if args.encoding:
        try:
            multi_encoding_specs = _parse_encoding_specs(args.encoding)
        except ValueError as exc:
            print(f"sae-forge sweep-capability-progressive: {exc}",
                  file=sys.stderr)
            return 2
        if (rc := _validate_encoding_specs(
            multi_encoding_specs, kind="sweep-capability-progressive",
        )) != 0:
            return rc
        print(
            f"sae-forge sweep-capability-progressive: --encoding flag(s) "
            f"provided; YAML config's encoder_checkpoint "
            f"({encoder_checkpoint}) will be ignored. Multi-encoding "
            f"mode active with {len(multi_encoding_specs)} encodings: "
            f"{[label for label, _ in multi_encoding_specs]!r}.",
            file=sys.stderr,
        )
    legacy_encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]

    if args.dry_run:
        return _emit_dry_run_projection(
            kind="sweep-capability-progressive",
            encodings=multi_encoding_specs or [("raw_slice", encoder_checkpoint)],
            widths=candidate_widths,
            scale_boosts=scale_boosts,
            schedule=schedule,
            dollars_per_gpu_hr=args.dollars_per_gpu_hr,
            output_dir=args.output_dir,
        )

    try:
        if multi_encoding_specs:
            history = sweep_pareto_capability_progressive(
                encodings=multi_encoding_specs,
                host_model_id=args.host,
                dataset=dataset,
                candidate_widths=candidate_widths,
                n_proteins_schedule=schedule,
                output_dir=args.output_dir,
                scale_boosts=scale_boosts,
                retained_mauc_tolerance=args.retained_mauc_tolerance,
                plateau_tolerance=args.plateau_tolerance,
                min_plateau_widths=args.min_plateau_widths,
                convergence_n_stages=args.convergence_n_stages,
                cache_host=(not args.no_host_cache),
                max_seq_len=args.max_seq_len,
                device=args.device,
            )
        else:
            history = sweep_pareto_capability_progressive(
                sae_checkpoint=encoder_checkpoint,
                host_model_id=args.host,
                dataset=dataset,
                candidate_widths=candidate_widths,
                n_proteins_schedule=schedule,
                output_dir=args.output_dir,
                encodings=legacy_encodings,
                scale_boosts=scale_boosts,
                retained_mauc_tolerance=args.retained_mauc_tolerance,
                plateau_tolerance=args.plateau_tolerance,
                min_plateau_widths=args.min_plateau_widths,
                convergence_n_stages=args.convergence_n_stages,
                cache_host=(not args.no_host_cache),
                max_seq_len=args.max_seq_len,
                device=args.device,
            )
    except ValueError as exc:
        print(f"sae-forge sweep-capability-progressive: {exc}",
              file=sys.stderr)
        return 2

    rec = history.recommendation
    summary_path = Path(args.output_dir) / "progressive_summary.json"
    encoding_summary = (
        f"; winning encoding: {rec.winning_encoding!r}"
        if rec.winning_encoding else ""
    )
    print(f"sae-forge sweep-capability-progressive: "
          f"{len(history.stages)} stage(s); converged={rec.converged}; "
          f"recommendation n={rec.target_n_features_kept}, "
          f"retained_mauc={rec.retained_mauc_vs_host:.4f}{encoding_summary}; "
          f"summary: {summary_path}")
    print(f"Rationale: {rec.rationale}")
    if rec.per_encoding_recommendations:
        print("\nPer-encoding recommendations:")
        for label, per_rec in rec.per_encoding_recommendations.items():
            converged_marker = "✓" if per_rec.converged else "✗"
            print(f"  {converged_marker} {label}: n={per_rec.target_n_features_kept}, "
                  f"retained_mauc={per_rec.retained_mauc_vs_host:.4f}, "
                  f"converged={per_rec.converged}")
    return 0 if rec.converged else 1


# Predicate parser for `sae-forge recommend`.
_RECOMMEND_OPS = (">=", "<=", "==", "<", ">")  # check 2-char before 1-char


def _parse_recommend_predicate(expr: str) -> tuple[str, str, float]:
    """Parse ``FIELD<OP>VALUE`` into (field_name, op, value).

    Field names accept kebab-case (``retained-mauc``) or snake_case
    (``retained_mauc_vs_host``). Kebab→snake conversion is mechanical:
    replace ``-`` with ``_``. Special case: bare ``retained-mauc``
    resolves to ``retained_mauc_vs_host`` (the common shorthand);
    same for ``retained-cov95``.

    Resolved field name SHALL be a real attribute on
    :class:`saeforge.sweep.ParetoFrontierRow`. Unknown fields raise
    ``ValueError`` at parse time (early failure) rather than silently
    skipping every row at predicate-application time.
    """
    op_used = None
    op_idx = -1
    for op in _RECOMMEND_OPS:
        idx = expr.find(op)
        if idx >= 0:
            op_used = op
            op_idx = idx
            break
    if op_used is None:
        raise ValueError(
            f"sae-forge recommend: predicate {expr!r} has no comparison "
            f"operator; expected one of {_RECOMMEND_OPS!r}"
        )
    field_raw = expr[:op_idx].strip()
    value_raw = expr[op_idx + len(op_used):].strip()
    if not field_raw or not value_raw:
        raise ValueError(
            f"sae-forge recommend: predicate {expr!r} missing field or value"
        )
    field = field_raw.replace("-", "_")
    # Shorthand aliases for the load-bearing capability fields.
    aliases = {
        "retained_mauc":  "retained_mauc_vs_host",
        "retained_cov95": "retained_cov95_vs_host",
    }
    field = aliases.get(field, field)
    # Validate the field exists on ParetoFrontierRow now — surfaces
    # typos at parse time rather than during the predicate loop where
    # the error would be indistinguishable from "every row's field
    # value is None".
    from saeforge.sweep import ParetoFrontierRow

    known_fields = {f.name for f in ParetoFrontierRow.__dataclass_fields__.values()}
    if field not in known_fields:
        raise ValueError(
            f"sae-forge recommend: predicate {expr!r} references unknown "
            f"field {field!r}. Available capability fields: "
            f"retained-mauc, retained-cov95, host-baseline-mauc, "
            f"forge-mauc, forge-cov95, gap-median, gap-p25, gap-p75, "
            f"gap-p95, n-features-gap-above-0-1, n-features-negative-gap."
        )
    return field, op_used, float(value_raw)


def _cmd_recommend(args: argparse.Namespace) -> int:
    """Filter a sweep frontier by predicate(s); return the smallest-
    parameter row matching all predicates.

    Output: tabular summary by default; JSON via ``--json``. Exits
    non-zero when no row matches any predicate.
    """
    from saeforge.sweep import ParetoFrontierRow

    frontier_path = Path(args.frontier)
    if not frontier_path.exists():
        print(f"sae-forge recommend: frontier not found: {frontier_path}",
              file=sys.stderr)
        return 2

    predicates: list[tuple[str, str, float]] = []
    for expr in args.target:
        try:
            predicates.append(_parse_recommend_predicate(expr))
        except ValueError as exc:
            print(f"sae-forge recommend: {exc}", file=sys.stderr)
            return 2

    rows: list[ParetoFrontierRow] = []
    for line in frontier_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(ParetoFrontierRow.from_json_dict(json.loads(line)))

    # Progressive-frontier detection: if any row carries the `stage`
    # field, the frontier was emitted by sweep-capability-progressive.
    # Check the companion progressive_summary.json's
    # recommendation.converged; refuse to recommend on
    # converged=False unless --accept-unconverged.
    is_progressive = any(r.stage is not None for r in rows)
    if is_progressive:
        summary_path = frontier_path.parent / "progressive_summary.json"
        if not summary_path.exists():
            print(
                f"sae-forge recommend: frontier {frontier_path} carries "
                f"stage fields (progressive sweep output) but the "
                f"companion progressive_summary.json was not found at "
                f"{summary_path}. Either copy both files together or "
                f"pass a single-shot frontier.",
                file=sys.stderr,
            )
            return 2
        summary = json.loads(summary_path.read_text())
        rec_meta = summary.get("recommendation", {})
        if not rec_meta.get("converged", False) and not args.accept_unconverged:
            traj = rec_meta.get("convergence_trajectory", [])
            shifted_stages = [
                e["stage"] for e in traj
                if e.get("shifted_from_prev_stage", False)
            ]
            rationale = rec_meta.get("rationale", "(no rationale on disk)")
            print(
                f"sae-forge recommend: progressive frontier at "
                f"{frontier_path} did NOT converge.\n"
                f"\n"
                f"  Recommended n: {rec_meta.get('target_n_features_kept')}\n"
                f"  Retained mAUC: {rec_meta.get('retained_mauc_vs_host')}\n"
                f"  Stages run:    {len(traj)}\n"
                f"  Shifted stages: {shifted_stages}\n"
                f"\n"
                f"  Rationale: {rationale}\n"
                f"\n"
                f"Use --accept-unconverged to recommend anyway, OR re-run "
                f"sweep-capability-progressive with a longer schedule / "
                f"looser plateau_tolerance / convergence_n_stages=1 "
                f"(see openspec/changes/add-progressive-capability-"
                f"sweep/design.md Decision 6 for the informed-opt-out "
                f"alternatives).",
                file=sys.stderr,
            )
            return 1

    survivors: list[ParetoFrontierRow] = []
    for row in rows:
        if row.error_message is not None:
            continue
        passes_all = True
        for field, op, value in predicates:
            attr = getattr(row, field, None)
            if attr is None:
                passes_all = False
                break
            actual = float(attr)
            comparisons = {
                ">=": actual >= value,
                "<=": actual <= value,
                "==": actual == value,
                ">":  actual > value,
                "<":  actual < value,
            }
            if not comparisons[op]:
                passes_all = False
                break
        if passes_all:
            survivors.append(row)

    if not survivors:
        print(
            f"sae-forge recommend: no row satisfies all predicates "
            f"({args.target!r}). Closest rows by predicate failure point "
            f"are not yet implemented; inspect the frontier directly.",
            file=sys.stderr,
        )
        return 1

    # Multi-encoding detection: any frontier carrying multiple
    # distinct encoding_label values triggers per-encoding ranking.
    distinct_encoding_labels = sorted(
        {r.encoding_label for r in rows if r.encoding_label}
    )
    is_multi_encoding = len(distinct_encoding_labels) > 1

    # Pick smallest target_n_features_kept among survivors. For
    # multi-encoding frontiers the tiebreaker is (smallest n,
    # CLI-flag-order-of-first-appearance) per spec.md "recommend over
    # multi-encoding frontiers".
    encoding_first_seen: dict[str, int] = {}
    for idx, r in enumerate(rows):
        if r.encoding_label and r.encoding_label not in encoding_first_seen:
            encoding_first_seen[r.encoding_label] = idx

    def _key(r: ParetoFrontierRow) -> tuple[int, int]:
        return (
            int(r.target_n_features_kept),
            encoding_first_seen.get(r.encoding_label or "", 0),
        )
    picked = min(survivors, key=_key)

    # Capability-trained-encoder preference (add-capability-trained-encoder, task 4.2): for the picked
    # cell, recommend the trained encoder ONLY when its held-out margin clears --trained-margin and it
    # did not overfit; otherwise keep the simpler pinv forge (ties default to pinv).
    trained_margin = getattr(args, "trained_margin", 0.02)
    use_trained = bool(
        picked.encoder_trained
        and picked.delta_heldout is not None
        and picked.delta_heldout > trained_margin
        and not picked.overfit_flag
    )
    recommended_encoder = "trained" if use_trained else "pinv"
    effective_retained = (
        picked.retained_mauc_trained if use_trained else
        (picked.retained_mauc_pinv_baseline
         if picked.retained_mauc_pinv_baseline is not None
         else picked.retained_mauc_vs_host)
    )

    if args.emit_json:
        out = picked.to_json_dict()
        if picked.encoder_trained:
            out["recommended_encoder"] = recommended_encoder
            out["effective_retained_mauc"] = effective_retained
            out["trained_margin"] = trained_margin
        print(json.dumps(out, indent=2))
        return 0

    # Tabular: emit the load-bearing fields per the spec.
    print(f"recommended config (smallest target_n_features_kept among "
          f"{len(survivors)} survivor row(s)):")
    print(f"  encoding_label:           {picked.encoding_label}")
    print(f"  target_n_features_kept:   {picked.target_n_features_kept}")
    if picked.host_baseline_mauc is not None:
        print(f"  host_baseline_mauc:       {picked.host_baseline_mauc:.4f}")
    if picked.forge_mauc is not None:
        print(f"  forge_mauc:               {picked.forge_mauc:.4f}")
    if picked.retained_mauc_vs_host is not None:
        print(f"  retained_mauc_vs_host:    {picked.retained_mauc_vs_host:.4f}")
    if picked.forge_cov95 is not None:
        print(f"  forge_cov95:              {picked.forge_cov95:.4f}")
    if picked.gap_median is not None:
        print(f"  gap_median:               {picked.gap_median:+.4f}")
    if picked.gap_p95 is not None:
        print(f"  gap_p95:                  {picked.gap_p95:+.4f}")
    if picked.encoder_trained:
        print(f"  recommended_encoder:      {recommended_encoder}"
              f"  (trained Δ_heldout {picked.delta_heldout:+.4f} vs margin "
              f"{trained_margin:+.4f}, overfit={picked.overfit_flag})")
        if effective_retained is not None:
            print(f"  effective_retained_mauc:  {effective_retained:.4f}")
        if recommended_encoder == "trained" and picked.encoder_artifact_path:
            print(f"  trained_encoder_artifact: {picked.encoder_artifact_path}")

    # Multi-encoding: emit the per-encoding ranking table. Per the
    # openspec, ALWAYS print on multi-encoding frontiers — the
    # winner-vs-runner-up gap is itself diagnostic.
    if is_multi_encoding:
        print()
        print(f"Per-encoding ranking (over {len(survivors)} survivors "
              f"after predicate filtering)")
        print("  Ranking: smallest target_n_features_kept WINS; "
              "ties broken by CLI --encoding flag order.")
        print(f"  {'rank':<5} {'encoding':<20} {'n':>5} "
              f"{'retained_mauc':>14} {'converged':>10}")
        # For each encoding, find its smallest-n survivor.
        per_encoding_winners: dict[str, ParetoFrontierRow] = {}
        for row in survivors:
            label = row.encoding_label or ""
            if not label:
                continue
            existing = per_encoding_winners.get(label)
            if existing is None or row.target_n_features_kept < existing.target_n_features_kept:
                per_encoding_winners[label] = row
        # Also list encodings WITH rows in the frontier but no
        # survivors (failed the predicate).
        encodings_with_no_survivors = [
            label for label in distinct_encoding_labels
            if label not in per_encoding_winners
        ]
        # Rank by smallest n, then CLI-flag order.
        ranked = sorted(
            per_encoding_winners.items(),
            key=lambda kv: (
                kv[1].target_n_features_kept,
                encoding_first_seen.get(kv[0], 0),
            ),
        )
        for rank, (label, row) in enumerate(ranked, start=1):
            retained = (
                f"{row.retained_mauc_vs_host:.4f}"
                if row.retained_mauc_vs_host is not None else "—"
            )
            print(f"  {rank:<5} {label:<20} "
                  f"{row.target_n_features_kept:>5} "
                  f"{retained:>14} {'—':>10}")
        for label in encodings_with_no_survivors:
            print(f"  {'—':<5} {label:<20} "
                  f"{'—':>5} {'—':>14} {'(no row meets predicate)':>10}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "forge":
        return _cmd_forge(args)
    if args.command == "sweep-pareto":
        return _cmd_sweep_pareto(args)
    if args.command == "sweep-capability":
        return _cmd_sweep_capability(args)
    if args.command == "sweep-capability-progressive":
        return _cmd_sweep_capability_progressive(args)
    if args.command == "recommend":
        return _cmd_recommend(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
