"""End-to-end Gemma-2-2B forge against a Gemma Scope SAE + recipe fine-tune.

Headline v0.3 demo:
1. Pull a Gemma Scope SAE from HuggingFace (default:
   ``google/gemma-scope-2b-pt-res``, configurable layer)
2. Slice to a configurable feature subset (default 256)
3. Run polygram.EpochCompressor against Gemma-2-2B forward passes
4. Forge with v0 host-attention (v0.2 feature_native is a separate flag)
5. Fine-tune via the v0.3 recipe: cosine LR + warmup, gradient
   clipping, optional gradient checkpointing, optional bf16/fp16
6. Periodic faithfulness eval; periodic checkpoint saves

CPU is not realistic for this script — Gemma-2-2B at ~5GB fp16 needs
GPU/MPS. Defaults assume `device="mps"` (Apple Silicon 24GB+) or
`device="cuda"` (24GB+ NVIDIA).

Wall-clock targets:
- M4 Pro 24GB MPS: ~30-90 min for 1k-step fine-tune
- RTX 4090 24GB CUDA: ~10-30 min for 1k-step fine-tune

Pre-conditions:
- Accept the Gemma license at https://huggingface.co/google/gemma-2-2b
  and run `huggingface-cli login` with a token that has read access
- ~10GB free under ~/.cache/huggingface/ for Gemma weights + SAE
- ``polygram>=0.1.0`` and ``sae-forge[torch]`` installed.
  multi-architecture-support landed in 0.2; older sae-forge silently
  loaded Gemma weights into a GPT-2 config and produced random output
- For the corpus: either pass --corpus /path/to/local.txt or accept
  the default streaming Fineweb-edu (lazy-imports `datasets`)

Run:
    python examples/forge_gemma2_2b.py /tmp/run --device mps --n-features 256 --steps 1000

For a fast smoke test (no fine-tune, just forge + KL eval):
    python examples/forge_gemma2_2b.py /tmp/run --device mps --n-features 64 --steps 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


SAE_REPO = "google/gemma-scope-2b-pt-res"
# Gemma Scope publishes a small set of L0 variants per layer. Layer 12
# has {22, 41, 82, 176, 445}; the previous hard-coded ``average_l0_71``
# does not exist for any layer 12 release. Pick via ``--l0``; default
# 82 is the layer-12 default that closest matches the recommended
# coverage target. See https://huggingface.co/google/gemma-scope-2b-pt-res
# for the full list.
SAE_FILE_TEMPLATE = "layer_{layer}/width_16k/average_l0_{l0}/params.npz"
DEFAULT_L0 = 82
HOST_MODEL = "google/gemma-2-2b"
DEFAULT_LAYER = 12

EVAL_PROMPTS = [
    "The mitochondrion is the powerhouse of the",
    "To be or not to be, that is the",
    "All happy families are alike; each unhappy family is",
    "In the beginning God created the heavens and the",
]


def slice_sae_to_features(input_path: Path, output_path: Path, feature_indices: list[int]) -> None:
    """Slice a Gemma Scope SAE (.npz format) to a feature subset, output as safetensors."""
    import numpy as np
    from safetensors.numpy import save_file

    with np.load(str(input_path)) as state:
        keys = list(state.keys())
        sliced: dict = {}
        # Gemma Scope npz convention: W_dec / W_enc / b_enc / b_dec / threshold
        # (TopK-style SAE has a `threshold` row tensor too)
        if "W_dec" in keys:
            sliced["W_dec"] = state["W_dec"][feature_indices]
        if "W_enc" in keys:
            we = state["W_enc"]
            if we.shape[1] == state["W_dec"].shape[0]:
                sliced["W_enc"] = we[:, feature_indices]
            else:
                sliced["W_enc"] = we[feature_indices]
        if "b_enc" in keys:
            sliced["b_enc"] = state["b_enc"][feature_indices]
        if "b_dec" in keys:
            sliced["b_dec"] = state["b_dec"]
        if "threshold" in keys:
            sliced["threshold"] = state["threshold"][feature_indices]
    save_file(sliced, str(output_path))


def main(args) -> dict:
    import torch  # noqa: F401
    from huggingface_hub import hf_hub_download

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: download SAE -----------------------------------
    sae_file = SAE_FILE_TEMPLATE.format(layer=args.layer, l0=args.l0)
    print(f"[1/5] downloading Gemma Scope SAE: {SAE_REPO} :: {sae_file}")
    t0 = time.monotonic()
    sae_path = Path(hf_hub_download(repo_id=SAE_REPO, filename=sae_file))
    print(f"      cached at {sae_path} ({time.monotonic() - t0:.1f}s)")

    # ---- Stage 2: slice ------------------------------------------
    print(f"[2/5] slicing SAE to first {args.n_features} features")
    sliced_path = output_dir / "sae_sliced.safetensors"
    slice_sae_to_features(sae_path, sliced_path, list(range(args.n_features)))

    # ---- Stage 3: polygram compression ---------------------------
    print(f"[3/5] polygram EpochCompressor on layer {args.layer}, "
          f"{args.n_features} features, {args.n_compress_prompts} prompts")
    from polygram import EpochCompressionConfig, EpochCompressor

    compressed_path = output_dir / "sae_compressed.safetensors"
    compress_prompts = _load_corpus_lines(args.corpus, n=args.n_compress_prompts) \
        if args.corpus and Path(args.corpus).exists() \
        else _default_compression_prompts(args.n_compress_prompts)
    # ``EpochCompressor.fast()`` is a preset wrapper that supplies
    # ``config=`` internally and forwards ``**overrides`` to the
    # constructor; passing ``config=`` as an override collides
    # (TypeError: got multiple values for keyword argument 'config').
    # Use the constructor directly when carrying a custom
    # EpochCompressionConfig.
    epoch = EpochCompressor(
        sae_checkpoint=sliced_path,
        prompts=compress_prompts,
        layer=args.layer,
        model_name=HOST_MODEL,
        strategy="zero",
        device=args.device,
        config=EpochCompressionConfig(
            coverage_target=args.coverage_target,
            cosine_threshold=0.30,
            n_visits_per_feature=1,
            max_iterations=args.compress_max_iterations,
        ),
    )
    t0 = time.monotonic()
    epoch_result = epoch.run(compressed_path)
    epoch_wall = time.monotonic() - t0
    print(f"      done in {epoch_wall/60:.1f}min; "
          f"zeroed={epoch_result.report.n_features_zeroed_total}/{args.n_features}, "
          f"panels={epoch_result.report.n_panels_total}")

    # ---- Stage 4: forge + fine-tune ------------------------------
    print(f"[4/5] forging Gemma-2-2B against compressed SAE; "
          f"fine-tune {args.steps} steps on {args.corpus or 'default Fineweb-edu'}")
    basis = FeatureBasis.from_polygram_checkpoint(compressed_path)
    if basis.n_features == 0:
        raise RuntimeError("compression zeroed every feature; raise --n-features or relax --coverage-target")
    print(f"      basis: n_features={basis.n_features}, d_model={basis.d_model}")

    projector = SubspaceProjector(basis, scale_boost=_coerce_scale_boost(args.scale_boost))

    # Surface forward_mode resolution BEFORE building the pipeline so the
    # user sees which path will run. The basis quality tier drives the
    # auto-dispatch: good/saturated → native_in_basis (the existing path);
    # undersized/degenerate → host_wrapped (the under-complete fallback
    # from add-host-wrapped-forge-fallback).
    from saeforge.forward_mode import resolve_forward_mode

    resolved_mode = resolve_forward_mode(basis, args.forward_mode)
    print(f"      forward_mode: requested={args.forward_mode!r}, "
          f"resolved={resolved_mode!r}")
    if resolved_mode == "host_wrapped" and args.steps > 0:
        # host_wrapped is inference-only in v1 — surface the incompatibility
        # before the pipeline raises. Auto-fall back to imperative orchestrator
        # with steps=0 so the demo at least produces a forge result.
        print(
            "      WARNING: host_wrapped is inference-only in v1; "
            "ignoring --steps and skipping fine-tune."
        )
        args.steps = 0
    # orchestrator="fsm" routes through the recipe action; the imperative
    # path (default) silently skips fine-tune. ``args.steps == 0`` is a
    # forge-only smoke run, so leave orchestrator at the imperative
    # default in that case to avoid the FSM's tokeniser/dataset round-trip
    # for an already-stubbed corpus.
    orchestrator = "fsm" if args.steps > 0 else "imperative"
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=HOST_MODEL,
        eval_prompts=EVAL_PROMPTS,
        dtype=args.dtype,
        device=args.device,
        attention_width=args.attention_width,
        orchestrator=orchestrator,
        forward_mode=args.forward_mode,
        finetune_corpus=args.corpus or "HuggingFaceFW/fineweb-edu",
        finetune_total_steps=args.steps,
        finetune_warmup_steps=max(1, args.steps // 10),
        finetune_peak_lr=args.lr,
        finetune_batch_size=args.batch_size,
        finetune_seq_len=args.seq_len,
        finetune_precision=args.precision,
        finetune_grad_checkpoint=args.grad_checkpoint,
        finetune_eval_every=max(1, args.steps // 10),
        finetune_save_every=max(1, args.steps // 4),
    )
    t0 = time.monotonic()
    if args.steps > 0:
        result = pipeline.run(output_dir / "forge")
    else:
        # Skip fine-tune entirely — pipeline.run still does the forge + eval
        pipeline.finetune_corpus = None
        result = pipeline.run(output_dir / "forge")
    forge_wall = time.monotonic() - t0
    print(f"      forged: n_params={result.n_params}, "
          f"final KL={result.faithfulness}, wall={forge_wall/60:.1f}min")

    # ---- Stage 5: summary ----------------------------------------
    # Surface polygram cluster diagnostics when the compression report
    # supplies them. Older polygram outputs may lack these fields; we
    # tolerate by reading them as Optional and skipping when absent.
    cluster_diag = {}
    for field in ("n_clusters", "n_zeroed"):
        value = basis.metadata.get(field) if hasattr(basis, "metadata") else None
        if value is not None:
            cluster_diag[field] = int(value)
    if "n_clusters" in cluster_diag and "n_zeroed" in cluster_diag:
        denom = cluster_diag["n_clusters"] + cluster_diag["n_zeroed"]
        if denom > 0:
            cluster_diag["redundancy_ratio"] = round(
                cluster_diag["n_zeroed"] / denom, 4
            )

    summary = {
        "host_model": HOST_MODEL,
        "sae_repo": SAE_REPO,
        "sae_layer": args.layer,
        "n_features_sliced": args.n_features,
        "n_features_kept": basis.n_features,
        "d_model": basis.d_model,
        "compression": {
            "zeroed": epoch_result.report.n_features_zeroed_total,
            "panels": epoch_result.report.n_panels_total,
            "wall_minutes": round(epoch_wall / 60, 2),
            **({"polygram_diagnostics": cluster_diag} if cluster_diag else {}),
        },
        "forge": {
            "attention_width": args.attention_width,
            "forward_mode_requested": args.forward_mode,
            "forward_mode_resolved": resolved_mode,
            "n_params": result.n_params,
            "faithfulness_kl": result.faithfulness,
            "wall_minutes": round(forge_wall / 60, 2),
        },
        "finetune": {
            "steps": args.steps,
            "corpus": args.corpus or "HuggingFaceFW/fineweb-edu",
            "precision": args.precision,
            "grad_checkpoint": args.grad_checkpoint,
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[5/5] summary written to {output_dir / 'run_summary.json'}")
    print(json.dumps(summary, indent=2))
    return summary


def _default_compression_prompts(n: int) -> list[str]:
    """Fallback compression prompts when no local corpus is supplied."""
    base = [
        "The capital of France is Paris.",
        "Photosynthesis converts light into chemical energy.",
        "Newton's third law: every action has an equal reaction.",
        "The mitochondrion is the powerhouse of the cell.",
        "Water freezes at zero degrees Celsius.",
        "DNA carries genetic information in living organisms.",
        "Computers manipulate symbols according to formal rules.",
        "Music expresses emotions through structured sound.",
    ]
    out: list[str] = []
    while len(out) < n:
        out.extend(base)
    return out[:n]


def _load_corpus_lines(path: str, n: int) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines or _default_compression_prompts(n)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output_dir", help="where to write the forge artifacts")
    parser.add_argument(
        "--device", default="cpu",
        help="cpu / mps (Apple) / cuda (NVIDIA). cpu is not realistic for Gemma-2-2B.",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"),
        help="host + projected model dtype; bfloat16 default keeps Gemma-2-2B near 5GB",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--l0",
        type=int,
        default=DEFAULT_L0,
        help=(
            "Gemma Scope L0 variant for the SAE checkpoint. Layer 12 has "
            "{22, 41, 82, 176, 445}; default is 82. Mismatch → 404 from "
            "huggingface_hub.hf_hub_download."
        ),
    )
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--n-compress-prompts", type=int, default=16)
    parser.add_argument("--coverage-target", type=float, default=0.5)
    parser.add_argument("--compress-max-iterations", type=int, default=1)
    parser.add_argument(
        "--attention-width", default="host", choices=("host", "feature_native"),
    )
    parser.add_argument(
        "--corpus", default=None,
        help="local file path for fine-tune corpus; defaults to HuggingFaceFW/fineweb-edu",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="default 1 to fit Gemma-2-2B on 24GB unified memory; bump on bigger GPUs",
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--precision", default="bf16", choices=("fp32", "bf16", "fp16"),
        help="bf16 recommended on M-series and modern CUDA",
    )
    parser.add_argument(
        "--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
        help="default on for Gemma-2-2B; pass --no-grad-checkpoint to disable on >40GB GPUs",
    )
    parser.add_argument(
        "--scale-boost",
        default="auto",
        help=(
            "SubspaceProjector scale_boost. Pass a float, or 'auto' "
            "(default) which picks min(1.0, d_model/n_features) for "
            "over-complete bases. Empirical anchor: GPT-2 (d=768) with "
            "1024 features needed ~0.25; if your run produces NaNs / "
            "saturated softmax / astronomical KL, hand-pick a value < 1.0."
        ),
    )
    parser.add_argument(
        "--forward-mode",
        default="auto",
        choices=("auto", "native_in_basis", "host_wrapped"),
        help=(
            "Forge forward implementation. 'auto' (default) picks "
            "native_in_basis for good/saturated basis quality and "
            "host_wrapped for undersized/degenerate. host_wrapped is "
            "GPT-2 only in v1 and inference-only — Gemma-2 + host_wrapped "
            "will raise from the adapter."
        ),
    )
    parser.add_argument(
        "--llm-scale",
        action="store_true",
        help=(
            "sm-sae provisional LLM-scale knob preset. Currently only "
            "informational on this example since the example builds the "
            "EpochCompressionConfig directly; the CLI surface "
            "(`sae-forge forge --llm-scale`) wires the actual knob bumps."
        ),
    )
    return parser


def _coerce_scale_boost(raw):
    """Accept 'auto' or a numeric string from argparse."""
    if isinstance(raw, str) and raw == "auto":
        return "auto"
    return float(raw)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(0 if main(args) else 1)
