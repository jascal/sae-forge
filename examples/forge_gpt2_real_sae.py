"""Real GPT-2 forge against a published SAE compressed via Polygram.

End-to-end research run (vs forge_gpt2_real.py's pipeline-only smoke run):

  1. Pull a published GPT-2-small SAE from HuggingFace
     (default: jbloom/GPT2-Small-SAEs-Reformatted, blocks.8.hook_resid_pre)
  2. Slice to a configurable feature subset (default: 64 features)
  3. Run polygram.EpochCompressor — panel-by-panel validate + compress
     against gpt2 forward passes on a small prompt set
  4. Forge gpt2 against the resulting compressed SAE
  5. Report params + faithfulness KL on held-out prompts

CPU-friendly defaults (16 features, 1 iteration, 8 prompts) target a
2–5 minute wall time. For meaningful capability claims, raise the
feature count and iteration budget and run on GPU.

Device note: pass `device="cuda"` for NVIDIA, `device="mps"` for
Apple/Intel-Mac Metal. CUDA gives the expected speedup. MPS on this
workload (short sequences, few prompts) is roughly break-even with
CPU because kernel-launch overhead beats the parallelism win — the
MPS payoff lives in long fine-tune runs (hundreds of steps, batch
≥16) and large host models, both of which are also constrained by
VRAM (4GB on a Radeon Pro 5500M caps you near GPT-2-small).

Run:
    python examples/forge_gpt2_real_sae.py [output_dir] [n_features] [device]

Example:
    python examples/forge_gpt2_real_sae.py /tmp/run 32 cpu
    python examples/forge_gpt2_real_sae.py /tmp/run 32 mps
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_FILE = "blocks.8.hook_resid_pre/sae_weights.safetensors"
HOST_MODEL = "gpt2"
LAYER = 8

VALIDATE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, never mind how long.",
    "The capital of France is Paris, a city renowned for its art.",
    "Newton's third law states that every action has an equal and opposite reaction.",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
]

EVAL_PROMPTS = [
    "The mitochondrion is the powerhouse of the",
    "To be or not to be, that is the",
    "All happy families are alike; each unhappy family is",
    "In the beginning God created the heavens and the",
]


def slice_sae_to_features(input_path: Path, output_path: Path, feature_indices: list[int]) -> None:
    """Slice a full SAE checkpoint to a feature subset, preserving HF format.

    Reads ``W_dec``, ``W_enc``, ``b_enc``, ``b_dec`` and writes a smaller
    checkpoint suitable for polygram's EpochCompressor.
    """
    from safetensors.numpy import load_file, save_file

    state = load_file(str(input_path))
    fid = list(feature_indices)

    sliced = {}
    if "W_dec" in state:
        sliced["W_dec"] = state["W_dec"][fid]
    if "W_enc" in state:
        # SAE-Lens convention: W_enc shape (d_model, n_features)
        if state["W_enc"].shape[1] == state["W_dec"].shape[0]:
            sliced["W_enc"] = state["W_enc"][:, fid]
        else:
            sliced["W_enc"] = state["W_enc"][fid]
    if "b_enc" in state:
        sliced["b_enc"] = state["b_enc"][fid]
    if "b_dec" in state:
        sliced["b_dec"] = state["b_dec"]
    save_file(sliced, str(output_path))


def main(
    output_dir: str | Path = "examples/output/gpt2_real_sae/",
    n_features: int = 16,
    max_iterations: int = 1,
    coverage_target: float = 0.5,
    device: str = "cpu",
    scale_boost: float | str = "auto",
) -> dict:
    import torch  # noqa: F401  (lazy-imported below)
    from huggingface_hub import hf_hub_download

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: download SAE -------------------------------------
    print(f"[1/5] downloading SAE: {SAE_REPO} :: {SAE_FILE}")
    t0 = time.monotonic()
    sae_path = Path(hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE))
    print(f"      cached at {sae_path} ({time.monotonic() - t0:.1f}s)")

    # ---- Stage 2: slice to feature subset --------------------------
    print(f"[2/5] slicing SAE to first {n_features} features")
    sliced_path = output_dir / "sae_sliced.safetensors"
    slice_sae_to_features(sae_path, sliced_path, list(range(n_features)))
    print(f"      wrote {sliced_path}")

    # ---- Stage 3: run polygram EpochCompressor ---------------------
    print(f"[3/5] polygram EpochCompressor on {n_features} features, "
          f"{len(VALIDATE_PROMPTS)} prompts, layer {LAYER}, max_iter={max_iterations}")
    from polygram import EpochCompressionConfig, EpochCompressor, ValidationConfig

    compressed_path = output_dir / "sae_compressed.safetensors"
    # Iterative-loop tuning. We pass a fully custom EpochCompressionConfig
    # (CLI-driven coverage_target / max_iterations + a pinned
    # ValidationConfig matching GPT-2-small calibration), so call the
    # constructor directly rather than the .fast() preset wrapper —
    # .fast() supplies its own preset as `config=` and would collide with
    # ours (polygram 0.1.0 raises TypeError: got multiple values for
    # keyword argument 'config').
    epoch = EpochCompressor(
        sae_checkpoint=sliced_path,
        prompts=VALIDATE_PROMPTS,
        layer=LAYER,
        model_name=HOST_MODEL,
        strategy="zero",  # EpochCompressor only supports 'zero' strategy
        device=device,
        config=EpochCompressionConfig(
            coverage_target=coverage_target,
            cosine_threshold=0.30,
            n_visits_per_feature=1,
            max_iterations=max_iterations,
            validation=ValidationConfig(
                polygram_overlap_threshold=0.7,
                jaccard_threshold=0.3,
            ),
        ),
    )
    t0 = time.monotonic()
    epoch_result = epoch.run(compressed_path)
    epoch_wall = time.monotonic() - t0
    print(f"      done in {epoch_wall:.1f}s; "
          f"convergence={epoch_result.report.convergence_reason}, "
          f"zeroed={epoch_result.report.n_features_zeroed_total}/{n_features}, "
          f"panels={epoch_result.report.n_panels_total}")

    # ---- Stage 4: forge -------------------------------------------
    print("[4/5] forging gpt2 against compressed SAE")
    basis = FeatureBasis.from_polygram_checkpoint(compressed_path)
    print(f"      basis: n_features={basis.n_features} (kept), d_model={basis.d_model}")
    if basis.n_features == 0:
        raise RuntimeError("compression zeroed every feature; raise n_features or relax thresholds")
    projector = SubspaceProjector(basis, scale_boost=scale_boost)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=HOST_MODEL,
        eval_prompts=EVAL_PROMPTS,
        dtype="float32",
        device=device,
    )
    t0 = time.monotonic()
    result = pipeline.run(output_dir / "forge")
    forge_wall = time.monotonic() - t0
    print(f"      forged: n_params={result.n_params}, KL={result.faithfulness:.3f}, "
          f"wall={forge_wall:.1f}s")

    # ---- Stage 5: summary ------------------------------------------
    summary = {
        "sae_repo": SAE_REPO,
        "sae_file": SAE_FILE,
        "host_model": HOST_MODEL,
        "n_features_sliced": n_features,
        "n_features_kept_after_compression": basis.n_features,
        "d_model": basis.d_model,
        "polygram_zeroed": epoch_result.report.n_features_zeroed_total,
        "polygram_panels": epoch_result.report.n_panels_total,
        "polygram_convergence": epoch_result.report.convergence_reason,
        "polygram_wall_s": round(epoch_wall, 2),
        "forge_wall_s": round(forge_wall, 2),
        "n_params_forged": result.n_params,
        "faithfulness_kl": result.faithfulness,
        "compressed_sae_path": str(compressed_path),
        "forged_dir": str(result.output_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("[5/5] summary:")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/output/gpt2_real_sae/"
    n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    main(out, n_features=n_features, device=device)
