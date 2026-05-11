"""Single-basis vs hybrid-bridge forge comparison on real GPT-2.

This is the cross-architecture defaults-validation surface documented in
``openspec/changes/hybrid-bridge-forge/design.md`` § "Cross-architecture
validation tiering" (T1). Runs on Intel Mac (CPU) in a couple of minutes.

What it does:

1. Loads real ``gpt2`` (cached HF checkpoint).
2. **Untying workaround**: GPT-2 ships with tied embeddings. Since hybrid
   forging refuses tied hosts, this script clones ``wte.weight`` into
   ``lm_head.weight`` (numerically identical, separate storage) and flips
   ``config.tie_word_embeddings = False``. The host's outputs are unchanged.
3. Builds three "synthetic SAE" bases by running a small prompt set through
   the host with ``output_hidden_states=True`` and PCA-ing each captured
   layer's residual to ``n_features`` directions. These are NOT trained SAEs —
   they are PCA proxies cheap enough to compute on CPU. Real SAE bases would
   be plugged in via ``--basis-{embed,mid,lm-head}`` paths in the CLI.
4. Forges three native models:
   - single (mid basis only),
   - single (embed basis only),
   - single (lm_head basis only),
   - hybrid (all three + bridges).
5. Computes faithfulness KL for each on a held-out eval set.
6. Prints a per-config comparison and writes a JSON record to ``--output``.

Usage:

    python scripts/compare_single_vs_hybrid_gpt2.py --output runs/comparison.json

Outputs:

    Pre-FT KL (lower = more faithful):
      single-mid:    <number>
      single-embed:  <number>
      single-lm:     <number>
      hybrid:        <number>

The point of the harness is *not* to settle whether bridges work. It's to give
a reproducible signal-vs-noise pattern that contributors with NVIDIA/CUDA
hosts (the T4 tier) can rerun on larger models to push the signal up. If
``hybrid`` is not measurably better than ``single-mid`` here, that's still a
valid result — documented in ``docs/hybrid_bridge_intel_gpt2.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# A short, fixed prompt set. Small enough for CPU, varied enough to surface
# distributional differences. Kept inline so the script is hermetic.
DEFAULT_PROMPTS = [
    "The capital of France is Paris, a city famous for its art, fashion, and cuisine.",
    "Once upon a time, in a land far away, there lived a kind old farmer who",
    "The mitochondria is the powerhouse of the cell, responsible for producing ATP",
    "She opened the door and saw a stranger standing in the rain, holding an umbrella.",
    "After running for hours through the forest, he finally reached the edge of the cliff.",
    "The recipe calls for two cups of flour, three eggs, half a cup of milk, and a pinch of salt.",
    "In a stunning turn of events, the long-shot candidate won the local mayoral race by a narrow margin.",
    "The professor explained that the equation governs the behavior of fluids under pressure.",
    "Climate change is the long-term shift in temperatures and weather patterns across the globe.",
    "Machine learning models trained on large datasets can capture complex statistical patterns.",
    "The novel opens with a description of a small town surrounded by misty mountains and dense forests.",
    "Scientists studying the human brain have discovered that different regions specialize in different tasks.",
    "Cooking pasta requires boiling water, adding salt, and stirring occasionally to prevent sticking.",
    "The detective examined the crime scene carefully, noting every fingerprint and footprint.",
    "Renewable energy sources like solar and wind are becoming increasingly cost-competitive.",
    "Mathematics is often called the language of the universe because it describes natural phenomena.",
    "The orchestra began with a soft melody from the strings, then built to a powerful crescendo.",
    "Throughout history, civilizations have risen and fallen in patterns that scholars still debate.",
    "Programming languages each have their own strengths, weaknesses, and idiomatic styles of expression.",
    "The wolf paced silently through the snowy forest, leaving tracks that quickly filled with fresh powder.",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales known.",
    "Doctors recommend exercising regularly, eating a balanced diet, and getting enough sleep.",
    "The artist mixed her paints carefully, blending blues and greens to capture the ocean's depth.",
    "Linguistic analysis can reveal hidden patterns in how different communities use language.",
    "The garden bloomed in spring with tulips, daffodils, and cherry blossoms in every color.",
    "Economists disagree about whether interest rate increases will slow inflation or cause a recession.",
    "The startup raised twenty million dollars in Series A funding from a top-tier venture firm.",
    "Astronauts on the International Space Station conduct experiments in microgravity every day.",
    "Children learn to read by gradually mapping letters to sounds and sounds to words.",
    "The mountain trail wound up through pine forests before opening onto alpine meadows.",
    "Ancient Roman engineers built aqueducts that still carry water in some parts of Europe today.",
    "Modern smartphones contain more computing power than the systems that landed humans on the moon.",
    "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen in plant leaves.",
    "The history of the printing press tracks closely with the rise of literacy across early modern Europe.",
    "Volcanic eruptions can dramatically alter the climate for years after the initial event by releasing aerosols.",
    "Reinforcement learning agents improve their behavior by maximizing a reward signal through repeated interaction.",
    "The chess grandmaster studied her opponent's openings for weeks before the championship match.",
    "Octopuses are remarkably intelligent invertebrates, capable of solving puzzles and using simple tools.",
    "The desert sun beat down mercilessly on the travelers as they searched for the next oasis.",
    "Modern cryptography relies on the difficulty of certain mathematical problems like integer factorization.",
    "The biologist labeled each specimen carefully, recording the date, location, and habitat conditions.",
    "Antibiotic resistance is a growing global health threat driven by overuse in both medicine and agriculture.",
    "She tied her hair back and stepped onto the climbing wall, chalk dust still on her fingers.",
    "Statistical mechanics bridges the gap between the microscopic behavior of particles and macroscopic phenomena.",
    "The carpenter measured twice and cut once, the boards stacked neatly against the workshop wall.",
    "Ocean currents redistribute heat around the planet, shaping the climate of distant coastlines.",
    "Architecture students learn to balance aesthetics, structural integrity, and the practical needs of occupants.",
    "The lighthouse keeper had not seen another human for three weeks when the supply boat finally arrived.",
    "Genome sequencing has revealed surprising evolutionary connections between species that look very different.",
    "He rolled the dice and moved his piece across the board, narrowly avoiding his opponent's trap.",
    "Sustainable agriculture seeks to feed humanity while preserving soil health and biodiversity for future generations.",
    "The hiking trail led them past waterfalls, through dense ferns, and onto a ridge with sweeping vistas.",
]

EVAL_PROMPTS = [
    "The weather today is",
    "Scientists have discovered that",
    "When you wake up in the morning,",
    "The most important lesson I learned was",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host-model", default="gpt2", help="HF model id (default: gpt2)")
    p.add_argument("--n-features", type=int, default=256, help="Basis size for all three bases.")
    p.add_argument(
        "--embed-layer",
        type=int,
        default=0,
        help="Host layer index whose residual seeds basis_embed (0 = post-embedding).",
    )
    p.add_argument(
        "--mid-layer",
        type=int,
        default=6,
        help="Host layer index whose residual seeds basis_mid (default ≈ middle of gpt2).",
    )
    p.add_argument(
        "--lm-layer",
        type=int,
        default=11,
        help="Host layer index whose residual seeds basis_lm_head (default = final block).",
    )
    p.add_argument(
        "--bridge-init",
        default="orthogonal",
        choices=("orthogonal", "identity", "zero"),
    )
    p.add_argument("--bridge-nonlin", default="none", choices=("none", "relu", "gelu"))
    p.add_argument("--bridge-no-pre-ln", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Write JSON results to this path.")
    return p.parse_args()


def untie_embeddings(host) -> None:
    """Clone wte→lm_head and flip ``tie_word_embeddings`` to False.

    The model behaves identically forward (lm_head.weight is numerically
    identical to wte.weight after the clone) — but the host is no longer
    flagged as tied, which lets hybrid forging proceed.
    """
    import torch

    with torch.no_grad():
        host.lm_head.weight = torch.nn.Parameter(
            host.transformer.wte.weight.detach().clone()
        )
    host.config.tie_word_embeddings = False


def capture_residuals(host, tokenizer, prompts, layer_indices):
    """Run ``prompts`` through ``host`` and return one ``(tokens, d_model)`` array per layer.

    Uses HF's ``output_hidden_states=True``. Layer index 0 corresponds to the
    post-embedding residual; subsequent indices are post-block residuals.
    """
    import torch

    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=48,
    )
    with torch.no_grad():
        out = host(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
    # hidden_states is a tuple of (n_layer+1,) tensors of shape (B, T, d_model).
    all_hidden = out.hidden_states
    captured = {}
    for idx in layer_indices:
        # ``idx`` is interpreted as the residual *after* block ``idx`` (idx=0 =
        # post-embedding, idx=11 = post-final-block).
        if idx < 0 or idx > len(all_hidden) - 1:
            raise ValueError(f"layer index {idx} out of range [0, {len(all_hidden)-1}]")
        h = all_hidden[idx]
        # Flatten across batch+time using only non-pad tokens.
        mask = inputs["attention_mask"].bool()
        flat = h[mask].detach().cpu().double().numpy()  # (tokens, d_model)
        captured[idx] = flat
    return captured


def pca_basis(activations: np.ndarray, n_features: int, seed: int):
    """Build a synthetic basis (PCA top-``n_features`` directions) from activations.

    The resulting W_dec has shape (n_features, d_model) with orthonormal rows.
    Returned as a ``FeatureBasis``.
    """
    from saeforge.basis import FeatureBasis

    # Center, then take SVD. With n_features ≤ d_model this is cheap (16-256
    # singular vectors over a ~few-hundred-token matrix).
    centered = activations - activations.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    if n_features > Vt.shape[0]:
        raise ValueError(
            f"requested n_features={n_features} > rank {Vt.shape[0]} of captured activations"
        )
    W_dec = Vt[:n_features]  # (n_features, d_model), orthonormal rows
    return FeatureBasis(
        kept_ids=np.arange(n_features),
        W_dec=W_dec,
        merged_norms=np.linalg.norm(W_dec, axis=1),
        original_norms=np.linalg.norm(W_dec, axis=1),
        scale_compression_ratio=1.0,
        metadata={"source": "pca_synthetic", "seed": int(seed)},
    )


def forge_and_kl(*, basis_mid, basis_embed, basis_lm_head, host, tokenizer, eval_prompts, hybrid: bool, bridge_init: str, bridge_nonlin: str, pre_layernorm: bool):
    """Run a single forge (hybrid or not) and return faithfulness KL on ``eval_prompts``."""
    from saeforge.bridges import BridgeConfig
    from saeforge.eval.faithfulness import faithfulness_kl
    from saeforge.model import NativeModel, _config_from_host
    from saeforge.projector import SubspaceProjector
    from saeforge.hybrid_basis import HybridBasisBundle

    proj = SubspaceProjector(basis_mid, scale_boost="auto")

    bundle = None
    if hybrid:
        bundle = HybridBasisBundle(
            basis_embed=basis_embed,
            basis_mid=basis_mid,
            basis_lm_head=basis_lm_head,
            n_layer=host.config.n_layer,
        )
    weights = proj.project_module(host, hybrid=bundle)
    config = _config_from_host(host, basis_mid.n_features)
    if hybrid:
        config.bridges = True
        config.bridge_init = bridge_init
        config.bridge_nonlin = bridge_nonlin
        config.bridge_pre_layernorm = pre_layernorm
    model = NativeModel.from_projected_weights(config, weights)
    kl = faithfulness_kl(model, host, eval_prompts, tokenizer=tokenizer, device="cpu")
    return kl


def main():
    args = parse_args()
    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.host_model} (cached)...", flush=True)
    t0 = time.time()
    host = AutoModelForCausalLM.from_pretrained(args.host_model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.host_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    if getattr(host.config, "tie_word_embeddings", False):
        print("Applying untying workaround (clone wte → lm_head)...", flush=True)
        untie_embeddings(host)

    print(
        f"Capturing residuals at layers {args.embed_layer}/{args.mid_layer}/{args.lm_layer}...",
        flush=True,
    )
    layer_indices = sorted({args.embed_layer, args.mid_layer, args.lm_layer})
    captured = capture_residuals(host, tokenizer, DEFAULT_PROMPTS, layer_indices)
    n = args.n_features
    print(
        f"  captured {captured[layer_indices[0]].shape[0]} tokens × "
        f"{captured[layer_indices[0]].shape[1]} dims at each layer",
        flush=True,
    )

    print(f"Building PCA bases (n_features={n})...", flush=True)
    basis_embed = pca_basis(captured[args.embed_layer], n, args.seed)
    basis_mid = pca_basis(captured[args.mid_layer], n, args.seed + 1)
    basis_lm = pca_basis(captured[args.lm_layer], n, args.seed + 2)

    results: dict[str, float] = {}
    configs = [
        ("single-mid", False, basis_mid, basis_mid, basis_mid),
        ("single-embed", False, basis_embed, basis_embed, basis_embed),
        ("single-lm", False, basis_lm, basis_lm, basis_lm),
        ("hybrid", True, basis_mid, basis_embed, basis_lm),
    ]
    for label, hybrid, b_mid, b_embed, b_lm in configs:
        t1 = time.time()
        kl = forge_and_kl(
            basis_mid=b_mid,
            basis_embed=b_embed,
            basis_lm_head=b_lm,
            host=host,
            tokenizer=tokenizer,
            eval_prompts=EVAL_PROMPTS,
            hybrid=hybrid,
            bridge_init=args.bridge_init,
            bridge_nonlin=args.bridge_nonlin,
            pre_layernorm=not args.bridge_no_pre_ln,
        )
        results[label] = float(kl)
        print(f"  {label:15s} KL = {kl:.4f}   ({time.time()-t1:.1f}s)", flush=True)

    record = {
        "host_model": args.host_model,
        "n_features": n,
        "embed_layer": args.embed_layer,
        "mid_layer": args.mid_layer,
        "lm_layer": args.lm_layer,
        "bridge_init": args.bridge_init,
        "bridge_nonlin": args.bridge_nonlin,
        "pre_layernorm": not args.bridge_no_pre_ln,
        "seed": args.seed,
        "results": results,
        "best": min(results, key=results.get),
        "hybrid_vs_single_mid_delta": results["hybrid"] - results["single-mid"],
    }
    print()
    print(f"Best: {record['best']}  (KL = {results[record['best']]:.4f})")
    print(
        f"Hybrid vs single-mid: ΔKL = {record['hybrid_vs_single_mid_delta']:+.4f}  "
        f"({'hybrid wins' if record['hybrid_vs_single_mid_delta'] < 0 else 'single-mid wins'})"
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(record, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
