"""Compare single-basis vs assertion / composition / two-basis forge on GPT-2.

Decision artifact for ``two-basis-forge`` (task 7.2/7.3, NOT a pytest). Forges
``gpt2`` four ways at matched ``n_features`` and seed, then reports, per config
and over a ``composition_rank`` / ``assertion_k`` sweep:

  - global KL(host ‖ forged)
  - induction-predictable KL (the circuit-faithfulness target)
  - assertion cov95 on the forged residual
  - preserved-dimension budget  +  dim(U_C ∩ basis) overlap

and emits a Pareto plot (preserved-dim % vs the three metrics) so the budget
knee is visible. Run on Intel/GPT-2; the numbers decide whether to default
either toggle on (see openspec/changes/two-basis-forge/tasks.md §9.3).

Usage:
    python scripts/compare_single_vs_two_basis_gpt2.py --basis path/to/polygram.safetensors
    python scripts/compare_single_vs_two_basis_gpt2.py --synthetic-features 1024   # smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _text(n_chars: int) -> str:
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return urllib.request.urlopen(url, timeout=8).read().decode("utf-8", "ignore")[:n_chars]


def _build_basis(args, d_model: int):
    from saeforge import FeatureBasis

    if args.basis:
        return FeatureBasis.from_polygram_checkpoint(args.basis)
    # synthetic over-complete basis (smoke only — reproduces the over-complete regime)
    rng = np.random.default_rng(0)
    n = args.synthetic_features
    W = rng.standard_normal((n, d_model)).astype(np.float64)
    norms = np.linalg.norm(W, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n), W_dec=W, merged_norms=norms, original_norms=norms,
        scale_compression_ratio=1.0,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="gpt2")
    p.add_argument("--basis", default=None, help="polygram checkpoint; omit for a synthetic smoke basis")
    p.add_argument("--synthetic-features", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--ranks", default="4,8,16,32", help="composition_rank sweep")
    p.add_argument("--assertion-k", type=int, default=64)
    p.add_argument("--scale-boost", default="auto")
    p.add_argument("--output", type=Path, default=Path("reports/two_basis_compare.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/two_basis_pareto.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from saeforge import ForgePipeline, SubspaceProjector
    from saeforge.eval.circuit_faithfulness import circuit_kl, induction_predictable

    host = GPT2LMHeadModel.from_pretrained(args.host).eval()
    d_model = host.config.n_embd
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok(_text(200000))["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    ind_mask = np.concatenate([induction_predictable(c)[1:] for c in chunks])

    sb = args.scale_boost if args.scale_boost == "auto" else float(args.scale_boost)

    def host_logits():
        out = []
        with torch.no_grad():
            for c in chunks:
                out.append(host(input_ids=torch.tensor([c])).logits[0, :-1].float().numpy())
        return np.concatenate(out, 0)

    H = host_logits()

    def forge_and_score(label, **kw):
        basis = _build_basis(args, d_model)
        pipe = ForgePipeline(basis=basis, projector=SubspaceProjector(basis, scale_boost=sb), **kw)
        eval_ids = torch.tensor([chunks[0][:16]])
        res = pipe.run_synthetic(host, Path("/tmp") / f"forge_{label}", eval_input_ids=eval_ids)
        mod = res.model.torch_module.eval()
        fl = []
        with torch.no_grad():
            for c in chunks:
                fl.append(mod(torch.tensor([c]))[0, :-1].float().numpy())
        F = np.concatenate(fl, 0)
        ck = circuit_kl(H, F, mask=ind_mask)
        rep = getattr(pipe, "_last_augmented_report", None)
        budget = (
            float(np.mean([v["preserved_fraction"] for v in rep["layers"].values()])) if rep else 0.0
        )
        overlap = (
            float(np.mean([v["U_C_overlap_with_basis"] for v in rep["layers"].values()])) if rep else 1.0
        )
        return {"label": label, "global_kl": ck["global_kl"], "induction_kl": ck["masked_kl"],
                "preserved_fraction": budget, "U_C_overlap": overlap}

    rows = [forge_and_score("single")]
    for r in [int(x) for x in args.ranks.split(",")]:
        rows.append(forge_and_score(f"comp_r{r}", composition_preserve=True, composition_rank=r))
    rows.append(forge_and_score("assert", assertion_preserve=True, assertion_k=args.assertion_k))
    for r in [int(x) for x in args.ranks.split(",")]:
        rows.append(forge_and_score(
            f"two_r{r}", composition_preserve=True, composition_rank=r,
            assertion_preserve=True, assertion_k=args.assertion_k))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"host": args.host, "rows": rows}, indent=2, default=float))
    print(f"\n{'config':>10} {'global_kl':>10} {'induction_kl':>13} {'preserved%':>11} {'U_C∩basis':>10}")
    for r in rows:
        print(f"{r['label']:>10} {r['global_kl']:>10.3f} {r['induction_kl']:>13.3f} "
              f"{r['preserved_fraction']:>10.1%} {r['U_C_overlap']:>10.2f}")
    single = rows[0]
    best = min((r for r in rows if r["label"].startswith("two")), key=lambda r: r["induction_kl"], default=single)
    print(f"\n[verdict] best two-basis induction_kl {best['induction_kl']:.3f} vs single {single['induction_kl']:.3f} "
          f"({'IMPROVED' if best['induction_kl'] < single['induction_kl'] else 'no improvement'}); "
          f"global_kl {best['global_kl']:.3f} vs {single['global_kl']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [r["preserved_fraction"] for r in rows]
        ax.scatter(xs, [r["induction_kl"] for r in rows], c="#d62728", label="induction-predictable KL")
        ax.scatter(xs, [r["global_kl"] for r in rows], c="#1f77b4", marker="s", label="global KL")
        for r in rows:
            ax.annotate(r["label"], (r["preserved_fraction"], r["induction_kl"]), fontsize=7)
        ax.set_xlabel("preserved-dimension budget (fraction of d_model)")
        ax.set_ylabel("KL(host ‖ forged)")
        ax.set_title("two-basis forge: budget vs faithfulness (Pareto)")
        ax.legend()
        fig.tight_layout()
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130)
        print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
