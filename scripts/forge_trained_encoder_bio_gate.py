#!/usr/bin/env python3
"""Formal acceptance gate for the capability-trained encoder (change add-capability-trained-encoder,
task 6.1) on bio-sae's REAL fixtures — the real-data confirmation of the synthetic de-risk (task 4.4).

Runs `sweep_pareto_capability(train_encoder=True)` against bio-sae's two fixtures at the gate widths and
compares, COMPRESSION-CONTROLLED (same width / same kept rows, only E differs), the held-out retained-mAUC
of the trained encoder vs the always-computed pinv baseline:

  spread        runs/uniref50_n5000/pooled_w1024_k64  @ n=512   (writeup peak retained ≈ 0.93)
  concentrated  runs/uniref50_small/residue           @ n=16    (writeup peak retained ≈ 1.03)

Gate: trained-E held-out retained-mAUC **≥ pinv baseline** (a tie is a descriptive pass; trained < baseline
is the documented overfit mode — surfaced via overfit_flag, never hidden). No "closes the tax" claim.

Needs bio-sae checked out + installed (`uv pip install -e .` in bio-sae) and the ESM-2 host cached. Writes
<out>/bio_gate_summary.json. Usage: python scripts/forge_trained_encoder_bio_gate.py [--bio-root PATH]
"""
import argparse
import json
from pathlib import Path

REGIMES = {
    "spread": dict(run_dir="runs/uniref50_n5000/pooled_w1024_k64",
                   bundle="data/bio_bundle_uniref50.safetensors",
                   sequences="data/uniref50_sample__n5000_seed0.parquet",
                   feed="pooled", n_proteins=500, min_prevalence=10, sae_k=64, width=512),
    "concentrated": dict(run_dir="runs/uniref50_small/residue",
                         bundle="data/bio_bundle_uniref50_n100.safetensors",
                         sequences="data/uniref50_sample__n100_seed0.parquet",
                         feed="residue", n_proteins=10, min_prevalence=0, sae_k=32, width=16),
}


def run_regime(name, cfg, bio_root, out_dir, device, steps, widths, train_objective="proxy", seeds=(0,)):
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir = bio_root / cfg["run_dir"]
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir, bundle_path=bio_root / cfg["bundle"],
        sequences_path=bio_root / cfg["sequences"], feed=cfg["feed"],
        n_proteins=cfg["n_proteins"], max_seq_len=512,
        min_prevalence=cfg["min_prevalence"], sae_k=cfg["sae_k"],
    )
    import numpy as np
    wlist = widths or [cfg["width"]]
    # per-(width) list of (delta, pinv, trained, overfit) across seeds
    per_width = {w: [] for w in wlist}
    for seed in seeds:
        rows = sweep_pareto_capability(
            sae_checkpoint=run_dir / "sae.pt", host_model_id="facebook/esm2_t6_8M_UR50D",
            dataset=dataset, widths=wlist, scale_boosts=[1.0],
            output_dir=out_dir / f"{name}_s{seed}", cache_host=True, device=device,
            train_encoder=True, train_objective=train_objective, train_steps=steps, train_seed=seed,
        )
        for r in rows:
            if r.error_message is not None:
                print(f"   [seed {seed}] n={r.target_n_features_kept}: ERROR {r.error_message}")
                continue
            per_width[r.target_n_features_kept].append(
                (r.delta_heldout, r.retained_mauc_pinv_baseline, r.retained_mauc_trained, r.overfit_flag))
    out = []
    print(f"\n== {name} ({cfg['feed']} feed, objective={train_objective}, seeds={seeds}) ==")
    for w in wlist:
        recs = per_width[w]
        if not recs:
            out.append({"regime": name, "width": w, "error": "all seeds errored"}); continue
        deltas = np.array([x[0] for x in recs], float)
        mean_d, std_d = float(deltas.mean()), float(deltas.std())
        gate = "PASS" if mean_d >= -1e-4 else "FAIL"
        print(f"   n={w:>4}: held-out retained-mAUC  pinv {np.mean([x[1] for x in recs]):.4f} → trained "
              f"{np.mean([x[2] for x in recs]):.4f}   Δ mean {mean_d:+.4f} ± {std_d:.4f} (n_seed={len(recs)})  "
              f"[{gate}]")
        out.append({"regime": name, "width": w, "feed": cfg["feed"], "objective": train_objective,
                    "n_seeds": len(recs), "delta_mean": mean_d, "delta_std": std_d,
                    "pinv_mean": float(np.mean([x[1] for x in recs])),
                    "trained_mean": float(np.mean([x[2] for x in recs])),
                    "any_overfit": bool(any(x[3] for x in recs)), "gate": gate})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio-root", type=Path, default=Path("/home/allans/code/bio-sae"))
    ap.add_argument("--regimes", nargs="+", default=["spread", "concentrated"])
    ap.add_argument("--out", type=Path, default=Path("/tmp/bio_gate"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--widths", default=None, help="comma-separated widths (override the regime default)")
    ap.add_argument("--train-objective", choices=["proxy", "full_forge"], default="proxy")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds (multi-seed gate, e.g. 0,1,2)")
    ap.add_argument("--n-proteins", type=int, default=None, help="override regime n_proteins (full_forge is heavy)")
    args = ap.parse_args()
    if args.n_proteins is not None:
        for cfg in REGIMES.values():
            cfg["n_proteins"] = args.n_proteins
    args.out.mkdir(parents=True, exist_ok=True)
    widths = [int(w) for w in args.widths.split(",")] if args.widths else None
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"== capability-trained-encoder FORMAL bio gate (compression-controlled, held-out; "
          f"objective={args.train_objective}, seeds={seeds}) ==")
    results = []
    for n in args.regimes:
        results.extend(run_regime(n, REGIMES[n], args.bio_root, args.out, args.device, args.steps, widths,
                                  train_objective=args.train_objective, seeds=seeds))
    json.dump(results, open(args.out / "bio_gate_summary.json", "w"), indent=2)
    ok = [r for r in results if "error" not in r]
    print("\n== verdict ==")
    for r in ok:
        print(f"   {r['regime']:<13} n={r['width']:>4}  Δ mean {r['delta_mean']:+.4f} ± {r['delta_std']:.4f}  "
              f"[{r['gate']}]  (any_overfit={r['any_overfit']}, n_seeds={r['n_seeds']})")
    print(f"   wrote {args.out / 'bio_gate_summary.json'}")


if __name__ == "__main__":
    main()
