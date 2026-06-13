#!/usr/bin/env python3
"""Pythia-ladder causal forge gate — does the trained-encoder win (which SURVIVES the GPT-2 forge) hold on a
DIFFERENT causal architecture (GPT-NeoX), across model SCALE?

Runs the same measurement as ``causal_host_forge_gate.py`` (proxy-train ``E`` + score on the full multi-layer
forge, trained vs ``pinv`` retained-mAUC) but on **Pythia** hosts via the ``gpt_neox`` adapter, using
**EleutherAI sparsify SAEs** fetched from HF. The cross-scale complement to R1's ``τ*`` ladder and the
cross-architecture complement to the GPT-2 forge gate.

Ladder: pythia-70m + pythia-160m (the EleutherAI ``sae-pythia-{size}-32k`` repos that exist). The sparsify
``layers.{i}`` residual SAE hooks the **output** of block ``i`` (== ``hidden_states[i+1]`` ==
``resid_pre[i+1]``), so ``host_layer = i + 1`` (verified by reconstruction cosine).

Labels are derived from each SAE's own prevalence-band features (same self-referential protocol as the GPT-2
gate); the downstream-task encoder is the SAE's feature directions restricted to those labels + ReLU
(consistent across hosts — see ``causal_lm_forge_gate.py``). Descriptive verdict; SAE-type / protocol caveats
identical to the GPT-2 gate.

Usage:
  python scripts/pythia_ladder_gate.py --widths 64,128,256 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_lm_forge_gate import _EXTRA_CORPUS, _derive_labels_from_sae

# (model_id, sae_repo, sae_subdir, sae_block_index). host_layer = sae_block_index + 1.
LADDER = [
    ("EleutherAI/pythia-70m", "EleutherAI/sae-pythia-70m-32k", "layers.3", 3),
    ("EleutherAI/pythia-160m", "EleutherAI/sae-pythia-160m-32k", "layers.6", 6),
]


def _load_sae(sae_path: str):
    """Return (W_enc[d_model,d_sae], b_enc[d_sae], b_dec[d_model], k, sae_path) for sparsify or SAELens SAEs."""
    import torch
    from safetensors.torch import load_file

    sd = load_file(sae_path)
    if "encoder.weight" in sd:  # EleutherAI sparsify: encoder.weight (d_sae, d_model)
        W_enc = sd["encoder.weight"].float().T.contiguous()  # -> (d_model, d_sae)
        b_enc = sd["encoder.bias"].float()
    elif "W_enc" in sd:  # SAELens: W_enc (d_model, d_sae)
        W_enc = sd["W_enc"].float()
        b_enc = sd["b_enc"].float()
    else:
        raise ValueError(f"{sae_path}: no encoder.weight (sparsify) or W_enc (SAELens) key")
    b_dec = sd["b_dec"].float() if "b_dec" in sd else torch.zeros(W_enc.shape[0])
    return W_enc, b_enc, b_dec


def build_dataset(*, model_id, sae_path, host_layer, k, min_prev, max_prev, max_labels, max_seq_len,
                  n_tokens, device):
    import torch
    from transformers import AutoTokenizer

    from saeforge.calibration import _BUILTIN_CALIBRATION_TEXT
    from saeforge.datasets import CapabilityDataset
    from saeforge.sweep_capability import _extract_host_activations

    W_enc, b_enc, b_dec = _load_sae(sae_path)

    tok = AutoTokenizer.from_pretrained(model_id)
    text = _BUILTIN_CALIBRATION_TEXT + "\n" + _EXTRA_CORPUS
    sentences, total = [], 0
    for line in text.replace("\n", " ").split("."):
        s = line.strip()
        if not s:
            continue
        sentences.append(s + ".")
        total += len(tok(s + ".").input_ids)
        if total >= n_tokens:
            break

    host_X = _extract_host_activations(
        host_model_id=model_id, sequences=sentences, aggregator="pool_then_encode",
        max_seq_len=max_seq_len, device=device, feed="residue", host_layer=host_layer,
    ).numpy().astype(np.float64)

    def _full_topk(x):  # the SAE's real TopK latents (for label derivation only)
        pre = (x - b_dec) @ W_enc + b_enc
        if k and k > 0:
            topv, topi = pre.topk(int(k), dim=-1)
            z = torch.zeros_like(pre)
            return z.scatter(-1, topi, topv.relu())
        return pre.relu()

    with torch.no_grad():
        Z_full = _full_topk(torch.tensor(host_X, dtype=torch.float32)).numpy()
    feat_idx = _derive_labels_from_sae(Z_full, min_prev=min_prev, max_prev=max_prev, max_labels=max_labels)
    labels = (Z_full[:, feat_idx] > 0).astype(np.float64)

    # Downstream-task encoder = the SAE's feature directions restricted to the labels + ReLU (fast, consistent
    # with the GPT-2 gate). TopK gating is dropped here so training is tractable; the directions are the SAE's.
    We = W_enc[:, feat_idx].contiguous()
    be = b_enc[feat_idx].contiguous()
    bd = b_dec.contiguous()

    def encoder(x):
        return torch.relu((x - bd) @ We + be)

    ds = CapabilityDataset(
        sequences=sentences, labels=labels, encoder=encoder, tokenizer_id=model_id, feed="residue",
        aggregator="pool_then_encode", min_prevalence=0, decode_via_basis=True,
        metadata={"host": model_id, "host_layer": host_layer, "n_tokens": int(host_X.shape[0]),
                  "n_labels": int(feat_idx.size)},
    )
    return ds


def run_rung(model_id, sae_repo, sae_subdir, block_idx, *, widths, seeds, steps, args):
    from huggingface_hub import hf_hub_download

    from saeforge import sweep_pareto_capability

    import json as _json
    sae_path = hf_hub_download(sae_repo, filename=f"{sae_subdir}/sae.safetensors")
    cfg_path = hf_hub_download(sae_repo, filename=f"{sae_subdir}/cfg.json")
    k = int(_json.load(open(cfg_path)).get("k", 0))
    host_layer = block_idx + 1
    ds = build_dataset(model_id=model_id, sae_path=sae_path, host_layer=host_layer, k=k,
                       min_prev=args.min_prev, max_prev=args.max_prev, max_labels=args.max_labels,
                       max_seq_len=args.max_seq_len, n_tokens=args.n_tokens, device=args.device)
    print(f"\n== {model_id} ({sae_subdir} → host_layer {host_layer}, TopK k={k}): "
          f"{ds.metadata['n_tokens']} tokens, {ds.metadata['n_labels']} labels ==")
    per_width = {w: [] for w in widths}
    for seed in seeds:
        rows = sweep_pareto_capability(
            sae_checkpoint=sae_path, host_model_id=model_id, dataset=ds, widths=widths, scale_boosts=[1.0],
            output_dir=args.out / f"{model_id.split('/')[-1]}_s{seed}", cache_host=True, device=args.device,
            max_seq_len=args.max_seq_len, host_layer=host_layer, train_encoder=True,
            train_objective="proxy", train_steps=steps, train_seed=seed,
        )
        for r in rows:
            if r.error_message is not None:
                print(f"   [seed {seed}] n={r.target_n_features_kept}: ERROR {r.error_message}")
                continue
            per_width[r.target_n_features_kept].append(
                (r.delta_heldout, r.retained_mauc_pinv_baseline, r.retained_mauc_trained, r.overfit_flag))
    out = []
    for w in widths:
        recs = [x for x in per_width[w] if x[0] is not None]
        if not recs:
            print(f"   n={w:>4}: all seeds errored/none")
            continue
        d = np.array([x[0] for x in recs]); mean_d, std_d = float(d.mean()), float(d.std())
        gate = "SURVIVES" if mean_d > 1e-4 else ("TIE" if mean_d >= -1e-4 else "ERASED")
        pinv_m = float(np.mean([x[1] for x in recs])); tr_m = float(np.mean([x[2] for x in recs]))
        any_of = bool(any(x[3] for x in recs))
        print(f"   n={w:>4}: pinv {pinv_m:.4f} -> trained {tr_m:.4f}   Δ {mean_d:+.4f} ± {std_d:.4f} "
              f"(n={len(recs)})  [{gate}]{'  overfit' if any_of else ''}")
        out.append({"host": model_id, "host_layer": host_layer, "width": w, "n_seeds": len(recs),
                    "delta_mean": mean_d, "delta_std": std_d, "pinv_mean": pinv_m, "trained_mean": tr_m,
                    "any_overfit": any_of, "gate": gate})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", default="64,128,256")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-tokens", type=int, default=1400)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--min-prev", type=float, default=0.04)
    ap.add_argument("--max-prev", type=float, default=0.40)
    ap.add_argument("--max-labels", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="substring filter on model_id (e.g. '70m')")
    ap.add_argument("--out", type=Path, default=Path("/tmp/pythia_ladder_gate"))
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"== Pythia-ladder forge gate (gpt_neox; proxy-train + full-forge-score; steps={args.steps}, "
          f"seeds={seeds}) ==")
    results = []
    for model_id, repo, sub, bidx in LADDER:
        if args.only and args.only not in model_id:
            continue
        results.extend(run_rung(model_id, repo, sub, bidx, widths=widths, seeds=seeds, steps=args.steps,
                                args=args))
    json.dump(results, open(args.out / "pythia_ladder_summary.json", "w"), indent=2)
    print("\n== verdict (trained − pinv, FULL-forge retained-mAUC) ==")
    for r in results:
        print(f"   {r['host'].split('/')[-1]:<14} n={r['width']:>4}  Δ {r['delta_mean']:+.4f} ± "
              f"{r['delta_std']:.4f}  [{r['gate']}]{'  overfit' if r['any_overfit'] else ''}")
    print(f"   wrote {args.out / 'pythia_ladder_summary.json'}")


if __name__ == "__main__":
    main()
