#!/usr/bin/env python3
"""Capability-ceiling diagnostic + the MATCHED-SAE control (add-capability-ceiling-diagnostic, PR #122).

Two experiments in one, both on **GPT-2 `hidden_states[8]`** (so the host + activation point are identical):

1. **Capability-ceiling decomposition** (`saeforge.capability_ceiling`): per width `N`, the four activation-side
   retained-mAUC references (random / activation-PCA `svd` / `pinv`-top-atoms / capability-supervised
   `best_atoms` / trained `ceiling`) and the three gaps (`selection_gap`, `interpretability_tax`,
   `ceiling_gap`). Everything is **encoder-side** geometry (the PR #122 correction — NOT readout).

2. **Matched-SAE control** — the clean isolation the X2/forge story needed: run (1) on the SAME host with a
   **ReLU** SAE (jbloom `blocks.8.hook_resid_pre`, 24576) vs a **TopK** SAE (OpenAI v5 `blocks.7.hook_resid_post`
   == hidden_states[8], k=32, 32768). If the `pinv`-leaves-room gaps (`selection_gap` + `interpretability_tax`)
   are large on the ReLU dictionary but small on the TopK one, the X2 "trained beats pinv" win was
   **SAE-dictionary-specific**, not host-class — isolating the confound with host held fixed.

Usage: python scripts/capability_ceiling_gate.py --widths 64,128,256 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from causal_lm_forge_gate import _EXTRA_CORPUS, _derive_labels_from_sae

SAES = {
    "relu": dict(repo="jbloom/GPT2-Small-SAEs-Reformatted",
                 sub="blocks.8.hook_resid_pre/sae_weights.safetensors", k=None, layer=8),
    "topk": dict(repo="jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs",
                 sub="v5_32k_layer_7.pt/sae_weights.safetensors", k=32, layer=8),
}


def _gpt2_hidden(layer: int, n_tokens: int, ctx: int, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from saeforge.calibration import _BUILTIN_CALIBRATION_TEXT

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(device).eval()
    ids = tok(_BUILTIN_CALIBRATION_TEXT + "\n\n" + _EXTRA_CORPUS, return_tensors="pt").input_ids[0]
    chunks = []
    with torch.no_grad():
        for i in range(0, ids.shape[0], ctx):
            out = model(ids[i:i + ctx].unsqueeze(0).to(device), output_hidden_states=True)
            chunks.append(out.hidden_states[layer][0].cpu().float())
            if sum(c.shape[0] for c in chunks) >= n_tokens:
                break
    return torch.cat(chunks, 0)[:n_tokens].numpy().astype(np.float64)


def _load_sae(repo: str, sub: str):
    from safetensors.torch import load_file
    path = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/{sub}"))
    if not path:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, filename=sub)
    else:
        p = path[0]
    sd = load_file(p)
    return (sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float(), sd["b_dec"].float())


def build_bundle(name: str, X, *, min_prev, max_prev, max_labels):
    import torch
    cfg = SAES[name]
    W_enc, b_enc, W_dec, b_dec = _load_sae(cfg["repo"], cfg["sub"])
    k = cfg["k"]
    Xt = torch.tensor(X, dtype=torch.float32)

    def native(x):  # the SAE's real activation, for label derivation
        pre = (x - b_dec) @ W_enc + b_enc
        if k:
            v, i = pre.topk(int(k), dim=-1)
            z = torch.zeros_like(pre)
            return z.scatter(-1, i, v.relu())
        return pre.relu()

    with torch.no_grad():
        Z = native(Xt).numpy()
    feat = _derive_labels_from_sae(Z, min_prev=min_prev, max_prev=max_prev, max_labels=max_labels)
    labels = (Z[:, feat] > 0).astype(np.float64)
    We, be, bd = W_enc[:, feat].contiguous(), b_enc[feat].contiguous(), b_dec.contiguous()

    def encoder(x):  # restricted-ReLU readout of the label features (consistent scoring across both SAEs)
        return torch.relu((x - bd) @ We + be)

    return encoder, W_dec.numpy().astype(np.float64), labels, int(feat.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", default="64,128,256")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-tokens", type=int, default=1400)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--min-prev", type=float, default=0.04)
    ap.add_argument("--max-prev", type=float, default=0.40)
    ap.add_argument("--max-labels", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("/tmp/capability_ceiling_gate"))
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    from saeforge.capability_ceiling import capability_ceiling_decomposition

    X = _gpt2_hidden(8, args.n_tokens, args.ctx, args.device)
    print(f"== capability-ceiling + matched-SAE control (GPT-2 hidden_states[8], {X.shape[0]} tokens, "
          f"steps={args.steps}, seeds={seeds}) ==")
    results = {}
    for name in ("relu", "topk"):
        enc, W_dec, labels, nlab = build_bundle(name, X, min_prev=args.min_prev, max_prev=args.max_prev,
                                                max_labels=args.max_labels)
        print(f"\n== GPT-2 + {name.upper()} SAE ({W_dec.shape[0]} atoms, {nlab} labels) ==")
        rows = []
        for w in widths:
            seed_dec = [capability_ceiling_decomposition(X, enc, labels, W_dec, w, steps=args.steps, seed=s)
                        for s in seeds]
            agg = {kf: float(np.mean([getattr(d, kf) for d in seed_dec]))
                   for kf in ("retained_mauc_random", "retained_mauc_svd", "retained_mauc_pinv",
                              "retained_mauc_best_atoms", "retained_mauc_ceiling",
                              "selection_gap", "interpretability_tax", "ceiling_gap")}
            agg["width"] = w
            agg["any_overfit"] = bool(any(d.overfit_flag for d in seed_dec))
            rows.append(agg)
            print(f"   n={w:>4}: rand {agg['retained_mauc_random']:.3f} svd {agg['retained_mauc_svd']:.3f} "
                  f"pinv {agg['retained_mauc_pinv']:.3f} best_atoms {agg['retained_mauc_best_atoms']:.3f} "
                  f"ceiling {agg['retained_mauc_ceiling']:.3f}  |  selection_gap {agg['selection_gap']:+.3f} "
                  f"interp_tax {agg['interpretability_tax']:+.3f} ceiling_gap {agg['ceiling_gap']:+.3f}")
        results[name] = rows

    json.dump(results, open(args.out / "capability_ceiling_summary.json", "w"), indent=2)
    print("\n== matched-SAE control verdict (selection_gap + interpretability_tax, ReLU vs TopK) ==")
    for w_i, w in enumerate(widths):
        r, t = results["relu"][w_i], results["topk"][w_i]
        r_room = r["selection_gap"] + r["interpretability_tax"]
        t_room = t["selection_gap"] + t["interpretability_tax"]
        print(f"   n={w:>4}: pinv-leaves-room  ReLU {r_room:+.3f}   TopK {t_room:+.3f}   "
              f"(ReLU−TopK {r_room - t_room:+.3f})")
    print(f"   wrote {args.out / 'capability_ceiling_summary.json'}")


if __name__ == "__main__":
    main()
