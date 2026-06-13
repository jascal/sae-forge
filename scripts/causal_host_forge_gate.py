#!/usr/bin/env python3
"""Causal-host FULL-FORGE gate — does the causal trained-encoder win SURVIVE the forge? (the forge-level
falsifier the activation-level `causal_lm_forge_gate.py` left open).

The GPT-2 analog of `forge_trained_encoder_bio_gate.py`: it runs the **same** measurement the ESM bio gate
ran — **proxy-train `E`, score on the full multi-layer forge** — but on a **causal LM** host, using the
mid-layer forged-hidden extraction added by `add-causal-host-capability-sweep`.

  host        gpt2 (causal), residual at blocks.8.hook_resid_pre (hidden_states[8])
  SAE         jbloom GPT2-Small-SAEs-Reformatted, blocks.8 (SAELens W_dec/W_enc, 24576x768, ReLU/L1)
  labels      derived from the SAE's own prevalence-band features (same protocol as causal_lm_forge_gate)
  metric      trained-E vs pinv FULL-forge retained-mAUC (held-out, compression-controlled, multi-seed)

Outcome (descriptive, both first-class):
  * SURVIVES  — trained-E full-forge retained-mAUC > pinv, noise-clearing → the causal projection win is
                forge-level, not just activation-level. Strongest vindication on a real LM forge.
  * ERASED    — trained-E ties/loses through the full GPT-2 forge (as on ESM) → the forge tax (LayerNorm /
                TopK rank-shuffle) erases the causal activation-level gain too.

Usage:
  python scripts/causal_host_forge_gate.py --widths 64,128,256 --seeds 0,1,2 --steps 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

# Reuse the corpus + label-derivation from the activation-level gate (no duplication).
from causal_lm_forge_gate import _EXTRA_CORPUS, _derive_labels_from_sae


def _jbloom_sae_path(layer: int) -> str:
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/"
        f"snapshots/*/blocks.{layer}.hook_resid_pre/sae_weights.safetensors"
    ))
    if not hits:
        raise FileNotFoundError(f"jbloom GPT-2 SAE for layer {layer} not cached.")
    return hits[0]


def build_dataset(*, layer: int, ctx: int, n_tokens: int, min_prev: float, max_prev: float,
                  max_labels: int, max_seq_len: int, device: str):
    """Build a per-token CapabilityDataset for GPT-2 whose labels align with the sweep's own
    `host_layer=layer, feed='residue'` extraction (same tokenizer/sequences/layer → same row order)."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from saeforge.calibration import _BUILTIN_CALIBRATION_TEXT
    from saeforge.datasets import CapabilityDataset
    from saeforge.sweep_capability import _extract_host_activations

    sae_path = _jbloom_sae_path(layer)
    sd = load_file(sae_path)
    W_enc, b_enc, b_dec = sd["W_enc"].float(), sd["b_enc"].float(), sd["b_dec"].float()

    # Split the corpus into sentence-sized sequences (the sweep tokenizes each separately, per-token).
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = _BUILTIN_CALIBRATION_TEXT + "\n" + _EXTRA_CORPUS
    sentences, total = [], 0
    for line in text.replace("\n", " ").split("."):
        s = line.strip()
        if not s:
            continue
        s = s + "."
        sentences.append(s)
        total += len(tok(s).input_ids)
        if total >= n_tokens:
            break

    # Extract host activations EXACTLY as the sweep will (causal, layer, residue) → align labels 1:1.
    host_X = _extract_host_activations(
        host_model_id="gpt2", sequences=sentences, aggregator="pool_then_encode",
        max_seq_len=max_seq_len, device=device, feed="residue", host_layer=layer,
    ).numpy().astype(np.float64)

    with torch.no_grad():
        Z_full = torch.relu((torch.tensor(host_X, dtype=torch.float32) - b_dec) @ W_enc + b_enc).numpy()
    feat_idx = _derive_labels_from_sae(Z_full, min_prev=min_prev, max_prev=max_prev, max_labels=max_labels)
    labels = (Z_full[:, feat_idx] > 0).astype(np.float64)

    We = W_enc[:, feat_idx].contiguous()
    be = b_enc[feat_idx].contiguous()
    bd = b_dec.contiguous()

    def encoder(x):
        return torch.relu((x - bd) @ We + be)

    ds = CapabilityDataset(
        sequences=sentences, labels=labels, encoder=encoder, tokenizer_id="gpt2",
        feed="residue", aggregator="pool_then_encode", min_prevalence=0, decode_via_basis=True,
        metadata={"host": "gpt2", "host_class": "causal", "layer": layer,
                  "n_tokens": int(host_X.shape[0]), "n_labels": int(feat_idx.size)},
    )
    return ds, sae_path, host_X


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", default="64,128,256")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=1400)
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="char-truncation cap (ESM-ism, applied as seq[:N]); keep >= longest sentence so "
                         "GPT-2 token counts stay consistent between label-build and the sweep's extraction")
    ap.add_argument("--min-prev", type=float, default=0.04)
    ap.add_argument("--max-prev", type=float, default=0.40)
    ap.add_argument("--max-labels", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("/tmp/causal_host_forge_gate"))
    args = ap.parse_args()

    from saeforge import sweep_pareto_capability

    widths = [int(w) for w in args.widths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    ds, sae_path, host_X = build_dataset(
        layer=args.layer, ctx=args.ctx, n_tokens=args.n_tokens, min_prev=args.min_prev,
        max_prev=args.max_prev, max_labels=args.max_labels, max_seq_len=args.max_seq_len, device=args.device,
    )
    print(f"== causal-host FULL-FORGE gate (GPT-2 layer {args.layer}, proxy-train + full-forge-score; "
          f"held-out, compression-controlled; steps={args.steps}, seeds={seeds}) ==")
    print(f"   {ds.metadata['n_tokens']} per-token items, {ds.metadata['n_labels']} labels, "
          f"{len(ds.sequences)} sequences")

    per_width = {w: [] for w in widths}
    for seed in seeds:
        rows = sweep_pareto_capability(
            sae_checkpoint=sae_path, host_model_id="gpt2", dataset=ds, widths=widths,
            scale_boosts=[1.0], output_dir=args.out / f"s{seed}", cache_host=True, device=args.device,
            max_seq_len=args.max_seq_len, host_layer=args.layer, train_encoder=True,
            train_objective="proxy", train_steps=args.steps, train_seed=seed,
        )
        for r in rows:
            if r.error_message is not None:
                print(f"   [seed {seed}] n={r.target_n_features_kept}: ERROR {r.error_message}")
                continue
            per_width[r.target_n_features_kept].append(
                (r.delta_heldout, r.retained_mauc_pinv_baseline, r.retained_mauc_trained, r.overfit_flag))

    out = []
    print("\n== verdict (trained − pinv, FULL-forge retained-mAUC) ==")
    for w in widths:
        recs = per_width[w]
        if not recs:
            print(f"   n={w:>4}: all seeds errored")
            continue
        deltas = np.array([x[0] for x in recs if x[0] is not None], float)
        mean_d, std_d = float(deltas.mean()), float(deltas.std())
        gate = "SURVIVES" if mean_d > 1e-4 else ("TIE" if mean_d >= -1e-4 else "ERASED")
        pinv_m = float(np.mean([x[1] for x in recs]))
        tr_m = float(np.mean([x[2] for x in recs]))
        any_of = bool(any(x[3] for x in recs))
        print(f"   n={w:>4}: pinv {pinv_m:.4f} -> trained {tr_m:.4f}   Δ {mean_d:+.4f} ± {std_d:.4f} "
              f"(n_seed={len(deltas)})  [{gate}]{'  overfit' if any_of else ''}")
        out.append({"host": "gpt2", "layer": args.layer, "width": w, "n_seeds": len(deltas),
                    "delta_mean": mean_d, "delta_std": std_d, "pinv_mean": pinv_m, "trained_mean": tr_m,
                    "any_overfit": any_of, "gate": gate})
    json.dump(out, open(args.out / "causal_host_gate_summary.json", "w"), indent=2)
    print(f"   wrote {args.out / 'causal_host_gate_summary.json'}")


if __name__ == "__main__":
    main()
