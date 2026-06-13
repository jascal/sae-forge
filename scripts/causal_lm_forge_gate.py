#!/usr/bin/env python3
"""Causal-LM forge gate — does the trained-encoder learning transfer to a CAUSAL host?

The follow-up the `add-full-forge-encoder-training` gate (and its host-class caveat) pointed to.
`add-capability-trained-encoder` was *motivated* by lm-sae's R2 result — a trained rank-`r`
projection beats frozen SVD (+13pp) — measured on **causal autoregressive LMs**. But both
sae-forge gates ran on **ESM-2, a non-causal bidirectional masked encoder**, and returned a null
(trained-`E` does not beat `pinv`). The honest reading of that null is *host-class-specific*; the
decisive untested case is a **causal** host. This script runs it.

## What it tests (and why this level)

The exact analog of the ESM gate — the *full multi-layer NativeModel forge* — is **blocked for
GPT-2**: the cached jbloom GPT-2 SAEs live on a *mid-layer* residual stream
(`blocks.8.hook_resid_pre`), but sae-forge's `ForgedGPT2.forward` only emits *final-layer logits*
with no intermediate-hidden-state extraction (`adapters/gpt2.py`; the sweep's `_extract_*` helpers
are ESM-shaped: `host.esm` / `last_hidden_state` / `[1:-1]`). Generalising that is a separate
plumbing change (tracked in the proposal's "What this does NOT solve").

What IS testable now — and what R2 is actually *about* — is the **projection** question at the
**activation level**: at a matched compressed rank `N`, does a *trained* encoder `E`
(`d_model -> N`) beat the closed-form `pinv(W_dec)` at preserving a downstream task through the
decode∘encode bottleneck `(x @ E) @ W_dec`? `saeforge.training.train_encoder(objective="distill")`
runs exactly that comparison — held-out, compression-controlled, `overfit_flag`-guarded — and is
**host-agnostic** (it consumes activations + an SAE encoder + labels, no NativeModel forge). So we
run the *identical* activation-level gate on:

  * a **causal** host  — GPT-2 layer-8 residual + the jbloom GPT-2-Small SAE, and
  * a **non-causal** control — ESM-2 + bio-sae's SAE (the substrate the null was measured on),

at matched widths, and compare `delta_heldout` (trained − pinv). If trained beats pinv on the
causal host but not the non-causal one, the null is host-class-specific (R2's structure is causal).
If trained *also* fails to beat pinv on the causal host, the projection-near-optimal reading
generalises past host class — a deeper, equally-reportable finding.

## Protocol (identical across hosts)

GPT-2 has no external ground-truth feature labels (bio-sae has GO/Pfam/EC), so to keep the two
hosts on the *same* protocol the labels are derived from each host's **own SAE features**: encode
the activations, keep the features whose token/residue prevalence sits in a band
`[min_prev, max_prev]` (drops always-off and always-on), binarise (`active = latent > 0`) → the
label matrix `Y`. The SAE encoder restricted to those features is the downstream task encoder. This
is *self-referential* (distill target ⊇ scored features) — a real caveat, stated in the verdict —
but it is *the same* self-reference on both hosts, so the trained-vs-pinv **delta** remains a valid
host-class comparison. (SAE-type is a secondary confound: GPT-2's SAE is ReLU/L1, ESM-2's is TopK.)

## Outcomes (descriptive, both first-class — no "irreducible"/"closes the tax" language)

  * **Host-class signal** — causal `delta_heldout` clears noise positive while non-causal sits at
    ≈0/negative → the ESM null was host-class-specific; trained-`E` helps on causal hosts.
  * **Generalises** — causal `delta_heldout` is also ≈0/negative → the projection is near-optimal
    past host class; the trained-encoder thesis does not transfer to the forge even on causal hosts.

Usage:
  python scripts/causal_lm_forge_gate.py --host gpt2 --widths 64,128,256,512 --seeds 0,1,2
  python scripts/causal_lm_forge_gate.py --host both --widths 64,128,256 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# A small, diverse, deterministic English corpus appended to sae-forge's built-in calibration text.
# Purpose: more (and more lexically varied) GPT-2 token-rows so held-out AUC is stable. Factual /
# narrative sentences spanning many semantic domains → many distinct SAE features fire. Offline:
# neither wikipedia nor lambada is actually downloaded in this environment (README only).
_EXTRA_CORPUS = """\
The harbor froze solid in January, and the fishing boats sat trapped in the grey ice until spring.
Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen inside the chloroplast.
She tightened the final bolt, wiped the grease from her hands, and the old engine coughed back to life.
The treaty was signed in 1648, ending three decades of war that had emptied the villages of central Europe.
Prime numbers grow sparser as you count higher, yet they never run out, a fact proved by Euclid long ago.
The chef folded chopped basil into the warm tomato sauce and let it simmer while the pasta finished cooking.
Migrating geese navigate by the stars, the setting sun, and the faint magnetic field of the planet itself.
He inherited a small vineyard on a steep southern slope where the soil was thin but the grapes ripened sweet.
The orchestra fell silent, the conductor raised her baton, and the first violins began the slow opening theme.
Erosion carved the canyon over millions of years as the river ground patiently through layer after layer of stone.
The startup ran out of money in the spring, but two engineers kept building the software in their spare evenings.
A single honeybee may visit several thousand flowers in one day, carrying pollen from blossom to blossom.
The judge listened to both lawyers, reread the contract twice, and adjourned the court until the following morning.
Cold mountain streams carry oxygen that trout need, which is why the fish vanish when the water grows too warm.
The library kept the oldest manuscripts in a cool vault, away from sunlight, humidity, and curious fingers.
Investors panicked when the central bank raised interest rates, and stock prices fell sharply across every sector.
The children built a crooked sandcastle near the tide line and watched the waves erase it before sunset.
Antibiotics kill bacteria but do nothing against viruses, so doctors warn against using them for a common cold.
The old railway line, abandoned for decades, was slowly being reclaimed by birch saplings and tangled brambles.
He translated the poem three different ways before deciding none of them captured the music of the original.
Volcanic ash drifted across the continent, dimming the sun and chilling the summer harvest in distant farmlands.
The marathon runner hit the wall at mile twenty, but the roar of the crowd carried her the final stretch home.
Glaciers store most of the planet's fresh water, locked in ice that took thousands of winters to accumulate.
The negotiations dragged on past midnight until both sides, exhausted, agreed to a compromise neither one liked.
A spider rebuilds its web each morning, eating the old silk to recover the protein it spent the day before.
The painter mixed a little black into the blue to capture the heavy shadow under the approaching thunderstorm.
Coral reefs shelter a quarter of all marine species, yet they bleach and die when the ocean grows even slightly hotter.
The accountant found the error on the third page, a single transposed digit that had cost the firm a fortune.
Wolves returning to the valley changed the rivers, because the deer no longer grazed the young willows bare.
She memorized the periodic table in a week, reciting the elements aloud on the long bus ride to school.
The bridge swayed gently in the wind, a deliberate design that let it bend instead of snapping under the strain.
Ancient sailors feared the open ocean at night, steering by coastlines and praying for the morning star to rise.
The bakery opened before dawn, and the smell of warm bread drifted down the empty street to the train station.
Quantum particles can be linked so that measuring one instantly fixes the state of the other, however far apart.
The committee rejected the proposal twice, then approved a nearly identical version after the budget was renamed.
Desert plants store water in thick waxy leaves and open their pores only at night to lose as little as possible.
"""


@dataclass
class HostBundle:
    """Everything the host-agnostic activation-level gate needs for one host."""

    name: str
    X: np.ndarray  # (N, d_model) host activations (one row per token/residue item)
    sae_encoder: Callable[..., Any]  # torch (M, d_model) -> (M, V) downstream-task latents
    W_dec_full: np.ndarray  # (n_features, d_model) full SAE decoder
    labels: np.ndarray  # (N, V) binary labels derived from the SAE's own features
    meta: dict


# --------------------------------------------------------------------------- labels


def _derive_labels_from_sae(
    Z_full: np.ndarray, *, min_prev: float, max_prev: float, max_labels: int
) -> np.ndarray:
    """Pick SAE features whose activation prevalence sits in ``[min_prev, max_prev]`` (drops
    always-off / always-on), most-prevalent first, capped at ``max_labels``. Returns the feature
    index array. Binarisation (``active = latent > 0``) is applied by the caller to ``Z_full``."""
    prev = (Z_full > 0).mean(axis=0)  # (n_features,)
    band = np.flatnonzero((prev >= min_prev) & (prev <= max_prev))
    if band.size == 0:
        raise ValueError(
            f"no SAE features with prevalence in [{min_prev}, {max_prev}]; widen the band"
        )
    band = band[np.argsort(-prev[band])][:max_labels]
    return np.sort(band)


# --------------------------------------------------------------------------- GPT-2


def build_gpt2_bundle(
    *, layer: int, n_tokens: int, ctx: int, min_prev: float, max_prev: float, max_labels: int,
    device: str,
) -> HostBundle:
    """CAUSAL host: GPT-2 layer-``layer`` residual stream + the jbloom GPT-2-Small SAE.

    Activations are the per-token residual at ``blocks.{layer}.hook_resid_pre`` (= HF
    ``hidden_states[layer]``), gathered over the corpus in ``ctx``-token windows up to ``n_tokens``
    rows. The SAE is loaded in SAELens convention (``W_enc``/``W_dec``/``b_enc``/``b_dec``); its
    encoder is ``relu((x - b_dec) @ W_enc + b_enc)`` restricted to the label features.
    """
    import glob
    import os

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from saeforge.calibration import _BUILTIN_CALIBRATION_TEXT

    sae_glob = os.path.expanduser(
        "~/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/"
        f"snapshots/*/blocks.{layer}.hook_resid_pre"
    )
    hits = glob.glob(sae_glob)
    if not hits:
        raise FileNotFoundError(
            f"jbloom GPT-2 SAE for layer {layer} not cached (looked under {sae_glob}). "
            "Fetch models--jbloom--GPT2-Small-SAEs-Reformatted first."
        )
    sd = load_file(os.path.join(hits[0], "sae_weights.safetensors"))
    W_enc = sd["W_enc"].float()       # (d_model, n_features)
    W_dec = sd["W_dec"].float()       # (n_features, d_model)
    b_enc = sd["b_enc"].float()       # (n_features,)
    b_dec = sd["b_dec"].float()       # (d_model,)

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(device).eval()

    text = _BUILTIN_CALIBRATION_TEXT + "\n\n" + _EXTRA_CORPUS
    ids = tok(text, return_tensors="pt").input_ids[0]
    windows = [ids[i:i + ctx] for i in range(0, ids.shape[0], ctx)]
    chunks: list = []
    with torch.no_grad():
        for w in windows:
            out = model(w.unsqueeze(0).to(device), output_hidden_states=True)
            chunks.append(out.hidden_states[layer][0].cpu().float())  # (Lw, d_model)
            if sum(c.shape[0] for c in chunks) >= n_tokens:
                break
    X = torch.cat(chunks, dim=0)[:n_tokens]  # (N, d_model)

    # Full latents (once, no grad) to choose label features by prevalence.
    with torch.no_grad():
        Z_full = torch.relu((X - b_dec) @ W_enc + b_enc).numpy()
    feat_idx = _derive_labels_from_sae(
        Z_full, min_prev=min_prev, max_prev=max_prev, max_labels=max_labels
    )
    labels = (Z_full[:, feat_idx] > 0).astype(np.float64)

    # Downstream-task encoder = SAE restricted to the label features (cheap matmul; ReLU is
    # elementwise so column-slicing W_enc/b_enc == slicing the full relu output).
    We = W_enc[:, feat_idx].contiguous()
    be = b_enc[feat_idx].contiguous()
    bd = b_dec.contiguous()

    def sae_encoder(x):
        return torch.relu((x - bd) @ We + be)

    return HostBundle(
        name="gpt2",
        X=X.numpy().astype(np.float64),
        sae_encoder=sae_encoder,
        W_dec_full=W_dec.numpy().astype(np.float64),
        labels=labels,
        meta={"host": "gpt2", "host_class": "causal", "sae_type": "relu_l1",
              "layer": layer, "n_items": int(X.shape[0]), "n_features": int(W_dec.shape[0]),
              "n_labels": int(feat_idx.size), "d_model": int(W_dec.shape[1])},
    )


# --------------------------------------------------------------------------- ESM-2 control


def build_esm2_bundle(
    *, bio_root: Path, run_dir: str, sequences: str, n_proteins: int, sae_k: int,
    min_prev: float, max_prev: float, max_labels: int, device: str,
) -> HostBundle:
    """NON-CAUSAL control: ESM-2 per-residue activations + bio-sae's SAE, SAME label protocol as
    GPT-2 (labels derived from the SAE's own features, not bio ground truth) so the only headline
    difference between the two bundles is host class."""
    import pandas as pd
    import torch

    from saeforge.datasets.capability import _build_topk_encoder
    from saeforge.sweep_capability import _extract_host_activations

    rd = bio_root / run_dir
    state = torch.load(rd / "sae.pt", map_location="cpu", weights_only=True)
    enc_weight = state["encoder.weight"]  # (n_features, d_model)
    enc_bias = state["encoder.bias"]
    W_dec_full = state["decoder.weight"].numpy().T.astype(np.float64)  # (n_features, d_model)
    full_encoder = _build_topk_encoder(enc_weight, enc_bias, variant="topk", k=sae_k)

    seqs_df = pd.read_parquet(bio_root / sequences)
    seqs = [s[:512] for s in seqs_df["sequence"].head(n_proteins)]
    X_t = _extract_host_activations(
        host_model_id="facebook/esm2_t6_8M_UR50D", sequences=seqs,
        aggregator="pool_then_encode", max_seq_len=512, device=device, feed="residue",
    )  # (N_res, d_model)
    X = X_t.numpy().astype(np.float64)

    with torch.no_grad():
        Z_full = full_encoder(torch.tensor(X, dtype=torch.float32)).numpy()
    feat_idx = _derive_labels_from_sae(
        Z_full, min_prev=min_prev, max_prev=max_prev, max_labels=max_labels
    )
    labels = (Z_full[:, feat_idx] > 0).astype(np.float64)

    def sae_encoder(x):
        return full_encoder(x)[:, feat_idx]  # TopK over full width, then select label columns

    return HostBundle(
        name="esm2",
        X=X,
        sae_encoder=sae_encoder,
        W_dec_full=W_dec_full,
        labels=labels,
        meta={"host": "esm2", "host_class": "non_causal", "sae_type": "topk",
              "n_items": int(X.shape[0]), "n_features": int(W_dec_full.shape[0]),
              "n_labels": int(feat_idx.size), "d_model": int(W_dec_full.shape[1])},
    )


# --------------------------------------------------------------------------- the gate


def cap_items(bundle: HostBundle, max_items: int, *, seed: int = 0) -> HostBundle:
    """Subsample a bundle to ``max_items`` rows (deterministic) so causal vs non-causal run at a
    matched N — a cleaner comparison and a bound on the per-cell AUC cost. No-op if N <= max_items."""
    n = bundle.X.shape[0]
    if max_items <= 0 or n <= max_items:
        return bundle
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_items, replace=False))
    meta = dict(bundle.meta)
    meta["n_items"] = int(max_items)
    meta["n_items_full"] = int(n)
    return HostBundle(
        name=bundle.name, X=bundle.X[idx], sae_encoder=bundle.sae_encoder,
        W_dec_full=bundle.W_dec_full, labels=bundle.labels[idx], meta=meta,
    )


def run_gate(
    bundle: HostBundle, *, widths: list[int], seeds: list[int], steps: int, lr: float,
) -> list[dict]:
    """Activation-level trained-vs-pinv gate per (width, seed) via train_encoder(objective='distill').
    Returns one summary dict per width: mean ± std of delta_heldout (trained − pinv) over seeds."""
    from saeforge.basis import FeatureBasis
    from saeforge.training import train_encoder

    W = bundle.W_dec_full
    row_norms = np.linalg.norm(W, axis=1)
    order = np.argsort(-row_norms)  # top-N atoms by L2 norm (the sweep's row_norm slicing)

    out: list[dict] = []
    print(f"\n== {bundle.name} ({bundle.meta['host_class']}, sae={bundle.meta['sae_type']}, "
          f"N={bundle.meta['n_items']} items, {bundle.meta['n_labels']} labels, "
          f"d_model={bundle.meta['d_model']}) ==")
    for w in widths:
        if w > W.shape[0]:
            print(f"   n={w:>4}: SKIP (exceeds SAE width {W.shape[0]})")
            continue
        kept = np.sort(order[:w])
        basis = FeatureBasis(
            kept_ids=kept.astype(np.int64), W_dec=W[kept],
            merged_norms=row_norms[kept].astype(np.float64),
            original_norms=row_norms[kept].astype(np.float64),
        )
        recs = []
        for seed in seeds:
            _, report = train_encoder(
                basis=basis, host_acts=bundle.X, host_encoder=bundle.sae_encoder,
                labels=bundle.labels, objective="distill", loss="cosine",
                steps=steps, lr=lr, seed=seed,
            )
            recs.append((report.delta_heldout, report.retained_mauc_pinv_baseline,
                         report.retained_mauc_trained, report.overfit_flag, report.steps_run))
        deltas = np.array([r[0] for r in recs], float)
        mean_d, std_d = float(deltas.mean()), float(deltas.std())
        gate = "PASS" if mean_d > 1e-4 else ("TIE" if mean_d >= -1e-4 else "FAIL")
        pinv_m = float(np.mean([r[1] for r in recs]))
        tr_m = float(np.mean([r[2] for r in recs]))
        any_of = bool(any(r[3] for r in recs))
        print(f"   n={w:>4}: held-out retained-mAUC  pinv {pinv_m:.4f} -> trained {tr_m:.4f}   "
              f"Δ mean {mean_d:+.4f} ± {std_d:.4f} (n_seed={len(recs)})  "
              f"[{gate}]{'  overfit' if any_of else ''}")
        out.append({
            "host": bundle.name, "host_class": bundle.meta["host_class"],
            "sae_type": bundle.meta["sae_type"], "width": w, "n_seeds": len(recs),
            "delta_mean": mean_d, "delta_std": std_d, "pinv_mean": pinv_m,
            "trained_mean": tr_m, "any_overfit": any_of, "gate": gate,
            "n_items": bundle.meta["n_items"], "n_labels": bundle.meta["n_labels"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", choices=["gpt2", "esm2", "both"], default="gpt2")
    ap.add_argument("--widths", default="64,128,256,512", help="comma-separated compressed widths N")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated train seeds (multi-seed gate)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("/tmp/causal_lm_forge_gate"))
    # GPT-2 knobs
    ap.add_argument("--gpt2-layer", type=int, default=8)
    ap.add_argument("--gpt2-n-tokens", type=int, default=1400)
    ap.add_argument("--gpt2-ctx", type=int, default=128)
    # ESM-2 control knobs
    ap.add_argument("--bio-root", type=Path, default=Path("/home/allans/code/bio-sae"))
    ap.add_argument("--esm-run-dir", default="runs/uniref50_n5000/pooled_w1024_k64")
    ap.add_argument("--esm-sequences", default="data/uniref50_sample__n5000_seed0.parquet")
    ap.add_argument("--esm-n-proteins", type=int, default=40)
    ap.add_argument("--esm-sae-k", type=int, default=64)
    # shared label-band knobs
    ap.add_argument("--min-prev", type=float, default=0.04)
    ap.add_argument("--max-prev", type=float, default=0.40)
    ap.add_argument("--max-labels", type=int, default=400)
    ap.add_argument("--max-items", type=int, default=1400,
                    help="cap rows per host so causal vs non-causal run at matched N (0 = no cap)")
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    print("== causal-LM forge gate (activation-level trained-vs-pinv; held-out, "
          f"compression-controlled; steps={args.steps}, seeds={seeds}) ==")

    results: list[dict] = []
    if args.host in ("gpt2", "both"):
        gpt2 = build_gpt2_bundle(
            layer=args.gpt2_layer, n_tokens=args.gpt2_n_tokens, ctx=args.gpt2_ctx,
            min_prev=args.min_prev, max_prev=args.max_prev, max_labels=args.max_labels,
            device=args.device,
        )
        results += run_gate(cap_items(gpt2, args.max_items), widths=widths, seeds=seeds,
                            steps=args.steps, lr=args.lr)
    if args.host in ("esm2", "both"):
        esm2 = build_esm2_bundle(
            bio_root=args.bio_root, run_dir=args.esm_run_dir, sequences=args.esm_sequences,
            n_proteins=args.esm_n_proteins, sae_k=args.esm_sae_k, min_prev=args.min_prev,
            max_prev=args.max_prev, max_labels=args.max_labels, device=args.device,
        )
        results += run_gate(cap_items(esm2, args.max_items), widths=widths, seeds=seeds,
                            steps=args.steps, lr=args.lr)

    json.dump(results, open(args.out / "causal_gate_summary.json", "w"), indent=2)
    print("\n== verdict (descriptive; trained − pinv held-out Δ) ==")
    for r in results:
        print(f"   {r['host']:<5} ({r['host_class']:<10}) n={r['width']:>4}  "
              f"Δ {r['delta_mean']:+.4f} ± {r['delta_std']:.4f}  [{r['gate']}]"
              f"{'  overfit' if r['any_overfit'] else ''}")
    print(f"   wrote {args.out / 'causal_gate_summary.json'}")


if __name__ == "__main__":
    main()
