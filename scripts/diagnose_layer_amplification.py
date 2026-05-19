"""Per-layer divergence diagnostic for the GPT-2 L8 KL blow-up.

Loads the four forged NativeModels from the 2026-05-16 smoke run
(scale_boost=1.0, K ∈ {25, 103, 163, 211}) alongside the host GPT-2,
runs both on a fixed calibration batch with per-layer hooks, and
writes a JSON + CSV trajectory of:

    forged_norm           mean ||forged_residual||_F across positions
    host_norm             mean ||host_residual||_F across positions
    norm_ratio            forged_norm / host_norm  (≈1 iff magnitudes match)
    decoded_rel_err       ||decode(forged) - host|| / ||host||  (host coords)
    decoded_cosine        mean cosine(decode(forged), host) per position
    via_host_kl           KL(p_host || p_forged_decoded) using host's
                          ln_f + lm_head applied to BOTH streams

Layer index 0 = post-(wte+wpe) embedding output, 1..N = post-block-i.
Layer N+1 = post-ln_f (i.e. the input to lm_head).

Output: reports/layer_amplification/{K}.json + summary.csv.

Run:
    .venv/bin/python scripts/diagnose_layer_amplification.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saeforge.basis import FeatureBasis
from saeforge.calibration import _BUILTIN_CALIBRATION_TEXT
from saeforge.model import NativeModel

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "smoke_fix_scale_boost"
OUT = ROOT / "reports" / "layer_amplification"

K_VALUES = [25, 103, 163, 211]
HOST_ID = "gpt2"
N_TOKENS = 256  # one forward pass; small enough to keep this <60s on Intel CPU


def collect_host_residuals(host_model, input_ids):
    """Return list of (n_layer+1) tensors (B, T, 768): hidden_states[0..N]."""
    with torch.no_grad():
        out = host_model(input_ids, output_hidden_states=True)
    # Index 0 = post-embedding, 1..N = post-block-i. HF returns this BEFORE
    # final ln_f. We need the post-ln_f state too — compute it ourselves
    # from the last hidden state.
    states = list(out.hidden_states)
    post_ln_f = host_model.transformer.ln_f(states[-1])
    states.append(post_ln_f)
    return states  # length N+2


def collect_forged_residuals(forged_module, input_ids):
    """Hook the forged transformer at the same points as host hidden_states.

    Returns list of (B, T, n_features) tensors matching host indexing:
        [0]                  post-(wte+wpe)
        [1..N]               post-block-i
        [N+1]                post-ln_f
    """
    captured = {}
    handles = []
    transformer = forged_module.transformer

    # Post-(wte+wpe). The forged Transformer.forward computes
    # x = wte(input_ids) + wpe(pos) THEN runs blocks. Hook on the first
    # block's pre-input via a forward_pre_hook on transformer.h[0].
    def pre_block0_hook(module, inputs):
        captured["embed"] = inputs[0].detach().clone()

    handles.append(transformer.h[0].register_forward_pre_hook(pre_block0_hook))

    # Post-block-i, for every block.
    def make_block_hook(i):
        def hook(module, inputs, output):
            captured[f"block_{i}"] = output.detach().clone()
        return hook

    for i, block in enumerate(transformer.h):
        handles.append(block.register_forward_hook(make_block_hook(i)))

    # Post-ln_f (the Transformer.forward returns ln_f(x), so capture the
    # transformer module's overall output).
    def ln_f_hook(module, inputs, output):
        captured["ln_f"] = output.detach().clone()

    handles.append(transformer.register_forward_hook(ln_f_hook))

    try:
        with torch.no_grad():
            _ = forged_module(input_ids)
    finally:
        for h in handles:
            h.remove()

    n_layers = len(transformer.h)
    out = [captured["embed"]]
    for i in range(n_layers):
        out.append(captured[f"block_{i}"])
    out.append(captured["ln_f"])
    return out  # length N+2


def per_layer_metrics(
    host_states, forged_states, W_dec, host_ln_f, host_lm_head_W
):
    """Compute per-layer divergence metrics.

    host_states: list of (B, T, d_model=768) tensors, length L
    forged_states: list of (B, T, n_features) tensors, length L
    W_dec: (n_features, d_model) numpy array — basis decode matrix
    host_ln_f: nn.LayerNorm from host (for fair-comparison logit projection)
    host_lm_head_W: (vocab, d_model) tensor

    Returns list of dicts, one per layer.
    """
    assert len(host_states) == len(forged_states), (
        len(host_states), len(forged_states),
    )
    W_dec_t = torch.from_numpy(W_dec.astype(np.float32))
    rows = []
    for layer_idx, (h, f) in enumerate(zip(host_states, forged_states)):
        h32 = h.float()
        f32 = f.float()

        # Frobenius norms averaged over batch/position.
        host_norm = h32.norm(dim=-1).mean().item()
        forged_norm = f32.norm(dim=-1).mean().item()

        # Decode forged residual back into host coordinates.
        f_decoded = f32 @ W_dec_t  # (B, T, d_model)

        diff = f_decoded - h32
        rel_err = (diff.norm(dim=-1) / h32.norm(dim=-1).clamp_min(1e-8)).mean().item()
        cos = F.cosine_similarity(f_decoded, h32, dim=-1).mean().item()

        # Per-layer KL via host's final ln_f + lm_head. Both streams pass
        # through the same final-norm + unembed so the comparison is
        # apples-to-apples — what would the next-token distribution look
        # like if THIS layer's residual was the final residual?
        with torch.no_grad():
            host_logits = host_ln_f(h32) @ host_lm_head_W.float().T
            forged_logits = host_ln_f(f_decoded) @ host_lm_head_W.float().T
        # KL(host || forged) — what the host distribution loses by being
        # replaced with the forged one.
        log_p_host = F.log_softmax(host_logits, dim=-1)
        log_p_forged = F.log_softmax(forged_logits, dim=-1)
        kl_per_pos = F.kl_div(
            log_p_forged, log_p_host, reduction="none", log_target=True
        ).sum(dim=-1)
        kl = kl_per_pos.mean().item()

        rows.append(
            {
                "layer": layer_idx,
                "host_norm": host_norm,
                "forged_norm": forged_norm,
                "norm_ratio": forged_norm / max(host_norm, 1e-8),
                "decoded_rel_err": rel_err,
                "decoded_cosine": cos,
                "via_host_kl": kl,
            }
        )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(HOST_ID)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        _BUILTIN_CALIBRATION_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=N_TOKENS,
    )
    input_ids = enc.input_ids

    print(f"[load] host {HOST_ID} ({input_ids.shape[1]} tokens)")
    host_model = transformers.AutoModelForCausalLM.from_pretrained(
        HOST_ID, torch_dtype=torch.float32
    ).eval()
    host_lm_head_W = host_model.lm_head.weight.detach()
    host_ln_f = host_model.transformer.ln_f

    print(f"[load] host hidden_states + post-ln_f")
    host_states = collect_host_residuals(host_model, input_ids)
    print(f"       captured {len(host_states)} states; last shape={tuple(host_states[-1].shape)}")

    summary_rows = []
    for K in K_VALUES:
        sae_path = SMOKE / "baseline" / "_materialised" / "hea" / "pareto" / f"k_{K}.safetensors"
        forged_dir = SMOKE / "baseline" / "hea" / f"k_{K}" / "forged"
        if not sae_path.exists():
            print(f"[skip] K={K}: missing {sae_path}")
            continue
        if not forged_dir.exists():
            print(f"[skip] K={K}: missing {forged_dir}")
            continue

        print(f"[K={K}] loading basis + forged model")
        basis = FeatureBasis.from_polygram_checkpoint(sae_path)
        forged = NativeModel.load_pretrained(forged_dir)
        forged.torch_module.eval()
        print(f"       n_features={basis.n_features} d_model={basis.d_model}")

        forged_states = collect_forged_residuals(forged.torch_module, input_ids)
        assert len(forged_states) == len(host_states), (
            f"length mismatch: forged={len(forged_states)} host={len(host_states)}"
        )

        rows = per_layer_metrics(
            host_states, forged_states, basis.W_dec, host_ln_f, host_lm_head_W
        )
        for r in rows:
            r["K"] = K
            r["n_features"] = basis.n_features
        summary_rows.extend(rows)

        # Per-K JSON.
        out_json = OUT / f"k_{K}.json"
        out_json.write_text(json.dumps({"K": K, "n_features": basis.n_features, "layers": rows}, indent=2))
        print(f"       wrote {out_json}")

        # Print compact per-K table.
        print(f"\n  layer | host_norm  forged_norm  norm_ratio  rel_err  cosine    KL_via_host")
        print(f"  ------+------------------------------------------------------------------")
        for r in rows:
            print(
                f"  {r['layer']:>5} | {r['host_norm']:>9.3f}  {r['forged_norm']:>10.3f}  "
                f"{r['norm_ratio']:>9.3f}  {r['decoded_rel_err']:>6.3f}  {r['decoded_cosine']:>+5.3f}  "
                f"{r['via_host_kl']:>10.3f}"
            )
        print()

    # Combined CSV.
    csv_path = OUT / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        fieldnames = [
            "K", "n_features", "layer",
            "host_norm", "forged_norm", "norm_ratio",
            "decoded_rel_err", "decoded_cosine", "via_host_kl",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[write] {csv_path}")


if __name__ == "__main__":
    main()
