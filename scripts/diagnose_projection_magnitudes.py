"""Inspect projected LayerNorm / linear weight magnitudes vs host originals.

Follow-up to diagnose_layer_amplification.py. The per-layer trajectory showed
block 0 amplifies forged residual norms by up to 270x at K=211. Hypothesis:
the projected ``ln_1.weight``/``bias`` (and ``c_attn.weight``, ``c_proj.weight``)
have explosive magnitudes when the basis pseudoinverse is poorly conditioned.

For each K and each block parameter, this prints:
    host_norm        ||W_host||_F
    forged_norm      ||W_forged||_F  (after projection)
    ratio            forged_norm / host_norm
    cond(W_dec)      pseudoinverse conditioning of the basis

Run:
    .venv/bin/python scripts/diagnose_projection_magnitudes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import transformers

from saeforge.basis import FeatureBasis
from saeforge.model import NativeModel

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "smoke_fix_scale_boost"
K_VALUES = [25, 103, 163, 211]
HOST_ID = "gpt2"


def main():
    host_model = transformers.AutoModelForCausalLM.from_pretrained(
        HOST_ID, torch_dtype=torch.float32
    ).eval()
    host_t = host_model.transformer

    # Host parameter Frobenius norms at block 0.
    host_norms = {
        "ln_1.weight": host_t.h[0].ln_1.weight.detach().norm().item(),
        "ln_1.bias": host_t.h[0].ln_1.bias.detach().norm().item(),
        "attn.c_attn.weight": host_t.h[0].attn.c_attn.weight.detach().norm().item(),
        "attn.c_attn.bias": host_t.h[0].attn.c_attn.bias.detach().norm().item(),
        "attn.c_proj.weight": host_t.h[0].attn.c_proj.weight.detach().norm().item(),
        "attn.c_proj.bias": host_t.h[0].attn.c_proj.bias.detach().norm().item(),
        "ln_2.weight": host_t.h[0].ln_2.weight.detach().norm().item(),
        "ln_2.bias": host_t.h[0].ln_2.bias.detach().norm().item(),
        "mlp.c_fc.weight": host_t.h[0].mlp.c_fc.weight.detach().norm().item(),
        "mlp.c_proj.weight": host_t.h[0].mlp.c_proj.weight.detach().norm().item(),
        "mlp.c_proj.bias": host_t.h[0].mlp.c_proj.bias.detach().norm().item(),
    }

    print(f"\nHost gpt2 block 0 parameter norms (||·||_F):")
    for k, v in host_norms.items():
        print(f"  {k:<24} {v:>10.4f}")
    print()

    print(f"{'param':<24} {'K':>4} {'n_feat':>7} {'forged_norm':>12} {'ratio':>8} {'pinv_max':>10}")
    print("-" * 76)

    for K in K_VALUES:
        sae_path = SMOKE / "baseline" / "_materialised" / "hea" / "pareto" / f"k_{K}.safetensors"
        forged_dir = SMOKE / "baseline" / "hea" / f"k_{K}" / "forged"
        if not sae_path.exists() or not forged_dir.exists():
            continue

        basis = FeatureBasis.from_polygram_checkpoint(sae_path)
        pinv = basis.pseudoinverse()
        # Conditioning of W_dec via SVD: max-singular-value of pinv = 1/min-singular-value of W_dec
        # already in the basis cache via pinv. Just report ||pinv|| (op norm).
        pinv_max_sv = float(np.linalg.norm(pinv, ord=2))

        forged = NativeModel.load_pretrained(forged_dir)
        fm = forged.torch_module.transformer.h[0]

        forged_norms = {
            "ln_1.weight": fm.ln_1.weight.detach().norm().item(),
            "ln_1.bias": fm.ln_1.bias.detach().norm().item(),
            "attn.c_attn.weight": fm.attn.c_attn.weight.detach().norm().item(),
            "attn.c_attn.bias": fm.attn.c_attn.bias.detach().norm().item(),
            "attn.c_proj.weight": fm.attn.c_proj.weight.detach().norm().item(),
            "attn.c_proj.bias": fm.attn.c_proj.bias.detach().norm().item(),
            "ln_2.weight": fm.ln_2.weight.detach().norm().item(),
            "ln_2.bias": fm.ln_2.bias.detach().norm().item(),
            "mlp.c_fc.weight": fm.mlp.c_fc.weight.detach().norm().item(),
            "mlp.c_proj.weight": fm.mlp.c_proj.weight.detach().norm().item(),
            "mlp.c_proj.bias": fm.mlp.c_proj.bias.detach().norm().item(),
        }
        for k, host_v in host_norms.items():
            fv = forged_norms[k]
            ratio = fv / max(host_v, 1e-10)
            print(
                f"{k:<24} {K:>4} {basis.n_features:>7} {fv:>12.4f} {ratio:>8.3f} {pinv_max_sv:>10.3f}"
            )
        print()


if __name__ == "__main__":
    main()
