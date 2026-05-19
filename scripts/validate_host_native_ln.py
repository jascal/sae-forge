"""Validate the host-native-LayerNorm fix experimentally.

For each K, monkey-patch the forged NativeModel's LayerNorms with a
DecodedLayerNorm that performs decode -> host_LN -> encode, using the
ORIGINAL host LN parameters (gamma, beta) rather than the projected
versions. Then re-run the per-layer divergence diagnostic and compare
forge-output KL against the baseline.

The host LN parameters for each block are loaded directly from the host
gpt2 model. The encode/decode matrices come from the per-K
FeatureBasis. scale_boost is hard-coded to 1.0 to match the smoke
baseline.

Expected outcome: per-layer norm_ratio drops to ~1.0, decoded_rel_err
drops, and final via_host_kl becomes monotone in kept-features K.

Run:
    .venv/bin/python scripts/validate_host_native_ln.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from saeforge.basis import FeatureBasis
from saeforge.calibration import _BUILTIN_CALIBRATION_TEXT
from saeforge.model import NativeModel

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "smoke_fix_scale_boost"
K_VALUES = [25, 103, 163, 211]
HOST_ID = "gpt2"
N_TOKENS = 256


class DecodedLayerNorm(nn.Module):
    """LayerNorm that runs in host space: decode -> LN_host -> encode.

    The forged residual ``z`` lives in basis space (n_features). To apply
    LayerNorm faithfully, decode to host space, apply host's LN (with
    host's original gamma/beta), re-encode.

    pinv and W_dec are stored as buffers so the module is portable.
    host_gamma/host_beta are nn.Parameters but initialised from the host
    model and not trained here.
    """

    def __init__(self, W_dec_np: np.ndarray, pinv_np: np.ndarray,
                 host_gamma: torch.Tensor, host_beta: torch.Tensor,
                 eps: float):
        super().__init__()
        self.eps = eps
        # W_dec: (n_features, d_model). pinv: (d_model, n_features).
        self.register_buffer("W_dec", torch.from_numpy(W_dec_np.astype(np.float32)))
        self.register_buffer("pinv", torch.from_numpy(pinv_np.astype(np.float32)))
        self.register_buffer("host_gamma", host_gamma.detach().float().clone())
        self.register_buffer("host_beta", host_beta.detach().float().clone())

    def forward(self, z):
        # z: (..., n_features). Decode to host space.
        x = z @ self.W_dec  # (..., d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_n = (x - mean) / torch.sqrt(var + self.eps)
        y = x_n * self.host_gamma + self.host_beta
        # Re-encode.
        return y @ self.pinv  # (..., n_features)


def patch_forged_layernorms(forged_module, host_model, basis):
    """Replace ln_1, ln_2, ln_f with DecodedLayerNorm using host params."""
    W_dec_np = basis.W_dec
    pinv_np = basis.pseudoinverse()
    host_t = host_model.transformer
    eps = forged_module.config.layer_norm_epsilon

    transformer = forged_module.transformer
    for i, block in enumerate(transformer.h):
        host_block = host_t.h[i]
        block.ln_1 = DecodedLayerNorm(
            W_dec_np, pinv_np,
            host_block.ln_1.weight, host_block.ln_1.bias, eps,
        )
        block.ln_2 = DecodedLayerNorm(
            W_dec_np, pinv_np,
            host_block.ln_2.weight, host_block.ln_2.bias, eps,
        )
    transformer.ln_f = DecodedLayerNorm(
        W_dec_np, pinv_np,
        host_t.ln_f.weight, host_t.ln_f.bias, eps,
    )
    return forged_module


def compute_forge_kl(host_model, forged_module, input_ids):
    """Compute forge-output KL: KL(host_softmax || forged_softmax) on input_ids.

    Mirrors saeforge's ``evaluate_faithfulness`` for LM hosts: per-token
    KL averaged over positions.
    """
    with torch.no_grad():
        host_logits = host_model(input_ids).logits
        forged_logits = forged_module(input_ids)
    log_p_host = F.log_softmax(host_logits, dim=-1)
    log_p_forged = F.log_softmax(forged_logits, dim=-1)
    kl_per_pos = F.kl_div(
        log_p_forged, log_p_host, reduction="none", log_target=True
    ).sum(dim=-1)
    return kl_per_pos.mean().item()


def per_layer_quick(forged_module, host_states, input_ids, W_dec_np):
    """Quick per-layer norm + decoded_rel_err comparison (no KL)."""
    W_dec_t = torch.from_numpy(W_dec_np.astype(np.float32))
    captured = {}
    handles = []
    transformer = forged_module.transformer

    def pre_block0_hook(module, inputs):
        captured["embed"] = inputs[0].detach().clone()

    handles.append(transformer.h[0].register_forward_pre_hook(pre_block0_hook))

    def make_hook(i):
        def hook(module, inputs, output):
            captured[f"block_{i}"] = output.detach().clone()
        return hook

    for i, block in enumerate(transformer.h):
        handles.append(block.register_forward_hook(make_hook(i)))

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
    forged_states = [captured["embed"]]
    for i in range(n_layers):
        forged_states.append(captured[f"block_{i}"])
    forged_states.append(captured["ln_f"])

    rows = []
    for i, (h, f) in enumerate(zip(host_states, forged_states)):
        h32 = h.float()
        f32 = f.float()
        host_norm = h32.norm(dim=-1).mean().item()
        forged_norm = f32.norm(dim=-1).mean().item()
        f_decoded = f32 @ W_dec_t
        rel_err = (
            (f_decoded - h32).norm(dim=-1)
            / h32.norm(dim=-1).clamp_min(1e-8)
        ).mean().item()
        rows.append({
            "layer": i,
            "host_norm": host_norm,
            "forged_norm": forged_norm,
            "norm_ratio": forged_norm / max(host_norm, 1e-8),
            "decoded_rel_err": rel_err,
        })
    return rows


def main():
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

    with torch.no_grad():
        host_out = host_model(input_ids, output_hidden_states=True)
    host_states = list(host_out.hidden_states)
    host_states.append(host_model.transformer.ln_f(host_states[-1]))

    print(f"\n{'K':>5} {'n_feat':>7} {'KL_baseline':>12} {'KL_fixed':>10} "
          f"{'L1_ratio_base':>14} {'L1_ratio_fix':>13}")
    print("-" * 76)

    for K in K_VALUES:
        sae_path = SMOKE / "baseline" / "_materialised" / "hea" / "pareto" / f"k_{K}.safetensors"
        forged_dir = SMOKE / "baseline" / "hea" / f"k_{K}" / "forged"
        if not sae_path.exists() or not forged_dir.exists():
            continue

        basis = FeatureBasis.from_polygram_checkpoint(sae_path)

        # Baseline arm.
        forged_base = NativeModel.load_pretrained(forged_dir)
        forged_base.torch_module.eval()
        kl_base = compute_forge_kl(host_model, forged_base.torch_module, input_ids)
        rows_base = per_layer_quick(
            forged_base.torch_module, host_states, input_ids, basis.W_dec
        )
        l1_ratio_base = rows_base[1]["norm_ratio"]  # post-block-0

        # Patched arm — fresh load (we mutated the module above).
        forged_fix = NativeModel.load_pretrained(forged_dir)
        forged_fix.torch_module.eval()
        patch_forged_layernorms(forged_fix.torch_module, host_model, basis)
        forged_fix.torch_module.eval()
        kl_fix = compute_forge_kl(host_model, forged_fix.torch_module, input_ids)
        rows_fix = per_layer_quick(
            forged_fix.torch_module, host_states, input_ids, basis.W_dec
        )
        l1_ratio_fix = rows_fix[1]["norm_ratio"]

        print(f"{K:>5} {basis.n_features:>7} {kl_base:>12.3f} {kl_fix:>10.3f} "
              f"{l1_ratio_base:>14.3f} {l1_ratio_fix:>13.3f}")

        # Save per-K layer trajectories for inspection.
        out_dir = ROOT / "reports" / "layer_amplification"
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        (out_dir / f"k_{K}_host_native_ln.json").write_text(json.dumps(
            {
                "K": K,
                "n_features": basis.n_features,
                "kl_baseline": kl_base,
                "kl_host_native_ln": kl_fix,
                "layers_baseline": rows_base,
                "layers_host_native_ln": rows_fix,
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
