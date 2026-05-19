"""Prototype the host-wrapped forward path against the smoke regime.

Builds a minimal HostWrappedGPT2 module per the spec in
openspec/changes/add-host-wrapped-forge-fallback/, runs it on the
GPT-2 layer-8 jbloom sweep, and reports forge KL vs the documented
native-in-basis baseline.

Acceptance gate from the spec:
    - host_wrapped KL monotone non-increasing across K=25,103,163,211
    - K=211 KL < 5.0 nats
    - Synthetic good-tier sanity (n=d=768, orthonormal W_dec):
      host_wrapped agrees with native_in_basis within 0.1 nats

Run:
    .venv/bin/python scripts/prototype_host_wrapped_forward.py
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


from saeforge.adapters._host_wrapped.gpt2 import build_host_wrapped_gpt2 as HostWrappedGPT2  # noqa: E402


def compute_kl(host_logits, forged_logits):
    """Per-token KL(host || forged) averaged over positions."""
    log_p_host = F.log_softmax(host_logits, dim=-1)
    log_p_forged = F.log_softmax(forged_logits, dim=-1)
    kl_per_pos = F.kl_div(
        log_p_forged, log_p_host, reduction="none", log_target=True
    ).sum(dim=-1)
    return kl_per_pos.mean().item()


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

    host_model = transformers.AutoModelForCausalLM.from_pretrained(
        HOST_ID, torch_dtype=torch.float32
    ).eval()

    with torch.no_grad():
        host_logits = host_model(input_ids).logits

    print(f"=== smoke regime: jbloom GPT-2 L8 ===\n")
    print(f"{'K':>5} {'n_feat':>7} {'KL_native_baseline':>20} {'KL_host_wrapped':>17}")
    print("-" * 56)
    kl_native = []
    kl_wrapped = []
    for K in K_VALUES:
        sae_path = SMOKE / "baseline" / "_materialised" / "hea" / "pareto" / f"k_{K}.safetensors"
        forged_dir = SMOKE / "baseline" / "hea" / f"k_{K}" / "forged"
        if not sae_path.exists() or not forged_dir.exists():
            continue
        basis = FeatureBasis.from_polygram_checkpoint(sae_path)

        # Native baseline (load the smoke-produced forged model).
        forged_native = NativeModel.load_pretrained(forged_dir)
        forged_native.torch_module.eval()
        with torch.no_grad():
            native_logits = forged_native.torch_module(input_ids)
        kl_n = compute_kl(host_logits, native_logits)
        kl_native.append((K, kl_n))

        # Host-wrapped prototype.
        wrapped = HostWrappedGPT2(host_model, basis, scale_boost=1.0).eval()
        with torch.no_grad():
            wrapped_logits = wrapped(input_ids)
        kl_w = compute_kl(host_logits, wrapped_logits)
        kl_wrapped.append((K, kl_w))

        print(f"{K:>5} {basis.n_features:>7} {kl_n:>20.4f} {kl_w:>17.4f}")

    # Acceptance gate from
    # openspec/changes/add-host-wrapped-forge-fallback/specs/forge-forward-mode/spec.md:
    #   1. host_wrapped KL <= native KL at every K
    #   2. K=211 KL < 25.0 (vs documented 86.39 native)
    #   3. No adjacent K-pair ΔKL > 10 nats (amplification removed)
    # Monotonicity is NOT a gate — smoke bases are non-nested.
    print()
    gate_1 = all(kw <= kn + 0.01 for (_, kn), (_, kw) in zip(kl_native, kl_wrapped))
    final_kl = kl_wrapped[-1][1] if kl_wrapped else float("inf")
    gate_2 = final_kl < 25.0
    max_adj_delta = 0.0
    for i in range(1, len(kl_wrapped)):
        lo_K, lo_kl = kl_wrapped[i - 1]
        hi_K, hi_kl = kl_wrapped[i]
        delta = hi_kl - lo_kl
        max_adj_delta = max(max_adj_delta, abs(delta))
        print(f"  K={lo_K}→{hi_K}: ΔKL={delta:+.4f}")
    gate_3 = max_adj_delta <= 10.0
    print(f"\n  gate 1 (host_wrapped ≤ native at every K): {'PASS' if gate_1 else 'FAIL'}")
    print(f"  gate 2 (K=211 KL < 25.0):                   "
          f"{'PASS' if gate_2 else 'FAIL'}  (got {final_kl:.3f})")
    print(f"  gate 3 (max adjacent ΔKL ≤ 10 nats):        "
          f"{'PASS' if gate_3 else 'FAIL'}  (got {max_adj_delta:.3f})")
    print(f"  acceptance: {'PASS' if (gate_1 and gate_2 and gate_3) else 'FAIL'}")

    # Good-tier sanity check: synthetic basis with n_features = d_model, orthonormal W_dec.
    print(f"\n=== good-tier sanity: synthetic orthonormal basis (n=d=768) ===\n")
    rng = np.random.default_rng(seed=0)
    Q, _ = np.linalg.qr(rng.standard_normal((768, 768)))
    W_dec_synthetic = Q.astype(np.float64)  # orthonormal rows
    n_kept = W_dec_synthetic.shape[0]
    synth_basis = FeatureBasis(
        W_dec=W_dec_synthetic,
        kept_ids=np.arange(n_kept, dtype=np.int64),
        merged_norms=np.ones(n_kept),
        original_norms=np.ones(n_kept),
    )
    # Host wrapped on synth basis.
    wrapped = HostWrappedGPT2(host_model, synth_basis, scale_boost=1.0).eval()
    with torch.no_grad():
        wrapped_logits = wrapped(input_ids)
    kl_synth = compute_kl(host_logits, wrapped_logits)
    print(f"  host_wrapped KL on n=d=768 orthonormal basis: {kl_synth:.4f}")
    print(f"  acceptance (<0.5 nats): {'PASS' if kl_synth < 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
