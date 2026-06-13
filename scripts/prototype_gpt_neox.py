#!/usr/bin/env python3
"""GPT-NeoX / Pythia adapter faithfulness gate (``add-gpt-neox-adapter``).

Identity-basis forge: with ``W_dec = I`` (basis width == d_model) every projection is the identity, so the
forged ``NativeModel`` is a pure re-implementation of the host — its logits MUST match the host's. This
isolates the native forward (parallel residual + partial rotary + LayerNorm + fused QKV + GELU MLP) from any
compression. Two checks:

  1. **tiny random** GPT-NeoX (various dims / rotary fractions) → exact match (max|Δ| < 1e-4, float32).
  2. **real Pythia-70m** → float32-precision relative match (rel < 1e-4) + 100% argmax agreement. (Absolute
     |Δlogits| is ~7e-3 only because real Pythia logits have magnitude ~1e3; the *relative* error is ~6e-6,
     i.e. the float32 accumulation floor over 6 layers — NOT a math error.)

Run: ``.venv/bin/python scripts/prototype_gpt_neox.py``
"""
from __future__ import annotations

import numpy as np


def _identity_forge_logits(host, ids):
    import torch

    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    d = host.config.hidden_size
    basis = FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64), W_dec=np.eye(d),
        merged_norms=np.ones(d), original_norms=np.ones(d),
    )
    adapter = adapter_for(host)
    proj = SubspaceProjector(basis, scale_boost=1.0)
    walk = adapter.walk(host, proj)
    cfg = adapter.build_native_config(host, d)
    cfg.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(cfg, walk)
    model._move(dtype="float32", device="cpu")
    with torch.no_grad():
        host_logits = host(ids).logits[0].double()
        forged_logits = model.torch_module(ids)[0].double()
    return host_logits, forged_logits, walk, model


def tiny_random_gate() -> bool:
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    ok = True
    print("== tiny-random GPT-NeoX identity forge (exact) ==")
    for hid, heads, nl, pct in [(64, 4, 3, 0.25), (512, 8, 6, 0.25), (128, 4, 2, 1.0), (96, 6, 4, 0.5)]:
        torch.manual_seed(0)
        cfg = GPTNeoXConfig(
            hidden_size=hid, num_attention_heads=heads, num_hidden_layers=nl, intermediate_size=4 * hid,
            vocab_size=512, max_position_embeddings=64, rotary_pct=pct, rotary_emb_base=10000,
            use_parallel_residual=True, layer_norm_eps=1e-5, tie_word_embeddings=False, hidden_act="gelu",
        )
        host = GPTNeoXForCausalLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 10))
        hl, fl, walk, model = _identity_forge_logits(host, ids)
        unreached = [n for n, _ in model.torch_module.named_parameters() if n not in set(walk)]
        md = (hl - fl).abs().max().item()
        passed = md < 1e-4 and not unreached
        ok &= passed
        print(f"   hid={hid:>4} heads={heads} head_dim={hid // heads:>2} pct={pct} {nl}L: "
              f"max|Δ|={md:.2e}  unreached={len(unreached)}  [{'OK' if passed else 'FAIL'}]")
    return ok


def real_pythia_gate(model_id: str = "EleutherAI/pythia-70m") -> bool:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n== real {model_id} identity forge (float32-precision relative match) ==")
    try:
        host = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    except Exception as exc:  # noqa: BLE001
        print(f"   SKIP — could not load {model_id}: {exc}")
        return True
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok("The capital of France is Paris. The capital of Italy is", return_tensors="pt").input_ids
    hl, fl, _, _ = _identity_forge_logits(host, ids)
    md = (hl - fl).abs().max().item()
    rel = md / hl.abs().max().item()
    agree = bool((hl.argmax(-1) == fl.argmax(-1)).all())
    passed = rel < 1e-4 and agree
    print(f"   max|Δ|={md:.3e}  rel={rel:.2e}  argmax_agree={agree}  [{'OK' if passed else 'FAIL'}]")
    return passed


def main() -> None:
    ok = tiny_random_gate()
    ok &= real_pythia_gate()
    print(f"\n== GPT-NeoX adapter gate: {'PASS' if ok else 'FAIL'} ==")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
