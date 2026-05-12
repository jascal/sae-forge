"""Real-Qwen3 smoke: load a Qwen3 host, project + construct + forward, sanity-check.

This is the load-bearing M4 confirmation for the ``qwen3-dense-support`` change
(PR #22). The Intel ``[intel]`` extra is capped at ``transformers<4.50`` and
cannot load Qwen3, so end-to-end validation against a real Qwen3 host has to
happen on a box with ``transformers >= 4.51`` (M4 / CUDA / fresh ``[torch]``).

What it does:

1. Loads the host via ``AutoModelForCausalLM``. Default: ``Qwen/Qwen3-0.6B``
   (smallest Qwen3 dense, ~1.2GB at bf16).
2. Confirms ``adapter_for(host).family == "qwen3"``.
3. Builds a random ``n_features``-wide basis and runs the projector.
4. Builds the forged ``NativeModel``. Confirms ``qk_norm=True`` and
   ``qkv_bias=False`` are set in the native config.
5. Confirms the forged attention blocks have ``q_norm`` / ``k_norm`` modules
   constructed on every layer.
6. Runs one forward pass. Confirms output shape and finite logits.

Usage:

    python scripts/smoke_qwen3.py                       # default Qwen3-0.6B
    python scripts/smoke_qwen3.py --host-model Qwen/Qwen3-1.7B
    python scripts/smoke_qwen3.py --n-features 128

Expected final line on success: ``SMOKE OK``.

Failure modes (most → least likely):

- ``Cannot import Qwen3ForCausalLM`` → ``transformers < 4.51``. Upgrade with
  ``pip install -U 'transformers>=4.51'``.
- ``adapter family: llama`` instead of ``qwen3`` → registration didn't take.
  Check ``import saeforge.adapters.qwen3`` did not raise.
- ``NaN/Inf in logits`` → likely a scale_boost issue with
  ``n_features << d_model``. The script uses ``scale_boost='auto'`` to handle
  this, but if it still triggers, the issue is mechanism-level. Paste the
  traceback in PR #22.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--host-model",
        default="Qwen/Qwen3-0.6B",
        help="HF model id (default: Qwen/Qwen3-0.6B, smallest Qwen3 dense)",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=64,
        help="Random basis size (default: 64; well below d_model so scale_boost=auto applies)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print(f"FAIL: cannot import torch/transformers: {e}", file=sys.stderr)
        return 2

    try:
        from transformers import Qwen3ForCausalLM  # noqa: F401
    except ImportError as e:
        print(
            f"FAIL: Qwen3 not available in this transformers install ({e}). "
            f"Need transformers >= 4.51.",
            file=sys.stderr,
        )
        return 2

    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    print(f"Loading {args.host_model}...", flush=True)
    host = AutoModelForCausalLM.from_pretrained(args.host_model, dtype=torch.bfloat16).eval()
    cfg = host.config
    print(
        f"  hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"head_dim={getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}, "
        f"vocab={cfg.vocab_size}",
        flush=True,
    )

    adapter = adapter_for(host)
    print(f"  adapter family: {adapter.family} (expect: qwen3)", flush=True)
    if adapter.family != "qwen3":
        print(f"FAIL: expected family=qwen3, got {adapter.family}", file=sys.stderr)
        return 1

    d = cfg.hidden_size
    rng = np.random.default_rng(args.seed)
    W = rng.standard_normal((args.n_features, d)).astype(np.float64)
    norms = np.linalg.norm(W, axis=1)
    basis = FeatureBasis(
        kept_ids=np.arange(args.n_features),
        W_dec=W,
        merged_norms=norms,
        original_norms=norms,
    )
    projector = SubspaceProjector(basis, scale_boost="auto")

    print(f"  projecting weights through {args.n_features}-feature basis...", flush=True)
    weights = projector.project_module(host)
    native_cfg = adapter.build_native_config(host, args.n_features)
    print(
        f"  native cfg: family={native_cfg.family}, "
        f"qk_norm={native_cfg.qk_norm}, qkv_bias={native_cfg.qkv_bias}",
        flush=True,
    )
    if native_cfg.qk_norm is not True:
        print(f"FAIL: expected qk_norm=True, got {native_cfg.qk_norm}", file=sys.stderr)
        return 1
    if native_cfg.qkv_bias is not False:
        print(f"FAIL: expected qkv_bias=False, got {native_cfg.qkv_bias}", file=sys.stderr)
        return 1

    model = NativeModel.from_projected_weights(native_cfg, weights)
    n_layers = len(model.torch_module.model.layers)
    for i, layer in enumerate(model.torch_module.model.layers):
        if layer.self_attn.q_norm is None:
            print(f"FAIL: layer {i} missing q_norm module", file=sys.stderr)
            return 1
        if layer.self_attn.k_norm is None:
            print(f"FAIL: layer {i} missing k_norm module", file=sys.stderr)
            return 1
    print(f"  forged module has q_norm/k_norm on all {n_layers} layers", flush=True)

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model.forward(ids)
    print(f"  forward output shape: {tuple(logits.shape)}", flush=True)
    if logits.shape != (1, 16, cfg.vocab_size):
        print(
            f"FAIL: expected shape (1, 16, {cfg.vocab_size}), got {tuple(logits.shape)}",
            file=sys.stderr,
        )
        return 1
    finite = bool(torch.isfinite(logits).all())
    print(f"  output is finite: {finite}", flush=True)
    if not finite:
        print("FAIL: logits contain NaN/Inf", file=sys.stderr)
        return 1

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
