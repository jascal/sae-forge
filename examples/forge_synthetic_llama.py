"""Synthetic Llama-3 forge example — no HF token required.

Constructs a tiny ``LlamaForCausalLM`` from a hand-rolled
``LlamaConfig`` (random weights — no pretrained download), builds a
small ``FeatureBasis``, and runs the forge end-to-end through the
Llama adapter introduced in multi-architecture-support.

Useful for:

- Smoke-testing the Llama adapter / native-module path on any
  machine, including CI / CPU-only laptops.
- Demonstrating GQA (``num_key_value_heads < num_attention_heads``)
  end-to-end without depending on a particular pretrained checkpoint.
- Reproducing the "no parameter is randomly initialised" invariant
  that multi-architecture-support pinned in tests.

Usage:

    python examples/forge_synthetic_llama.py /tmp/forged_llama

Run without arguments to default to ``./forged_synthetic_llama/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _build_tiny_basis(d_model: int, n_features: int):
    """A small random ``FeatureBasis`` over the host's residual width."""
    from saeforge.basis import FeatureBasis

    rng = np.random.default_rng(0)
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
    norms = np.linalg.norm(W_dec, axis=1).astype(np.float32)
    return FeatureBasis(
        kept_ids=np.arange(n_features, dtype=np.int64),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=Path("forged_synthetic_llama"),
        type=Path,
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        print(
            "examples/forge_synthetic_llama.py needs sae-forge[torch] "
            "(or [intel] on x86_64 macOS).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from saeforge import NativeModel, SubspaceProjector
    from saeforge.adapters import adapter_for

    print(f"[1/4] building tiny LlamaForCausalLM "
          f"(hidden={args.hidden_size}, layers={args.num_layers}, "
          f"heads={args.num_heads}, kv_heads={args.num_kv_heads})")
    cfg = LlamaConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        head_dim=args.hidden_size // args.num_heads,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    host = LlamaForCausalLM(cfg).eval()
    n_host_params = sum(p.numel() for p in host.parameters())
    print(f"      host params: {n_host_params:,}")

    print(f"[2/4] building synthetic FeatureBasis ({args.n_features} features over "
          f"d_model={args.hidden_size})")
    basis = _build_tiny_basis(args.hidden_size, args.n_features)
    projector = SubspaceProjector(basis)

    print("[3/4] dispatching adapter + projecting weights ...")
    adapter = adapter_for(host)
    walk = adapter.walk(host, projector, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    print(f"      adapter family: {adapter.family}; emitted {len(walk)} keys")

    print("[4/4] assembling NativeModel and saving ...")
    nm = NativeModel.from_projected_weights(config, walk)
    nm._move(dtype="float32", device=args.device)
    n_native_params = sum(p.numel() for p in nm.parameters())
    print(f"      native params: {n_native_params:,} "
          f"(compression: {n_host_params / n_native_params:.1f}x)")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    nm.save_pretrained(out / "forged")

    summary = {
        "host_class": type(host).__name__,
        "adapter_family": adapter.family,
        "n_walk_keys": len(walk),
        "host_params": n_host_params,
        "native_params": n_native_params,
        "tied_embeddings": config.tied_embeddings,
        "n_kv_heads": config.n_kv_heads,
    }
    (out / "forge_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"forge_summary.json -> {out / 'forge_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
