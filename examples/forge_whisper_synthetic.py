"""Synthetic Whisper-encoder forge example — no HF download, no audio file.

Constructs a tiny ``WhisperModel`` from a hand-rolled
``WhisperConfig`` (random weights — no pretrained download), builds
a small ``FeatureBasis`` over the encoder's residual width, walks
the Whisper adapter to project every residual-touching weight into
the basis, assembles the forged encoder, and evaluates per-frame
cosine faithfulness against the host on a synthetic sine-sweep mel
spectrogram.

Useful for:

- Smoke-testing the Whisper-encoder adapter / native module path on
  any CPU-only machine, including CI without ``[audio]`` extras
  (mel features are synthesised pure-numpy via
  :func:`saeforge.audio_data.synthetic_mel_features`).
- Demonstrating the conv-stem ε_conv accounting end-to-end: the
  forged encoder's conv1 / conv2 / embed_positions are frozen-copied
  from the host bit-for-bit; the d → f basis projection is applied
  at the conv-stem → first-block boundary via the ``basis_encode``
  buffer.
- Validating the no-randomly-initialised-weights invariant on a
  non-LM architecture (the eval-only audio path).

Usage:

    python examples/forge_whisper_synthetic.py /tmp/forged_whisper

Run without arguments to default to ``./forged_whisper_synthetic/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _build_tiny_basis(d_model: int, n_features: int):
    """A small random ``FeatureBasis`` over the encoder's residual width."""
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
        default=Path("forged_whisper_synthetic"),
        type=Path,
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-attention-heads", type=int, default=4)
    parser.add_argument("--encoder-ffn-dim", type=int, default=128)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument(
        "--max-source-positions",
        type=int,
        default=1500,
        help=(
            "Whisper's positional table size. The conv stem expects "
            "input length 2 * max_source_positions due to conv2 stride=2."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--scale-boost",
        default="auto",
        help=(
            "SubspaceProjector scale_boost (float or 'auto'). 'auto' "
            "picks min(1.0, d_model/n_features) for over-complete bases."
        ),
    )
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import WhisperConfig, WhisperModel
    except ImportError as exc:
        print(
            "examples/forge_whisper_synthetic.py needs sae-forge[torch] "
            "(or [intel] on x86_64 macOS).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from saeforge import NativeModel, SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.audio_data import synthetic_mel_features
    from saeforge.audio_eval import cosine_faithfulness

    print(
        f"[1/5] building tiny WhisperModel "
        f"(d_model={args.d_model}, encoder_layers={args.encoder_layers}, "
        f"heads={args.encoder_attention_heads}, "
        f"ffn={args.encoder_ffn_dim})"
    )
    cfg = WhisperConfig(
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_dim,
        # Decoder fields populated to the WhisperConfig validation
        # minimum; the encoder-only forge never touches them.
        decoder_layers=1,
        decoder_attention_heads=1,
        decoder_ffn_dim=8,
        vocab_size=51865,
        num_mel_bins=80,
        max_source_positions=args.max_source_positions,
    )
    host = WhisperModel(cfg).eval()
    n_host_params = sum(p.numel() for p in host.parameters())
    print(f"      host params (full model, encoder + decoder): {n_host_params:,}")

    print(
        f"[2/5] building synthetic FeatureBasis ({args.n_features} features "
        f"over d_model={args.d_model})"
    )
    basis = _build_tiny_basis(args.d_model, args.n_features)
    sb = args.scale_boost if args.scale_boost == "auto" else float(args.scale_boost)
    projector = SubspaceProjector(basis, scale_boost=sb)

    print("[3/5] dispatching adapter + projecting encoder weights ...")
    adapter = adapter_for(host)
    walk = adapter.walk(host, projector, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    assert adapter.family == "whisper_encoder", (
        f"expected whisper_encoder, got {adapter.family!r}"
    )
    print(
        f"      adapter family: {adapter.family}; emitted {len(walk)} keys "
        f"(includes the basis_encode buffer for the d → f bridge)"
    )

    print("[4/5] assembling ForgedWhisperEncoder ...")
    nm = NativeModel.from_projected_weights(config, walk)
    nm._move(dtype="float32", device=args.device)
    n_native_params = sum(p.numel() for p in nm.parameters())
    print(
        f"      forged encoder params: {n_native_params:,} "
        f"(decoder is out of scope — forge-whisper-decoder is a follow-up)"
    )

    print(
        f"[5/5] synthesising mel features (seed={args.seed}, "
        f"1 × 80 × {args.max_source_positions * 2}) and running cosine eval ..."
    )
    n_frames = args.max_source_positions * 2
    mel = synthetic_mel_features(args.seed, n_frames=n_frames).to(args.device)
    with torch.no_grad():
        score = cosine_faithfulness(nm, host, mel, device=args.device)
    print(f"      cosine faithfulness vs host (per-frame mean): {score:.4f}")
    print(
        "      note: with random host weights + a random basis, ε from "
        "LayerNorm projection compounds across blocks and the cosine "
        "decorrelates quickly. A real polygram-compressed Whisper SAE — "
        "where the basis is trained to align with the host's natural "
        "feature directions — gives a much higher score."
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    nm.save_pretrained(out / "forged")

    summary = {
        "host_class": type(host).__name__,
        "adapter_family": adapter.family,
        "n_walk_keys": len(walk),
        "host_params_full_model": n_host_params,
        "native_params_encoder_only": n_native_params,
        "n_features": args.n_features,
        "d_model": args.d_model,
        "encoder_layers": args.encoder_layers,
        "cosine_faithfulness": score,
    }
    (out / "forge_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"forge_summary.json -> {out / 'forge_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
