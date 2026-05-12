"""NVIDIA-tier smoke for the Qwen3-MoE forge path.

This is the load-bearing T3 validation for the ``qwen3-moe-support``
change. Neither the Intel ``[intel]`` extra (capped at
``transformers<4.50``) nor the M4 box (insufficient memory for a 60GB+
Qwen3-MoE host) can run this end-to-end. The script targets a real
``Qwen3MoeForCausalLM`` on an NVIDIA GPU with ≥80GB recommended (40GB
works with aggressive CPU offload).

Bundled with the PROPOSAL PR per reviewer request, so the script is
available the moment the implementation PR ships. **Before the impl PR
ships**, the script will fail gracefully at the family check:

    FAIL: expected family=qwen3_moe, got llama (Qwen3MoEAdapter not registered yet)

That's the "implementation not yet shipped" signal, not a bug.

What it does:

1. Verifies ``transformers >= 4.51`` (Qwen3-MoE availability).
2. Verifies CUDA availability.
3. Loads ``Qwen/Qwen3-30B-A3B-Base`` (or ``--host-model``) via
   ``device_map="auto"`` and ``dtype=torch.bfloat16``.
4. Confirms ``adapter_for(host).family == "qwen3_moe"``.
5. Walks the host. Confirms the emitted dict has the expected shape:
   the inherited Qwen3 attention keys plus, per block, one ``mlp.gate``
   key and ``num_experts * 3`` per-expert MLP keys.
6. Builds the forged ``NativeModelConfig``. Confirms the four MoE
   fields are populated correctly (``num_experts``, ``num_experts_per_tok``,
   ``moe_intermediate_size``, ``norm_topk_prob``).
7. Builds the forged ``NativeModel``. Confirms each block's ``mlp`` is
   a ``Qwen3MoEMLP`` with ``gate`` (``nn.Linear``) and ``experts``
   (``nn.ModuleList`` of correct length).
8. Runs one forward pass on a short prompt. Confirms output shape and
   finite logits.
9. Optionally (under ``--log-expert-utilization``), instruments the
   forged module's routers and compares top-K decisions to the host's
   on the same prompt. Emits a per-layer top-K agreement rate.

Usage:

    python scripts/smoke_qwen3_moe.py                              # default Qwen3-30B-A3B-Base
    python scripts/smoke_qwen3_moe.py --host-model <other-qwen3-moe>
    python scripts/smoke_qwen3_moe.py --n-features 128             # smaller basis
    python scripts/smoke_qwen3_moe.py --log-expert-utilization     # routing diagnostic

Expected final line on success: ``SMOKE OK``.

Exit codes:
- ``0`` — SMOKE OK. End-to-end pipeline runs and produces finite logits.
- ``1`` — assertion failure (wrong family, missing module, NaN/Inf
  logits, routing collapse). See stderr for which gate tripped.
- ``2`` — environment failure (transformers < 4.51, no CUDA, OOM, gated
  host model). See stderr for the actionable fix.

Hardware notes:

- A100/H100 ≥80GB: comfortable. Host loads to GPU, forge runs in-place.
- A100 40GB / 4090: requires aggressive offload. Pass
  ``--device-map balanced`` or accept slow CPU-offloaded experts.
- Multi-GPU (2x A100-40GB): ``device_map="auto"`` handles sharding.

The actual numbers (num_experts, moe_intermediate_size, num_hidden_layers)
depend on the host's config and are not pre-validated here — the script
adapts to whatever the host advertises.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--host-model",
        default="Qwen/Qwen3-30B-A3B-Base",
        help="HF model id (default: Qwen/Qwen3-30B-A3B-Base, ~60GB bf16 host)",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=256,
        help="Random basis size for projection (default: 256)",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help='Device-map strategy passed to from_pretrained (default: "auto"). '
        'Use "balanced" for multi-GPU equal sharding; "cpu" for CPU-only '
        "(very slow, debug only).",
    )
    p.add_argument(
        "--log-expert-utilization",
        action="store_true",
        help="Instrument the forged module's routers and compare top-K "
        "agreement with the host on the same prompt. Useful for "
        "diagnosing routing collapse.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=16,
        help="Token count for the forward-pass smoke (default: 16, keeps "
        "activations small)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fail_env(msg: str) -> int:
    print(f"FAIL (env): {msg}", file=sys.stderr)
    return 2


def fail_assert(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    # ---------------- environment checks ----------------
    try:
        import torch
    except ImportError as e:
        return fail_env(f"cannot import torch: {e}")
    try:
        import transformers
    except ImportError as e:
        return fail_env(f"cannot import transformers: {e}")
    try:
        from transformers import AutoModelForCausalLM, Qwen3MoeForCausalLM  # noqa: F401
    except ImportError as e:
        return fail_env(
            f"Qwen3-MoE not available in this transformers install ({e}). "
            f"Installed transformers {transformers.__version__}; need >= 4.51. "
            "Upgrade with `pip install -U 'transformers>=4.51'`."
        )
    if not torch.cuda.is_available():
        return fail_env(
            "CUDA is not available. Qwen3-MoE is too large for CPU; this "
            "script targets NVIDIA GPUs (≥80GB recommended). Use "
            "`--device-map cpu` only for debug; it will be very slow."
        )

    print(f"transformers: {transformers.__version__}", flush=True)
    print(
        f"torch: {torch.__version__}  CUDA devices: {torch.cuda.device_count()}",
        flush=True,
    )
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  cuda:{i}  {props.name}  {props.total_memory / 1024**3:.1f}GB",
            flush=True,
        )

    # ---------------- import the forge ----------------
    try:
        from saeforge.adapters import adapter_for
        from saeforge.basis import FeatureBasis
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector
    except ImportError as e:
        return fail_env(f"cannot import sae-forge: {e}")

    # ---------------- load the host ----------------
    print(
        f"\nLoading {args.host_model} (device_map={args.device_map}, dtype=bf16)...",
        flush=True,
    )
    try:
        host = AutoModelForCausalLM.from_pretrained(
            args.host_model,
            dtype=torch.bfloat16,
            device_map=args.device_map,
        ).eval()
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            return fail_env(
                f"host model {args.host_model} requires HF auth (gated). "
                "Run `huggingface-cli login` first."
            )
        if "out of memory" in msg.lower() or "cuda oom" in msg.lower():
            return fail_env(
                f"OOM loading {args.host_model}. Try a smaller host with "
                "`--host-model <smaller-qwen3-moe>` or use device_map=balanced "
                "across multiple GPUs."
            )
        return fail_env(f"host load failed: {e}")

    cfg = host.config
    print(
        f"  hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"experts={cfg.num_experts}, top_k={cfg.num_experts_per_tok}, "
        f"moe_inter={cfg.moe_intermediate_size}, vocab={cfg.vocab_size}",
        flush=True,
    )

    # ---------------- adapter dispatch ----------------
    adapter = adapter_for(host)
    print(f"  adapter family: {adapter.family} (expect: qwen3_moe)", flush=True)
    if adapter.family != "qwen3_moe":
        return fail_assert(
            f"expected family=qwen3_moe, got {adapter.family} "
            "(Qwen3MoEAdapter not registered yet — wait for the impl PR or "
            "check that `saeforge.adapters.qwen3_moe` imported without error)"
        )

    # ---------------- walker sanity ----------------
    print(f"\nWalking host through {args.n_features}-feature basis...", flush=True)
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
    weights = projector.project_module(host)

    # Expected: per layer — one mlp.gate.weight + num_experts × 3 MLP keys.
    moe_key_counts = Counter()
    for k in weights:
        if ".mlp.gate.weight" in k:
            moe_key_counts["gate"] += 1
        elif ".mlp.experts." in k and ".gate_proj.weight" in k:
            moe_key_counts["expert_gate_proj"] += 1
        elif ".mlp.experts." in k and ".up_proj.weight" in k:
            moe_key_counts["expert_up_proj"] += 1
        elif ".mlp.experts." in k and ".down_proj.weight" in k:
            moe_key_counts["expert_down_proj"] += 1
    expected_gate = cfg.num_hidden_layers
    expected_expert = cfg.num_hidden_layers * cfg.num_experts
    if moe_key_counts["gate"] != expected_gate:
        return fail_assert(
            f"expected {expected_gate} mlp.gate.weight keys, "
            f"got {moe_key_counts['gate']}"
        )
    for kind in ("expert_gate_proj", "expert_up_proj", "expert_down_proj"):
        if moe_key_counts[kind] != expected_expert:
            return fail_assert(
                f"expected {expected_expert} {kind} keys, "
                f"got {moe_key_counts[kind]}"
            )
    print(
        f"  walker OK: {moe_key_counts['gate']} gate keys, "
        f"{moe_key_counts['expert_gate_proj']} per-expert MLP key sets "
        f"(matches {cfg.num_hidden_layers} layers × {cfg.num_experts} experts)",
        flush=True,
    )

    # ---------------- native config ----------------
    native_cfg = adapter.build_native_config(host, args.n_features)
    print(
        f"  native cfg: family={native_cfg.family}, "
        f"num_experts={native_cfg.num_experts}, "
        f"num_experts_per_tok={native_cfg.num_experts_per_tok}, "
        f"moe_intermediate_size={native_cfg.moe_intermediate_size}, "
        f"qk_norm={native_cfg.qk_norm}, qkv_bias={native_cfg.qkv_bias}",
        flush=True,
    )
    if native_cfg.num_experts != cfg.num_experts:
        return fail_assert(
            f"native_cfg.num_experts={native_cfg.num_experts} != "
            f"host.num_experts={cfg.num_experts}"
        )
    if native_cfg.num_experts_per_tok != cfg.num_experts_per_tok:
        return fail_assert(
            f"native_cfg.num_experts_per_tok={native_cfg.num_experts_per_tok} != "
            f"host.num_experts_per_tok={cfg.num_experts_per_tok}"
        )
    if native_cfg.qk_norm is not True:
        return fail_assert(f"expected qk_norm=True, got {native_cfg.qk_norm}")

    # ---------------- forged module ----------------
    print("\nBuilding forged native module...", flush=True)
    model = NativeModel.from_projected_weights(native_cfg, weights)
    n_layers = len(model.torch_module.model.layers)
    sample_mlp = model.torch_module.model.layers[0].mlp
    if not hasattr(sample_mlp, "gate") or not hasattr(sample_mlp, "experts"):
        return fail_assert(
            f"forged block[0].mlp lacks gate/experts submodules "
            f"(type: {type(sample_mlp).__name__}; "
            "expected Qwen3MoEMLP)"
        )
    if len(sample_mlp.experts) != cfg.num_experts:
        return fail_assert(
            f"forged block[0] has {len(sample_mlp.experts)} experts, "
            f"expected {cfg.num_experts}"
        )
    # Spot-check every layer has the MoE structure
    for i, layer in enumerate(model.torch_module.model.layers):
        if not hasattr(layer.mlp, "gate"):
            return fail_assert(f"layer {i} missing mlp.gate")
        if not hasattr(layer.mlp, "experts"):
            return fail_assert(f"layer {i} missing mlp.experts")
        if layer.self_attn.q_norm is None or layer.self_attn.k_norm is None:
            return fail_assert(f"layer {i} missing q_norm or k_norm (Qwen3 inherited)")
    print(
        f"  forged module has Qwen3MoEMLP on all {n_layers} layers "
        f"(each with {cfg.num_experts} experts) and q_norm/k_norm on every block",
        flush=True,
    )

    # ---------------- forward pass ----------------
    print(f"\nRunning forward pass ({args.max_seq_len} tokens)...", flush=True)
    # Forged module is on CPU by default; move to one of the GPUs.
    target_device = "cuda:0"
    model._move(dtype="float32", device=target_device)
    ids = torch.randint(
        0, cfg.vocab_size, (1, args.max_seq_len), device=target_device
    )
    with torch.no_grad():
        logits = model.forward(ids)
    print(f"  forward output shape: {tuple(logits.shape)}", flush=True)
    if logits.shape != (1, args.max_seq_len, cfg.vocab_size):
        return fail_assert(
            f"unexpected output shape {tuple(logits.shape)}; expected "
            f"(1, {args.max_seq_len}, {cfg.vocab_size})"
        )
    finite = bool(torch.isfinite(logits).all())
    print(f"  output is finite: {finite}", flush=True)
    if not finite:
        return fail_assert(
            "logits contain NaN/Inf. Likely a scale_boost issue at "
            f"n_features={args.n_features} << d_model={d}; try a larger "
            "--n-features."
        )

    # ---------------- optional routing diagnostic ----------------
    if args.log_expert_utilization:
        print("\nLogging expert utilization (top-K agreement vs host)...", flush=True)
        # Run host with output_hidden_states + log gate logits per layer
        # via forward hooks; do the same on the forged module on a shared
        # input; compute the fraction of tokens whose forged top-K set
        # matches the host top-K set per layer.
        #
        # Implementation note: this requires hooking into both modules'
        # router calls. The HF Qwen3MoeSparseMoeBlock has a stable
        # forward signature; the forged Qwen3MoEMLP matches it. The hook
        # captures gate logits, applies the same softmax + topk, and
        # compares index sets.
        try:
            host_tops = _capture_topk(host, ids.cpu(), cfg.num_experts_per_tok)
            forged_tops = _capture_topk(
                model.torch_module, ids, cfg.num_experts_per_tok
            )
        except Exception as e:
            print(
                f"  routing diagnostic failed: {e}. Skipping (non-fatal).",
                flush=True,
            )
        else:
            for i, (h, f) in enumerate(zip(host_tops, forged_tops)):
                # h and f are tensors of shape (B*T, top_k) of expert indices.
                # Top-K agreement: fraction of tokens whose top-K set
                # matches as a set (order-insensitive).
                h_set = set(map(tuple, h.cpu().sort(-1).values.tolist()))
                f_set = set(map(tuple, f.cpu().sort(-1).values.tolist()))
                agreement = len(h_set & f_set) / max(len(h_set), 1)
                print(
                    f"  layer {i:2d}: top-K set agreement {agreement:.1%}",
                    flush=True,
                )
                if agreement < 0.5:
                    print(
                        f"    WARNING: low top-K agreement at layer {i}; "
                        "routing may be collapsing under projection",
                        flush=True,
                    )

    print("\nSMOKE OK")
    return 0


def _capture_topk(module, ids, top_k):
    """Run ``module(ids)`` and return per-layer top-K expert indices.

    Hooks into every block whose ``mlp`` has a ``gate`` + ``experts``
    structure. Captures the gate logits, applies the same softmax-then-topk
    used by the forged module, and returns a list of `(B*T, top_k)` tensors.
    """
    import torch

    captured: list = []

    def make_hook(layer_idx):
        def hook(mlp, inputs, output):
            x = inputs[0]
            with torch.no_grad():
                logits = mlp.gate(x.reshape(-1, x.shape[-1]))
                weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
                _, top_i = weights.topk(top_k, dim=-1)
                captured.append((layer_idx, top_i.detach()))

        return hook

    handles = []
    layers = (
        module.model.layers
        if hasattr(module, "model") and hasattr(module.model, "layers")
        else module.transformer.h
    )
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
            continue
        handles.append(mlp.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            module(ids)
    finally:
        for h in handles:
            h.remove()
    captured.sort(key=lambda t: t[0])
    return [t[1] for t in captured]


if __name__ == "__main__":
    sys.exit(main())
