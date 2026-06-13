"""GPT-NeoX / Pythia adapter tests (``add-gpt-neox-adapter``).

CI-safe: builds tiny random ``GPTNeoXForCausalLM`` instances (no downloads). The headline check is the
identity-basis forge — with ``W_dec = I`` every projection is the identity, so the forged ``NativeModel`` is a
pure re-implementation of the host and its logits MUST match. This exercises the three features GPT-NeoX adds
over existing adapters: parallel residual, partial rotary, and LayerNorm-with-bias + fused QKV + GELU MLP.
Real-Pythia faithfulness lives in ``scripts/prototype_gpt_neox.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saeforge.adapters import adapter_for  # noqa: E402
from saeforge.basis import FeatureBasis  # noqa: E402
from saeforge.model import NativeModel  # noqa: E402
from saeforge.projector import SubspaceProjector  # noqa: E402


def _tiny_gpt_neox(hidden=64, heads=4, layers=3, pct=0.25, seed=0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(
        hidden_size=hidden, num_attention_heads=heads, num_hidden_layers=layers,
        intermediate_size=4 * hidden, vocab_size=256, max_position_embeddings=64,
        rotary_pct=pct, rotary_emb_base=10000, use_parallel_residual=True,
        layer_norm_eps=1e-5, tie_word_embeddings=False, hidden_act="gelu",
    )
    return GPTNeoXForCausalLM(cfg).eval()


def _identity_basis(d):
    return FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64), W_dec=np.eye(d),
        merged_norms=np.ones(d), original_norms=np.ones(d),
    )


def test_dispatch_to_gpt_neox_adapter():
    host = _tiny_gpt_neox()
    assert adapter_for(host).family == "gpt_neox"


def test_build_native_config_reads_partial_rotary_from_rope_parameters():
    """Modern transformers stores rotary knobs in cfg.rope_parameters; the adapter must read it (a regression
    guard for the 'partial_rotary_factor silently 1.0' bug found during bring-up)."""
    host = _tiny_gpt_neox(hidden=64, heads=4, pct=0.25)
    cfg = adapter_for(host).build_native_config(host, 64)
    assert cfg.family == "gpt_neox"
    assert cfg.partial_rotary_factor == pytest.approx(0.25)
    assert cfg.rope_theta == pytest.approx(10000.0)
    assert cfg.qkv_bias is True  # GPT-NeoX has attention biases


def test_walk_emits_expected_keys_and_reaches_every_param():
    host = _tiny_gpt_neox()
    d = host.config.hidden_size
    proj = SubspaceProjector(_identity_basis(d), scale_boost=1.0)
    adapter = adapter_for(host)
    walk = adapter.walk(host, proj)
    cfg = adapter.build_native_config(host, d)
    cfg.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(cfg, walk)

    # No native parameter left randomly initialised.
    unreached = [n for n, _ in model.torch_module.named_parameters() if n not in set(walk)]
    assert unreached == [], f"unreached params: {unreached}"

    # GPT-NeoX-specific keys present: two LayerNorms (weight+bias), fused QKV (+bias), GELU MLP (+biases),
    # untied embed_out.
    for k in (
        "gpt_neox.embed_in.weight",
        "gpt_neox.layers.0.input_layernorm.weight", "gpt_neox.layers.0.input_layernorm.bias",
        "gpt_neox.layers.0.post_attention_layernorm.weight",
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.attention.query_key_value.bias",
        "gpt_neox.layers.0.attention.dense.weight", "gpt_neox.layers.0.attention.dense.bias",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight", "gpt_neox.layers.0.mlp.dense_h_to_4h.bias",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight", "gpt_neox.layers.0.mlp.dense_4h_to_h.bias",
        "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias",
        "embed_out.weight",
    ):
        assert k in walk, f"missing walk key {k}"


@pytest.mark.parametrize("hidden,heads,layers,pct", [
    (64, 4, 3, 0.25),   # partial rotary (Pythia's fraction)
    (128, 4, 2, 1.0),   # full rotary
    (96, 6, 4, 0.5),    # half rotary, head_dim=16
])
def test_identity_forge_reproduces_host_logits(hidden, heads, layers, pct):
    """Identity basis (W_dec=I) → the forge is lossless; forged logits must equal the host's to float32
    precision. Covers parallel residual + partial/full rotary + LayerNorm + fused QKV + GELU MLP end-to-end."""
    host = _tiny_gpt_neox(hidden=hidden, heads=heads, layers=layers, pct=pct)
    d = host.config.hidden_size
    adapter = adapter_for(host)
    proj = SubspaceProjector(_identity_basis(d), scale_boost=1.0)
    walk = adapter.walk(host, proj)
    cfg = adapter.build_native_config(host, d)
    cfg.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(cfg, walk)
    model._move(dtype="float32", device="cpu")

    ids = torch.randint(0, host.config.vocab_size, (1, 12))
    with torch.no_grad():
        host_logits = host(ids).logits[0]
        forged_logits = model.torch_module(ids)[0]
    assert forged_logits.shape == host_logits.shape
    assert (host_logits - forged_logits).abs().max().item() < 1e-4


def test_compressed_forge_runs_and_shrinks_residual():
    """A genuinely compressed basis (n_features < d_model) forges and runs a forward (smoke — not a
    faithfulness claim; that's the capability sweep's job)."""
    host = _tiny_gpt_neox(hidden=64, heads=4, layers=2)
    d = host.config.hidden_size
    n = 32
    rng = np.random.default_rng(0)
    W_dec = rng.standard_normal((n, d))
    basis = FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64), W_dec=W_dec,
        merged_norms=np.linalg.norm(W_dec, axis=1), original_norms=np.linalg.norm(W_dec, axis=1),
    )
    adapter = adapter_for(host)
    proj = SubspaceProjector(basis, scale_boost=1.0)
    walk = adapter.walk(host, proj)
    cfg = adapter.build_native_config(host, n)
    cfg.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(cfg, walk)
    model._move(dtype="float32", device="cpu")
    ids = torch.randint(0, host.config.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model.torch_module(ids)
    assert logits.shape == (1, 8, host.config.vocab_size)
    assert torch.isfinite(logits).all()
