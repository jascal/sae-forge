"""Tests for forward_mode dispatch and the GPT-2 host-wrapped path.

Covers `saeforge.forward_mode.resolve_forward_mode` (pure unit tests)
and the GPT-2 host-wrapped module end-to-end on a tiny synthetic
basis (no network access, no large host downloads).

The smoke regime (jbloom GPT-2 layer-8) is exercised by the
acceptance-gate script `scripts/prototype_host_wrapped_forward.py`,
not the test suite, because it requires the gitignored smoke
checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from saeforge.basis import FeatureBasis
from saeforge.forward_mode import resolve_forward_mode
from saeforge.model import NativeModelConfig


def _make_basis(n_features: int, d_model: int, *, orthonormal: bool = False):
    rng = np.random.default_rng(seed=42)
    if orthonormal and n_features <= d_model:
        Q, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
        W_dec = Q[:n_features].astype(np.float64)
    else:
        W_dec = rng.standard_normal((n_features, d_model)).astype(np.float64)
    return FeatureBasis(
        W_dec=W_dec,
        kept_ids=np.arange(n_features, dtype=np.int64),
        merged_norms=np.ones(n_features),
        original_norms=np.ones(n_features),
    )


# ---- resolve_forward_mode unit tests --------------------------------------


def test_resolve_explicit_native_in_basis_passes_through():
    basis = _make_basis(50, 768)  # degenerate
    assert resolve_forward_mode(basis, "native_in_basis") == "native_in_basis"


def test_resolve_explicit_host_wrapped_passes_through():
    basis = _make_basis(768, 768, orthonormal=True)  # good
    assert resolve_forward_mode(basis, "host_wrapped") == "host_wrapped"


def test_resolve_auto_on_good_basis_picks_native():
    # n_features = d_model with orthonormal rows -> basis_rank = 768, ratio = 1.0
    # -> SATURATED -> native_in_basis
    basis = _make_basis(768, 768, orthonormal=True)
    assert resolve_forward_mode(basis, "auto") == "native_in_basis"


def test_resolve_auto_on_degenerate_basis_picks_host_wrapped():
    # 50 features in d_model=768 -> ratio = 50/768 = 0.065 -> DEGENERATE
    basis = _make_basis(50, 768)
    assert resolve_forward_mode(basis, "auto") == "host_wrapped"


def test_resolve_auto_on_undersized_basis_picks_host_wrapped():
    # 300 features in d_model=768 -> ratio = 300/768 = 0.39 -> UNDERSIZED
    basis = _make_basis(300, 768)
    assert resolve_forward_mode(basis, "auto") == "host_wrapped"


def test_resolve_invalid_mode_raises():
    basis = _make_basis(50, 768)
    with pytest.raises(ValueError, match="forward_mode must be one of"):
        resolve_forward_mode(basis, "other")


# ---- NativeModelConfig validation -----------------------------------------


def test_config_rejects_invalid_forward_mode():
    with pytest.raises(ValueError, match="forward_mode must be one of"):
        NativeModelConfig(
            family="gpt2",
            hidden_size=64,
            qkv_inner_size=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            intermediate_size=256,
            vocab_size=100,
            forward_mode="garbage",
        )


def test_config_round_trips_forward_mode():
    cfg = NativeModelConfig(
        family="gpt2",
        hidden_size=64,
        qkv_inner_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        intermediate_size=256,
        vocab_size=100,
        forward_mode="host_wrapped",
    )
    assert cfg.to_dict()["forward_mode"] == "host_wrapped"
    rt = NativeModelConfig.from_dict(cfg.to_dict())
    assert rt.forward_mode == "host_wrapped"


def test_config_from_dict_tolerates_legacy_payload_without_forward_mode():
    # Older serialised configs predate this field. Drop it from a fresh
    # config and confirm reconstruction defaults to "auto".
    cfg = NativeModelConfig(
        family="gpt2",
        hidden_size=64,
        qkv_inner_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        intermediate_size=256,
        vocab_size=100,
    )
    payload = cfg.to_dict()
    payload.pop("forward_mode", None)
    rt = NativeModelConfig.from_dict(payload)
    assert rt.forward_mode == "auto"


# ---- GPT-2 host-wrapped module --------------------------------------------


def test_host_wrapped_gpt2_forward_shape():
    """End-to-end: load a tiny pre-trained GPT-2-like host (use gpt2), wrap
    with a synthetic basis, run forward. Asserts logits shape only —
    correctness against host KL is exercised by the prototype script.
    """
    from saeforge.adapters._host_wrapped.gpt2 import build_host_wrapped_gpt2

    host = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    d_model = host.config.n_embd
    basis = _make_basis(d_model, d_model, orthonormal=True)
    wrapped = build_host_wrapped_gpt2(host, basis, scale_boost=1.0).eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        logits = wrapped(input_ids)
    assert logits.shape == (1, 5, host.config.vocab_size)
    # No host parameters should require grad after wrapping.
    assert all(not p.requires_grad for p in wrapped.host.parameters())


def test_host_wrapped_gpt2_matches_host_on_full_orthonormal_basis():
    """On a full-rank orthonormal basis (n=d, W_dec @ W_dec.T = I), the
    encode/decode round-trip is exact and host-wrapped forward should
    equal host forward up to float precision.
    """
    from saeforge.adapters._host_wrapped.gpt2 import build_host_wrapped_gpt2

    host = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    d_model = host.config.n_embd
    basis = _make_basis(d_model, d_model, orthonormal=True)
    wrapped = build_host_wrapped_gpt2(host, basis, scale_boost=1.0).eval()

    input_ids = torch.tensor([[10, 20, 30, 40]])
    with torch.no_grad():
        host_logits = host(input_ids).logits
        wrapped_logits = wrapped(input_ids)
    # Per-token KL between host and wrapped distributions.
    import torch.nn.functional as F

    log_p_host = F.log_softmax(host_logits, dim=-1)
    log_p_wrapped = F.log_softmax(wrapped_logits, dim=-1)
    kl = (
        F.kl_div(log_p_wrapped, log_p_host, reduction="none", log_target=True)
        .sum(dim=-1)
        .mean()
        .item()
    )
    assert kl < 0.05, f"host-wrapped on orthonormal n=d basis must equal host (KL≈0), got {kl:.4f}"


# ---- Non-GPT-2 adapter stub -----------------------------------------------


def test_forge_pipeline_rejects_host_wrapped_with_finetune():
    """ForgePipeline construction must refuse host_wrapped + finetune_steps>0
    with a clear error pointing at the queued follow-up.
    """
    from saeforge import ForgePipeline, SubspaceProjector

    basis = _make_basis(64, 64, orthonormal=True)
    projector = SubspaceProjector(basis)
    with pytest.raises(ValueError, match="add-host-wrapped-finetune-recipe"):
        ForgePipeline(
            basis=basis,
            projector=projector,
            host_model_id="gpt2",
            forward_mode="host_wrapped",
            finetune_steps=10,
        )


def test_forge_pipeline_rejects_host_wrapped_with_hybrid_bridge():
    """host_wrapped has no projected blocks for hybrid bridges to attach to."""
    from saeforge import ForgePipeline, SubspaceProjector

    basis = _make_basis(64, 64, orthonormal=True)
    projector = SubspaceProjector(basis)
    with pytest.raises(ValueError, match="hybrid_bridge=True"):
        ForgePipeline(
            basis=basis,
            projector=projector,
            host_model_id="gpt2",
            forward_mode="host_wrapped",
            hybrid_bridge=True,
            basis_embed=basis,
            basis_lm_head=basis,
        )


def test_cli_forward_mode_default_auto():
    """`sae-forge forge --forward-mode auto` is the default. Confirm parser
    accepts all three values and rejects garbage.
    """
    from saeforge.cli import _build_parser

    parser = _build_parser()
    base = ["forge", "x", "--host-model", "gpt2", "--output-dir", "y"]
    for mode in ("auto", "native_in_basis", "host_wrapped"):
        ns = parser.parse_args(base + ["--forward-mode", mode])
        assert ns.forward_mode == mode
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--forward-mode", "garbage"])


def test_cli_llm_scale_flag_present():
    """`sae-forge forge --llm-scale` is parsed; default is off."""
    from saeforge.cli import _build_parser

    parser = _build_parser()
    base = ["forge", "x", "--host-model", "gpt2", "--output-dir", "y"]
    ns = parser.parse_args(base)
    assert ns.llm_scale is False
    assert ns.regrow_n_init is None
    ns = parser.parse_args(base + ["--llm-scale"])
    assert ns.llm_scale is True


def test_non_gpt2_adapter_raises_clear_error():
    """Adapters that haven't shipped host_wrapped_module yet must raise
    NotImplementedError with a message pointing at the openspec change.
    """
    from saeforge.adapters import adapter_for_family

    adapter = adapter_for_family("llama")
    with pytest.raises(NotImplementedError, match="add-host-wrapped-forge-fallback"):
        adapter.host_wrapped_module(host=None, basis=None, scale_boost=1.0)
