"""Adapter assertion: every Llama-family adapter applies RoPE at default.

Walks the registered adapters; for those with a Llama-family
`family` identifier, builds a tiny synthetic host, forges it
through the saeforge pipeline, and asserts the forged attention
produces position-sensitive output at default `rope_mode="standard"`
and position-invariant output at `rope_mode="none"` (the regression-
diff arm).

Catches future regressions where someone adds a Llama-family
adapter and forgets to wire RoPE — see
``openspec/changes/add-llama-family-rope/`` for the why.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from saeforge.basis import FeatureBasis  # noqa: E402
from saeforge.model import NativeModel, NativeModelConfig  # noqa: E402


_LLAMA_FAMILY = ("llama", "gemma2", "qwen2", "qwen3", "qwen3_moe")


def _identity_basis(d_model: int) -> FeatureBasis:
    """Identity W_dec — projection is exact identity. The forge's only
    deviation from host is the RoPE step (or its absence).
    """
    return FeatureBasis(
        W_dec=np.eye(d_model, dtype=np.float64),
        kept_ids=np.arange(d_model, dtype=np.int64),
        merged_norms=np.ones(d_model),
        original_norms=np.ones(d_model),
    )


def _last_token_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    return out[0, -1].float() if isinstance(out, torch.Tensor) else out.logits[0, -1].float()


@pytest.fixture
def tiny_llama_for_rope():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval(), cfg


def _forge_with_rope_mode(host, basis, rope_mode: str) -> NativeModel:
    """Helper: forge an HF host through identity basis at a chosen rope_mode."""
    from saeforge.adapters import adapter_for
    from saeforge.projector import SubspaceProjector

    projector = SubspaceProjector(basis, scale_boost=1.0)
    adapter = adapter_for(host)
    weights = projector.project_module(host, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    config.rope_mode = rope_mode
    # Suppress the rope_mode='none' UserWarning during the regression-
    # arm test; the warning is informative for users but expected here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return NativeModel.from_projected_weights(config, weights)


def test_llama_forge_with_rope_recovers_host_on_identity_basis(tiny_llama_for_rope):
    """The headline gate from the prototype, pinned as a test:
    with W_dec = I and rope_mode='standard' (default), the forge
    matches host within float precision (<= 1e-4 L2). The 22,671×
    improvement factor over no-RoPE is the production-relevant signal.
    """
    host, cfg = tiny_llama_for_rope
    basis = _identity_basis(cfg.hidden_size)
    ids = torch.tensor([[1, 2, 3, 7]])

    host_logits = _last_token_logits(host, ids)
    rope_forge = _forge_with_rope_mode(host, basis, rope_mode="standard")
    forge_logits = _last_token_logits(rope_forge.torch_module, ids)

    gap = (forge_logits - host_logits).norm().item()
    assert gap < 1e-4, (
        f"RoPE-enabled forge should match host on identity basis "
        f"(prototype measured 7.5e-7); got L2 = {gap:.6f}. "
        f"See openspec/changes/archive/<date>-add-llama-family-rope/"
        f"smoke-results.md for the baseline."
    )


def test_llama_forge_without_rope_diverges_from_host(tiny_llama_for_rope):
    """rope_mode='none' must produce a forge that differs from host
    by far more than the rope_mode='standard' case. Pins the bug's
    existence and the regression-diff arm.
    """
    host, cfg = tiny_llama_for_rope
    basis = _identity_basis(cfg.hidden_size)
    ids = torch.tensor([[1, 2, 3, 7]])

    host_logits = _last_token_logits(host, ids)
    no_rope_forge = _forge_with_rope_mode(host, basis, rope_mode="none")
    forge_logits = _last_token_logits(no_rope_forge.torch_module, ids)

    gap = (forge_logits - host_logits).norm().item()
    # Above float noise; below would mean rotation wasn't actually skipped.
    assert gap > 1e-4, (
        f"rope_mode='none' should produce a meaningfully-different forge "
        f"(prototype measured 1.7e-2); got L2 = {gap:.6f}. "
        f"If this is < 1e-4 then either rope_mode='none' isn't being "
        f"honored, or the fixture is degenerate."
    )


def test_rope_fix_improves_host_fidelity_by_orders_of_magnitude(tiny_llama_for_rope):
    """The headline 22,671× improvement from the prototype, pinned at
    a conservative ≥100× threshold to absorb fixture noise."""
    host, cfg = tiny_llama_for_rope
    basis = _identity_basis(cfg.hidden_size)
    ids = torch.tensor([[1, 2, 3, 7]])

    host_logits = _last_token_logits(host, ids)
    rope_forge = _forge_with_rope_mode(host, basis, rope_mode="standard")
    no_rope_forge = _forge_with_rope_mode(host, basis, rope_mode="none")

    rope_gap = (_last_token_logits(rope_forge.torch_module, ids) - host_logits).norm().item()
    no_rope_gap = (_last_token_logits(no_rope_forge.torch_module, ids) - host_logits).norm().item()

    improvement = no_rope_gap / max(rope_gap, 1e-12)
    assert improvement >= 100.0, (
        f"RoPE should improve forge-vs-host fidelity by >= 100x on identity "
        f"basis (prototype measured 22,671x); got {improvement:.1f}x"
    )


def test_rope_mode_none_emits_user_warning():
    """Configuring rope_mode='none' on a Llama-family family must
    emit a UserWarning so accidental production use is loud.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        NativeModelConfig(
            family="llama",
            hidden_size=32,
            qkv_inner_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=128,
            rope_mode="none",
        )
    rope_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "rope_mode='none'" in str(w.message)
    ]
    assert rope_warnings, (
        "rope_mode='none' on a Llama-family config should emit a UserWarning"
    )


def test_rope_mode_none_no_warning_for_gpt2():
    """rope_mode is silent on GPT-2 (which uses absolute embeddings via
    wpe and doesn't care about the rope_* knobs)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        NativeModelConfig(
            family="gpt2",
            hidden_size=32,
            qkv_inner_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=128,
            rope_mode="none",
        )
    rope_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "rope_mode='none'" in str(w.message)
    ]
    assert not rope_warnings, (
        "GPT-2 doesn't read rope_mode; rope_mode='none' should not warn"
    )


def test_native_model_config_rope_mode_invalid_raises():
    with pytest.raises(ValueError, match="rope_mode must be one of"):
        NativeModelConfig(
            family="llama",
            hidden_size=32,
            qkv_inner_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=128,
            rope_mode="garbage",
        )


def test_native_model_config_rope_fields_round_trip():
    """NativeModelConfig round-trips through to_dict / from_dict with
    the new RoPE fields populated.
    """
    cfg = NativeModelConfig(
        family="llama",
        hidden_size=32,
        qkv_inner_size=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        rope_mode="standard",
        rope_theta=500000.0,
        rope_scaling={"type": "default", "factor": 1.0},
        partial_rotary_factor=0.5,
    )
    rt = NativeModelConfig.from_dict(cfg.to_dict())
    assert rt.rope_mode == "standard"
    assert rt.rope_theta == 500000.0
    assert rt.rope_scaling == {"type": "default", "factor": 1.0}
    assert rt.partial_rotary_factor == 0.5


def test_native_model_config_from_dict_tolerates_missing_rope_fields():
    """Legacy serialised configs (from before add-llama-family-rope)
    won't have rope_mode etc. Reconstruction must default-fill them.
    """
    cfg_dict = NativeModelConfig(
        family="llama",
        hidden_size=32,
        qkv_inner_size=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
    ).to_dict()
    # Strip the new fields to simulate a pre-fix payload.
    for k in ("rope_mode", "rope_theta", "rope_scaling", "partial_rotary_factor"):
        cfg_dict.pop(k, None)
    rt = NativeModelConfig.from_dict(cfg_dict)
    assert rt.rope_mode == "standard"
    assert rt.rope_theta == 10000.0
    assert rt.rope_scaling is None
    assert rt.partial_rotary_factor == 1.0


def test_forge_result_positional_encoding_field_validation():
    """ForgeResult.positional_encoding must be one of the legal values
    or None."""
    from pathlib import Path

    from saeforge.forge import ForgeResult

    # Legal values pass.
    for value in ("absolute_projected", "rotary", "none_skipped", "sinusoidal", None):
        r = ForgeResult(
            model=None,
            output_dir=Path("/tmp/x"),
            positional_encoding=value,
        )
        assert r.positional_encoding == value

    # Garbage raises.
    with pytest.raises(ValueError, match="positional_encoding must be one of"):
        ForgeResult(
            model=None,
            output_dir=Path("/tmp/x"),
            positional_encoding="garbage",
        )
