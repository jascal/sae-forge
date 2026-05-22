"""Tests for the ESM-2 adapter.

The identity-basis byte-equivalence test is the load-bearing one:
with ``W_dec = I`` and ``scale_boost = 1.0``, every ``project_*`` call
in the projector is a no-op, so the walked weights are bit-identical
to the host's. The forged ``ForgedEsm2`` module's forward must then
reproduce HF's ``EsmModel.last_hidden_state`` exactly (max-abs diff
== 0.0). This pins:

- Walk key shapes (every key the adapter emits has a slot in the
  forged module's state_dict).
- Forward semantics (pre-LN attention sublayer, pre-LN FFN sublayer,
  ESM-specific GELU, ESM-specific query-scaling-before-RoPE order,
  bidirectional attention with no causal mask, final
  ``emb_layer_norm_after``).

Without this test the adapter could silently produce a model that
looks right structurally but diverges semantically — the bug class
the byte-identity test is designed to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _build_tiny_esm_model(seed: int = 0):
    """Construct a small EsmModel with all the ESM-2 defaults that
    matter for the adapter (rotary position embeddings, no pre-emb
    LayerNorm, token-dropout disabled in eval)."""
    from transformers import EsmConfig, EsmModel

    torch.manual_seed(seed)
    cfg = EsmConfig(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        position_embedding_type="rotary",
        emb_layer_norm_before=False,
        token_dropout=False,
        mask_token_id=32,
        pad_token_id=1,
    )
    return EsmModel(cfg).eval(), cfg


def _identity_basis(d: int):
    from saeforge.basis import FeatureBasis

    return FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64),
        W_dec=np.eye(d, dtype=np.float64),
        merged_norms=np.ones(d, dtype=np.float64),
        original_norms=np.ones(d, dtype=np.float64),
    )


def test_adapter_registers_for_esm_model():
    from transformers import EsmForMaskedLM, EsmModel

    from saeforge.adapters import adapter_for, registered_families

    assert "esm2" in registered_families()

    host, _ = _build_tiny_esm_model()
    assert adapter_for(host).family == "esm2"

    # EsmForMaskedLM also dispatches to the same adapter (uses host.esm
    # to extract the encoder root).
    cfg = host.config
    masked_host = EsmForMaskedLM(cfg).eval()
    assert adapter_for(masked_host).family == "esm2"


def test_adapter_rejects_non_rotary_position_embedding():
    """ESM-1 (position_embedding_type='absolute') is out of scope —
    the adapter raises rather than silently misprojecting the
    position-embeddings table that ESM-1 has and ESM-2 doesn't."""
    from transformers import EsmConfig, EsmModel

    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.projector import SubspaceProjector

    cfg = EsmConfig(
        vocab_size=33, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64,
        max_position_embeddings=128,
        position_embedding_type="absolute",  # ESM-1 style
        emb_layer_norm_before=False, token_dropout=False,
        mask_token_id=32, pad_token_id=1,
    )
    host = EsmModel(cfg).eval()
    adapter = adapter_for(host)
    projector = SubspaceProjector(basis=_identity_basis(32))
    with pytest.raises(NotImplementedError, match="rotary"):
        adapter.walk(host, projector)


def test_native_config_pins_encoder_states_and_mha():
    from saeforge.adapters import adapter_for

    host, _ = _build_tiny_esm_model()
    adapter = adapter_for(host)
    cfg = adapter.build_native_config(host, n_features=32)
    assert cfg.family == "esm2"
    assert cfg.output_kind == "encoder_states"
    assert cfg.hidden_size == 32
    assert cfg.vocab_size == 33
    assert cfg.num_heads == cfg.n_kv_heads, "ESM-2 is MHA, not GQA"
    assert cfg.rope_theta == 10000.0, "ESM-2 always uses theta=10000"


def test_native_config_validation_rejects_vocab_zero_for_esm2():
    """ESM-2 has a word-embeddings table — vocab_size must be > 0.
    (Whisper-encoder has the opposite invariant: no embeddings, vocab=0.)"""
    from saeforge.model import NativeModelConfig

    with pytest.raises(ValueError, match="vocab_size > 0"):
        NativeModelConfig(
            family="esm2",
            hidden_size=32, qkv_inner_size=32, num_layers=1, num_heads=4,
            head_dim=8, intermediate_size=64, vocab_size=0,
            output_kind="encoder_states",
        )


def test_identity_basis_reproduces_host_bit_for_bit():
    """The load-bearing semantic test. With W_dec = I, the forged
    ESM-2 forward must equal HF's ``EsmModel.last_hidden_state``
    exactly (max abs diff == 0.0)."""
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    host, cfg = _build_tiny_esm_model(seed=42)
    projector = SubspaceProjector(basis=_identity_basis(cfg.hidden_size))
    adapter = adapter_for(host)
    weights = adapter.walk(host, projector)
    native_cfg = adapter.build_native_config(host, n_features=cfg.hidden_size)
    model = NativeModel.from_projected_weights(native_cfg, weights)

    # Include CLS / EOS to mirror how bio-sae feeds real protein sequences.
    input_ids = torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10, 2]], dtype=torch.long)
    with torch.no_grad():
        host_h = host(input_ids=input_ids).last_hidden_state
        forged_h = model.torch_module(input_ids)
    assert host_h.shape == forged_h.shape
    max_diff = (host_h - forged_h).abs().max().item()
    assert max_diff == 0.0, (
        f"identity-basis forge diverged from host by {max_diff}; "
        f"adapter walk or ForgedEsm2 forward is misaligned with HF "
        f"modeling_esm.py"
    )


def test_walk_emits_every_expected_key():
    """Sanity check that the walk doesn't silently drop a slot. The
    forged module must accept the walked dict via load_state_dict
    without missing keys."""
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    host, cfg = _build_tiny_esm_model()
    adapter = adapter_for(host)
    projector = SubspaceProjector(basis=_identity_basis(cfg.hidden_size))
    weights = adapter.walk(host, projector)

    # Expected key shapes (n_layers = 2).
    expected = {
        "embeddings.word_embeddings.weight",
        "encoder.emb_layer_norm_after.weight",
        "encoder.emb_layer_norm_after.bias",
    }
    for i in range(cfg.num_hidden_layers):
        for k in (
            f"encoder.layer.{i}.attention.LayerNorm.weight",
            f"encoder.layer.{i}.attention.LayerNorm.bias",
            f"encoder.layer.{i}.attention.self.query.weight",
            f"encoder.layer.{i}.attention.self.query.bias",
            f"encoder.layer.{i}.attention.self.key.weight",
            f"encoder.layer.{i}.attention.self.key.bias",
            f"encoder.layer.{i}.attention.self.value.weight",
            f"encoder.layer.{i}.attention.self.value.bias",
            f"encoder.layer.{i}.attention.output.dense.weight",
            f"encoder.layer.{i}.attention.output.dense.bias",
            f"encoder.layer.{i}.LayerNorm.weight",
            f"encoder.layer.{i}.LayerNorm.bias",
            f"encoder.layer.{i}.intermediate.dense.weight",
            f"encoder.layer.{i}.intermediate.dense.bias",
            f"encoder.layer.{i}.output.dense.weight",
            f"encoder.layer.{i}.output.dense.bias",
        ):
            expected.add(k)
    missing = expected - set(weights)
    assert not missing, f"adapter walk missing expected keys: {missing}"

    native_cfg = adapter.build_native_config(host, n_features=cfg.hidden_size)
    model = NativeModel.from_projected_weights(native_cfg, weights)
    # No key drift — the model accepted every walk key.
    assert model.num_parameters() > 0


def test_default_faithfulness_target_is_token_cosine():
    from saeforge.adapters import adapter_for_family

    adapter = adapter_for_family("esm2")
    target = adapter.default_faithfulness_target()
    assert target.name == "token_cosine"
    assert target.better_when == "higher"


def test_token_cosine_score_at_identity_basis():
    """End-to-end test: identity-basis forge fed to the default
    faithfulness target returns cosine ≈ 1.0 (the forged and host
    encoder states are bit-identical on every position; the CLS+EOS
    strip leaves real residues only, both still identical)."""
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    host, cfg = _build_tiny_esm_model(seed=7)
    projector = SubspaceProjector(basis=_identity_basis(cfg.hidden_size))
    adapter = adapter_for(host)
    weights = adapter.walk(host, projector)
    native_cfg = adapter.build_native_config(host, n_features=cfg.hidden_size)
    model = NativeModel.from_projected_weights(native_cfg, weights)

    input_ids = torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10, 2]], dtype=torch.long)
    target = adapter.default_faithfulness_target()
    cosine, perplexity = target.score(
        forged=model, host=host,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )
    assert cosine == pytest.approx(1.0, abs=1e-5)
    assert perplexity == pytest.approx(0.0, abs=1e-5)
