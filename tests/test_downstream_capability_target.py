"""Tests for DownstreamCapabilityTarget.

The identity-basis test is load-bearing: forge identity-W_dec, the
target decodes back via basis_decode, the encoder produces the *same
latents host would* on the same activations. The version-independent
invariant is therefore "forge reproduces host" — forge's per-label AUC
must equal the host-reference AUC computed by ``_host_reference_pf_auc``,
bit-for-bit. (It is *not* "AUC == 1.0": whether the fixture's random
labels are linearly separable by the random encoder directions depends
on the exact ESM activations, which drift between torch/transformers
builds — so a hardcoded 1.0 is a flaky proxy for the real invariant.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _build_tiny_esm_model(seed: int = 0):
    from transformers import EsmConfig, EsmModel

    torch.manual_seed(seed)
    cfg = EsmConfig(
        vocab_size=33, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64,
        max_position_embeddings=128,
        position_embedding_type="rotary",
        emb_layer_norm_before=False, token_dropout=False,
        mask_token_id=32, pad_token_id=1,
    )
    return EsmModel(cfg).eval(), cfg


def _identity_basis(d):
    from saeforge.basis import FeatureBasis
    return FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64),
        W_dec=np.eye(d, dtype=np.float64),
        merged_norms=np.ones(d, dtype=np.float64),
        original_norms=np.ones(d, dtype=np.float64),
    )


def _identity_forge(seed=0):
    """Build (forged_module, host, input_ids) — a 5-row eval fixture."""
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    host, cfg = _build_tiny_esm_model(seed=seed)
    d = cfg.hidden_size
    proj = SubspaceProjector(basis=_identity_basis(d))
    adapter = adapter_for(host)
    weights = adapter.walk(host, proj)
    ncfg = adapter.build_native_config(host, n_features=d)
    model = NativeModel.from_projected_weights(ncfg, weights)
    input_ids = torch.tensor([
        [0, 4, 5, 6, 7, 8, 9, 10, 2],
        [0, 4, 4, 5, 5, 6, 6, 6, 2],
        [0, 8, 9, 8, 9, 8, 9, 4, 2],
        [0, 4, 5, 6, 7, 8, 9, 10, 2],
        [0, 5, 6, 7, 8, 9, 4, 4, 2],
    ], dtype=torch.long)
    return model, host, input_ids, d


def _make_target(d, latent_width=8, n_labels=4, seed=0):
    from saeforge.eval.targets import DownstreamCapabilityTarget

    rng = np.random.default_rng(seed)
    W_enc = torch.from_numpy(
        rng.standard_normal((latent_width, d)).astype(np.float32) * 0.1
    )
    b_enc = torch.zeros(latent_width)
    encoder = lambda x: x @ W_enc.T + b_enc  # noqa: E731
    labels = rng.integers(0, 2, size=(5, n_labels)).astype(np.uint8)
    # Guarantee every column has at least one positive and one negative.
    labels[0] = 1
    labels[-1] = 0
    return DownstreamCapabilityTarget(encoder=encoder, labels=labels), encoder


def _host_reference_pf_auc(host, input_ids, encoder, labels,
                           aggregator="pool_then_encode"):
    """Per-label best-feature AUC computed straight from the *host* model.

    Mirrors ``DownstreamCapabilityTarget.score``'s pipeline on an
    identity basis — where the host's last_hidden_state *is* the basis
    coordinate, so no decode step is needed. This is the
    version-independent reference the forge must reproduce exactly: on an
    identity basis the whole forge construction (walk → native config →
    forward → W_dec decode) is a no-op, so forge's per-label AUC must
    equal this array regardless of whether the random labels happen to be
    linearly separable in a given torch/transformers build.
    """
    from saeforge.eval.targets.downstream_capability import _best_auc_per_feature

    rows = []
    host.eval()
    with torch.no_grad():
        for i in range(int(input_ids.shape[0])):
            h = host(input_ids[i:i + 1]).last_hidden_state[0, 1:-1, :].float()
            if aggregator == "pool_then_encode":
                z = encoder(h.mean(dim=0, keepdim=True)).squeeze(0)
            else:  # encode_then_pool
                z = encoder(h).mean(dim=0)
            rows.append(z.detach().cpu())
    Z = torch.stack(rows, dim=0).numpy()
    return _best_auc_per_feature(Z, labels)


def test_identity_basis_yields_perfect_score():
    """W_dec = I + same encoder on host and forge → the forge is an exact
    no-op: its reconstructed activations equal host's last_hidden_state
    bit-for-bit, so the reported per-label AUC and score match the host
    reference. (Replaces the flaky ``score == 1.0`` — see module docstring.)
    """
    model, host, input_ids, d = _identity_forge()
    target, encoder = _make_target(d)
    score, perp = target.score(
        forged=model, host=host,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )

    # Primary, full-teeth invariant: on an identity basis the forge
    # reconstructs host activations exactly. Any forge divergence shows up
    # here, whereas the coarse 5-point AUC summary can mask it.
    fm = model.torch_module.eval()
    host.eval()
    with torch.no_grad():
        for i in range(int(input_ids.shape[0])):
            row = input_ids[i:i + 1]
            forge_act = fm(row)[0, 1:-1, :].float().numpy()
            host_act = host(row).last_hidden_state[0, 1:-1, :].float().numpy()
            np.testing.assert_allclose(
                forge_act, host_act, atol=1e-6,
                err_msg=f"forge must reproduce host activations (row {i})",
            )

    # Scorer-path coverage: the reported AUC/score equal the host reference.
    host_ref = _host_reference_pf_auc(host, input_ids, encoder, target.labels)
    assert target.forge_pf_auc is not None
    assert target.forge_pf_auc.shape == (4,)
    np.testing.assert_allclose(target.forge_pf_auc, host_ref, atol=1e-6)
    assert score == pytest.approx(float(np.nanmean(host_ref)), abs=1e-6)
    assert perp == pytest.approx(max(0.0, 1.0 - score), abs=1e-6)


def test_path_b_default_no_pinv_warning():
    """basis_decode buffer is the default decode path; no pinv warning."""
    model, host, input_ids, d = _identity_forge()
    target, _ = _make_target(d)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        target.score(
            forged=model, host=host,
            ctx={"_eval_input_ids": input_ids, "device": "cpu"},
        )
    pinv_msgs = [str(w.message) for w in w_list if "pinv" in str(w.message).lower()]
    assert not pinv_msgs, (
        f"Expected no pinv-fallback warning; got: {pinv_msgs!r}"
    )


def test_warn_on_pinv_false_silences_fallback_warning():
    """warn_on_pinv=False suppresses the path-(c) warning while still
    using the pinv fallback (added per #76 review)."""
    from saeforge.eval.targets import DownstreamCapabilityTarget

    model, host, input_ids, d = _identity_forge()
    # Force path (c) by zeroing the buffer.
    model.torch_module.basis_decode.zero_()

    rng = np.random.default_rng(0)
    W_enc = torch.from_numpy(rng.standard_normal((8, d)).astype(np.float32) * 0.1)
    b_enc = torch.zeros(8)
    encoder = lambda x: x @ W_enc.T + b_enc  # noqa: E731
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.uint8)

    target = DownstreamCapabilityTarget(
        encoder=encoder, labels=labels, warn_on_pinv=False,
    )
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        target.score(
            forged=model, host=host,
            ctx={"_eval_input_ids": input_ids, "device": "cpu"},
        )
    pinv_msgs = [str(w.message) for w in w_list if "pinv" in str(w.message).lower()]
    assert not pinv_msgs, (
        f"warn_on_pinv=False should silence the fallback warning; "
        f"got: {pinv_msgs!r}"
    )


def test_path_c_pinv_fallback_emits_warning():
    """Forged module without basis_decode falls back to pinv with warning."""
    model, host, input_ids, d = _identity_forge()
    # Zero out basis_decode to force path (c). The pinv recovery on
    # the identity basis_encode will reproduce W_dec=I correctly.
    model.torch_module.basis_decode.zero_()
    target, _ = _make_target(d)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        target.score(
            forged=model, host=host,
            ctx={"_eval_input_ids": input_ids, "device": "cpu"},
        )
    pinv_msgs = [str(w.message) for w in w_list if "pinv" in str(w.message).lower()]
    assert len(pinv_msgs) >= 1, (
        f"Expected at least one pinv-fallback warning; got: "
        f"{[str(w.message) for w in w_list]!r}"
    )


def test_path_a_explicit_basis_in_ctx_wins():
    """ctx['basis'] precedes both buffer and pinv paths."""
    model, host, input_ids, d = _identity_forge()
    # Corrupt the buffer; ctx['basis'] should still win.
    model.torch_module.basis_decode.zero_()
    from saeforge.basis import FeatureBasis
    basis = FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64),
        W_dec=np.eye(d, dtype=np.float64),
        merged_norms=np.ones(d, dtype=np.float64),
        original_norms=np.ones(d, dtype=np.float64),
    )
    target, encoder = _make_target(d)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        score, _ = target.score(
            forged=model, host=host,
            ctx={"_eval_input_ids": input_ids, "device": "cpu", "basis": basis},
        )
    pinv_msgs = [str(w.message) for w in w_list if "pinv" in str(w.message).lower()]
    assert not pinv_msgs, "ctx['basis'] should prevent pinv fallback"
    # ctx['basis'] (identity) decodes to the same coords as host → forge
    # reproduces the host-reference score exactly (not necessarily 1.0).
    host_ref = _host_reference_pf_auc(host, input_ids, encoder, target.labels)
    assert score == pytest.approx(float(np.nanmean(host_ref)), abs=1e-6)


def test_aggregator_dispatch():
    """pool_then_encode and encode_then_pool both work on identity basis;
    on identity basis they should agree because encoder is linear."""
    model, host, input_ids, d = _identity_forge()
    target_pool_enc, _ = _make_target(d)
    target_enc_pool, _ = _make_target(d)
    target_enc_pool.aggregator = "encode_then_pool"
    s1, _ = target_pool_enc.score(
        forged=model, host=host,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )
    s2, _ = target_enc_pool.score(
        forged=model, host=host,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )
    # Linear encoder + mean commute exactly: pool_then_encode and
    # encode_then_pool produce identical aggregated vectors.
    assert s1 == pytest.approx(s2, abs=1e-5)


def test_construction_validates_inputs():
    from saeforge.eval.targets import DownstreamCapabilityTarget

    with pytest.raises(TypeError, match="callable"):
        DownstreamCapabilityTarget(encoder="not a callable", labels=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="2-D"):
        DownstreamCapabilityTarget(encoder=lambda x: x, labels=np.zeros((3,)))
    with pytest.raises(ValueError, match="shape"):
        DownstreamCapabilityTarget(encoder=lambda x: x, labels=np.zeros((0, 4)))
    with pytest.raises(ValueError, match="unsupported"):
        DownstreamCapabilityTarget(
            encoder=lambda x: x, labels=np.zeros((1, 1)),
            aggregator="not_a_strategy",
        )
    with pytest.raises(ValueError, match="min_prevalence"):
        DownstreamCapabilityTarget(
            encoder=lambda x: x, labels=np.zeros((1, 1)),
            min_prevalence=-1,
        )


def test_protocol_metadata():
    from saeforge.eval.targets import DownstreamCapabilityTarget

    assert DownstreamCapabilityTarget.name == "downstream_capability"
    assert DownstreamCapabilityTarget.better_when == "higher"


def test_default_target_dispatch_does_not_return_capability():
    """DownstreamCapabilityTarget must never be a family default;
    it requires caller-supplied encoder + labels."""
    from saeforge.eval.targets import (
        DownstreamCapabilityTarget,
        _default_target_for,
    )

    for family in ("gpt2", "llama", "gemma2", "qwen2", "whisper_encoder", "esm2"):
        try:
            target = _default_target_for(family)
        except ValueError:
            # Family may not be registered in this env; that's fine.
            continue
        assert not isinstance(target, DownstreamCapabilityTarget), (
            f"family={family!r} unexpectedly defaults to "
            f"DownstreamCapabilityTarget"
        )


def test_prevalence_filter_drops_columns():
    """min_prevalence drops label columns whose positive count is below
    the threshold."""
    from saeforge.eval.targets import DownstreamCapabilityTarget

    model, host, input_ids, d = _identity_forge()
    rng = np.random.default_rng(0)
    W_enc = torch.from_numpy(rng.standard_normal((8, d)).astype(np.float32) * 0.1)
    b_enc = torch.zeros(8)
    encoder = lambda x: x @ W_enc.T + b_enc  # noqa: E731

    # 5 rows × 4 cols. Column 0 has 5 positives (kept under min_prevalence=3).
    # Column 1 has 1 positive (dropped). Columns 2-3 mixed.
    labels = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
    ], dtype=np.uint8)
    target = DownstreamCapabilityTarget(
        encoder=encoder, labels=labels, min_prevalence=3,
    )
    target.score(
        forged=model, host=host,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )
    # Two columns survive (col 0 n_pos=5 ≥ 3; col 3 n_pos=3 ≥ 3); col 1
    # (n_pos=1) and col 2 (n_pos=2) drop.
    assert target.forge_pf_auc.shape == (2,)
