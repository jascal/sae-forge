"""Tests for ``saeforge._positional.rope``.

Pins the RoPE helper math against HF's reference implementation
and exercises the round-trip shape invariants. The end-to-end
host-vs-forge validation (forge-vs-host distance 7.5e-7 on
identity basis) lives in ``scripts/prototype_llama_rope.py``;
this file is the unit-level pinning.

See ``openspec/changes/add-llama-family-rope/`` for the proposal,
design, and smoke results.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from saeforge._positional.rope import (  # noqa: E402
    apply_rotary_pos_emb,
    compute_rope_cache,
)


def test_compute_rope_cache_shape_and_dtype():
    cos, sin = compute_rope_cache(seq_len=16, head_dim=8, theta=10000.0)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_compute_rope_cache_first_row_identity():
    """At position t=0, cos=1 and sin=0 everywhere — RoPE is identity."""
    cos, sin = compute_rope_cache(seq_len=4, head_dim=8, theta=10000.0)
    assert torch.allclose(cos[0], torch.ones(8))
    assert torch.allclose(sin[0], torch.zeros(8))


def test_compute_rope_cache_matches_hf_reference():
    """Reference: HF's `LlamaRotaryEmbedding` produces (cos, sin) where
    inv_freq = 1 / theta**(2i/d). Validate the exact values for a small
    case so any future refactor catches drift from the reference.
    """
    cos, sin = compute_rope_cache(seq_len=4, head_dim=4, theta=10000.0)
    # head_dim=4 -> inv_freq has 2 entries: [1, 1/100].
    # emb[t, :] = [t*1, t*(1/100), t*1, t*(1/100)] (concat duplicates).
    expected_emb = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.01, 1.0, 0.01],
            [2.0, 0.02, 2.0, 0.02],
            [3.0, 0.03, 3.0, 0.03],
        ]
    )
    assert torch.allclose(cos, expected_emb.cos(), atol=1e-6)
    assert torch.allclose(sin, expected_emb.sin(), atol=1e-6)


def test_apply_rotary_pos_emb_identity_at_position_zero():
    """At seq_len=1 (single position at t=0), the rotation is identity."""
    q = torch.randn(1, 2, 1, 8)  # (B, n_heads, T, head_dim)
    k = torch.randn(1, 2, 1, 8)
    cos, sin = compute_rope_cache(seq_len=1, head_dim=8, theta=10000.0)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_apply_rotary_pos_emb_preserves_norms():
    """RoPE is a rotation — per-position L2 norm of q (and k) must be
    preserved exactly (up to float precision). Pinning this catches
    sign errors in rotate_half.
    """
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)  # (B, n_heads, T, head_dim)
    k = torch.randn(2, 4, 8, 16)
    cos, sin = compute_rope_cache(seq_len=8, head_dim=16, theta=10000.0)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    # Norms per (B, n_heads, T) should match.
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_apply_rotary_pos_emb_position_dependent():
    """Same vector at two different positions produces different rotated
    output. Pinning the "rotation depends on position" contract.
    """
    base = torch.randn(1, 1, 1, 16)
    q = base.expand(1, 1, 4, 16).clone()  # same vector at 4 positions
    k = q.clone()
    cos, sin = compute_rope_cache(seq_len=4, head_dim=16, theta=10000.0)
    q_rot, _ = apply_rotary_pos_emb(q, k, cos, sin)
    # Position 0 is identity (cos=1, sin=0). Positions 1, 2, 3 differ.
    assert torch.allclose(q_rot[0, 0, 0], base[0, 0, 0], atol=1e-6)
    assert (q_rot[0, 0, 1] - q_rot[0, 0, 0]).norm() > 1e-3
    assert (q_rot[0, 0, 2] - q_rot[0, 0, 0]).norm() > 1e-3
    assert (q_rot[0, 0, 3] - q_rot[0, 0, 0]).norm() > 1e-3


def test_compute_rope_cache_theta_scaling():
    """Larger theta -> lower frequencies -> rotation accumulates more
    slowly with position. Cos at small position with large theta
    should be very close to 1.
    """
    cos_default, _ = compute_rope_cache(seq_len=4, head_dim=8, theta=10000.0)
    cos_large, _ = compute_rope_cache(seq_len=4, head_dim=8, theta=1_000_000.0)
    # At larger theta, position-1 cos is closer to 1 than with default theta.
    assert cos_large[1].mean() > cos_default[1].mean()
