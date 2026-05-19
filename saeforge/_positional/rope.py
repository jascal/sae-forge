"""Rotary positional embedding helpers for Llama-family forged attention.

Mirrors HuggingFace's reference implementation. Validated against a
running HF ``LlamaForCausalLM`` by the
``scripts/prototype_llama_rope.py`` smoke gate (forge-vs-host
distance 7.5e-7 on identity basis — at float precision, confirming
the math is exactly right).

See ``openspec/changes/add-llama-family-rope/`` for the proposal,
design, smoke results, and per-family rollout contract.

Pure torch; lazy-imported so the no-``[torch]`` install path stays
clean.
"""

from __future__ import annotations

from saeforge.utils.lazy import require_extra


def compute_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device=None,
    dtype=None,
):
    """Build the ``(cos, sin)`` cache for RoPE on a sequence of length ``seq_len``.

    Returns ``(cos, sin)`` both of shape ``(seq_len, head_dim)``.
    Matches HF's reference math:

    ::

        inv_freq[i] = 1 / theta**(2i/head_dim)   for i in [0, head_dim/2)
        freqs[t, i] = t * inv_freq[i]
        emb         = concat(freqs, freqs)        # along the last axis
        cos         = emb.cos()
        sin         = emb.sin()

    The "duplicate then rotate-half" pattern is HF's encoding of the
    `(x_i, x_{i+d/2})` 2D rotation per pair of dimensions. See
    ``apply_rotary_pos_emb`` for the application step.
    """
    torch = require_extra("torch", "torch")
    if dtype is None:
        dtype = torch.float32
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
        )
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    """HF-compatible rotate-half: split the last axis in two halves and
    return ``concat(-second, first)``. Pairs with the duplicated
    ``(cos, sin)`` cache produced by :func:`compute_rope_cache`."""
    torch = require_extra("torch", "torch")
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to Q and K.

    Args:
        q, k: ``(..., n_heads, seq_len, head_dim)`` query and key tensors.
        cos, sin: ``(seq_len, head_dim)`` rotation cache from
            :func:`compute_rope_cache`. Broadcast across the leading
            batch and head dims.

    Returns:
        Rotated ``(q_rot, k_rot)`` with the same shapes as ``q`` and ``k``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
