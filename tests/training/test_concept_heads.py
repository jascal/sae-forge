"""Unit tests for `saeforge.training.heads`."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from saeforge.training.heads import (
    PerChannelConceptHead,
    PooledConceptHead,
    focal_bce_loss,
)


def test_pooled_head_shape():
    head = PooledConceptHead(d_model=8, n_concepts=4)
    out = head(torch.randn(2, 16, 8))
    assert out.shape == (2, 4)


def test_per_channel_head_shape():
    head = PerChannelConceptHead(n_concepts=6)
    out = head(torch.randn(2, 16, 6))
    assert out.shape == (2, 16, 6)


def test_per_channel_head_applies_per_concept_affine():
    head = PerChannelConceptHead(n_concepts=3)
    with torch.no_grad():
        head.scale.copy_(torch.tensor([2.0, -1.0, 0.5]))
        head.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    x = torch.tensor([[1.0, 1.0, 1.0]])
    out = head(x)
    assert torch.allclose(out, torch.tensor([[2.1, -0.8, 0.8]]))


def test_focal_bce_gamma_zero_equals_bce():
    logits = torch.tensor([[0.5, -0.3, 1.2], [-1.0, 0.0, 0.8]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    focal = focal_bce_loss(logits, labels, gamma=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_bce_gamma_two_down_weights_confident():
    """A confidently-correct prediction (p_t ≈ 0.993) gets ~5e-5×
    its plain BCE under γ=2."""
    logits = torch.tensor([[5.0]])  # sigmoid(5) ≈ 0.993
    labels = torch.tensor([[1.0]])
    focal = focal_bce_loss(logits, labels, gamma=2.0)
    plain = focal_bce_loss(logits, labels, gamma=0.0)
    ratio = focal.item() / plain.item()
    # Expected ratio ≈ (1 - sigmoid(5)) ** 2 ≈ 4.5e-5
    assert ratio < 1e-3, (focal, plain, ratio)


def test_focal_bce_gradient_flows():
    logits = torch.tensor([[0.5, -0.3]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    loss = focal_bce_loss(logits, labels, gamma=2.0)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()


def test_focal_bce_reduction_none_returns_per_element():
    logits = torch.zeros(3, 4)
    labels = torch.zeros(3, 4)
    out = focal_bce_loss(logits, labels, gamma=2.0, reduction="none")
    assert out.shape == (3, 4)


def test_focal_bce_negative_gamma_rejected():
    import pytest

    with pytest.raises(ValueError, match="gamma must be"):
        focal_bce_loss(torch.zeros(2), torch.zeros(2), gamma=-1.0)


def test_focal_bce_unknown_reduction_rejected():
    import pytest

    with pytest.raises(ValueError, match="reduction must be"):
        focal_bce_loss(torch.zeros(2), torch.zeros(2), gamma=0.0, reduction="bogus")
