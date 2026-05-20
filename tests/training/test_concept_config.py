"""TrainingConfig validation tests for concept-anchoring fields."""

from __future__ import annotations

import pytest

from saeforge.training.config import TrainingConfig


def test_concept_alpha_default_zero():
    cfg = TrainingConfig()
    assert cfg.concept_alpha == 0.0
    assert cfg.concept_pool_weight == 1.0
    assert cfg.concept_channel_weight == 1.0
    assert cfg.concept_focal_gamma == 2.0
    assert cfg.concept_label_source == "polygram-clusters"
    assert cfg.concept_label_source_kwargs == {}


def test_concept_alpha_out_of_range_rejected():
    with pytest.raises(ValueError, match="concept_alpha"):
        TrainingConfig(concept_alpha=-0.1)
    with pytest.raises(ValueError, match="concept_alpha"):
        TrainingConfig(concept_alpha=1.5)


def test_concept_alpha_positive_requires_nonzero_weight():
    with pytest.raises(ValueError, match="at least one of"):
        TrainingConfig(
            concept_alpha=0.5,
            concept_pool_weight=0.0,
            concept_channel_weight=0.0,
        )


def test_concept_focal_gamma_negative_rejected():
    with pytest.raises(ValueError, match="concept_focal_gamma"):
        TrainingConfig(concept_focal_gamma=-0.5)


def test_concept_pool_weight_negative_rejected():
    with pytest.raises(ValueError, match="concept_pool_weight"):
        TrainingConfig(concept_pool_weight=-0.1)


def test_concept_channel_weight_negative_rejected():
    with pytest.raises(ValueError, match="concept_channel_weight"):
        TrainingConfig(concept_channel_weight=-0.1)


def test_concept_label_source_unknown_rejected_when_alpha_positive():
    with pytest.raises(ValueError, match="not registered"):
        TrainingConfig(concept_alpha=0.5, concept_label_source="bogus")


def test_concept_label_source_not_validated_when_alpha_zero():
    # Defensive: when alpha is 0 the branch is inactive and we skip
    # registry validation to keep the default path cheap.
    cfg = TrainingConfig(concept_alpha=0.0, concept_label_source="bogus")
    assert cfg.concept_label_source == "bogus"


def test_pool_only_config_valid():
    cfg = TrainingConfig(
        concept_alpha=0.1,
        concept_pool_weight=1.0,
        concept_channel_weight=0.0,
    )
    assert cfg.concept_channel_weight == 0.0


def test_channel_only_config_valid():
    cfg = TrainingConfig(
        concept_alpha=0.1,
        concept_pool_weight=0.0,
        concept_channel_weight=1.0,
    )
    assert cfg.concept_pool_weight == 0.0
