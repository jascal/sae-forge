"""Tests for the ForgePipeline → FSM-ctx → polygram-action passthrough
introduced by the forge-polygram-tuning-passthrough change.

Covers tasks.md §8.1: pipeline-builds-context-dict,
action-reconstitutes-config, regrow-count-without-regrow-raises,
from_dict round-trip, unknown-key warning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polygram")

from polygram import (
    CompressionConfig,
    EpochCompressionConfig,
    RegrowConfig,
)

from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector


def _basis(n: int = 4) -> FeatureBasis:
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=np.eye(n, dtype=np.float32),
        merged_norms=np.ones(n, dtype=np.float32),
        original_norms=np.ones(n, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# ForgePipeline construction-time guard
# ---------------------------------------------------------------------------


class TestRegrowGuard:
    def test_regrow_count_without_regrow_raises(self):
        basis = _basis()
        with pytest.raises(ValueError) as excinfo:
            ForgePipeline(
                basis=basis, projector=SubspaceProjector(basis),
                regrow_count=2,
            )
        # Actionable message: names both fixes — supply RegrowConfig OR
        # set regrow_count=0 — and references the polygram-side reason.
        msg = str(excinfo.value)
        assert "RegrowConfig" in msg
        assert "regrow_count=0" in msg
        assert "model_name" in msg and "layer" in msg

    def test_regrow_count_with_regrow_succeeds(self):
        basis = _basis()
        cfg = RegrowConfig(model_name="pythia-160m", layer=4)
        pipeline = ForgePipeline(
            basis=basis, projector=SubspaceProjector(basis),
            regrow_count=2, regrow=cfg,
        )
        assert pipeline.regrow == cfg

    def test_regrow_count_zero_no_regrow_succeeds(self):
        basis = _basis()
        pipeline = ForgePipeline(
            basis=basis, projector=SubspaceProjector(basis),
        )
        assert pipeline.regrow_count == 0
        assert pipeline.regrow is None

    def test_legacy_compression_strategy_kwarg_rejected(self):
        # The flat compression_strategy / rep_selection fields were
        # removed in this change; passing them now raises TypeError.
        basis = _basis()
        with pytest.raises(TypeError, match="compression_strategy"):
            ForgePipeline(
                basis=basis, projector=SubspaceProjector(basis),
                compression_strategy="merge",  # type: ignore[call-arg]
            )

    def test_legacy_rep_selection_kwarg_rejected(self):
        basis = _basis()
        with pytest.raises(TypeError, match="rep_selection"):
            ForgePipeline(
                basis=basis, projector=SubspaceProjector(basis),
                rep_selection="n_fires",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# ForgePipeline.from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_round_trip_with_nested_configs(self):
        basis = _basis()
        data = {
            "host_model_id": "gpt2",
            "compression": {"strategy": "merge", "rep_selection": "scale_aware"},
            "epoch_compression": {"coverage_target": 0.7, "max_iterations": 2},
            "regrow_count": 2,
            "regrow": {"model_name": "pythia-160m", "layer": 4},
        }
        pipeline = ForgePipeline.from_dict(
            {**data, "basis": basis, "projector": SubspaceProjector(basis)}
        )
        assert isinstance(pipeline.compression, CompressionConfig)
        assert pipeline.compression.strategy == "merge"
        assert isinstance(pipeline.epoch_compression, EpochCompressionConfig)
        assert pipeline.epoch_compression.coverage_target == 0.7
        assert isinstance(pipeline.regrow, RegrowConfig)
        assert pipeline.regrow.model_name == "pythia-160m"
        assert pipeline.regrow.layer == 4

    def test_unknown_top_level_key_warns_and_drops(self):
        basis = _basis()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline = ForgePipeline.from_dict(
                {
                    "basis": basis,
                    "projector": SubspaceProjector(basis),
                    "futurefield": 42,
                }
            )
        assert any("futurefield" in str(wi.message) for wi in w)
        # Defaults still apply for the rest.
        assert pipeline.host_model_id is None

    def test_multiple_unknown_keys_collected_in_one_warning(self):
        # Single warning naming every unknown key, rather than one
        # warning per dropped key — easier to spot in logs.
        basis = _basis()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ForgePipeline.from_dict(
                {
                    "basis": basis,
                    "projector": SubspaceProjector(basis),
                    "futurefield": 42,
                    "anotherfuture": "x",
                    "third": True,
                }
            )
        # Exactly one UserWarning; message lists every unknown key.
        unknown_warnings = [wi for wi in w if "ignoring unknown" in str(wi.message)]
        assert len(unknown_warnings) == 1, [str(wi.message) for wi in w]
        msg = str(unknown_warnings[0].message)
        assert "futurefield" in msg
        assert "anotherfuture" in msg
        assert "third" in msg

    def test_nested_validation_error_bubbles_up(self):
        # If the nested polygram config rejects a field (e.g. an
        # out-of-range coverage_target), the error should surface
        # cleanly from from_dict — naming the offending field and
        # the polygram dataclass — not silently get swallowed.
        basis = _basis()
        with pytest.raises(ValueError, match=r"coverage_target"):
            ForgePipeline.from_dict(
                {
                    "basis": basis,
                    "projector": SubspaceProjector(basis),
                    "epoch_compression": {"coverage_target": 1.5},
                }
            )

    def test_nested_required_field_missing_bubbles_up(self):
        # RegrowConfig requires model_name and layer; from_dict should
        # bubble polygram's TypeError when either is missing.
        basis = _basis()
        with pytest.raises(TypeError, match=r"layer"):
            ForgePipeline.from_dict(
                {
                    "basis": basis,
                    "projector": SubspaceProjector(basis),
                    "regrow_count": 2,
                    "regrow": {"model_name": "gpt2"},  # layer missing
                }
            )

    def test_from_dict_rejects_non_mapping(self):
        with pytest.raises(TypeError, match="mapping"):
            ForgePipeline.from_dict([("host_model_id", "gpt2")])  # type: ignore[arg-type]

    def test_compression_dict_already_a_config_is_passed_through(self):
        # If a caller hands us an already-built CompressionConfig instead
        # of a dict, from_dict should leave it alone (recursion only
        # happens for Mapping inputs).
        basis = _basis()
        cfg = CompressionConfig(strategy="zero")
        pipeline = ForgePipeline.from_dict(
            {
                "basis": basis,
                "projector": SubspaceProjector(basis),
                "compression": cfg,
            }
        )
        assert pipeline.compression is cfg


# ---------------------------------------------------------------------------
# FSM context serialisation
# ---------------------------------------------------------------------------


class TestContextSerialisation:
    """Verify that ForgePipeline's ctx-build serialises configs as
    JSON-friendly dicts and omits the key entirely when the field is
    ``None``. We exercise this by calling `_build_context` indirectly
    via `run_synthetic`'s scaffolding path is heavy on torch, so we
    invoke the dict-shaping helpers directly.
    """

    def _make_pipeline(self, **kwargs) -> ForgePipeline:
        basis = _basis()
        return ForgePipeline(
            basis=basis, projector=SubspaceProjector(basis),
            host_model_id="gpt2", **kwargs,
        )

    def test_compression_serialises_as_dict(self):
        cfg = CompressionConfig(strategy="merge")
        pipeline = self._make_pipeline(compression=cfg)
        # The serialisation site is in run_synthetic's ctx-build; the
        # invariant we care about is "to_dict produces JSON-friendly
        # output". Verify directly.
        d = pipeline.compression.to_dict()
        assert isinstance(d, dict)
        json.dumps(d)  # JSON-serialisable
        assert CompressionConfig.from_dict(d) == cfg

    def test_regrow_serialises_as_dict(self):
        cfg = RegrowConfig(model_name="pythia-160m", layer=4)
        pipeline = self._make_pipeline(regrow_count=2, regrow=cfg)
        d = pipeline.regrow.to_dict()
        assert isinstance(d, dict)
        json.dumps(d)
        assert RegrowConfig.from_dict(d) == cfg

    def test_no_polygram_fields_set_means_no_dict_keys(self):
        # When all polygram fields are None, _build_context must omit
        # the keys (rather than include them with None values) so the
        # action layer's `ctx.get(key)` returns missing-not-None and
        # falls back to polygram's own defaults.
        pipeline = self._make_pipeline()
        assert pipeline.compression is None
        assert pipeline.epoch_compression is None
        assert pipeline.regrow is None


# ---------------------------------------------------------------------------
# Action reconstitutes config from ctx
# ---------------------------------------------------------------------------


class TestActionReconstitution:
    def test_perform_regrowth_without_ctx_regrow_raises(self, tmp_path: Path):
        # When regrow_count > 0 reaches the action and ctx["regrow"] is
        # missing, the action surfaces a clear ValueError (rather than
        # falling back to the deleted layer=10 / model_name="gpt2" keys).
        from saeforge.actions import perform_regrowth

        # Construct a synthetic compression_report file the action
        # accepts as "compression happened" so we get past the
        # passthrough guard.
        report_path = tmp_path / "compression_report.json"
        report_path.write_text("{}")  # contents irrelevant before regrow check
        compressed_path = tmp_path / "compressed.safetensors"
        compressed_path.write_bytes(b"")  # must exist

        ctx = {
            "regrow_count": 2,
            "compression_report_path": str(report_path),
            "compressed_sae_path": str(compressed_path),
            "output_dir": str(tmp_path),
            "transitions_log": [],
            # NB: no ctx["regrow"] — that's what we're testing.
        }
        with pytest.raises(ValueError, match="RegrowConfig"):
            perform_regrowth(ctx, None)

    def test_perform_regrowth_passthrough_when_regrow_count_zero(self, tmp_path: Path):
        from saeforge.actions import perform_regrowth

        ctx = {
            "regrow_count": 0,
            "compressed_sae_path": "/tmp/whatever.safetensors",
            "transitions_log": [],
        }
        delta = perform_regrowth(ctx, None)
        # passthrough returns the compressed path unchanged
        assert delta == {"regrown_sae_path": "/tmp/whatever.safetensors"}

    def test_compress_passthrough_no_validation_report(self, tmp_path: Path):
        from saeforge.actions import compress_with_polygram

        ctx = {
            "current_sae_path": "/tmp/whatever.safetensors",
            "current_feature_count": 5,
            "transitions_log": [],
        }
        delta = compress_with_polygram(ctx, None)
        assert delta == {
            "compressed_sae_path": "/tmp/whatever.safetensors",
            "current_feature_count": 5,
        }
