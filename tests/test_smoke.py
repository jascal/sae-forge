"""Smoke tests for the bootstrap-package change. No torch / polygram required."""

from __future__ import annotations

import re

import numpy as np
import pytest


def test_package_imports_without_optional_extras() -> None:
    import saeforge

    assert hasattr(saeforge, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+", saeforge.__version__)


def test_public_surface_is_frozen() -> None:
    import saeforge

    expected = {
        "FeatureBasis",
        # ForgeFailed added in 0.2.3 — surfaces FSM failures as
        # exceptions instead of silent KL=0.0 returns.
        "ForgeFailed",
        "ForgePipeline",
        "ForgeResult",
        "NativeModel",
        # ParetoFrontierRow + sweep_pareto added by add-pareto-sweep-driver.
        "ParetoFrontierRow",
        "SubspaceProjector",
        "__version__",
        "sweep_pareto",
    }
    assert set(saeforge.__all__) == expected


def _make_basis(n_kept: int = 8, d_model: int = 64):
    from saeforge import FeatureBasis

    rng = np.random.default_rng(0)
    W_dec = rng.standard_normal((n_kept, d_model)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n_kept),
        W_dec=W_dec,
        merged_norms=np.linalg.norm(W_dec, axis=1),
        original_norms=np.linalg.norm(W_dec, axis=1),
        scale_compression_ratio=1.0,
    )


def test_feature_basis_shape_validation() -> None:
    from saeforge import FeatureBasis

    rng = np.random.default_rng(1)
    W_dec = rng.standard_normal((8, 64))
    with pytest.raises(ValueError, match="merged_norms"):
        FeatureBasis(
            kept_ids=np.arange(8),
            W_dec=W_dec,
            merged_norms=np.ones(7),
            original_norms=np.ones(8),
        )


def test_feature_basis_pseudoinverse_is_cached() -> None:
    basis = _make_basis()
    a = basis.pseudoinverse()
    b = basis.pseudoinverse()
    assert a is b
    assert a.shape == (basis.d_model, basis.n_features)


def test_feature_basis_summary_keys() -> None:
    basis = _make_basis()
    summary = basis.to_summary()
    assert summary["n_features"] == basis.n_features
    assert summary["d_model"] == basis.d_model
    assert "scale_compression_ratio" in summary


def test_subspace_projector_roundtrip() -> None:
    from saeforge import SubspaceProjector

    basis = _make_basis()
    projector = SubspaceProjector(basis)
    z = np.eye(basis.n_features)
    reconstructed = projector.encode(projector.decode(z))
    assert np.allclose(reconstructed, z, atol=1e-6)


def test_subspace_projector_rejects_non_positive_scale_boost() -> None:
    from saeforge import SubspaceProjector

    basis = _make_basis()
    with pytest.raises(ValueError, match="scale_boost"):
        SubspaceProjector(basis, scale_boost=0.0)


def test_native_model_config_constructs() -> None:
    from saeforge.model import NativeModelConfig

    config = NativeModelConfig(
        family="gpt2", hidden_size=64,
        qkv_inner_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        vocab_size=50257,
    )
    assert config.hidden_size == 64
    assert config.num_heads * config.head_dim == config.qkv_inner_size


def test_from_polygram_checkpoint_missing_file_raises() -> None:
    from saeforge import FeatureBasis

    with pytest.raises(FileNotFoundError):
        FeatureBasis.from_polygram_checkpoint("does-not-exist.safetensors")


def test_forge_pipeline_run_requires_host_model_id() -> None:
    # ``pipeline.run()`` reaches the host_model_id ValueError only after
    # touching transformers; without the [torch] extra installed, the
    # import fails first. Pin the validation behaviour for torch-equipped
    # runners and let the no-extras install skip cleanly.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from saeforge import ForgePipeline, SubspaceProjector

    basis = _make_basis()
    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(basis=basis, projector=projector)
    with pytest.raises(ValueError, match="host_model_id"):
        pipeline.run("does-not-matter/")


def test_cli_parser_builds() -> None:
    from saeforge.cli import _build_parser

    parser = _build_parser()
    assert parser.prog == "sae-forge"


def test_cli_version_exits_zero(capsys) -> None:
    from saeforge.cli import main

    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    captured = capsys.readouterr()
    assert "sae-forge" in captured.out
