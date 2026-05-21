"""Tests for full-sae-keys-in-synth-basis.

Three behaviours under test:

  1. ``FeatureBasis`` accepts optional ``W_enc`` / ``b_enc`` / ``b_dec``
     fields and validates their shape + dtype on construction.
  2. ``from_polygram_checkpoint`` reads these keys from the safetensors
     when present and slices them to the kept-feature subset; when
     absent, all three default to ``None`` (back-compat with synth
     bases that only ship ``W_dec``).
  3. ``_write_basis_as_checkpoint`` writes all four SAE keys, with
     placeholder synthesis when the basis lacks the encoder fields,
     and records the placeholder names in the
     ``__synthesised_keys__`` safetensors-header metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from saeforge import FeatureBasis
from saeforge.forge import _write_basis_as_checkpoint


# ---------------------------------------------------------------------------
# §1 — FeatureBasis optional-field schema + validation
# ---------------------------------------------------------------------------


def _basis(n_kept: int = 4, d_model: int = 8, **extra) -> FeatureBasis:
    rng = np.random.default_rng(0)
    W_dec = rng.standard_normal((n_kept, d_model)).astype(np.float64)
    return FeatureBasis(
        kept_ids=np.arange(n_kept),
        W_dec=W_dec,
        merged_norms=np.linalg.norm(W_dec, axis=1),
        original_norms=np.linalg.norm(W_dec, axis=1),
        scale_compression_ratio=1.0,
        **extra,
    )


def test_optional_keys_default_none():
    basis = _basis()
    assert basis.W_enc is None
    assert basis.b_enc is None
    assert basis.b_dec is None


def test_optional_keys_accepted_with_correct_shapes():
    rng = np.random.default_rng(1)
    basis = _basis(
        n_kept=4, d_model=8,
        W_enc=rng.standard_normal((8, 4)).astype(np.float64),
        b_enc=rng.standard_normal(4).astype(np.float64),
        b_dec=rng.standard_normal(8).astype(np.float64),
    )
    assert basis.W_enc.shape == (8, 4)
    assert basis.b_enc.shape == (4,)
    assert basis.b_dec.shape == (8,)


def test_optional_w_enc_wrong_shape_raises():
    with pytest.raises(ValueError, match="W_enc shape"):
        _basis(W_enc=np.zeros((4, 4), dtype=np.float64))  # wrong (n_kept, n_kept)


def test_optional_b_enc_wrong_shape_raises():
    with pytest.raises(ValueError, match="b_enc shape"):
        _basis(b_enc=np.zeros(8, dtype=np.float64))  # length d_model not n_kept


def test_optional_b_dec_wrong_shape_raises():
    with pytest.raises(ValueError, match="b_dec shape"):
        _basis(b_dec=np.zeros(4, dtype=np.float64))  # length n_kept not d_model


def test_optional_dtype_cast_to_w_dec_dtype():
    """Optional fields supplied at float32 SHALL be cast to W_dec's
    dtype (float64) so downstream consumers don't see mixed dtypes."""
    basis = _basis(
        n_kept=4, d_model=8,
        W_enc=np.zeros((8, 4), dtype=np.float32),
        b_enc=np.zeros(4, dtype=np.float32),
        b_dec=np.zeros(8, dtype=np.float32),
    )
    assert basis.W_enc.dtype == np.float64
    assert basis.b_enc.dtype == np.float64
    assert basis.b_dec.dtype == np.float64


# ---------------------------------------------------------------------------
# §2 — from_polygram_checkpoint reads optional keys
# ---------------------------------------------------------------------------


def test_from_polygram_checkpoint_back_compat_w_dec_only(synthetic_compressed_sae):
    """The existing test fixture writes only W_dec. After this change,
    loading it should still succeed with all three optional fields as
    None — preserving back-compat with pre-change synth-basis files."""
    basis = FeatureBasis.from_polygram_checkpoint(
        synthetic_compressed_sae["checkpoint"]
    )
    assert basis.W_enc is None
    assert basis.b_enc is None
    assert basis.b_dec is None


def test_from_polygram_checkpoint_with_full_sae_keys(tmp_path):
    """A safetensors file that DOES carry all four keys should load
    them into the basis, sliced down to the kept-feature subset."""
    from safetensors.numpy import save_file
    import json

    rng = np.random.default_rng(2)
    n_total, d_model = 8, 16
    W_dec_full = rng.standard_normal((n_total, d_model)).astype(np.float32)
    W_dec_full[5] = 0.0
    W_dec_full[7] = 0.0
    W_enc_full = rng.standard_normal((d_model, n_total)).astype(np.float32)
    b_enc_full = rng.standard_normal(n_total).astype(np.float32)
    b_dec_full = rng.standard_normal(d_model).astype(np.float32)

    ckpt = tmp_path / "full_keys.safetensors"
    save_file(
        {
            "W_dec": W_dec_full,
            "W_enc": W_enc_full,
            "b_enc": b_enc_full,
            "b_dec": b_dec_full,
        },
        str(ckpt),
    )
    report_path = tmp_path / "full_keys_compression_report.json"
    report_path.write_text(json.dumps({
        "n_features_kept": 6,
        "n_clusters": 0,
        "clusters": [],
    }))

    basis = FeatureBasis.from_polygram_checkpoint(ckpt)
    # 6 kept features (0,1,2,3,4,6) sliced from 8 total
    assert basis.n_features == 6
    assert basis.W_enc is not None
    assert basis.W_enc.shape == (d_model, 6)  # column-sliced to kept
    assert basis.b_enc is not None
    assert basis.b_enc.shape == (6,)
    assert basis.b_dec is not None
    assert basis.b_dec.shape == (d_model,)  # invariant under feature slice
    # All optional fields cast to W_dec's dtype (float64)
    assert basis.W_enc.dtype == np.float64
    assert basis.b_enc.dtype == np.float64
    assert basis.b_dec.dtype == np.float64
    # Spot-check that the kept-feature slice matches the full-array
    # columns at kept_ids
    expected_W_enc = W_enc_full[:, basis.kept_ids].astype(np.float64)
    np.testing.assert_allclose(basis.W_enc, expected_W_enc)


# ---------------------------------------------------------------------------
# §3 — _write_basis_as_checkpoint produces all four keys + metadata
# ---------------------------------------------------------------------------


def test_write_basis_synthesises_placeholders_for_synth_basis(tmp_path):
    """A basis with only W_dec → on-disk safetensors has all four
    keys; the three placeholders are recorded in metadata."""
    from safetensors import safe_open

    basis = _basis(n_kept=4, d_model=8)
    out = tmp_path / "synth.safetensors"
    _write_basis_as_checkpoint(basis, out)

    with safe_open(str(out), framework="numpy") as f:
        keys = set(f.keys())
        md = f.metadata() or {}
    assert keys == {"W_dec", "W_enc", "b_enc", "b_dec"}
    synthesised = (md.get("__synthesised_keys__") or "").split(",")
    assert set(synthesised) == {"W_enc", "b_enc", "b_dec"}


def test_write_basis_placeholder_shapes_and_dtype(tmp_path):
    from safetensors import safe_open

    basis = _basis(n_kept=4, d_model=8)
    out = tmp_path / "synth.safetensors"
    _write_basis_as_checkpoint(basis, out)

    with safe_open(str(out), framework="numpy") as f:
        W_dec = f.get_tensor("W_dec")
        W_enc = f.get_tensor("W_enc")
        b_enc = f.get_tensor("b_enc")
        b_dec = f.get_tensor("b_dec")
    assert W_dec.shape == (4, 8)
    assert W_enc.shape == (8, 4)
    assert b_enc.shape == (4,)
    assert b_dec.shape == (8,)
    assert W_enc.dtype == W_dec.dtype == basis.W_dec.dtype
    # b_enc / b_dec are zeros (placeholder synthesis)
    np.testing.assert_array_equal(b_enc, np.zeros(4, dtype=basis.W_dec.dtype))
    np.testing.assert_array_equal(b_dec, np.zeros(8, dtype=basis.W_dec.dtype))
    # W_enc placeholder is W_dec.T
    np.testing.assert_allclose(W_enc, basis.W_dec.T)


def test_write_basis_real_keys_not_marked_synthesised(tmp_path):
    """When the basis carries real W_enc/b_enc/b_dec, the write
    SHALL NOT mark them as synthesised."""
    from safetensors import safe_open

    rng = np.random.default_rng(3)
    basis = _basis(
        n_kept=4, d_model=8,
        W_enc=rng.standard_normal((8, 4)).astype(np.float64),
        b_enc=rng.standard_normal(4).astype(np.float64),
        b_dec=rng.standard_normal(8).astype(np.float64),
    )
    out = tmp_path / "real.safetensors"
    _write_basis_as_checkpoint(basis, out)
    with safe_open(str(out), framework="numpy") as f:
        md = f.metadata() or {}
    # Empty string when all four are real
    assert (md.get("__synthesised_keys__") or "") == ""


def test_write_basis_partial_real_keys(tmp_path):
    """When some real, some synthesised, only the synthesised set is
    recorded in metadata."""
    from safetensors import safe_open

    rng = np.random.default_rng(4)
    basis = _basis(
        n_kept=4, d_model=8,
        W_enc=rng.standard_normal((8, 4)).astype(np.float64),
        # b_enc, b_dec left as None
    )
    out = tmp_path / "partial.safetensors"
    _write_basis_as_checkpoint(basis, out)
    with safe_open(str(out), framework="numpy") as f:
        md = f.metadata() or {}
    synthesised = set((md.get("__synthesised_keys__") or "").split(",")) - {""}
    assert synthesised == {"b_enc", "b_dec"}


def test_write_basis_round_trip_real_via_from_polygram_checkpoint(tmp_path):
    """Write a basis with real keys → read back via
    from_polygram_checkpoint → all four arrays round-trip bit-exactly."""
    import json

    rng = np.random.default_rng(5)
    basis = _basis(
        n_kept=4, d_model=8,
        W_enc=rng.standard_normal((8, 4)).astype(np.float64),
        b_enc=rng.standard_normal(4).astype(np.float64),
        b_dec=rng.standard_normal(8).astype(np.float64),
    )
    out = tmp_path / "real_rt.safetensors"
    _write_basis_as_checkpoint(basis, out)
    # Inject a trivial compression report so the loader picks kept_ids
    # consistently with the in-memory basis (4 kept features, none zeroed)
    (tmp_path / "real_rt_compression_report.json").write_text(json.dumps({
        "n_features_kept": 4, "n_clusters": 0, "clusters": [],
    }))

    rt = FeatureBasis.from_polygram_checkpoint(out)
    np.testing.assert_allclose(rt.W_dec, basis.W_dec)
    assert rt.W_enc is not None
    assert rt.b_enc is not None
    assert rt.b_dec is not None
    np.testing.assert_allclose(rt.W_enc, basis.W_enc)
    np.testing.assert_allclose(rt.b_enc, basis.b_enc)
    np.testing.assert_allclose(rt.b_dec, basis.b_dec)
