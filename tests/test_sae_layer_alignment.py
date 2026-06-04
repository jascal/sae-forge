"""Unit tests for the SAE hook-point → polygram ``layer`` alignment guard.

Pure-Python: no network, no torch. Covers the resid_pre/post/mid mapping,
the warning behaviour, and ``hook_name`` recovery from cfg.json / path.
"""

import json
import warnings

import pytest

from saeforge.utils.sae_layer import (
    check_sae_layer_alignment,
    expected_polygram_layer,
    resolve_hook_name,
)


# ---- expected_polygram_layer -----------------------------------------
@pytest.mark.parametrize(
    "hook_name, expected",
    [
        ("blocks.8.hook_resid_pre", 8),       # resid_pre → same index
        ("blocks.6.hook_resid_post", 7),      # resid_post → block + 1
        ("blocks.0.hook_resid_post", 1),
        ("blocks.23.hook_resid_pre", 23),
        ("blocks.6.hook_resid_mid", None),    # no clean block boundary
        ("blocks.6.hook_mlp_out", None),      # not a residual hook
        ("transformer.h.5", None),            # unparseable
        ("", None),
        (None, None),
    ],
)
def test_expected_polygram_layer(hook_name, expected):
    assert expected_polygram_layer(hook_name) == expected


# ---- check_sae_layer_alignment: warning behaviour --------------------
def test_warns_on_resid_post_offbyone():
    # blocks.6.hook_resid_post wants layer=7; supplying 6 must warn.
    with pytest.warns(UserWarning, match=r"layer=7"):
        expected = check_sae_layer_alignment("blocks.6.hook_resid_post", 6)
    assert expected == 7


def test_no_warn_when_layer_matches_post():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        assert check_sae_layer_alignment("blocks.6.hook_resid_post", 7) == 7


def test_no_warn_when_layer_matches_pre():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_sae_layer_alignment("blocks.8.hook_resid_pre", 8) == 8


def test_warns_on_resid_pre_mismatch():
    with pytest.warns(UserWarning, match=r"layer=8"):
        check_sae_layer_alignment("blocks.8.hook_resid_pre", 5)


def test_label_appears_in_warning():
    with pytest.warns(UserWarning, match=r"my-sae"):
        check_sae_layer_alignment("blocks.6.hook_resid_post", 6, sae_label="my-sae")


@pytest.mark.parametrize("hook_name", ["blocks.6.hook_resid_mid", "blocks.6.hook_mlp_out", None])
def test_no_warn_when_unadvisable(hook_name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_sae_layer_alignment(hook_name, 3) is None


def test_no_warn_when_layer_is_none():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # expected is still computed/returned, but with no layer there's
        # nothing to compare against, so no warning.
        assert check_sae_layer_alignment("blocks.6.hook_resid_post", None) == 7


# ---- resolve_hook_name -----------------------------------------------
def test_resolve_from_cfg_json(tmp_path):
    d = tmp_path / "blocks.6.hook_resid_post"
    d.mkdir()
    (d / "cfg.json").write_text(json.dumps({"hook_name": "blocks.6.hook_resid_post"}))
    weights = d / "sae_weights.safetensors"
    weights.write_bytes(b"")
    # passed a file path → reads the sibling cfg.json
    assert resolve_hook_name(weights) == "blocks.6.hook_resid_post"
    # passed the directory → reads cfg.json inside
    assert resolve_hook_name(d) == "blocks.6.hook_resid_post"


def test_resolve_falls_back_to_path_string(tmp_path):
    # No cfg.json, but the path encodes the hook point.
    d = tmp_path / "blocks.3.hook_resid_pre"
    d.mkdir()
    weights = d / "sae_weights.safetensors"
    weights.write_bytes(b"")
    assert resolve_hook_name(weights) == "blocks.3.hook_resid_pre"


def test_resolve_prefers_cfg_over_path(tmp_path):
    # cfg.json (authoritative) disagrees with the directory name; cfg wins.
    d = tmp_path / "blocks.3.hook_resid_pre"
    d.mkdir()
    (d / "cfg.json").write_text(json.dumps({"hook_name": "blocks.9.hook_resid_post"}))
    assert resolve_hook_name(d / "sae_weights.safetensors") == "blocks.9.hook_resid_post"


def test_resolve_returns_none_when_unknown(tmp_path):
    d = tmp_path / "mystery"
    d.mkdir()
    assert resolve_hook_name(d / "sae_weights.safetensors") is None
    assert resolve_hook_name(None) is None


def test_resolve_tolerates_malformed_cfg(tmp_path):
    d = tmp_path / "blocks.4.hook_resid_post"
    d.mkdir()
    (d / "cfg.json").write_text("{not valid json")
    # malformed cfg → falls back to the path-string heuristic
    assert resolve_hook_name(d) == "blocks.4.hook_resid_post"
