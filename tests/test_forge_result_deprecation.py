"""ForgeResult.faithfulness_kl deprecation tests.

The property and the constructor kwarg both emit DeprecationWarning.
The property returns the stored faithfulness when the active target
is "kl"; ``None`` otherwise.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from saeforge.forge import ForgeResult


def test_property_read_when_target_is_kl_returns_value_and_warns():
    result = ForgeResult(
        model=None,
        output_dir=Path("/tmp/test"),
        n_params=0,
        faithfulness=0.123,
        faithfulness_target_name="kl",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = result.faithfulness_kl
    assert value == pytest.approx(0.123)
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert ".faithfulness" in str(caught[0].message)


def test_property_read_when_target_is_not_kl_returns_none_and_warns():
    result = ForgeResult(
        model=None,
        output_dir=Path("/tmp/test"),
        n_params=0,
        faithfulness=0.91,
        faithfulness_target_name="cosine",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = result.faithfulness_kl
    assert value is None
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning


def test_constructor_kwarg_shim_sets_fields_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ForgeResult(
            model=None,
            output_dir=Path("/tmp/test"),
            n_params=0,
            faithfulness_kl=0.2,
        )
    assert result.faithfulness == pytest.approx(0.2)
    assert result.faithfulness_target_name == "kl"
    assert any(w.category is DeprecationWarning for w in caught)


def test_property_setter_writes_to_faithfulness_and_warns():
    result = ForgeResult(
        model=None,
        output_dir=Path("/tmp/test"),
        n_params=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result.faithfulness_kl = 0.5
    assert result.faithfulness == pytest.approx(0.5)
    assert result.faithfulness_target_name == "kl"
    assert any(w.category is DeprecationWarning for w in caught)


def test_property_read_warns_on_every_access():
    """The spec uses ``warnings.warn(..., DeprecationWarning)`` without
    ``once=True`` so tests can observe the warning every time. Default
    test-runner config still filters to one-per-source-line; we
    re-enable ``always`` here to confirm the warn() call itself fires
    on each access.
    """
    result = ForgeResult(
        model=None,
        output_dir=Path("/tmp/test"),
        n_params=0,
        faithfulness=0.1,
        faithfulness_target_name="kl",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = result.faithfulness_kl
        _ = result.faithfulness_kl
        _ = result.faithfulness_kl
    deprecation_count = sum(1 for w in caught if w.category is DeprecationWarning)
    assert deprecation_count == 3
