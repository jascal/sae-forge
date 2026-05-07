"""Lazy-import helper — raise a clear actionable error when an optional extra is missing."""

from __future__ import annotations

import importlib
from typing import Any


def require_extra(module_name: str, extra: str) -> Any:
    """Import ``module_name`` or raise an ImportError naming the pip extra to install."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"sae-forge feature requires the '{extra}' extra; "
            f"install it with `pip install sae-forge[{extra}]`."
        ) from e
