"""SAE hook-point → polygram ``layer`` alignment.

polygram interprets ``layer=N`` as the **input** to transformer block N —
it registers a ``forward_pre_hook`` on ``model.<...>.layers[N]``, so the
captured residual stream is ``blocks.N.hook_resid_pre``. sae-forge's own
calibration path (``out.hidden_states[N]``) uses the same convention.

A SAE trained on ``blocks.N.hook_resid_post`` therefore needs ``layer = N+1``
(a block's ``resid_post`` *is* the next block's ``resid_pre``), while a
``blocks.N.hook_resid_pre`` SAE needs ``layer = N``. Getting this wrong does
**not** raise — the forge still runs, but the basis is measured one or more
blocks away from where the SAE was trained and faithfulness silently
degrades. This module turns that silent footgun into a ``UserWarning``.

Everything here is pure-Python (``json`` + ``re`` + ``warnings``); it does
not import torch or transformers, so it is safe to call from the no-torch
inspection paths.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

__all__ = [
    "expected_polygram_layer",
    "resolve_hook_name",
    "check_sae_layer_alignment",
]

# Matches the SAELens residual-stream hook naming, e.g.
# ``blocks.6.hook_resid_post`` / ``blocks.8.hook_resid_pre``.
_HOOK_RE = re.compile(r"blocks\.(\d+)\.hook_resid_(pre|post|mid)")


def expected_polygram_layer(hook_name: str | None) -> int | None:
    """Return the polygram ``layer`` index matching a SAE's ``hook_name``.

    - ``blocks.N.hook_resid_pre``  → ``N``   (polygram hooks resid_pre of N)
    - ``blocks.N.hook_resid_post`` → ``N+1`` (== resid_pre of block N+1)
    - ``blocks.N.hook_resid_mid``  → ``None`` (not a clean block boundary)
    - anything else / unparseable  → ``None``

    ``None`` means "can't advise" — callers should skip the check rather
    than guess.
    """
    if not hook_name:
        return None
    m = _HOOK_RE.search(hook_name)
    if m is None:
        return None
    block, kind = int(m.group(1)), m.group(2)
    if kind == "pre":
        return block
    if kind == "post":
        return block + 1
    # resid_mid sits between attention and MLP — polygram only hooks the
    # block input, so there is no faithful single-layer mapping.
    return None


def resolve_hook_name(sae_path: str | Path | None) -> str | None:
    """Best-effort recovery of a SAE's ``hook_name``.

    Tries, in order:

    1. A sibling ``cfg.json`` (SAELens drops one next to ``sae_weights``)
       and reads its ``hook_name`` field.
    2. The path string itself (a ``blocks.N.hook_resid_*`` directory/file
       component, as published SAE repos are typically laid out).

    Returns ``None`` when neither yields a residual-stream hook name.
    """
    if sae_path is None:
        return None
    p = Path(sae_path)

    # 1. sibling cfg.json. If the path points at a file (e.g.
    #    ``.../sae_weights.safetensors``) look in its parent; if a dir,
    #    look inside it.
    search_dir = p.parent if (p.suffix or p.is_file()) else p
    cfg = search_dir / "cfg.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text())
            hook = data.get("hook_name")
            if isinstance(hook, str) and hook:
                return hook
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # fall through to the path-string heuristic

    # 2. path-string heuristic.
    m = _HOOK_RE.search(str(p))
    if m is not None:
        return m.group(0)
    return None


def check_sae_layer_alignment(
    hook_name: str | None,
    layer: int | None,
    *,
    sae_label: str | None = None,
    stacklevel: int = 2,
) -> int | None:
    """Warn if ``layer`` doesn't match the SAE's ``hook_name``.

    Returns the expected polygram layer (or ``None`` when no advice is
    possible). Emits a :class:`UserWarning` — never raises — so callers can
    drop it in front of any polygram layer-hooking call without changing
    control flow.
    """
    expected = expected_polygram_layer(hook_name)
    if expected is None or layer is None:
        return expected
    if int(layer) != expected:
        where = f" for {sae_label}" if sae_label else ""
        warnings.warn(
            f"SAE hook point {hook_name!r}{where} corresponds to polygram "
            f"layer={expected}, but layer={int(layer)} was supplied. "
            f"polygram's layer=N hooks resid_pre of block N, so a "
            f"hook_resid_post SAE needs layer = block + 1. This does not "
            f"error, but the basis will be measured a different block from "
            f"the SAE's training point and faithfulness will silently "
            f"degrade — pass layer={expected} to match the SAE.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return expected
