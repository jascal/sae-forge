"""Polygram concept-structure diagnostics.

Surfaces polygram-compressor signals onto each ``ParetoFrontierRow``:
``polygram_n_clusters``, ``polygram_n_zeroed``,
``polygram_redundancy_ratio``, and ``polygram_encoding_capacity``. The
first three describe how many distinct concepts the compressed
dictionary encodes and how concentrated those concepts are; the fourth
describes the encoding's structural cap (Rung3=16, Rung4=32, Rung5=128,
HEA_Rung2(n)=2**n).

This is the *content* counterpart to ``saeforge.forge_quality``'s
*structural* rank-ratio diagnostics. See
``openspec/changes/add-polygram-cluster-diagnostics/proposal.md`` for
the motivating empirical case (econ-sae Phase 7.2, supervised vs
unsupervised SAE at Rung5 cap=128) and ``AGENTS.md``'s "Polygram
dependency contract" section for the upstream report schema this
module relies on.

The module is intentionally thin: ``load_polygram_report`` wraps the
suffix-list logic that ``saeforge.basis._locate_report`` already
exposes; ``compute_redundancy_ratio`` is a pure-function arithmetic
helper; ``resolve_encoding_capacity`` parses the encoding spec string
the sweep CLI accepts; ``format_saturation_note`` formats the advisory
template surfaced by ``saeforge.forge_quality.advise_sweep_quality``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from saeforge.basis import _CANDIDATE_REPORT_SUFFIXES

_LOGGER = logging.getLogger(__name__)

# Encoding spec parsing. Bare rungs are case-insensitive; the parametric
# ``HEA_Rung2`` form accepts ``n_qubits=N`` or ``n_qubits:N`` (matching
# the LABEL:VALUE separator used by other sweep flags), tolerates
# whitespace, and is also case-insensitive.
_BARE_RUNG_CAPACITIES: dict[str, int] = {
    "rung3": 16,
    "rung4": 32,
    "rung5": 128,
}
_HEA_RUNG2_PATTERN = re.compile(
    r"^\s*hea_rung2\s*\(\s*n_qubits\s*[:=]\s*(\d+)\s*\)\s*$",
    re.IGNORECASE,
)


def load_polygram_report(checkpoint_path: str | Path) -> dict | None:
    """Load the polygram ``compression_report.json`` colocated with a SAE.

    ``checkpoint_path`` is the polygram-compressed ``.safetensors`` file.
    The companion report is located by trying the suffix variants
    ``saeforge.basis`` already uses (``_compression_report.json``,
    ``.compression_report.json``, ``_report.json``) on the checkpoint's
    stem.

    Returns the parsed JSON dict on success; ``None`` on any failure
    (missing report file, JSON decode error, unreadable file). Failures
    are logged at INFO — the sweep should proceed with the polygram
    diagnostic fields populated as ``None`` rather than aborting.
    """
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    stem = path.with_suffix("")
    candidate: Path | None = None
    for suffix in _CANDIDATE_REPORT_SUFFIXES:
        c = Path(str(stem) + suffix)
        if c.is_file():
            candidate = c
            break
    if candidate is None:
        _LOGGER.info(
            "saeforge.polygram_diagnostics: no compression report found "
            "for %s (tried suffixes %s)",
            path,
            _CANDIDATE_REPORT_SUFFIXES,
        )
        return None
    try:
        return json.loads(candidate.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _LOGGER.info(
            "saeforge.polygram_diagnostics: failed to load report %s (%s); "
            "polygram diagnostics will be None for this row",
            candidate,
            exc,
        )
        return None


def compute_redundancy_ratio(
    n_clusters: int | None,
    n_zeroed: int | None,
) -> float | None:
    """Return ``n_zeroed / (n_clusters + n_zeroed)``, or ``None``.

    Returns a non-None float only when both inputs are non-None,
    non-negative, and their sum is strictly positive. The both-zero
    case returns ``None`` (no signal — the dictionary had no
    compressible structure to report on).
    """
    if n_clusters is None or n_zeroed is None:
        return None
    n_c = int(n_clusters)
    n_z = int(n_zeroed)
    if n_c < 0 or n_z < 0:
        return None
    total = n_c + n_z
    if total <= 0:
        return None
    return float(n_z) / float(total)


def resolve_encoding_capacity(encoding_spec: str) -> int | None:
    """Return the encoding's structural capacity, or ``None``.

    Supported spec strings (case-insensitive, whitespace-tolerant):

    - ``"rung3"`` → 16
    - ``"rung4"`` → 32
    - ``"rung5"`` → 128
    - ``"hea_rung2(n_qubits=N)"`` / ``"HEA_Rung2(n_qubits:N)"`` → ``2 ** N``

    Anything else returns ``None``. This is conservative on purpose:
    new encoding families that polygram may add downstream parse to
    ``None`` here, the row's capacity field stays ``None``, and the
    saturation advisory does not fire (no false positives).
    """
    if encoding_spec is None:
        return None
    s = str(encoding_spec).strip()
    if not s:
        return None
    lower = s.lower()
    if lower in _BARE_RUNG_CAPACITIES:
        return _BARE_RUNG_CAPACITIES[lower]
    m = _HEA_RUNG2_PATTERN.match(s)
    if m is not None:
        n_qubits = int(m.group(1))
        if n_qubits < 0:
            return None
        return 1 << n_qubits  # 2 ** n_qubits
    return None


def format_saturation_note(
    n_clusters: int,
    capacity: int,
    suggested_next_encoding: str,
) -> str:
    """Format the cluster-saturation advisory line.

    The wording is normative — the spec freezes it. ``advise_sweep_quality``
    is the only intended caller.
    """
    return (
        f"Note: polygram_n_clusters ({n_clusters}) equals encoding "
        f"capacity ({capacity}) — the encoding may be saturated. "
        f"Consider re-running polygram compress with a larger encoding "
        f"({suggested_next_encoding}) to see whether additional concepts "
        f"are present."
    )


__all__ = [
    "compute_redundancy_ratio",
    "load_polygram_report",
    "resolve_encoding_capacity",
]
