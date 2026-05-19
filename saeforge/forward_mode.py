"""Forward-mode dispatch for the forged transformer.

`forward_mode` selects between two implementations of the forged
transformer's forward pass:

- ``"native_in_basis"`` runs every op (LayerNorm, attention, MLP) in
  the basis-space residual stream using algebraically projected
  parameters. The existing v0.5.1 forward path. Mathematically
  faithful when the basis is high-fidelity (``quality_tier ∈ {good,
  saturated}``).
- ``"host_wrapped"`` runs every op host-native (host's exact,
  unprojected weights) with the residual stream encoded/decoded at
  every block boundary. Bounded by basis-approximation error rather
  than projection-amplification; appropriate for under-complete
  bases where the algebraic projection of LayerNorm parameters is a
  category error.

The default ``"auto"`` resolves via ``basis.quality_tier``:
``good``/``saturated`` → ``native_in_basis``; ``undersized``/
``degenerate`` → ``host_wrapped``.

See ``openspec/changes/add-host-wrapped-forge-fallback`` for the
falsifiable acceptance gate and prototype results.
"""

from __future__ import annotations

import logging
from typing import Literal

from saeforge.basis import FeatureBasis
from saeforge.forge_quality import (
    QualityTier,
    classify_quality,
    compute_basis_rank,
)

log = logging.getLogger(__name__)

ForwardModeLiteral = Literal["native_in_basis", "host_wrapped"]
_LEGAL = {"auto", "native_in_basis", "host_wrapped"}


def _classify_basis(basis: FeatureBasis) -> tuple[int, QualityTier]:
    """Return ``(basis_rank, quality_tier)`` for a feature basis."""
    rank = compute_basis_rank(basis.W_dec)
    _, tier = classify_quality(rank, basis.d_model)
    return rank, tier


def resolve_forward_mode(
    basis: FeatureBasis, requested: str = "auto"
) -> ForwardModeLiteral:
    """Resolve ``forward_mode`` for a (basis, request) pair.

    Pure function of basis + request. Does not load or run the host
    model. When ``requested == "auto"``, returns ``"native_in_basis"``
    for ``quality_tier ∈ {good, saturated}`` and ``"host_wrapped"``
    otherwise. When ``requested`` is one of ``"native_in_basis"`` /
    ``"host_wrapped"``, returns it unchanged.

    Logs the resolution at INFO once per call when source was
    ``"auto"``.
    """
    if requested not in _LEGAL:
        raise ValueError(
            f"forward_mode must be one of {sorted(_LEGAL)}; got {requested!r}"
        )
    if requested != "auto":
        return requested  # type: ignore[return-value]
    rank, tier = _classify_basis(basis)
    if tier in (QualityTier.GOOD, QualityTier.SATURATED):
        resolved: ForwardModeLiteral = "native_in_basis"
    else:
        resolved = "host_wrapped"
    log.info(
        "forward_mode resolved to %r (basis_rank=%d, d_model=%d, "
        "quality_tier=%r)",
        resolved,
        rank,
        basis.d_model,
        tier.value,
    )
    return resolved
