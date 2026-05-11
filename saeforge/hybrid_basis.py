"""HybridBasisBundle — three FeatureBasis instances anchored at embed / mid / lm-head.

The bundle is the single authority on which basis projects which host weight.
Adapter helpers (``saeforge.adapters._hybrid``) and any future
multi-anchor work consult ``basis_for_layer(idx)`` rather than re-deriving
the rule. Validation rejects shape-incompatible bases at construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from saeforge.basis import FeatureBasis


@dataclass
class HybridBasisBundle:
    """Three bases bound to a host's layer count.

    Region routing:
    - layer ``0``               → ``basis_embed``
    - layer ``[1, n_layer-2]``  → ``basis_mid``
    - layer ``n_layer-1``       → ``basis_lm_head``

    Non-block keys route to whichever basis owns their adjacent region:
    ``wte`` / ``wpe`` → embed; ``ln_f`` / ``lm_head`` → lm-head.
    """

    basis_embed: FeatureBasis
    basis_mid: FeatureBasis
    basis_lm_head: FeatureBasis
    n_layer: int

    def __post_init__(self) -> None:
        d_embed = self.basis_embed.d_model
        d_mid = self.basis_mid.d_model
        d_lm = self.basis_lm_head.d_model
        if not (d_embed == d_mid == d_lm):
            raise ValueError(
                f"HybridBasisBundle: d_model mismatch — "
                f"basis_embed.d_model={d_embed}, basis_mid.d_model={d_mid}, "
                f"basis_lm_head.d_model={d_lm}"
            )
        n_embed = self.basis_embed.n_features
        n_mid = self.basis_mid.n_features
        n_lm = self.basis_lm_head.n_features
        if not (n_embed == n_mid == n_lm):
            raise ValueError(
                f"HybridBasisBundle: n_features mismatch — "
                f"basis_embed.n_features={n_embed}, basis_mid.n_features={n_mid}, "
                f"basis_lm_head.n_features={n_lm}"
            )
        if self.n_layer < 3:
            raise ValueError(
                f"HybridBasisBundle: n_layer must be >= 3 for a non-degenerate "
                f"three-region split; got n_layer={self.n_layer}"
            )

    @property
    def d_model(self) -> int:
        return self.basis_mid.d_model

    @property
    def n_features(self) -> int:
        return self.basis_mid.n_features

    @property
    def boundaries(self) -> tuple[int, int]:
        """Layer indices ``(emb_to_mid, mid_to_lm)`` where bridges insert."""
        return (0, self.n_layer - 1)

    def basis_for_layer(self, idx: int) -> FeatureBasis:
        """Return the basis whose region contains layer ``idx``."""
        if idx < 0 or idx >= self.n_layer:
            raise IndexError(
                f"HybridBasisBundle.basis_for_layer: idx {idx} out of range "
                f"[0, {self.n_layer})"
            )
        if idx == 0:
            return self.basis_embed
        if idx == self.n_layer - 1:
            return self.basis_lm_head
        return self.basis_mid
