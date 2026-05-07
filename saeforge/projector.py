"""SubspaceProjector — project host weights into and out of a feature basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saeforge.basis import FeatureBasis


@dataclass
class SubspaceProjector:
    """Project a host model's weights into the feature basis defined by ``basis``.

    The projection math is pure-numpy. Real host models live behind the
    ``[torch]`` extra; ``project_module(...)`` lazy-imports torch on demand.
    """

    basis: FeatureBasis
    scale_boost: float = 1.0

    def __post_init__(self) -> None:
        if self.scale_boost <= 0.0:
            raise ValueError(f"scale_boost must be positive; got {self.scale_boost}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project ``x`` (..., d_model) into the basis (..., n_features)."""
        pinv = self.basis.pseudoinverse()
        return x @ pinv * self.scale_boost

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct (..., d_model) from basis-space ``z`` (..., n_features)."""
        return z @ self.basis.W_dec

    def project_embed(self, W_embed: np.ndarray) -> np.ndarray:
        """(V, d_model) -> (V, n_features)."""
        return self.encode(W_embed)

    def project_unembed(self, W_unembed: np.ndarray) -> np.ndarray:
        """(d_model, V) -> (n_features, V)."""
        return self.basis.W_dec @ W_unembed

    def project_qkv(self, W_qkv: np.ndarray) -> np.ndarray:
        """(d_model, 3 * d_head * n_heads) -> (n_features, 3 * d_head * n_heads)."""
        return self.basis.W_dec @ W_qkv

    def project_mlp_in(self, W_in: np.ndarray) -> np.ndarray:
        """(d_model, d_ff) -> (n_features, d_ff)."""
        return self.basis.W_dec @ W_in

    def project_mlp_out(self, W_out: np.ndarray) -> np.ndarray:
        """(d_ff, d_model) -> (d_ff, n_features)."""
        return self.encode(W_out)

    def project_module(self, host_model) -> dict:
        """Project every relevant weight of a torch host model into the basis.

        Returns a dict keyed by the host module path (e.g. ``"transformer.h.0.attn.c_attn"``)
        whose values are ``np.ndarray`` projected weights ready to feed into
        ``NativeModel.from_projected_weights``.
        """
        raise NotImplementedError(
            "SubspaceProjector.project_module is the change-3 deliverable; "
            "see openspec/changes/subspace-projector/proposal.md."
        )
