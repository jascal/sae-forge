"""AugmentedBasis — a FeatureBasis with two verbatim-preserved subspaces.

The single-basis forge keeps only ``span(W_dec)`` and discards residual
variance outside it. That silently drops two things a residual feature basis
cannot carry: the *sharp monosemantic atoms* (merged by Polygram → ``cov95``
collapse) and the *composition directions* the host attention reads/writes
(outside the retained atoms → circuits break).

``AugmentedBasis`` guarantees both are inside the kept subspace and reproduced
verbatim. To keep the forged weight shapes unchanged (``n_features`` fixed —
required for byte-equivalence and the over-complete Polygram case, where the
subspace cannot be orthonormalised to ``n_features`` rows), it **replaces the
least-important basis atoms** with the verbatim directions rather than growing
the basis. The Polygram basis carries the orthogonal remainder.

``kept_subspace(layer)`` is the single contract the projector consumes:
returns the effective decoder ``W_dec_eff`` (shape unchanged) and a
``preserve_mask`` marking which rows must be written verbatim (vs.
Polygram-merged). When neither subspace is supplied it returns
``(basis.W_dec, all-False)`` — the single-basis path, byte-identical.

See ``openspec/specs/composition-subspace-preserve``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.composition_subspace import CompositionSubspace


@dataclass
class AugmentedBasis:
    """A ``FeatureBasis`` plus optional verbatim assertion (``U_A``) and
    per-layer composition (``U_C``) subspaces."""

    basis: FeatureBasis
    assertion_atoms: np.ndarray | None = None  # (K_A, d_model) verbatim sharp-atom directions
    composition: dict[int, CompositionSubspace] | None = None

    def __post_init__(self) -> None:
        d = self.basis.W_dec.shape[1]
        if self.assertion_atoms is not None:
            if self.assertion_atoms.ndim != 2 or self.assertion_atoms.shape[1] != d:
                raise ValueError(
                    f"assertion_atoms must be 2-D (K_A, d_model) with d_model={d}; "
                    f"got shape {self.assertion_atoms.shape}"
                )
        if self.composition is not None:
            for ell, cs in self.composition.items():
                if cs.d_model != d:
                    raise ValueError(
                        f"composition[{ell}] d_model {cs.d_model} does not match "
                        f"basis d_model {d}"
                    )

    def _verbatim_rows(self, layer: int) -> np.ndarray | None:
        rows = []
        if self.assertion_atoms is not None:
            rows.append(self.assertion_atoms)
        if self.composition is not None and layer in self.composition:
            rows.append(self.composition[layer].U.T)  # (r, d_model)
        return np.concatenate(rows, axis=0) if rows else None

    def kept_subspace(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(W_dec_eff, preserve_mask)`` for the given layer.

        ``W_dec_eff`` has the same shape as ``basis.W_dec``; the
        ``preserve_mask`` rows hold the verbatim ``U_A``/``U_C`` directions
        (the least-important basis atoms they displaced), the rest are the
        surviving Polygram atoms.
        """
        W_dec = self.basis.W_dec
        n_features = W_dec.shape[0]
        V = self._verbatim_rows(layer)
        if V is None:
            return W_dec, np.zeros(n_features, dtype=bool)
        n_v = V.shape[0]
        if n_v > n_features:
            raise ValueError(
                f"preserved dimension {n_v} exceeds basis n_features {n_features}; "
                f"reduce assertion_k / composition_rank"
            )
        # displace the least-important (lowest decoder-norm) atoms with the verbatim rows
        norms = np.linalg.norm(W_dec, axis=1)
        repl = np.argsort(norms)[:n_v]
        W_eff = W_dec.copy()
        W_eff[repl] = V
        mask = np.zeros(n_features, dtype=bool)
        mask[repl] = True
        return W_eff, mask

    def preserved_dimension(self, layer: int) -> int:
        V = self._verbatim_rows(layer)
        return 0 if V is None else int(V.shape[0])

    def preserved_fraction(self, layer: int) -> float:
        """Preserved verbatim dimension as a fraction of ``d_model`` (the budget)."""
        return self.preserved_dimension(layer) / self.basis.W_dec.shape[1]
