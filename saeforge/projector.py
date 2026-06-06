"""SubspaceProjector — project host weights into and out of a feature basis.

Projection algebra (W_dec ≡ D, shape (n_features, d_model); pinv(W_dec) ≡ E,
shape (d_model, n_features); D @ E == I_n for full row-rank D):

- residual-input matrix (d_model -> m): W_n = D @ W; shape (n_features, m)
- residual-output matrix (m -> d_model): W_n = W @ E; shape (m, n_features)
- residual-output bias (d_model,): b_n = b @ E; shape (n_features,)
- residual-aligned scale/shift (γ, β ∈ R^d): γ_n = γ @ E; same for β

v0.2 feature-native attention adds two more identities for the both-sides
QKV projection (attention internal dimensions also become k-wide):

- both-sides-projected matrix (d_model -> d_model): W_n = D @ W @ E; shape (k, k)
- QKV-output triple (d_model -> 3*d_model): W_n = D @ W @ block_diag(E, E, E);
  shape (k, 3k); equivalent to splitting the 3d output into Q/K/V blocks,
  applying D @ W_block @ E to each, and concatenating.

Layer norm under linear projection is not equivariant; γ/β projection is
the lossy v0 fallback. Faithfulness drops are expected and tracked by the
forge-pipeline KL eval.

scale_boost notes
-----------------

``scale_boost`` multiplies the ``encode`` output (``x @ E * scale_boost``).
For a well-conditioned basis with ``n_features <= d_model``,
``scale_boost=1.0`` is the identity-preserving default — every linear map
is exactly preserved on the basis subspace. For *over-complete* bases
(``n_features > d_model``, common when a Polygram-compressed SAE keeps
more features than the host's residual width), the encode operation
no longer round-trips to identity in the n-dim subspace and empirical
activation magnitudes can blow up — overflowing bf16, saturating
softmax, or producing astronomical initial KL.

Empirical anchor: GPT-2 (d_model=768) with a 1024-feature basis
required ``scale_boost ≈ 0.25`` to keep training stable; the default
of 1.0 was too large.

``scale_boost="auto"`` picks ``min(1.0, d_model / n_features)`` — a
defensible starting heuristic for over-complete bases that scales
inversely with the over-completeness ratio. For the GPT-2 + 1024
case that gives ``768/1024 = 0.75``: directionally right (less than 1)
but not the empirical optimum. Treat ``"auto"`` as a starting point
and tune from there. A more principled per-basis calibrator is
deferred to a follow-up.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

from saeforge.basis import FeatureBasis

if TYPE_CHECKING:  # pragma: no cover — type-only import
    from saeforge.augmented_basis import AugmentedBasis
    from saeforge.hybrid_basis import HybridBasisBundle


@dataclass
class SubspaceProjector:
    """Project a host model's weights into the feature basis defined by ``basis``.

    The projection math is pure-numpy. Real host models live behind the
    ``[torch]`` extra; ``project_module(...)`` lazy-imports torch on demand.

    ``scale_boost`` accepts a positive float OR the string ``"auto"``:

    - Literal float: the classic path; the supplied value is used as-is.
    - ``"auto"``: resolved to ``min(1.0, d_model / n_features)`` — a
      heuristic for over-complete bases (see module docstring for
      empirical context).

    An earlier draft of the ``fix-scale-boost-calibration`` change
    explored a ``"calibrate"`` mode that auto-picked ``scale_boost`` from
    a fixed grid to minimise forward KL on a calibration corpus. The
    2026-05-16 smoke gate ([[project_fix_scale_boost_smoke]]) found that
    three successive proxies for the forge's faithfulness KL all
    diverged from the real target — including a "real" end-of-network
    KL implementation, because the forge's KL measures a fully-projected
    NativeModel and the residual-perturbation proxy can't see the
    stacked-projection compounding. The calibrate mode was dropped; the
    change's diagnostic surface (row fields, advisories) survives.

    When a literal ``1.0`` (the default) is supplied with an
    over-complete basis, a ``UserWarning`` surfaces so the silent
    activation-magnitude footgun can't recur.
    """

    basis: FeatureBasis
    scale_boost: Union[float, str] = 1.0

    def __post_init__(self) -> None:
        # Resolve "auto" first so the float invariants below see a numeric value.
        if isinstance(self.scale_boost, str):
            if self.scale_boost != "auto":
                raise ValueError(
                    f"scale_boost must be a positive float or 'auto'; "
                    f"got {self.scale_boost!r}"
                )
            self.scale_boost = self._auto_scale_boost()
        if self.scale_boost <= 0.0:
            raise ValueError(f"scale_boost must be positive; got {self.scale_boost}")
        # Footgun warning: the GPT-2 + 1024-feature anchor showed
        # scale_boost=1.0 was too large on over-complete bases.
        if (
            self.basis.n_features > self.basis.d_model
            and float(self.scale_boost) == 1.0
        ):
            warnings.warn(
                f"SubspaceProjector: over-complete basis detected "
                f"(n_features={self.basis.n_features} > d_model="
                f"{self.basis.d_model}) with scale_boost=1.0. The default "
                f"is often too large in this regime — empirically GPT-2 "
                f"(d_model=768) with 1024 features needed scale_boost≈0.25 "
                f"to train stably. Consider scale_boost='auto' or a "
                f"hand-picked value < 1.0; tune from there if needed.",
                UserWarning,
                stacklevel=2,
            )

    def _auto_scale_boost(self) -> float:
        """Heuristic default for over-complete bases.

        For ``n_features <= d_model`` the basis can in principle represent
        every direction in the residual stream and ``scale_boost=1.0``
        preserves linear maps exactly — no down-scaling needed.

        For ``n_features > d_model`` the encode-decode round-trip is a
        rank-d projection in n-dim space; activations spread across more
        coordinates and per-row magnitudes of ``pinv(W_dec)`` grow. We
        return ``d_model / n_features`` as a defensible starting
        heuristic that scales inversely with the over-completeness.
        """
        n = self.basis.n_features
        d = self.basis.d_model
        if n <= d:
            return 1.0
        return float(d) / float(n)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project ``x`` (..., d_model) into the basis (..., n_features)."""
        return x @ self.basis.pseudoinverse() * self.scale_boost

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct (..., d_model) from basis-space ``z`` (..., n_features)."""
        return z @ self.basis.W_dec

    def project_residual_input(self, W: np.ndarray) -> np.ndarray:
        """(d_model, m) -> (n_features, m). Used for QKV, MLP up, any layer that reads the residual."""
        return self.basis.W_dec @ W

    def project_residual_output(self, W: np.ndarray) -> np.ndarray:
        """(m, d_model) -> (m, n_features). Used for attn output, MLP down, any layer that writes the residual."""
        return self.encode(W)

    def project_residual_full(self, W: np.ndarray) -> np.ndarray:
        """(d_model, d_model) -> (n_features, n_features). Both-sides projection: D @ W @ E.

        v0.2 feature-native attention path. Used for c_proj when attention
        internal width equals n_features.
        """
        if W.ndim != 2 or W.shape[0] != self.basis.d_model or W.shape[1] != self.basis.d_model:
            raise ValueError(
                f"project_residual_full expects (d_model, d_model) = "
                f"({self.basis.d_model}, {self.basis.d_model}); got {W.shape}"
            )
        return self.basis.W_dec @ W @ self.basis.pseudoinverse() * self.scale_boost

    def project_qkv_full(self, W: np.ndarray) -> np.ndarray:
        """(d_model, 3 * d_model) -> (n_features, 3 * n_features). v0.2 feature-native c_attn.

        Splits the input into Q/K/V blocks along axis=1, applies
        ``project_residual_full`` to each block, and concatenates the
        results back along axis=1.
        """
        d = self.basis.d_model
        if W.ndim != 2 or W.shape[0] != d or W.shape[1] != 3 * d:
            raise ValueError(
                f"project_qkv_full expects (d_model, 3*d_model) = ({d}, {3*d}); got {W.shape}"
            )
        q, k, v = np.split(W, 3, axis=1)
        return np.concatenate(
            [
                self.project_residual_full(q),
                self.project_residual_full(k),
                self.project_residual_full(v),
            ],
            axis=1,
        )

    def project_residual_bias(self, b: np.ndarray) -> np.ndarray:
        """(d_model,) -> (n_features,). Used for any bias added to the residual."""
        return self.encode(b)

    def project_residual_aligned(self, v: np.ndarray) -> np.ndarray:
        """(d_model,) -> (n_features,). Used for LN γ / β (lossy under projection)."""
        return self.encode(v)

    def project_embed(self, W_embed: np.ndarray) -> np.ndarray:
        """(V, d_model) -> (V, n_features). Token / position embeddings."""
        return self.encode(W_embed)

    def project_unembed(self, W_unembed: np.ndarray) -> np.ndarray:
        """(V, d_model) -> (V, n_features). HF lm_head weight (Linear stores as (vocab, d_model))."""
        return W_unembed @ self.basis.W_dec.T

    def project_qkv(self, W_qkv: np.ndarray) -> np.ndarray:
        """(d_model, 3 * d_head * n_heads) -> (n_features, ...). HF GPT-2 c_attn weight."""
        return self.project_residual_input(W_qkv)

    def project_mlp_in(self, W_in: np.ndarray) -> np.ndarray:
        """(d_model, d_ff) -> (n_features, d_ff). HF GPT-2 mlp.c_fc weight."""
        return self.project_residual_input(W_in)

    def project_mlp_out(self, W_out: np.ndarray) -> np.ndarray:
        """(d_ff, d_model) -> (d_ff, n_features). HF GPT-2 mlp.c_proj weight."""
        return self.project_residual_output(W_out)

    def project_module(
        self,
        host_model,
        *,
        attention_width: str = "host",
        hybrid: "HybridBasisBundle | None" = None,
        augmented: "AugmentedBasis | None" = None,
    ) -> dict[str, np.ndarray]:
        """Project every relevant host-model weight into the basis.

        Dispatches via ``saeforge.adapters.adapter_for(host_model)``. The
        v0.1 GPT-2 walker now lives in
        :class:`saeforge.adapters.gpt2.GPT2Adapter`; Llama and Gemma-2
        adapters are bundled too. Unregistered architectures raise
        ``NotImplementedError`` naming the host's type and the
        registered class set.

        ``attention_width`` is forwarded to the adapter; what each
        adapter does with it depends on the architecture (the GPT-2
        adapter's "feature_native" mode applies both-sides projection
        to ``c_attn`` / ``c_proj``; Llama-family adapters currently
        accept only ``"host"``).

        When ``hybrid`` is a :class:`~saeforge.hybrid_basis.HybridBasisBundle`,
        dispatch is routed through ``saeforge.adapters._hybrid.walk_hybrid``:
        three per-basis walks are run and each emitted key is routed to the
        basis owning its layer region. The keyset matches the single-basis
        output for the same host exactly. ``self`` (the mid basis projector)
        supplies the shared ``scale_boost``. See
        ``openspec/specs/hybrid-bridge-forge`` for the routing contract.
        """
        if attention_width not in ("host", "feature_native"):
            raise ValueError(
                f"attention_width must be 'host' or 'feature_native'; got {attention_width!r}"
            )
        try:
            import transformers  # noqa: F401  — checked here for friendlier ImportError
        except ImportError as e:
            raise ImportError(
                "SubspaceProjector.project_module needs the [torch] extra; "
                "install it with `pip install sae-forge[torch]`."
            ) from e

        from saeforge.adapters import adapter_for

        adapter = adapter_for(host_model)
        if hybrid is not None and augmented is not None:
            raise ValueError(
                "project_module: hybrid= and augmented= are independent v1 paths and "
                "cannot be combined; pass at most one."
            )
        if hybrid is not None:
            from saeforge.adapters._hybrid import walk_hybrid

            return walk_hybrid(
                host_model,
                adapter,
                self,
                bundle=hybrid,
                attention_width=attention_width,
            )
        if augmented is not None:
            from saeforge.adapters._augmented import walk_augmented

            return walk_augmented(
                host_model,
                adapter,
                self,
                augmented=augmented,
                attention_width=attention_width,
            )
        return adapter.walk(host_model, self, attention_width=attention_width)


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to float64 numpy without requiring torch at import time.

    Goes via ``.float()`` because numpy has no native bfloat16 dtype — a direct
    ``.numpy()`` on a bf16 tensor raises ``TypeError: Got unsupported ScalarType BFloat16``.
    """
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().float().numpy().astype(np.float64, copy=False)
    return np.asarray(tensor).astype(np.float64, copy=False)
