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
"""

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

    def project_module(self, host_model, *, attention_width: str = "host") -> dict[str, np.ndarray]:
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
        return adapter.walk(host_model, self, attention_width=attention_width)


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to float64 numpy without requiring torch at import time."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(tensor).astype(np.float64, copy=False)
