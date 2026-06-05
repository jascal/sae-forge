"""``sae-moe-forge`` — turn a polygram-compressed SAE into a routed MoE.

Public surface (all re-exported lazily from ``saeforge`` so a torch-free
``import saeforge`` stays cheap — this module imports torch at load time):

- :func:`forge_to_moe` — the single entry point. Takes a
  :class:`~saeforge.basis.FeatureBasis` (+ optional polygram
  ``ExpertDictionary``) and returns a :class:`ForgedMoE`.
- :class:`ForgedMoE` — an inference-only ``nn.Module`` whose decode
  cost scales as ``k_experts / n_experts`` of the flat SAE. No
  trainable parameters in v1.
- :class:`ForgedMoEConfig` — the frozen contract surface.

The v1 contract is *structurally correct + computationally honest +
inference-only*. Each expert is a deterministic slice of the SAE
decoder (``sub_dictionary``); routing wraps polygram's summed-activation
heuristic (``polygram_heuristic``). Quality versus a learned router or a
distilled expert belongs to the queued follow-up proposals named in
``openspec/changes/add-sae-moe-forge``.

The encoder that maps the residual stream into feature activations is
the basis pseudo-inverse ``pinv(W_dec)`` — the same convention
:class:`~saeforge.projector.SubspaceProjector` uses on the input side,
and the encoder the 2026-05-19 acceptance prototype measured its bands
against. Using the SAE's native ``W_enc``/``b_enc`` (with its
nonlinearity) is deferred to ``add-moe-encoder-side``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from saeforge._moe.routers import PolygramHeuristicRouter
from saeforge._moe.sub_dictionary import SubDictionaryExpertSet

if TYPE_CHECKING:  # pragma: no cover — type-only import
    from saeforge.basis import FeatureBasis

_LEGAL_EXPERT_TYPES = ("sub_dictionary",)
_LEGAL_ROUTER_TYPES = ("polygram_heuristic",)
# expert/router values that are deliberately deferred → NotImplementedError
# pointing at the queued follow-up proposal that will land them.
_DEFERRED_EXPERT_TYPES = {
    "tiny_mlp": "add-moe-tiny-mlp-experts",
    "residual_block": "add-moe-residual-block-experts",
}
_DEFERRED_ROUTER_TYPES = {
    "linear": "add-moe-trained-router",
    "mlp": "add-moe-trained-router",
}

_CONFIG_FILENAME = "forged_moe_config.json"
_WEIGHTS_FILENAME = "forged_moe_buffers.safetensors"


@dataclass(frozen=True)
class ForgedMoEConfig:
    """The v1 forged-MoE contract surface (frozen, JSON-round-trippable).

    There is deliberately no ``encoder_type`` field: v1 always uses the
    basis pseudo-inverse ``pinv(W_dec)`` (the ``SubspaceProjector``
    convention the acceptance prototype measured its bands against). The
    SAE's native ``W_enc``/``b_enc`` encoder — which would add an
    ``encoder_type`` knob here — is the queued ``add-moe-encoder-side``
    follow-up; it is held out of the frozen v1 surface so the contract
    stays small and falsifiable.
    """

    n_features: int
    d_model: int
    n_experts: int
    k_experts: int
    expert_type: str = "sub_dictionary"
    router_type: str = "polygram_heuristic"
    source_basis_checkpoint: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> ForgedMoEConfig:
        return cls(**payload)


@dataclass(frozen=True)
class CoherenceDiagnostic:
    """Basis cluster-quality signal computed at forge time.

    ``median_intra_cluster_cosine`` is the median over every
    within-expert decoder-row pair cosine; ``low_coherence`` is the
    Band-C-strict gate predicate (``median <= 0.5`` ⇒ faithfulness is
    advisory only on this basis). ``n_intra_cluster_pairs`` is 0 when
    every expert is a singleton, in which case both cosines default to
    0.0 and the basis counts as low-coherence.
    """

    median_intra_cluster_cosine: float
    max_intra_cluster_cosine: float
    n_intra_cluster_pairs: int

    @property
    def low_coherence(self) -> bool:
        return not (self.median_intra_cluster_cosine > 0.5)

    def to_dict(self) -> dict:
        return {**asdict(self), "low_coherence": self.low_coherence}


@dataclass(frozen=True)
class FaithfulnessReport:
    """Band-C-advisory diagnostic: routed reconstruction vs the flat SAE.

    ``ratio = routed_vs_flat_mse / flat_vs_host_mse`` — how much extra
    reconstruction error routing adds, in units of the flat SAE's own
    error against the host residual. On a clusterable basis this is
    ``<= 0.5`` (Band-C-strict); on a near-isotropic basis the
    2026-05-19 prototype measured ≈ 4.6 (advisory, not a forge bug).
    """

    routed_vs_flat_mse: float
    flat_vs_host_mse: float
    ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _coherence_diagnostic(
    W_dec: np.ndarray, feature_to_expert: np.ndarray, n_experts: int
) -> CoherenceDiagnostic:
    """Median/max within-expert decoder-row cosine over all intra-cluster pairs."""
    rows = np.asarray(W_dec, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit = rows / np.clip(norms, 1e-12, None)
    cosines: list[np.ndarray] = []
    for e in range(n_experts):
        members = np.flatnonzero(feature_to_expert == e)
        if members.shape[0] < 2:
            continue
        gram = unit[members] @ unit[members].T
        iu = np.triu_indices(members.shape[0], k=1)
        cosines.append(gram[iu])
    if not cosines:
        return CoherenceDiagnostic(0.0, 0.0, 0)
    allc = np.concatenate(cosines)
    return CoherenceDiagnostic(
        median_intra_cluster_cosine=float(np.median(allc)),
        max_intra_cluster_cosine=float(np.max(allc)),
        n_intra_cluster_pairs=int(allc.shape[0]),
    )


class ForgedMoE(nn.Module):
    """A flat SAE projected into a routed mixture-of-experts (inference-only).

    Construct via :func:`forge_to_moe`. The module holds only buffers —
    ``.parameters()`` is empty — so it is moved across devices with
    ``.to(device)`` but never trained in v1.
    """

    def __init__(
        self,
        *,
        encoder_weight: np.ndarray | torch.Tensor,
        experts: SubDictionaryExpertSet,
        router: PolygramHeuristicRouter,
        config: ForgedMoEConfig,
        coherence_diagnostic: CoherenceDiagnostic,
    ):
        super().__init__()
        self.config = config
        self.coherence_diagnostic = coherence_diagnostic
        self.experts = experts
        self.router = router
        enc = torch.as_tensor(np.asarray(encoder_weight), dtype=torch.float32)
        if enc.shape != (config.d_model, config.n_features):
            raise ValueError(
                f"encoder_weight shape {tuple(enc.shape)} != (d_model, "
                f"n_features) = ({config.d_model}, {config.n_features})"
            )
        self.register_buffer("encoder_weight", enc)
        self._last_load: torch.Tensor | None = None

    # -- core forward path --------------------------------------------------

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Map a ``(*batch, d_model)`` residual into ``(*batch, n_features)``."""
        return residual @ self.encoder_weight.to(residual.dtype)

    def forward(
        self, residual: torch.Tensor, *, track_load: bool = False
    ) -> torch.Tensor:
        """Routed reconstruction of ``residual`` ``(*batch, d_model)``."""
        features = self.encode(residual)
        top_k = self.router.route(features, self.config.k_experts)
        decoded = self.experts(features, top_k)
        self._last_load = self._expert_load(top_k) if track_load else None
        return decoded

    def route(self, residual: torch.Tensor) -> torch.Tensor:
        """``(*batch, k_experts)`` selected expert indices, best first."""
        return self.router.route(self.encode(residual), self.config.k_experts)

    def expert_load(self) -> torch.Tensor | None:
        """Per-expert token-slot fraction from the most recent
        ``forward(track_load=True)``; ``None`` otherwise."""
        return self._last_load

    def _expert_load(self, top_k_experts: torch.Tensor) -> torch.Tensor:
        flat = top_k_experts.reshape(-1)
        counts = torch.bincount(flat, minlength=self.config.n_experts).to(torch.float64)
        total = counts.sum()
        return counts / total if total > 0 else counts

    # -- diagnostics --------------------------------------------------------

    def faithfulness_report(self, host_residual: torch.Tensor) -> FaithfulnessReport:
        """Band-C-advisory: routed-vs-flat MSE ratioed against flat-vs-host."""
        features = self.encode(host_residual)
        flat = features @ self.experts.W_dec.to(features.dtype)
        top_k = self.router.route(features, self.config.k_experts)
        routed = self.experts(features, top_k)
        routed_vs_flat = float(((routed - flat) ** 2).mean().item())
        flat_vs_host = float(((flat - host_residual) ** 2).mean().item())
        ratio = routed_vs_flat / flat_vs_host if flat_vs_host > 0 else float("inf")
        return FaithfulnessReport(routed_vs_flat, flat_vs_host, ratio)

    # -- persistence (tasks §11) -------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """Write config + buffers to ``path`` (self-contained round-trip).

        v1 stores the decoder slice alongside the partition and encoder
        so a load needs no access to the original polygram checkpoint.
        The source-checkpoint-only variant (re-slice ``W_dec`` on load,
        avoiding the duplicate) is a queued optimisation.
        """
        import json

        from safetensors.torch import save_file

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / _CONFIG_FILENAME).write_text(json.dumps(self.config.to_dict(), indent=2))
        save_file(
            {
                "encoder_weight": self.encoder_weight.contiguous().cpu(),
                "W_dec": self.experts.W_dec.contiguous().cpu(),
                "feature_to_expert": self.experts.feature_to_expert.contiguous().cpu(),
            },
            str(out / _WEIGHTS_FILENAME),
        )

    @classmethod
    def load_pretrained(cls, path: str | Path) -> ForgedMoE:
        """Reconstruct a :class:`ForgedMoE` written by :meth:`save_pretrained`.

        Buffers are saved on CPU and reload on CPU at their stored dtypes
        (``encoder_weight`` / ``W_dec`` float32, ``feature_to_expert``
        int64) — dtype round-trips exactly. The reloaded module always
        lands on CPU regardless of the source module's device; call
        ``.to(device)`` afterwards to move it, as with any nn.Module.
        """
        import json

        from safetensors.torch import load_file

        src = Path(path)
        config = ForgedMoEConfig.from_dict(
            json.loads((src / _CONFIG_FILENAME).read_text())
        )
        buffers = load_file(str(src / _WEIGHTS_FILENAME))
        W_dec = buffers["W_dec"].numpy()
        feature_to_expert = buffers["feature_to_expert"].numpy()
        experts = SubDictionaryExpertSet(W_dec, feature_to_expert, config.n_experts)
        router = PolygramHeuristicRouter(feature_to_expert, config.n_experts)
        coherence = _coherence_diagnostic(W_dec, feature_to_expert, config.n_experts)
        return cls(
            encoder_weight=buffers["encoder_weight"].numpy(),
            experts=experts,
            router=router,
            config=config,
            coherence_diagnostic=coherence,
        )


def _auto_cluster(
    basis: FeatureBasis,
    coherence_threshold: float,
    max_features_per_expert: int | None,
):
    """Reload the polygram ``Dictionary`` from the basis checkpoint and cluster it.

    Aligns the reloaded ``Dictionary`` to the basis's kept-feature order
    (``basis.kept_ids``) so ``cluster_experts``' name→index map lines up
    with ``basis.W_dec`` row-for-row.
    """
    if basis.polygram_checkpoint_path is None:
        raise ValueError(
            "forge_to_moe(basis) needs either an explicit `expert_dictionary` "
            "or a basis carrying `polygram_checkpoint_path` (set by "
            "FeatureBasis.from_polygram_checkpoint). Auto-clustering from "
            "`basis.W_dec` alone is the queued follow-up "
            "`add-moe-explicit-cluster-construction`."
        )
    import polygram
    from polygram import HEA_Rung2

    records = polygram.load_sae_safetensors(basis.polygram_checkpoint_path)
    feature_ids = [int(i) for i in np.asarray(basis.kept_ids).tolist()]
    # from_sae_lens caps a flat Dictionary at the encoding's max_features
    # (MPSRung1 defaults to 8). Size an HEA_Rung2 so its 2**n_qubits cap
    # covers every kept feature — matching the prototype's encoding —
    # and request a flat (clustered=False) Dictionary for cluster_experts.
    n_qubits = max(1, int(np.ceil(np.log2(max(2, len(feature_ids))))))
    dictionary, _report = polygram.from_sae_lens(
        records,
        feature_ids=feature_ids,
        encoding=HEA_Rung2(depth=1, n_qubits=n_qubits),
        clustered=False,
    )
    return polygram.cluster_experts(
        dictionary,
        decoder_vectors=np.asarray(basis.W_dec),
        method="cosine",
        coherence_threshold=coherence_threshold,
        max_features_per_expert=max_features_per_expert,
    )


def forge_to_moe(
    basis: FeatureBasis,
    expert_dictionary=None,
    *,
    k_experts: int = 2,
    expert_type: str = "sub_dictionary",
    router_type: str = "polygram_heuristic",
    coherence_threshold: float = 0.3,
    max_features_per_expert: int | None = None,
) -> ForgedMoE:
    """Forge a polygram-compressed SAE basis into a routed :class:`ForgedMoE`.

    Parameters
    ----------
    basis:
        A :class:`~saeforge.basis.FeatureBasis`. Supplies ``W_dec`` (the
        expert decoder slices), the pseudo-inverse encoder, and — on the
        auto-cluster path — ``polygram_checkpoint_path``.
    expert_dictionary:
        An optional polygram ``ExpertDictionary``. When ``None``, the
        partition is computed by reloading the polygram ``Dictionary``
        from the basis checkpoint and calling ``polygram.cluster_experts``.
    k_experts:
        Top-k experts routed per token; ``1 <= k_experts <= n_experts``.
    expert_type / router_type:
        v1 accepts only ``"sub_dictionary"`` / ``"polygram_heuristic"``.
        Deferred values raise ``NotImplementedError`` naming the queued
        follow-up proposal.
    coherence_threshold / max_features_per_expert:
        Forwarded to ``polygram.cluster_experts`` on the auto-cluster path.
    """
    if expert_type not in _LEGAL_EXPERT_TYPES:
        if expert_type in _DEFERRED_EXPERT_TYPES:
            raise NotImplementedError(
                f"expert_type={expert_type!r} is deferred to the queued "
                f"`{_DEFERRED_EXPERT_TYPES[expert_type]}` proposal; v1 ships "
                f"only {_LEGAL_EXPERT_TYPES}."
            )
        raise ValueError(
            f"unknown expert_type={expert_type!r}; legal: {_LEGAL_EXPERT_TYPES}"
        )
    if router_type not in _LEGAL_ROUTER_TYPES:
        if router_type in _DEFERRED_ROUTER_TYPES:
            raise NotImplementedError(
                f"router_type={router_type!r} is deferred to the queued "
                f"`{_DEFERRED_ROUTER_TYPES[router_type]}` proposal; v1 ships "
                f"only {_LEGAL_ROUTER_TYPES}."
            )
        raise ValueError(
            f"unknown router_type={router_type!r}; legal: {_LEGAL_ROUTER_TYPES}"
        )

    if expert_dictionary is None:
        expert_dictionary = _auto_cluster(
            basis, coherence_threshold, max_features_per_expert
        )
    elif expert_dictionary.n_features != basis.n_features:
        raise ValueError(
            f"expert_dictionary.n_features ({expert_dictionary.n_features}) "
            f"!= basis.n_features ({basis.n_features})"
        )

    n_experts = int(expert_dictionary.n_experts)
    if not (1 <= k_experts <= n_experts):
        raise ValueError(
            f"k_experts={k_experts} must satisfy 1 <= k_experts <= "
            f"n_experts={n_experts} (legal range [1, {n_experts}])"
        )

    feature_to_expert = np.asarray(
        expert_dictionary._feature_to_expert, dtype=np.int64
    )
    experts = SubDictionaryExpertSet(basis.W_dec, feature_to_expert, n_experts)
    router = PolygramHeuristicRouter(feature_to_expert, n_experts)
    config = ForgedMoEConfig(
        n_features=basis.n_features,
        d_model=basis.d_model,
        n_experts=n_experts,
        k_experts=int(k_experts),
        expert_type=expert_type,
        router_type=router_type,
        source_basis_checkpoint=basis.polygram_checkpoint_path,
    )
    coherence = _coherence_diagnostic(
        np.asarray(basis.W_dec), feature_to_expert, n_experts
    )
    return ForgedMoE(
        encoder_weight=basis.pseudoinverse(),
        experts=experts,
        router=router,
        config=config,
        coherence_diagnostic=coherence,
    )
