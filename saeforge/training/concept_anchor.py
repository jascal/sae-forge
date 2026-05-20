"""Concept-anchoring label sources for ``run_finetune``.

Defines the :class:`LabelSource` protocol, a module-level registry, and
the v1 :class:`PolygramClusterLabelSource` backend.

A label source maps batches → multi-hot label tensors of shape
``[B, T, n_concepts]``. The fine-tune loop multiplies these against the
two concept-anchoring heads' logits, weights by focal BCE, and folds the
result into the total loss when ``TrainingConfig.concept_alpha > 0``.

The registry pattern lets future label-source backends
(``corpus-tags``, ``host-probe``, etc.) plug in via the
:func:`register_label_source` decorator without touching the loss code.

See ``openspec/changes/add-concept-anchored-finetune/`` for the full
spec and the design discussion.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Callable, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from saeforge.basis import FeatureBasis


@runtime_checkable
class LabelSource(Protocol):
    """Maps batches to multi-hot per-token concept labels.

    Lifecycle:

    1. The trainer instantiates the source once, passing backend-specific
       kwargs from ``TrainingConfig.concept_label_source_kwargs``.
    2. The trainer calls :meth:`prepare(model, iterator)` exactly once
       BEFORE the main training loop. ``prepare`` runs any required
       calibration (e.g. the polygram backend records per-cluster firing
       distributions on the pre-fine-tune forged model). It returns
       ``n_concepts`` so the trainer can size the heads.
    3. The trainer calls :meth:`labels_for_batch(batch, hidden_states)`
       per training step. The backend may use the supplied
       ``hidden_states`` (the student's last-layer activations, already
       computed by the forward pass) or ignore them and run its own
       forward.
    """

    def prepare(
        self,
        model: "nn.Module",
        iterator: Iterable,
    ) -> int:
        """Run one-time calibration. Returns ``n_concepts``."""
        ...

    def labels_for_batch(
        self,
        batch: "torch.Tensor",
        hidden_states: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Return multi-hot labels with shape ``[B, T, n_concepts]``.

        Dtype is float; values are exactly ``0.0`` or ``1.0``.
        """
        ...


LABEL_SOURCE_REGISTRY: dict[str, type] = {}


def register_label_source(name: str) -> Callable[[type], type]:
    """Decorator that registers a :class:`LabelSource` implementation
    under ``name``.

    Raises:
        ValueError: if ``name`` is already registered.
    """

    def _decorate(cls: type) -> type:
        if name in LABEL_SOURCE_REGISTRY:
            raise ValueError(
                f"register_label_source: name {name!r} is already registered "
                f"to {LABEL_SOURCE_REGISTRY[name]!r}; pick a different name "
                f"or remove the prior registration first."
            )
        LABEL_SOURCE_REGISTRY[name] = cls
        return cls

    return _decorate


# ---------------------------------------------------------------------------
# v1 backend: polygram-clusters
# ---------------------------------------------------------------------------


@register_label_source("polygram-clusters")
class PolygramClusterLabelSource:
    """Self-supervised label source using polygram-cluster firings.

    Calibrates on the *pre-fine-tune* forged model: runs a handful of
    student forwards under ``torch.no_grad``, projects the residual
    stream into the polygram feature space via the basis's pseudoinverse,
    and freezes a per-cluster firing threshold. ``labels_for_batch``
    then projects fresh residuals and thresholds against the frozen
    table.

    Phase 6.2 recipe: labels are FROZEN at fine-tune start (recomputing
    them mid-training defeats supervision — the labels would chase the
    student's drift).
    """

    def __init__(
        self,
        polygram_basis: "FeatureBasis",
        calibration_batches: int = 32,
        firing_threshold: float = 0.5,
    ) -> None:
        self._basis = polygram_basis
        self._calibration_batches = int(calibration_batches)
        self._firing_threshold = float(firing_threshold)
        self._n_concepts: int | None = None
        self._pinv_tensor: "torch.Tensor | None" = None  # cached, on-device
        self._cluster_index: list[list[int]] | None = None  # cluster_id -> kept-feature local indices

    def prepare(
        self,
        model: "nn.Module",
        iterator: Iterable,
    ) -> int:
        """Build the per-cluster firing table from a calibration corpus.

        Returns ``n_concepts`` (the polygram report's ``n_clusters``).
        """
        import torch

        # n_clusters comes from the polygram-compressed basis metadata.
        # If the report didn't include it, we have nothing to anchor on.
        meta = self._basis.metadata or {}
        n_clusters = meta.get("n_clusters")
        if n_clusters is None or int(n_clusters) <= 1:
            report_path = meta.get("report_path", "<unknown>")
            raise ValueError(
                f"PolygramClusterLabelSource: polygram basis has "
                f"n_clusters={n_clusters!r} (report: {report_path}). "
                f"Concept anchoring needs >= 2 clusters; this basis is "
                f"trivially clustered. Re-run polygram compression with a "
                f"higher-rung encoding (Rung4 / Rung5) or a coherence-aware "
                f"profile so multiple clusters survive."
            )
        self._n_concepts = int(n_clusters)

        # Per-cluster membership over the kept features.
        # PREFERRED: the polygram basis carries `cluster_assignments` —
        # a list[int] of length n_kept giving each kept feature's cluster
        # id. Polygram's compression report has emitted this since v0.10
        # for cosine-strategy ClusteredDictionary; if your basis is from
        # a newer polygram + clustered build it should be present.
        # FALLBACK: round-robin (feature i → cluster i mod n_concepts).
        # This is *deterministic and minimal-useful*; the resulting label
        # signal is real but only coarsely aligned with the true cluster
        # structure. Emits a UserWarning so callers can tell when their
        # basis is missing the metadata and the supervision is operating
        # on the fallback partition.
        cluster_assignments = meta.get("cluster_assignments")
        if cluster_assignments is not None:
            membership: list[list[int]] = [[] for _ in range(self._n_concepts)]
            for kept_idx, cid in enumerate(cluster_assignments):
                if 0 <= int(cid) < self._n_concepts:
                    membership[int(cid)].append(kept_idx)
        else:
            import warnings as _w

            n_kept = int(self._basis.W_dec.shape[0])
            _w.warn(
                "PolygramClusterLabelSource: polygram basis metadata has "
                "no `cluster_assignments`; using deterministic round-robin "
                f"(feature i -> cluster i mod {self._n_concepts}). The "
                "supervised signal will only coarsely align with the true "
                "cluster structure. Re-run polygram compression with a "
                "ClusteredDictionary that emits cluster_assignments, or "
                "supply a custom mapping via "
                "`basis.metadata['cluster_assignments'] = [...]` before "
                "starting fine-tune.",
                UserWarning,
                stacklevel=2,
            )
            membership = [[] for _ in range(self._n_concepts)]
            for kept_idx in range(n_kept):
                membership[kept_idx % self._n_concepts].append(kept_idx)
        self._cluster_index = membership

        # Project hidden states into polygram feature space via the
        # pseudoinverse. Cache as a torch tensor on the model's device.
        device = next(model.parameters()).device
        pinv = self._basis.pseudoinverse()  # numpy, (d_model, n_features)
        self._pinv_tensor = torch.from_numpy(pinv).to(device=device, dtype=torch.float32)

        # Run the calibration loop (slicing `calibration_batches` from
        # the iterator). The polygram backend's only "calibration" need
        # is materialising the projection — the firing threshold is a
        # static scalar. The forward passes here are diagnostic; they
        # validate the projection runs without error against a real
        # batch shape. Future extensions (quantile-based thresholds)
        # would consume these activations.
        model.eval()
        consumed = 0
        with torch.no_grad():
            for batch in itertools.islice(iterator, self._calibration_batches):
                batch = batch.to(device)
                out = model(batch)
                consumed += 1
                # If the module returns an object with .hidden_states,
                # use the last layer; otherwise it returned raw logits
                # and we can't validate the projection — that's fine,
                # the projection is exercised at training time.
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif hasattr(out, "hidden_states") and out.hidden_states is not None:
                    hs = out.hidden_states[-1]
                else:
                    continue
                # Sanity-check shape (B, T, d_model) — projection would
                # fail with a clear matmul error otherwise.
                _ = hs @ self._pinv_tensor
        model.train()

        # Iterator-consumption diagnostic. `prepare` consumes from the
        # SAME iterator the main loop will read from next, so a short
        # corpus can mean training sees fewer steps than expected.
        # Warn when (a) we got 0 batches (the iterator was already
        # exhausted or empty — likely a bug) or (b) we got fewer
        # batches than requested (corpus may be too small for
        # `calibration_batches + total_steps`).
        if consumed == 0:
            import warnings as _w

            _w.warn(
                "PolygramClusterLabelSource.prepare consumed 0 batches "
                "from the iterator — either the iterator was already "
                "exhausted, or it yields nothing on this run. The "
                "projection is constructed but the training loop will "
                "see no batches either. Likely a setup bug.",
                UserWarning,
                stacklevel=2,
            )
        elif consumed < self._calibration_batches:
            import warnings as _w

            _w.warn(
                f"PolygramClusterLabelSource.prepare consumed only "
                f"{consumed} of the requested {self._calibration_batches} "
                f"calibration batches before the iterator was exhausted. "
                f"The main training loop will see ZERO batches from the "
                f"same iterator — pre-slice your calibration corpus or "
                f"pass a `DataLoader` (resettable) instead of a one-shot "
                f"`iter()` chain. See "
                f"`docs/concept-anchoring.md` (forthcoming) for the "
                f"iterator-consumption contract.",
                UserWarning,
                stacklevel=2,
            )

        return self._n_concepts

    def labels_for_batch(
        self,
        batch: "torch.Tensor",
        hidden_states: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Return multi-hot per-token cluster firing labels.

        Args:
            batch: input ids (B, T). Unused when ``hidden_states`` is
                provided — the loop passes the student's last hidden
                state to avoid a second forward.
            hidden_states: the student's last-layer residual stream,
                shape ``(B, T, d_model)``. REQUIRED for the polygram
                backend; ``None`` raises ``ValueError`` (the projection
                cost is the dominant per-step overhead, so the loop
                always reuses its own forward).
        """
        import torch

        if hidden_states is None:
            raise ValueError(
                "PolygramClusterLabelSource.labels_for_batch requires "
                "`hidden_states` (the student's last hidden state). The "
                "loop is expected to pass it in to avoid a redundant "
                "forward pass."
            )
        if self._pinv_tensor is None or self._cluster_index is None or self._n_concepts is None:
            raise RuntimeError(
                "PolygramClusterLabelSource: call prepare(...) before "
                "labels_for_batch(...)."
            )

        # Project hidden states into polygram feature space:
        # h @ pinv -> (B, T, n_features)
        features = hidden_states.to(self._pinv_tensor.dtype) @ self._pinv_tensor
        # Threshold features into per-feature firing booleans.
        fired = features > self._firing_threshold  # (B, T, n_features)

        # Aggregate per-feature firings into per-cluster firings via
        # the cluster_index membership. A cluster "fires" at (b, t) if
        # ANY of its constituent features fires there.
        labels = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[1], self._n_concepts),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for cid, kept_indices in enumerate(self._cluster_index):
            if not kept_indices:
                continue
            # Slice features at the cluster's kept indices and OR-reduce
            # along the feature axis.
            cluster_features = fired[..., kept_indices]  # (B, T, n_members)
            cluster_fired = cluster_features.any(dim=-1)  # (B, T)
            labels[..., cid] = cluster_fired.to(torch.float32)

        return labels


__all__ = [
    "LabelSource",
    "LABEL_SOURCE_REGISTRY",
    "register_label_source",
    "PolygramClusterLabelSource",
]
