"""DownstreamCapabilityTarget — per-feature × per-label AUC through a
caller-supplied downstream task encoder.

The target answers the question "does the forged model retain the
features a downstream encoder has already learned to discriminate?"
— in contrast to ``KLTarget`` / ``CosineTarget`` / ``TokenCosineTarget``
which ask "are the forged hidden states numerically close to host?".
Bio-sae's 2026-05-22 capability-bottleneck investigation showed the
cosine question systematically misranks forges for capability-bound
users (cosine recommends n=256, capability recommends n=16, on the
same ESM-2 / bio-sae substrate). See
``openspec/changes/add-downstream-capability-target/proposal.md`` for
the empirical motivation.

Pipeline inside ``score()``:

    sequences (via ctx["_eval_input_ids"]) ->
    forged ESM-2 -> forged_h (basis coords) ->
    strip CLS / EOS ->
    decode via W_dec [three-path precedence: ctx["basis"] >
                      forged_module.basis_decode > pinv(basis_encode)] ->
    encoder (caller-supplied) ->
    aggregator (pool_then_encode | encode_then_pool) ->
    AUC per latent × per label (Mann-Whitney) ->
    mean_over_labels(max_over_latents(AUC))

Returns ``(score, perplexity_analog)`` per the FaithfulnessTarget
protocol. ``better_when = "higher"``. Never returned by
``_default_target_for(family)`` — opt-in only, must be passed via
``ForgePipeline(faithfulness=...)``.

Example — manual construction
=============================

::

    import numpy as np
    import torch
    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector
    from saeforge.eval.targets import DownstreamCapabilityTarget

    # A trained SAE wrapped as a "task encoder" callable. nn.Module
    # SAEs that return (reconstruction, latents) wrap as:
    #   encoder = lambda x: my_sae(x)[1]
    encoder = lambda x: x @ W_enc.T + b_enc      # any d_model -> latent_width

    # Binary label matrix (one row per eval sequence).
    labels = np.array([[1, 0, 1], [0, 1, 0], ...], dtype=np.uint8)

    target = DownstreamCapabilityTarget(
        encoder=encoder,
        labels=labels,
        aggregator="pool_then_encode",   # or "encode_then_pool"
        min_prevalence=10,               # drop singleton labels
    )

    pipeline = ForgePipeline(
        basis=basis, projector=projector,
        host_model_id="facebook/esm2_t6_8M_UR50D",
        eval_prompts=protein_sequences,
        faithfulness=target,
    )
    result = pipeline.run(output_dir)
    # result.faithfulness is mean-best-AUC over labels.
    # target.forge_pf_auc holds the per-feature AUC array for plotting.

Example — via ``CapabilityDataset.from_bio_sae``
================================================

::

    from saeforge.datasets import CapabilityDataset

    dataset = CapabilityDataset.from_bio_sae(
        run_dir="runs/uniref50_n5000/pooled_w1024_k64/",
        bundle_path="data/bio_bundle_uniref50.safetensors",
        sequences_path="data/uniref50_sample__n5000_seed0.parquet",
        feed="pooled",
        n_proteins=500,
        min_prevalence=10,
    )
    target = DownstreamCapabilityTarget(
        encoder=dataset.encoder,
        labels=dataset.labels,
        aggregator=dataset.aggregator,
        min_prevalence=dataset.min_prevalence,
    )
    # Use ``target`` with ForgePipeline as above.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, Mapping

import numpy as np


_AGGREGATOR_STRINGS = frozenset({"pool_then_encode", "encode_then_pool"})


class DownstreamCapabilityTarget:
    """Capability-aware faithfulness scorer.

    See module docstring for the pipeline and motivation. Constructor
    arguments:

    Parameters
    ----------
    encoder:
        Callable ``(Tensor (..., d_model)) -> Tensor (..., latent_width)``.
        Bio-sae's ``_ReferenceSAE.forward`` returns ``(reconstruction,
        latents)`` — wrap with ``lambda x: sae(x)[1]`` so this target
        receives just the latents.
    labels:
        ``(N_items, V)`` binary label matrix. Coerced to ``float64`` at
        construction time.
    aggregator:
        ``"pool_then_encode"`` (default — mean over residues, then
        encode), ``"encode_then_pool"`` (encode per residue, then
        mean), or a callable
        ``(host_coord_residues: Tensor (L, d_model)) -> Tensor
        (latent_width,)`` for custom reductions. See
        ``openspec/changes/add-downstream-capability-target/design.md``
        Decision 3 for the contract.
    min_prevalence:
        Drop label columns with positive-class count below this
        threshold at score time. Matches bio-sae's ``--min-n-pos``.
        Default 0 (no filter).
    decode_via_basis:
        When True (default), apply ``forged_h @ W_dec`` to bring forged
        basis-coord states into ``d_model`` coords the encoder expects.
        Set False when the encoder operates directly on basis-coord
        activations (skips the W_dec recovery entirely).
    warn_on_pinv:
        When True (default), emit a one-shot ``UserWarning`` per
        ``id(forged_module)`` if the target falls back to path (c) —
        recovering ``W_dec`` via ``pinv(basis_encode)`` because neither
        ``ctx["basis"]`` nor ``forged_module.basis_decode`` is
        available. Set False to silence the warning for advanced users
        who knowingly forge against legacy / third-party adapters
        without the buffer.
    """

    name = "downstream_capability"
    better_when = "higher"

    def __init__(
        self,
        *,
        encoder: Callable[[Any], Any],
        labels: np.ndarray,
        aggregator: "Literal['pool_then_encode', 'encode_then_pool'] | Callable" = "pool_then_encode",
        min_prevalence: int = 0,
        decode_via_basis: bool = True,
        warn_on_pinv: bool = True,
    ) -> None:
        if not callable(encoder):
            raise TypeError(
                f"DownstreamCapabilityTarget(encoder=...): expected a "
                f"callable; got {type(encoder).__name__!r}. nn.Module "
                f"SAEs that return (reconstruction, latents) need a "
                f"wrapper: pass `lambda x: sae(x)[1]`."
            )
        labels_arr = np.asarray(labels, dtype=float)
        if labels_arr.ndim != 2:
            raise ValueError(
                f"DownstreamCapabilityTarget(labels=...): expected a "
                f"2-D array; got shape {labels_arr.shape!r} "
                f"(ndim={labels_arr.ndim})."
            )
        if labels_arr.shape[0] < 1 or labels_arr.shape[1] < 1:
            raise ValueError(
                f"DownstreamCapabilityTarget(labels=...): expected "
                f"shape (N, V) with N>=1 and V>=1; got "
                f"{labels_arr.shape!r}."
            )
        if isinstance(aggregator, str):
            if aggregator not in _AGGREGATOR_STRINGS:
                raise ValueError(
                    f"DownstreamCapabilityTarget(aggregator="
                    f"{aggregator!r}): unsupported string. Built-ins: "
                    f"{sorted(_AGGREGATOR_STRINGS)}; or pass a callable."
                )
        elif not callable(aggregator):
            raise TypeError(
                f"DownstreamCapabilityTarget(aggregator=...): expected "
                f"a string or callable; got "
                f"{type(aggregator).__name__!r}."
            )
        if not isinstance(min_prevalence, int) or min_prevalence < 0:
            raise ValueError(
                f"DownstreamCapabilityTarget(min_prevalence="
                f"{min_prevalence!r}): expected a non-negative integer."
            )

        self.encoder = encoder
        self.labels = labels_arr
        self.aggregator = aggregator
        self.min_prevalence = min_prevalence
        self.decode_via_basis = decode_via_basis
        self.warn_on_pinv = warn_on_pinv

        # Caches keyed by id(forged_module). Populated lazily on first
        # score() call so repeat-sweeps amortise the W_dec recovery
        # cost. Cleared if the user constructs a new target instance
        # per sweep cell (the recommended pattern).
        self._w_dec_cache: dict[int, np.ndarray] = {}
        self._pinv_warning_emitted: set[int] = set()
        # Side-channel observability for sweep_pareto_capability — last
        # score() call leaves per-feature AUCs here. Public attributes
        # documented in the pareto-sweep spec.
        self.host_pf_auc: np.ndarray | None = None
        self.forge_pf_auc: np.ndarray | None = None

    # ------------------------------------------------------------------
    # FaithfulnessTarget.score
    # ------------------------------------------------------------------
    def score(
        self,
        *,
        forged: Any,
        host: Any,  # noqa: ARG002 — accepted for protocol conformance, never read
        ctx: Mapping[str, Any],
    ) -> tuple[float, float]:
        import torch  # lazy

        try:
            input_ids = ctx["_eval_input_ids"]
        except KeyError as exc:
            raise KeyError(
                "DownstreamCapabilityTarget.score requires "
                "ctx['_eval_input_ids'] (pre-tokenised eval inputs). "
                "Populate it via ForgePipeline(eval_prompts=...) which "
                "tokenises through the host's tokenizer."
            ) from exc
        if input_ids is None:
            raise KeyError(
                "DownstreamCapabilityTarget.score requires "
                "ctx['_eval_input_ids'] to be non-None"
            )
        device = ctx.get("device", "cpu")

        forged_module = (
            forged.torch_module if hasattr(forged, "torch_module") else forged
        )
        forged_module.to(device).eval()

        # ---- W_dec recovery (three-path precedence, Decision 2) ----
        if self.decode_via_basis:
            W_dec = self._resolve_W_dec(forged_module, ctx)
            W_dec_t = torch.from_numpy(np.ascontiguousarray(W_dec)).to(
                device=device, dtype=torch.float32
            )
        else:
            W_dec_t = None  # signal: skip decode, encoder reads basis coords

        # ---- per-item forward + aggregate ----
        latent_rows: list[Any] = []
        with torch.no_grad():
            for i in range(int(input_ids.shape[0])):
                row = input_ids[i:i + 1].to(device)
                h_basis = forged_module(row)[0, 1:-1, :].float()  # (L, n)
                if W_dec_t is not None:
                    h_d = h_basis @ W_dec_t  # (L, d_model)
                else:
                    h_d = h_basis
                z = self._apply_aggregator(h_d)  # (latent_width,)
                latent_rows.append(z.detach().cpu())
        Z = torch.stack(latent_rows, dim=0).numpy()  # (N_items, latent_width)

        # ---- prevalence filter ----
        Y = self.labels
        if self.min_prevalence > 0:
            n_pos = Y.sum(axis=0)
            keep_cols = np.flatnonzero(n_pos >= self.min_prevalence)
            Y = Y[:, keep_cols]
            if Y.shape[1] == 0:
                raise ValueError(
                    f"DownstreamCapabilityTarget.score: prevalence "
                    f"filter (min_prevalence={self.min_prevalence}) "
                    f"dropped every label column. Lower the threshold "
                    f"or pass a denser label matrix."
                )

        # ---- per-feature × per-label AUC ----
        forge_pf = _best_auc_per_feature(Z, Y)
        self.forge_pf_auc = forge_pf
        # host_pf_auc is populated by sweep_pareto_capability separately
        # (a different sweep call that scores the host); set to None
        # here so downstream consumers see only this target's data.
        score = float(np.nanmean(forge_pf))
        perplexity = max(0.0, 1.0 - score)
        return score, perplexity

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve_W_dec(self, forged_module: Any, ctx: Mapping[str, Any]) -> np.ndarray:
        """Three-path precedence per design.md Decision 2:
        (a) ctx['basis'].W_dec, (b) forged_module.basis_decode buffer,
        (c) pinv(forged_module.basis_encode) with a one-shot warning.
        """
        # Path (a): explicit basis in ctx — exact, free.
        basis = ctx.get("basis")
        if basis is not None and hasattr(basis, "W_dec"):
            return np.ascontiguousarray(basis.W_dec)

        # Path (b): basis_decode buffer on the forged module — exact,
        # no pinv. Cached per id() so repeat scores skip the
        # tensor→numpy copy.
        key = id(forged_module)
        if key in self._w_dec_cache:
            return self._w_dec_cache[key]
        bd = getattr(forged_module, "basis_decode", None)
        if bd is not None and bd.numel() > 0 and float(bd.abs().max()) > 0.0:
            W_dec = bd.detach().cpu().numpy().astype(np.float64)
            self._w_dec_cache[key] = W_dec
            return W_dec

        # Path (c): pinv(basis_encode) fallback.
        be = getattr(forged_module, "basis_encode", None)
        if be is None:
            raise RuntimeError(
                "DownstreamCapabilityTarget: forged module exposes "
                "neither basis_decode nor basis_encode buffers, and no "
                "ctx['basis'] was supplied. Cannot decode forged "
                "hidden states. Supported encoder-only families "
                "(esm2 / whisper_encoder as of v0.7+esm-2-adapter) "
                "emit both buffers; for other forge families the "
                "target needs decode_via_basis=False."
            )
        basis_encode = be.detach().cpu().numpy().astype(np.float64)
        W_dec = np.linalg.pinv(basis_encode)
        # One-shot warning per forged_module instance (silenceable via
        # constructor flag for advanced users).
        if self.warn_on_pinv and key not in self._pinv_warning_emitted:
            self._pinv_warning_emitted.add(key)
            rank = int(np.linalg.matrix_rank(basis_encode))
            n_features = basis_encode.shape[1]
            extra = (
                f" The basis_encode buffer has rank {rank} < n_features="
                f"{n_features}, so the recovered W_dec is approximate."
                if rank < n_features
                else ""
            )
            warnings.warn(
                "DownstreamCapabilityTarget: fell back to "
                "pinv(basis_encode) because the forged module has no "
                "basis_decode buffer. Bundled encoder-only adapters "
                "(esm2, whisper_encoder) emit basis_decode directly — "
                "this fallback fires for legacy or third-party "
                "adapters." + extra,
                UserWarning,
                stacklevel=3,
            )
        self._w_dec_cache[key] = W_dec
        return W_dec

    def _apply_aggregator(self, h_d):
        """Dispatch the aggregator on a per-protein residue tensor.

        ``h_d`` is shape ``(L, d_model)`` (or ``(L, n_features)`` when
        ``decode_via_basis=False``). Returns ``(latent_width,)``.
        """
        if self.aggregator == "pool_then_encode":
            pooled = h_d.mean(dim=0, keepdim=True)
            z = self.encoder(pooled)  # (1, latent_width)
            return z.squeeze(0)
        if self.aggregator == "encode_then_pool":
            z_per_residue = self.encoder(h_d)  # (L, latent_width)
            return z_per_residue.mean(dim=0)
        # Custom callable per the contract documented in design.md.
        return self.aggregator(h_d)


# ----------------------------------------------------------------------
# Scoring kernel — chunked Mann-Whitney AUC.
# Pulled from biosae.sae.evaluation / GroundTruthTarget for vector
# reuse; the math is identical. Kept inline so this module's scoring
# path doesn't depend on bio-sae the package.
# ----------------------------------------------------------------------
def _best_auc_per_feature(Z: np.ndarray, Y: np.ndarray, latent_chunk: int = 512) -> np.ndarray:
    """For each column of Y, return the max-over-latents symmetric AUC.

    ``Z``: ``(N, latent_width)``. ``Y``: ``(N, V)`` binary.
    Returns ``(V,)`` array; entries are ``NaN`` for label columns
    whose positives or negatives are empty.
    """
    n, n_latents = Z.shape
    V = Y.shape[1]
    Y_f = Y.astype(np.float64)
    n_pos = Y_f.sum(axis=0)
    n_neg = n - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    u_offset = n_pos * (n_pos + 1) / 2.0
    denom = np.where(valid, n_pos * n_neg, 1.0)
    rank_template = np.arange(1, n + 1, dtype=np.float64)[:, None]
    running_best = np.full(V, -np.inf, dtype=np.float64)
    for start in range(0, n_latents, latent_chunk):
        stop = min(start + latent_chunk, n_latents)
        chunk = Z[:, start:stop]
        k = chunk.shape[1]
        order = chunk.argsort(axis=0)
        ranks = np.empty((n, k), dtype=np.float64)
        col_idx = np.arange(k)[None, :]
        ranks[order, col_idx] = rank_template
        s_pos = Y_f.T @ ranks  # (V, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = (s_pos - u_offset[:, None]) / denom[:, None]
        sym = np.maximum(auc, 1.0 - auc)
        sym = np.where(valid[:, None], sym, -np.inf)
        running_best = np.maximum(running_best, sym.max(axis=1))
    return np.where(valid, running_best, np.nan)
