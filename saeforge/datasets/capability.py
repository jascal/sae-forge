"""CapabilityDataset — a labeled fixture for capability-aware forge tuning.

Bundles four things a :class:`DownstreamCapabilityTarget` + sweep
need:

1. Input ``sequences`` (protein FASTAs, prompts, mel features, …).
2. A binary ``labels`` matrix (one row per sequence).
3. A downstream task ``encoder`` (``d_model -> latent_width`` callable).
4. The host's ``tokenizer_id`` for re-extraction.

Plus a couple of knobs (``aggregator``, ``min_prevalence``,
``decode_via_basis``) the target also consumes — colocating them
keeps the dataset shape consistent across sm-sae / econ-sae / bio-sae
fixtures.

The ``from_bio_sae`` constructor parses bio-sae's bundle / sequences /
SAE format without depending on ``biosae`` the package. Other fixture
repos (sm-sae, econ-sae) provide their own ``from_<repo>``
constructors in their own codebases; the contract is what
``CapabilityDataset`` carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class CapabilityDataset:
    """Labeled fixture for :class:`DownstreamCapabilityTarget` + sweep.

    Attributes
    ----------
    sequences:
        List of inputs (protein sequences, text prompts, …) to forge
        against.
    labels:
        Binary label matrix. Under ``feed="pooled"`` (default) shape
        is ``(len(sequences), V)`` — one row per protein. Under
        ``feed="residue"`` shape is ``(n_total_residues, V)`` — one
        row per residue across all proteins, ordered protein-major
        per ``residue_index``-style flattening.
    encoder:
        Callable ``(Tensor (..., d_model)) -> Tensor (..., latent_width)``.
        Bio-sae's ``_ReferenceSAE.forward`` returns ``(reconstruction,
        latents)`` — wrap with ``lambda x: sae(x)[1]``.
    tokenizer_id:
        HF id of the host's tokenizer. Used by the sweep wrapper to
        re-extract host activations; not consumed by the dataset
        itself.
    feed:
        ``"pooled"`` (default) or ``"residue"``. Drives the sweep
        wrapper's extraction behaviour:

        - ``"pooled"``: mean-pool per protein → ``(n_proteins, d_model)``
          host activations; score against per-protein labels.
        - ``"residue"``: keep per-residue activations (CLS / EOS
          stripped, concatenated across proteins) →
          ``(n_total_residues, d_model)``; score against per-residue
          labels. Required to match bio-sae's residue-SAE measurements
          on categorical AA / SS3 features.
    aggregator:
        Pool-order under ``feed="pooled"``. ``"pool_then_encode"`` /
        ``"encode_then_pool"`` / callable. Ignored under
        ``feed="residue"`` (residue-feed never pools).
    min_prevalence:
        Forwarded to the target. Drops label columns with positive
        count below this threshold.
    decode_via_basis:
        Forwarded to the target. Set ``False`` when the encoder
        operates in basis coords directly.
    metadata:
        Free-form provenance dict (``run_dir``, ``bundle_path``, …).
        Not consumed by the target or sweep; surfaced on the
        :class:`ParetoFrontierRow` for downstream attribution.
    """

    sequences: list[str]
    labels: Any  # np.ndarray; typed as Any to avoid numpy import here
    encoder: Callable[..., Any]
    tokenizer_id: str
    feed: "Literal['pooled', 'residue']" = "pooled"
    aggregator: "Literal['pool_then_encode', 'encode_then_pool'] | Callable" = (
        "pool_then_encode"
    )
    min_prevalence: int = 0
    decode_via_basis: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feed not in ("pooled", "residue"):
            raise ValueError(
                f"CapabilityDataset(feed=...): expected 'pooled' or "
                f"'residue'; got {self.feed!r}."
            )
        if self.feed == "pooled" and self.labels.shape[0] != len(self.sequences):
            raise ValueError(
                f"CapabilityDataset(feed='pooled'): labels rows "
                f"({self.labels.shape[0]}) must equal len(sequences) "
                f"({len(self.sequences)})."
            )
        if self.feed == "residue" and self.labels.shape[0] < len(self.sequences):
            raise ValueError(
                f"CapabilityDataset(feed='residue'): labels rows "
                f"({self.labels.shape[0]}) must be >= len(sequences) "
                f"({len(self.sequences)}) — each protein contributes "
                f">=1 residue row."
            )

    @classmethod
    def from_bio_sae(
        cls,
        run_dir: "str | Path",
        bundle_path: "str | Path",
        sequences_path: "str | Path",
        *,
        feed: Literal["pooled", "residue"] = "pooled",
        n_proteins: int | None = None,
        max_seq_len: int = 512,
        tokenizer_id: str = "facebook/esm2_t6_8M_UR50D",
        aggregator: str = "pool_then_encode",
        min_prevalence: int = 0,
        sae_variant: Literal["topk", "jumprelu", "l1"] = "topk",
        sae_k: int = 64,
    ) -> "CapabilityDataset":
        """Build a :class:`CapabilityDataset` from a bio-sae bundle.

        Parses three artifacts:

        - ``run_dir/sae.pt`` — a bio-sae ``_ReferenceSAE`` state dict
          (``encoder.weight``, ``encoder.bias``, ``decoder.weight``,
          ``decoder.bias``). Wrapped into a callable ``encoder(x)``
          via the topk / jumprelu / l1 activation dispatch.
        - ``bundle_path`` — a ``bio_bundle_*.safetensors`` carrying
          ``labels_protein_Y`` (pooled feed) or ``labels_residue_Y``
          (residue feed).
        - ``sequences_path`` — a parquet with a ``"sequence"`` column
          (one row per protein). Truncated to ``max_seq_len``.

        Does NOT import ``biosae``; parses the artifacts directly so
        sae-forge stays self-contained on its own dependencies. The
        contract was lifted from bio-sae's
        ``scripts/forge_capability_eval.py``; future drift in
        ``_ReferenceSAE`` shape needs a matching update here.
        """
        import numpy as np
        import torch
        from safetensors.numpy import load_file

        # pandas is an optional dependency — only needed by this
        # constructor (parses bio-sae's sequences parquet). Surface a
        # clear ImportError pointing to the install rather than a bare
        # ModuleNotFoundError from deep inside the read call.
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "CapabilityDataset.from_bio_sae requires pandas to read "
                "the sequences parquet. Install with `pip install pandas` "
                "or `pip install bio-sae[labels]` (pulls pandas + the "
                "label-side stack)."
            ) from exc

        run_dir = Path(run_dir)
        bundle_path = Path(bundle_path)
        sequences_path = Path(sequences_path)

        # ---- Load SAE state + wrap as a callable encoder. ----
        state = torch.load(run_dir / "sae.pt", map_location="cpu", weights_only=True)
        enc_weight = state["encoder.weight"]   # (latent_width, d_model)
        enc_bias = state["encoder.bias"]       # (latent_width,)
        latent_width = enc_weight.shape[0]

        encoder = _build_topk_encoder(enc_weight, enc_bias, variant=sae_variant, k=sae_k)

        # ---- Load bundle's labels (protein or residue feed). ----
        bundle = load_file(str(bundle_path))
        if feed == "pooled":
            labels_full = bundle["labels_protein_Y"]
        elif feed == "residue":
            labels_full = bundle["labels_residue_Y"]
        else:
            raise ValueError(f"feed must be 'pooled' or 'residue'; got {feed!r}")

        # ---- Load protein sequences. ----
        seqs_df = pd.read_parquet(sequences_path)
        if "sequence" not in seqs_df.columns:
            raise ValueError(
                f"CapabilityDataset.from_bio_sae: {sequences_path} has no "
                f"'sequence' column; columns: {list(seqs_df.columns)!r}"
            )

        # ---- Optional n_proteins slice + alignment. ----
        if n_proteins is None:
            n_proteins = min(len(seqs_df), labels_full.shape[0])
        sequences = [
            s[: max_seq_len] for s in seqs_df["sequence"].head(n_proteins)
        ]
        if feed == "pooled":
            labels = labels_full[:n_proteins]
            if labels.shape[0] != len(sequences):
                raise ValueError(
                    f"protein-feed slice mismatch: {labels.shape[0]} "
                    f"labels vs {len(sequences)} sequences"
                )
        else:
            # Residue feed: labels are at residue scope; the dataset
            # rows correspond to per-residue tokens. The sweep wrapper
            # is responsible for concatenating per-residue forge
            # outputs across proteins; ``aggregator`` MUST be a
            # callable (or 'encode_then_pool' followed by no pooling)
            # — this constructor surfaces the labels but doesn't
            # restructure them.
            protein_ids = bundle["residue_index"][:, 0]
            mask = protein_ids < n_proteins
            labels = labels_full[mask]

        # ---- Residues-per-protein vector (residue-feed only). ----
        # The progressive sweep wrapper subsamples to fewer proteins
        # across stages and needs to know each protein's residue
        # count to align labels correctly. Bio-sae's bundles store
        # this implicitly in residue_index[:, 0] (protein_id column).
        residues_per_protein: list[int] | None = None
        if feed == "residue":
            kept_protein_ids = bundle["residue_index"][mask, 0]
            # Each protein contributes a contiguous block of residues;
            # count them in protein-id order.
            residues_per_protein = []
            for pid in range(int(n_proteins)):
                residues_per_protein.append(
                    int((kept_protein_ids == pid).sum())
                )

        metadata = {
            "source":          "bio_sae",
            "run_dir":         str(run_dir),
            "bundle_path":     str(bundle_path),
            "sequences_path":  str(sequences_path),
            "feed":            feed,
            "n_proteins":      int(n_proteins),
            "max_seq_len":     int(max_seq_len),
            "sae_latent_width": int(latent_width),
            "sae_variant":     sae_variant,
            "sae_k":           int(sae_k),
        }
        if residues_per_protein is not None:
            metadata["residues_per_protein"] = residues_per_protein

        return cls(
            sequences=sequences,
            labels=np.ascontiguousarray(labels),
            encoder=encoder,
            tokenizer_id=tokenizer_id,
            feed=feed,
            aggregator=aggregator,
            min_prevalence=min_prevalence,
            decode_via_basis=True,
            metadata=metadata,
        )


def _build_topk_encoder(W_enc, b_enc, *, variant: str, k: int):
    """Wrap an SAE encoder Linear + bias into a callable that applies the
    variant-specific activation function (TopK / JumpReLU / L1).

    Mirrors ``biosae.sae.trainers._ReferenceSAE.encode`` semantics
    exactly so latents from this wrapper match what a freshly-loaded
    bio-sae SAE would produce. The constants (``jumprelu`` theta=0.05)
    are pinned to match bio-sae's defaults.
    """
    import torch

    W_t = W_enc.detach() if hasattr(W_enc, "detach") else torch.as_tensor(W_enc)
    b_t = b_enc.detach() if hasattr(b_enc, "detach") else torch.as_tensor(b_enc)

    if variant == "topk":
        def _encoder(x):
            pre = x @ W_t.T + b_t
            topv, topi = pre.topk(int(k), dim=-1)
            z = torch.zeros_like(pre)
            return z.scatter(-1, topi, topv.relu())
    elif variant == "jumprelu":
        def _encoder(x):
            pre = x @ W_t.T + b_t
            return pre * (pre > 0.05).float()
    elif variant == "l1":
        def _encoder(x):
            pre = x @ W_t.T + b_t
            return pre.relu()
    else:
        raise ValueError(f"unsupported sae_variant: {variant!r}")
    return _encoder
