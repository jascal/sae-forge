"""CompositionSubspace — the host attention's QK/OV residual read+write geometry.

A single ``FeatureBasis`` is a *residual / feature* basis: it reads 1-operand
*assertions* (directions ``d_X``). The model's *computation* is 2-operand —
the bilinear forms ``M_h = W_Q^h W_K^h.T`` (QK, the "match") and
``OV_h = W_V^h W_O^h`` (OV, the "move"). A rule such as induction is a
property of *pairs* of residual directions, so it survives forging only if
the residual directions attention actually reads/writes are kept. This module
extracts that subspace ``U_C`` per layer, directly from the host weights:

- **read geometry** — a residual change ``Δr`` changes attention scores only
  through ``Δr W_Q^h`` and ``Δr W_K^h``; the dominant left-singular directions
  of the stacked ``[W_Q^h | W_K^h]`` are the directions attention *reads*.
- **write geometry** — attention adds ``Σ_h (attn_h · (r W_V^h)) W_O^h``; the
  column space of ``W_V^h W_O^h`` is what it *writes*.

``U_C`` is the orthonormalised union. Preserving it inside the forge
projection makes the forged QK/OV agree with the host on exactly the
directions attention uses (the circuit-faithfulness invariant). Pure-numpy;
``torch`` is lazy-imported only to read host parameters.

See ``openspec/specs/composition-subspace-preserve``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CompositionSubspace:
    """Orthonormal residual subspace of one layer's attention read+write geometry.

    ``U`` has shape ``(d_model, r)`` with orthonormal columns; ``rank`` is
    ``r``. ``source_heads`` records which heads contributed (``"all"`` or an
    explicit list). ``singular_tail`` is the discarded singular-value spectrum
    (read then write), kept so the rank choice is auditable. ``metadata``
    carries the LN-approximation flag (see ``extract_composition_subspace``).
    """

    U: np.ndarray
    layer: int
    rank: int
    source_heads: list[int] | str
    d_model: int
    singular_tail: np.ndarray = field(default_factory=lambda: np.empty(0))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.U.ndim != 2:
            raise ValueError(f"U must be 2-D (d_model, r); got shape {self.U.shape}")
        d, r = self.U.shape
        if d != self.d_model:
            raise ValueError(f"U rows {d} do not match d_model {self.d_model}")
        if r > self.d_model:
            raise ValueError(f"rank {r} exceeds d_model {self.d_model}")
        gram = self.U.T @ self.U
        off = float(np.linalg.norm(gram - np.eye(r)))
        if off >= 1e-5:
            raise ValueError(
                f"U columns are not orthonormal: ||U^T U - I||_F = {off:.2e} (>= 1e-5)"
            )
        self.rank = r

    def preserved_fraction(self) -> float:
        """``rank / d_model`` — the residual-capacity budget this subspace costs."""
        return self.rank / self.d_model


def _knee(svals: np.ndarray, energy: float = 0.99) -> int:
    """Smallest k whose top-k singular values capture ``energy`` of the spectrum."""
    if svals.size == 0:
        return 0
    e = svals ** 2
    cum = np.cumsum(e) / e.sum()
    return int(np.searchsorted(cum, energy) + 1)


def _top_left_singular(M: np.ndarray, rank: int | None, energy: float) -> tuple[np.ndarray, np.ndarray]:
    """Top-``rank`` left singular vectors of ``M`` (knee on energy when ``rank`` is None)."""
    # full_matrices=False -> U:(d, min(d,n)), s:(min,)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    k = _knee(s, energy) if rank is None else int(rank)
    k = max(1, min(k, U.shape[1]))
    return U[:, :k], s[k:]


def _gpt2_layer_geometry(block, n_head: int, heads, fold_ln1: bool):
    """Return (read_geometry R, write_geometry W) in residual coords for one GPT-2 block.

    ``R = [W_Q^h | W_K^h]`` over the selected heads (ln_1 gain folded into the
    residual side when ``fold_ln1``); ``W = [W_V^h W_O^h | ...]`` stacked over
    the selected heads (raw residual coords — the downstream reader applies its
    own ln_1).
    """
    Wc = block.attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)  # (d, 3d) Conv1D
    Wo = block.attn.c_proj.weight.detach().cpu().numpy().astype(np.float64)  # (d, d)
    d = Wo.shape[0]
    hd = d // n_head
    Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
    head_ids = range(n_head) if heads == "all" else heads
    qk_cols, ov_blocks = [], []
    for h in head_ids:
        sl = slice(h * hd, (h + 1) * hd)
        qk_cols.append(Wq[:, sl])
        qk_cols.append(Wk[:, sl])
        ov_blocks.append(Wv[:, sl] @ Wo[sl, :])           # (d, d) head OV map
    R = np.concatenate(qk_cols, axis=1)                   # (d, 2*|heads|*hd)
    W = np.concatenate(ov_blocks, axis=1)                 # (d, |heads|*d)
    if fold_ln1:
        ln_w = block.ln_1.weight.detach().cpu().numpy().astype(np.float64)
        R = R * ln_w[:, None]                             # residual-side gain fold
    return R, W


def extract_writer_subspace(host, *, writer_heads, rank=None) -> CompositionSubspace:
    """``U_C`` = orthonormalised union of the circuit WRITER heads' OV-OUTPUT row spaces.

    For head ``(L, h)``, ``OV = W_V^h W_O^h`` and its written subspace is ``rowspace(OV)``. This is the
    direction a forge must keep so the downstream circuit still reads what the writer wrote — validated to
    eliminate the induction forge tax where the aggregate reader geometry does not (see
    ``openspec/changes/two-basis-uc-writer-output``). Uniform union in v1.
    """
    cfg = host.config
    if getattr(cfg, "model_type", "") not in ("gpt2",):
        raise NotImplementedError(
            f"extract_writer_subspace supports gpt2 hosts in v1; got model_type="
            f"{getattr(cfg, 'model_type', '')!r}."
        )
    d = cfg.n_embd
    H = cfg.n_head
    hd = d // H
    blocks = host.transformer.h
    ovs = []
    for (L, h) in writer_heads:
        Wc = blocks[L].attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)
        Wo = blocks[L].attn.c_proj.weight.detach().cpu().numpy().astype(np.float64)
        sl = slice(h * hd, (h + 1) * hd)
        ovs.append(Wc[:, 2 * d:3 * d][:, sl] @ Wo[sl, :])          # (d, d) OV_A
    Vt = np.linalg.svd(np.concatenate(ovs, 0), full_matrices=False)[2]   # right singular vecs = written dirs
    r = min(rank or Vt.shape[0], Vt.shape[0], d)
    return CompositionSubspace(
        U=Vt[:r].T, layer=-1, rank=r, source_heads=[list(w) for w in writer_heads], d_model=d,
        metadata={"mode": "writer-output", "writer_heads": [list(w) for w in writer_heads]},
    )


def extract_composition_subspace(
    host,
    *,
    layers,
    rank: int | None = None,
    heads="all",
    fold_ln1: bool = True,
    energy: float = 0.99,
    mode: str = "writer-output",
) -> dict[int, CompositionSubspace]:
    """Extract per-layer ``U_C`` from a host model's attention weights.

    ``mode="writer-output"`` (default, the VALIDATED circuit-preserve): ``heads`` must be a list of
    ``(layer, head)`` writer tuples; the same writer OV-output subspace is preserved at each requested
    layer (dispatches to :func:`extract_writer_subspace`).

    ``mode="reader-geometry"`` (legacy/ablation): per-layer SVD of the aggregate ``[W_Q^h|W_K^h]`` read +
    ``W_V^h W_O^h`` write geometry; ``heads`` is ``"all"`` or a head-index list. This mode does NOT protect
    circuits (the fragile signal is the writers' OV output, not the readers' geometry) — kept for comparison.

    ``rank`` caps the rank; ``None`` uses an energy-knee. The ``ln_1`` mean-subtraction is recorded in each
    subspace's ``metadata`` (``ln_meansub_approx``).
    """
    if mode == "writer-output":
        if heads == "all" or not all(isinstance(h, (tuple, list)) and len(h) == 2 for h in heads):
            raise ValueError(
                "mode='writer-output' requires heads as a list of (layer, head) writer tuples; "
                "got heads=%r. Use circuit_heads to identify them, or mode='reader-geometry'." % (heads,)
            )
        ws = extract_writer_subspace(host, writer_heads=heads, rank=rank)
        return {int(L): CompositionSubspace(U=ws.U, layer=int(L), rank=ws.rank,
                                            source_heads=ws.source_heads, d_model=ws.d_model,
                                            metadata=dict(ws.metadata)) for L in layers}
    if mode != "reader-geometry":
        raise ValueError(f"mode must be 'writer-output' or 'reader-geometry'; got {mode!r}")
    cfg = host.config
    model_type = getattr(cfg, "model_type", "")
    if model_type not in ("gpt2",):
        raise NotImplementedError(
            f"extract_composition_subspace supports gpt2 hosts in v1; got model_type={model_type!r}. "
            f"Other architectures plug in via their adapter's head-geometry helper."
        )
    n_head = cfg.n_head
    d_model = cfg.n_embd
    blocks = host.transformer.h
    out: dict[int, CompositionSubspace] = {}
    for ell in layers:
        R, W = _gpt2_layer_geometry(blocks[ell], n_head, heads, fold_ln1)
        # magnitude of the (un-removed) ln_1 mean-subtraction direction in the read
        # geometry: fraction of ||R|| lying along the all-ones residual direction.
        ones = np.ones(R.shape[0]) / np.sqrt(R.shape[0])
        ln_mag = float(np.linalg.norm(ones @ R) / (np.linalg.norm(R) + 1e-12))
        read_dirs, read_tail = _top_left_singular(R, rank, energy)
        write_dirs, write_tail = _top_left_singular(W, rank, energy)
        # orthonormalise the union (read dirs first so they are preferred under rank pressure)
        stacked = np.concatenate([read_dirs, write_dirs], axis=1)
        Q, _ = np.linalg.qr(stacked)
        # QR pads to min(d, ncols) columns; keep only the independent ones
        r_eff = min(stacked.shape[1], d_model)
        U = Q[:, :r_eff]
        out[ell] = CompositionSubspace(
            U=U,
            layer=int(ell),
            rank=r_eff,
            source_heads="all" if heads == "all" else list(heads),
            d_model=d_model,
            singular_tail=np.concatenate([read_tail, write_tail]),
            metadata={
                "ln_meansub_approx": bool(fold_ln1),
                "ln_meansub_magnitude": ln_mag,
                "ln_meansub_note": "ln_1 gain folded; mean-subtraction (~rank-1) not removed",
                "read_rank": int(read_dirs.shape[1]),
                "write_rank": int(write_dirs.shape[1]),
            },
        )
    return out
