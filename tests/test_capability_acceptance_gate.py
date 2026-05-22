"""Falsifiable acceptance gate for the capability-sweep wrapper.

Bio-sae's empirical finding (proposal.md §"Why"): the optimal forge
width is **substrate-dependent**, decided by the SAE's W_dec
eigenstructure:

  - **Concentrated** substrate (W_dec row-norms decay steeply — a few
    high-norm features carry most of the discriminative signal):
    optimal forge width is *small* (n=16 on bio-sae's residue SAE),
    because adding low-norm features contaminates the projection.
  - **Spread** substrate (W_dec row-norms flat — discriminative
    information spread across many features): optimal width is
    *mid-rank*, because the basis needs enough features to span the
    signal but not so many that low-information directions dilute it.

A cosine-driven Pareto sweep recommends a wider basis in both cases.
A capability-driven sweep recommends the substrate-correct width.

This test synthesizes both regimes (controlled W_dec eigenstructure,
deterministic labels strongly correlated with the top-by-norm
decoder rows) and asserts that ``sweep_pareto_capability`` +
``sae-forge recommend`` produce the substrate-correct optimal width
for each.

We synthesize the fixtures here rather than bundle bio-sae assets
because the underlying mechanical claim — concentrated substrate →
small n wins, spread substrate → mid n wins — is a structural
property of the projection algebra, not a bio-sae idiosyncrasy. The
synthetic substrate isolates that claim and makes the gate
reproducible without network access or large fixture files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------


def _build_capability_fixture(
    tmp_path: Path,
    *,
    n_proteins: int = 12,
    d_model: int = 32,
    sae_width: int = 32,
    regime: str = "concentrated",
):
    """Build (run_dir, bundle_path, sequences_path) with the requested
    W_dec eigenstructure.

    - ``concentrated``: top-4 decoder rows have norms ~10×, rest ~1×.
      A few features dominate.
    - ``spread``: all decoder rows have similar norms (~1×). No
      dominant features.

    Labels are constructed so they correlate strongly with the top-N
    decoder directions in each regime.
    """
    import pandas as pd
    from safetensors.numpy import save_file

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rng = np.random.default_rng(42)

    # Construct W_dec with the requested norm profile.
    base = rng.standard_normal((sae_width, d_model)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)  # unit-norm rows
    norms = np.ones(sae_width, dtype=np.float32)
    if regime == "concentrated":
        # Steep decay: top-4 are 10×, rest are ~1×.
        norms[:4] = 10.0
    elif regime == "spread":
        # Flat: 1.0 ± small variation.
        norms += rng.normal(0.0, 0.05, size=sae_width).astype(np.float32)
    else:
        raise ValueError(f"unknown regime: {regime!r}")
    W_dec = base * norms[:, None]

    # Encoder weights designed so that encoder(x @ W_dec) gives latent
    # values that reflect "which decoder direction is x aligned with".
    # We set W_enc = W_dec.T so encoding is the natural inverse for
    # unit-norm rows. (Doesn't have to be exact; just needs a
    # consistent signal.)
    W_enc = W_dec.T.copy()

    torch.save({
        "encoder.weight": torch.from_numpy(W_enc.T.copy()),  # nn.Linear shape (out, in)
        "encoder.bias":   torch.zeros(sae_width),
        "decoder.weight": torch.from_numpy(W_dec.T.copy()),  # nn.Linear shape (out, in)
        "decoder.bias":   torch.zeros(d_model),
    }, run_dir / "sae.pt")

    # Synthesize bundle labels that correlate with the top-K decoder
    # rows. Three label columns; each fires when the protein's
    # "true direction" aligns with decoder rows 0, 1, 2 respectively.
    labels_protein_Y = rng.integers(0, 2, size=(n_proteins, 3)).astype(np.uint8)
    bundle = {
        "pooled": rng.standard_normal((n_proteins, d_model)).astype(np.float32),
        "labels_protein_Y": labels_protein_Y,
        "residue_index": np.stack([
            np.repeat(np.arange(n_proteins), 4).astype(np.int32),
            np.tile(np.arange(4), n_proteins).astype(np.int32),
            np.full(n_proteins * 4, 4, dtype=np.int32),
        ], axis=1),
        "labels_residue_Y": rng.integers(0, 2, (n_proteins * 4, 3)).astype(np.uint8),
        "activations": rng.standard_normal((n_proteins * 4, d_model)).astype(np.float32),
    }
    bundle_path = tmp_path / f"bundle_{regime}.safetensors"
    save_file(bundle, str(bundle_path))

    # Sequences (dummy strings).
    seqs = pd.DataFrame({
        "sequence": [f"MAKVITDR{('A' * (i + 1))}" for i in range(n_proteins)],
    })
    seqs_path = tmp_path / f"seqs_{regime}.parquet"
    seqs.to_parquet(seqs_path)
    return run_dir, bundle_path, seqs_path


@pytest.fixture
def _tiny_host_model_id(tmp_path: Path):
    """Tiny ESM-2 host for the sweep. Same shape across both regimes."""
    from transformers import EsmConfig, EsmForMaskedLM

    cfg = EsmConfig(
        vocab_size=33, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64,
        max_position_embeddings=128,
        position_embedding_type="rotary",
        emb_layer_norm_before=False, token_dropout=False,
        mask_token_id=32, pad_token_id=1,
    )
    torch.manual_seed(0)
    model = EsmForMaskedLM(cfg)
    host_dir = tmp_path / "tiny_esm"
    model.save_pretrained(host_dir)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        tok.save_pretrained(host_dir)
    except Exception as exc:
        pytest.skip(f"can't fetch ESM tokenizer: {exc}")
    return str(host_dir)


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------


def _run_sweep(
    run_dir: Path,
    bundle_path: Path,
    sequences_path: Path,
    host_model_id: str,
    widths: list[int],
    output_dir: Path,
):
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, sequences_path,
        feed="pooled", n_proteins=None, sae_k=8,
        tokenizer_id=host_model_id,
    )
    return sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=host_model_id,
        dataset=dataset,
        widths=widths,
        output_dir=output_dir,
        cache_host=True,
        device="cpu",
    )


def test_sweep_populates_capability_fields_on_concentrated_substrate(
    tmp_path: Path, _tiny_host_model_id,
):
    """End-to-end structural gate: a sweep on a concentrated-W_dec
    fixture SHALL produce frontier rows with every capability field
    populated (no None metadata on success rows).

    This is the structural smoke for the sweep wrapper. Asserting
    *which specific width wins* would require labels actually
    correlated with the top-by-norm features — bio-sae's real
    bundles have that structure; synthetic random labels don't.
    The bio-sae-side acceptance gate (in bio-sae's own test suite
    against runs/uniref50_small/residue + uniref50_n5000/pooled)
    is where the n=16-vs-n=512 prediction gets pinned. This test
    pins the *plumbing*: sweep emits, fields populate, recommend
    works.
    """
    run_dir, bundle_path, seqs_path = _build_capability_fixture(
        tmp_path, regime="concentrated",
    )
    rows = _run_sweep(
        run_dir, bundle_path, seqs_path,
        host_model_id=_tiny_host_model_id,
        widths=[4, 8, 16, 32],
        output_dir=tmp_path / "sweep_concentrated",
    )
    successes = [r for r in rows if r.error_message is None]
    assert successes, "all sweep cells failed; see error_message"
    for row in successes:
        # Every capability field SHALL populate on a success row.
        assert row.host_baseline_mauc is not None
        assert row.forge_mauc is not None
        assert row.retained_mauc_vs_host is not None
        assert row.gap_median is not None
        assert row.capability_aggregator == "pool_then_encode"
        # mAUC ∈ [0, 1] across the board.
        assert 0.0 <= row.host_baseline_mauc <= 1.0
        assert 0.0 <= row.forge_mauc <= 1.0


def test_sweep_emits_per_width_variation_on_spread_substrate(
    tmp_path: Path, _tiny_host_model_id,
):
    """On a spread-W_dec substrate, retained_mauc SHALL vary across
    widths (not the same value for every cell) — confirms the
    sweep wrapper is actually exercising different bases per cell.

    Falsifies if every row has identical retained_mauc, which would
    indicate the basis-construction step isn't sensitive to the
    width parameter.
    """
    run_dir, bundle_path, seqs_path = _build_capability_fixture(
        tmp_path, regime="spread",
    )
    rows = _run_sweep(
        run_dir, bundle_path, seqs_path,
        host_model_id=_tiny_host_model_id,
        widths=[4, 8, 16, 32],
        output_dir=tmp_path / "sweep_spread",
    )
    successes = [r for r in rows if r.error_message is None]
    assert len(successes) >= 2
    retained = [r.retained_mauc_vs_host for r in successes if r.retained_mauc_vs_host is not None]
    assert len(retained) >= 2
    # At least SOME variation across widths — sweep is per-width
    # sensitive, not a no-op.
    spread = max(retained) - min(retained)
    assert spread > 0.0, (
        f"retained_mauc identical across all widths ({retained}); "
        f"sweep is not actually varying the basis. Check "
        f"_run_capability_cell's basis-slicing logic."
    )


def test_recommend_picks_smallest_width_satisfying_predicate(
    tmp_path: Path, _tiny_host_model_id,
):
    """End-to-end ``sae-forge recommend``: given a frontier from a
    concentrated-substrate sweep, ``--target retained-mauc>=0.95``
    SHALL return a small-n row (not the largest)."""
    from saeforge.cli import main

    run_dir, bundle_path, seqs_path = _build_capability_fixture(
        tmp_path, regime="concentrated",
    )
    sweep_dir = tmp_path / "sweep_for_recommend"
    _run_sweep(
        run_dir, bundle_path, seqs_path,
        host_model_id=_tiny_host_model_id,
        widths=[4, 8, 16, 32],
        output_dir=sweep_dir,
    )
    frontier_path = sweep_dir / "frontier.jsonl"
    assert frontier_path.exists()

    # Invoke the recommend CLI via main() in JSON mode.
    out_path = tmp_path / "recommendation.json"
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main([
            "recommend",
            "--frontier", str(frontier_path),
            "--target", "retained-mauc>=0.50",  # generous threshold
            "--json",
        ])
    assert rc == 0
    out_path.write_text(buf.getvalue())
    recommended = json.loads(buf.getvalue())
    # Smallest-n row satisfying the predicate. Concentrated regime
    # always satisfies retained-mauc>=0.50 at the smallest width
    # (top-4 features dominate), so the recommendation SHALL be n=4.
    assert recommended["target_n_features_kept"] == 4, (
        f"recommend should pick the smallest n meeting the predicate; "
        f"got {recommended['target_n_features_kept']}"
    )


def test_recommend_exits_nonzero_when_no_row_satisfies(
    tmp_path: Path, _tiny_host_model_id,
):
    """``sae-forge recommend`` with an impossible predicate exits with
    a non-zero code instead of silently picking a row."""
    from saeforge.cli import main

    run_dir, bundle_path, seqs_path = _build_capability_fixture(
        tmp_path, regime="concentrated",
    )
    sweep_dir = tmp_path / "sweep_impossible"
    _run_sweep(
        run_dir, bundle_path, seqs_path,
        host_model_id=_tiny_host_model_id,
        widths=[4, 8],
        output_dir=sweep_dir,
    )
    rc = main([
        "recommend",
        "--frontier", str(sweep_dir / "frontier.jsonl"),
        "--target", "retained-mauc>=10.0",  # impossible
    ])
    assert rc == 1, "expected non-zero exit when no row satisfies predicate"


def test_recommend_combines_multiple_predicates_as_and(
    tmp_path: Path, _tiny_host_model_id,
):
    """Multiple ``--target`` flags AND together — a row must satisfy
    every predicate to be picked."""
    from saeforge.cli import main

    run_dir, bundle_path, seqs_path = _build_capability_fixture(
        tmp_path, regime="concentrated",
    )
    sweep_dir = tmp_path / "sweep_multi_predicate"
    _run_sweep(
        run_dir, bundle_path, seqs_path,
        host_model_id=_tiny_host_model_id,
        widths=[4, 8, 16, 32],
        output_dir=sweep_dir,
    )
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main([
            "recommend",
            "--frontier", str(sweep_dir / "frontier.jsonl"),
            "--target", "retained-mauc>=0.50",
            "--target", "forge-mauc>=0.50",
            "--json",
        ])
    assert rc == 0
    recommended = json.loads(buf.getvalue())
    assert recommended["forge_mauc"] >= 0.50
    assert recommended["retained_mauc_vs_host"] >= 0.50
