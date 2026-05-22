"""Tests for CapabilityDataset + from_bio_sae.

The from_bio_sae constructor parses a bio-sae bundle without
importing biosae. To keep the test self-contained, we synthesize a
minimal bundle / sequences / sae.pt fixture on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _build_bio_sae_fixture(tmp_path: Path, *, n_proteins=8, d_model=32, sae_width=64):
    """Synthesize the three artifacts from_bio_sae expects:
    sae.pt, bio_bundle_*.safetensors, sequences.parquet."""
    import pandas as pd
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)

    # SAE state dict in bio-sae's _ReferenceSAE shape.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sae_state = {
        "encoder.weight": torch.from_numpy(
            rng.standard_normal((sae_width, d_model)).astype(np.float32)
        ),
        "encoder.bias": torch.zeros(sae_width),
        "decoder.weight": torch.from_numpy(
            rng.standard_normal((d_model, sae_width)).astype(np.float32)
        ),
        "decoder.bias": torch.zeros(d_model),
    }
    torch.save(sae_state, run_dir / "sae.pt")

    # Bundle with both protein-scope and residue-scope labels.
    bundle = {
        "pooled": rng.standard_normal((n_proteins, d_model)).astype(np.float32),
        "labels_protein_Y": rng.integers(0, 2, size=(n_proteins, 12)).astype(np.uint8),
    }
    # Residue index: 5 residues per protein for the smoke. Three cols
    # per residue: protein_id, position, length.
    n_res_per = 5
    n_res = n_proteins * n_res_per
    bundle["residue_index"] = np.stack([
        np.repeat(np.arange(n_proteins), n_res_per).astype(np.int32),
        np.tile(np.arange(n_res_per), n_proteins).astype(np.int32),
        np.full(n_res, n_res_per, dtype=np.int32),
    ], axis=1)
    bundle["labels_residue_Y"] = rng.integers(0, 2, size=(n_res, 6)).astype(np.uint8)
    bundle["activations"] = rng.standard_normal((n_res, d_model)).astype(np.float32)
    bundle_path = tmp_path / "bio_bundle.safetensors"
    save_file(bundle, str(bundle_path))

    # Sequences parquet (just dummy protein strings).
    sequences = ["MAKVITDRLG" * 2 for _ in range(n_proteins)]
    sequences_df = pd.DataFrame({"sequence": sequences})
    sequences_path = tmp_path / "sequences.parquet"
    sequences_df.to_parquet(sequences_path)

    return run_dir, bundle_path, sequences_path


def test_from_bio_sae_pooled_feed(tmp_path):
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, sequences_path = _build_bio_sae_fixture(tmp_path)
    ds = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, sequences_path,
        feed="pooled", n_proteins=5, sae_k=16,
    )
    assert len(ds.sequences) == 5
    assert ds.labels.shape == (5, 12)  # protein-scope label vocab
    assert ds.tokenizer_id == "facebook/esm2_t6_8M_UR50D"
    assert ds.aggregator == "pool_then_encode"
    assert ds.decode_via_basis is True
    assert ds.metadata["source"] == "bio_sae"
    assert ds.metadata["feed"] == "pooled"
    assert ds.metadata["sae_latent_width"] == 64


def test_from_bio_sae_residue_feed(tmp_path):
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, sequences_path = _build_bio_sae_fixture(tmp_path)
    ds = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, sequences_path,
        feed="residue", n_proteins=5, sae_k=16,
    )
    # Residue-feed labels: 5 proteins × 5 residues = 25 rows in the
    # fixture, 6 label columns.
    assert ds.labels.shape == (25, 6)
    assert ds.metadata["feed"] == "residue"


def test_encoder_is_topk_with_correct_k(tmp_path):
    """The constructed encoder applies TopK with the configured k."""
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, sequences_path = _build_bio_sae_fixture(tmp_path)
    ds = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, sequences_path, n_proteins=3, sae_k=16,
    )
    x = torch.randn(4, 32)
    z = ds.encoder(x)
    assert z.shape == (4, 64)
    # Exactly k=16 nonzero entries per row.
    n_active = (z != 0).sum(dim=-1)
    assert (n_active == 16).all(), f"got n_active={n_active.tolist()}"


def test_unsupported_feed_raises(tmp_path):
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, sequences_path = _build_bio_sae_fixture(tmp_path)
    with pytest.raises(ValueError, match="feed must be"):
        CapabilityDataset.from_bio_sae(
            run_dir, bundle_path, sequences_path,
            feed="invalid",
        )


def test_missing_sequence_column_raises(tmp_path):
    from saeforge.datasets import CapabilityDataset

    import pandas as pd

    run_dir, bundle_path, _ = _build_bio_sae_fixture(tmp_path)
    bad_seqs = tmp_path / "bad_seqs.parquet"
    pd.DataFrame({"not_sequence": ["ABC"]}).to_parquet(bad_seqs)
    with pytest.raises(ValueError, match="'sequence' column"):
        CapabilityDataset.from_bio_sae(run_dir, bundle_path, bad_seqs)


def test_dataset_threads_through_target(tmp_path):
    """End-to-end: build the dataset, plug it into the target, call
    score() on an identity-basis forge."""
    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.datasets import CapabilityDataset
    from saeforge.eval.targets import DownstreamCapabilityTarget
    from saeforge.model import NativeModel
    from saeforge.projector import SubspaceProjector

    torch.manual_seed(13)
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel

    # Build a fixture matching d_model=32 so the bio-sae SAE's encoder
    # (which expects d=32 input) flows through correctly.
    d = 32
    run_dir, bundle_path, sequences_path = _build_bio_sae_fixture(
        tmp_path, d_model=d,
    )
    ds = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, sequences_path, n_proteins=5, sae_k=16,
    )

    # Set up the matching tiny ESM forge.
    cfg = EsmConfig(
        vocab_size=33, hidden_size=d, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64,
        max_position_embeddings=128,
        position_embedding_type="rotary",
        emb_layer_norm_before=False, token_dropout=False,
        mask_token_id=32, pad_token_id=1,
    )
    host = EsmModel(cfg).eval()
    basis = FeatureBasis(
        kept_ids=np.arange(d, dtype=np.int64),
        W_dec=np.eye(d, dtype=np.float64),
        merged_norms=np.ones(d), original_norms=np.ones(d),
    )
    proj = SubspaceProjector(basis=basis)
    adapter = adapter_for(host)
    weights = adapter.walk(host, proj)
    model = NativeModel.from_projected_weights(
        adapter.build_native_config(host, n_features=d), weights,
    )
    input_ids = torch.tensor([
        [0, 4, 5, 6, 7, 2],
        [0, 7, 8, 9, 10, 2],
        [0, 5, 5, 6, 7, 2],
        [0, 8, 9, 10, 11, 2],
        [0, 4, 4, 5, 5, 2],
    ], dtype=torch.long)

    target = DownstreamCapabilityTarget(
        encoder=ds.encoder, labels=ds.labels,
        aggregator=ds.aggregator,
        min_prevalence=ds.min_prevalence,
        decode_via_basis=ds.decode_via_basis,
    )
    score, perp = target.score(
        forged=model, host=host,
        ctx={"_eval_input_ids": input_ids, "device": "cpu"},
    )
    # Identity basis → forge identical to host → forge AUCs match the
    # mAUC the encoder would achieve on host activations. Compute the
    # host-side reference manually and assert equality (not necessarily
    # 1.0 — random labels on 5 rows don't guarantee any latent
    # perfectly discriminates every column).
    inner = host.esm if hasattr(host, "esm") else host
    with torch.no_grad():
        host_states = torch.cat([
            inner(input_ids=input_ids[i:i + 1]).last_hidden_state[0, 1:-1, :].mean(dim=0, keepdim=True)
            for i in range(input_ids.shape[0])
        ], dim=0)
        host_latents = ds.encoder(host_states.float()).numpy()
    from saeforge.eval.targets.downstream_capability import _best_auc_per_feature
    host_pf = _best_auc_per_feature(host_latents, ds.labels)
    np.testing.assert_allclose(
        target.forge_pf_auc, host_pf, atol=1e-5,
        err_msg=(
            "identity-basis forge must produce per-feature AUCs "
            "identical to host"
        ),
    )
    assert score == pytest.approx(float(np.nanmean(host_pf)), abs=1e-5)
