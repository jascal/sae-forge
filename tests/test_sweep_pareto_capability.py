"""Tests for sweep_pareto_capability + ParetoFrontierRow capability fields.

Three suites:
  1. ParetoFrontierRow schema back-compat — v0.7 rows without
     capability fields load cleanly; non-capability rows omit the new
     fields from to_json_dict so byte-equivalence with the old
     format holds.
  2. HostExtractionCache — hit/miss semantics, key invalidation,
     opt-out behaviour.
  3. End-to-end sweep — tiny ESM-2 forge across 2 widths, asserts
     frontier.jsonl is populated with capability fields and the
     host cache hit on the second cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Suite 1: ParetoFrontierRow schema back-compat
# ---------------------------------------------------------------------------


def _base_row_kwargs():
    return dict(
        encoding_label="Rung5(n_amp_qubits=2)",
        target_n_features_kept=128,
        n_features_kept_actual=128,
        pareto_reached_target=True,
        faithfulness_kl=0.5,
        perplexity=1.6,
        final_fine_tune_loss=None,
        sae_checkpoint="/tmp/sae.safetensors",
        forged_model_path=None,
        elapsed_seconds=4.2,
        error_message=None,
    )


def test_row_without_capability_fields_omits_them_from_json():
    """A v0.7-shape row (no capability fields populated) emits a JSON
    dict that does NOT carry any capability keys — preserves byte-
    equivalence with pre-change frontier files."""
    from saeforge.sweep import ParetoFrontierRow

    row = ParetoFrontierRow(**_base_row_kwargs())
    d = row.to_json_dict()
    capability_keys = {
        "host_baseline_mauc", "host_baseline_cov95",
        "forge_mauc", "forge_cov95",
        "retained_mauc_vs_host", "retained_cov95_vs_host",
        "gap_median", "gap_p25", "gap_p75", "gap_p95",
        "n_features_gap_above_0_1", "n_features_negative_gap",
        "capability_aggregator", "capability_min_prevalence",
    }
    assert capability_keys.isdisjoint(d.keys()), (
        f"v0.7-shape row leaked capability keys into JSON: "
        f"{capability_keys & d.keys()}"
    )


def test_row_with_capability_fields_round_trips():
    """Populated capability fields survive to_json_dict / from_json_dict."""
    from saeforge.sweep import ParetoFrontierRow

    row = ParetoFrontierRow(
        **_base_row_kwargs(),
        host_baseline_mauc=0.857,
        host_baseline_cov95=0.173,
        forge_mauc=0.799,
        forge_cov95=0.028,
        retained_mauc_vs_host=0.932,
        retained_cov95_vs_host=0.162,
        gap_median=0.052,
        gap_p25=0.013,
        gap_p75=0.099,
        gap_p95=0.159,
        n_features_gap_above_0_1=95,
        n_features_negative_gap=12,
        capability_aggregator="pool_then_encode",
        capability_min_prevalence=10,
    )
    d = row.to_json_dict()
    assert d["host_baseline_mauc"] == pytest.approx(0.857)
    assert d["capability_aggregator"] == "pool_then_encode"
    back = ParetoFrontierRow.from_json_dict(d)
    assert back.retained_mauc_vs_host == pytest.approx(0.932)
    assert back.gap_p95 == pytest.approx(0.159)
    assert back.n_features_negative_gap == 12
    assert back.capability_min_prevalence == 10


def test_v07_frontier_file_loads_without_capability_fields():
    """A row dict written by a pre-change sweep loads via from_json_dict
    with capability fields defaulting to None."""
    from saeforge.sweep import ParetoFrontierRow

    v07_payload = {
        **_base_row_kwargs(),
        # v0.7 also carried these optional diagnostic fields; included
        # to keep the fixture realistic.
        "host_d_model": 320,
        "basis_rank": 128,
        "quality_ratio": 0.8,
        "quality_tier": "good",
        "logit_std_ratio": 1.1,
        "top1_anomalous": False,
        "polygram_n_clusters": 64,
        "polygram_n_zeroed": 32,
        "polygram_redundancy_ratio": 0.5,
        "polygram_encoding_capacity": 128,
    }
    v07_payload["sae_checkpoint"] = str(v07_payload["sae_checkpoint"])
    row = ParetoFrontierRow.from_json_dict(v07_payload)
    assert row.host_baseline_mauc is None
    assert row.retained_mauc_vs_host is None
    assert row.gap_median is None
    assert row.capability_aggregator is None


def test_row_validation_rejects_bad_capability_values():
    from saeforge.sweep import ParetoFrontierRow

    # mAUC outside [0, 1]
    with pytest.raises(ValueError, match="host_baseline_mauc"):
        ParetoFrontierRow(**_base_row_kwargs(), host_baseline_mauc=1.5)
    # negative retained ratio
    with pytest.raises(ValueError, match="retained_mauc_vs_host"):
        ParetoFrontierRow(**_base_row_kwargs(), retained_mauc_vs_host=-0.1)
    # gap outside [-1, 1]
    with pytest.raises(ValueError, match="gap_p95"):
        ParetoFrontierRow(**_base_row_kwargs(), gap_p95=1.5)
    # negative count
    with pytest.raises(ValueError, match="n_features_gap_above_0_1"):
        ParetoFrontierRow(**_base_row_kwargs(), n_features_gap_above_0_1=-1)


def test_retained_can_exceed_one():
    """Bio-sae's concentrated substrate hit retained_mauc=103% at
    n=16; the validator MUST allow values above 1.0."""
    from saeforge.sweep import ParetoFrontierRow

    row = ParetoFrontierRow(**_base_row_kwargs(), retained_mauc_vs_host=1.032)
    assert row.retained_mauc_vs_host == pytest.approx(1.032)


# ---------------------------------------------------------------------------
# Suite 2: HostExtractionCache
# ---------------------------------------------------------------------------


def test_cache_key_is_content_addressed():
    """Same inputs → same key; different inputs → different key."""
    from saeforge.datasets._host_cache import HostCacheKey

    seqs = ["MAKVITDR", "GLEPVAGR"]
    k1 = HostCacheKey.from_inputs("esm2", seqs, "pool_then_encode", 512)
    k2 = HostCacheKey.from_inputs("esm2", seqs, "pool_then_encode", 512)
    assert k1 == k2
    # Different aggregator → different key.
    k3 = HostCacheKey.from_inputs("esm2", seqs, "encode_then_pool", 512)
    assert k1 != k3
    # Different sequence ordering → different sequences_hash.
    k4 = HostCacheKey.from_inputs("esm2", list(reversed(seqs)),
                                  "pool_then_encode", 512)
    assert k1.sequences_hash != k4.sequences_hash
    # Different max_seq_len → different key.
    k5 = HostCacheKey.from_inputs("esm2", seqs, "pool_then_encode", 256)
    assert k1 != k5


def test_cache_hit_miss_round_trip(tmp_path: Path):
    from saeforge.datasets._host_cache import HostCacheKey, HostExtractionCache

    cache = HostExtractionCache(tmp_path / "host_cache", enabled=True)
    key = HostCacheKey.from_inputs(
        "esm2", ["MAKVITDR"], "pool_then_encode", 512,
    )
    assert not cache.has(key)
    tensor = torch.randn(1, 32)
    cache.save(key, tensor)
    assert cache.has(key)
    loaded = cache.load(key)
    torch.testing.assert_close(loaded, tensor)


def test_cache_opt_out_skips_io(tmp_path: Path):
    from saeforge.datasets._host_cache import HostCacheKey, HostExtractionCache

    cache = HostExtractionCache(tmp_path / "host_cache", enabled=False)
    key = HostCacheKey.from_inputs(
        "esm2", ["MAKVITDR"], "pool_then_encode", 512,
    )
    tensor = torch.randn(1, 32)
    cache.save(key, tensor)  # no-op
    assert not cache.has(key)


def test_cache_corrupted_meta_raises(tmp_path: Path):
    """If the meta file's contents don't match the requested key
    (collision or manual edit), load() raises a clear RuntimeError."""
    from saeforge.datasets._host_cache import HostCacheKey, HostExtractionCache

    cache = HostExtractionCache(tmp_path / "host_cache", enabled=True)
    key = HostCacheKey.from_inputs(
        "esm2", ["MAKVITDR"], "pool_then_encode", 512,
    )
    cache.save(key, torch.randn(1, 32))
    # Corrupt the meta to something that doesn't match.
    meta_path = next(tmp_path.glob("host_cache/*.meta.json"))
    meta_path.write_text(json.dumps({
        "host_model_id": "wrong",
        "sequences_hash": "wrong",
        "aggregator": "wrong",
        "max_seq_len": 0,
    }))
    with pytest.raises(RuntimeError, match="meta mismatch"):
        cache.load(key)


# ---------------------------------------------------------------------------
# Suite 3: End-to-end sweep (tiny scale)
# ---------------------------------------------------------------------------


def _build_bio_sae_fixture(tmp_path: Path, *, n_proteins=4, d_model=32, sae_width=32):
    """Synthesize the three artifacts CapabilityDataset.from_bio_sae expects."""
    import pandas as pd
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({
        "encoder.weight": torch.from_numpy(
            rng.standard_normal((sae_width, d_model)).astype(np.float32)
        ),
        "encoder.bias": torch.zeros(sae_width),
        "decoder.weight": torch.from_numpy(
            rng.standard_normal((d_model, sae_width)).astype(np.float32)
        ),
        "decoder.bias": torch.zeros(d_model),
    }, run_dir / "sae.pt")
    bundle = {
        "pooled": rng.standard_normal((n_proteins, d_model)).astype(np.float32),
        "labels_protein_Y": np.array([
            [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1],
        ], dtype=np.uint8),
        "residue_index": np.stack([
            np.repeat(np.arange(n_proteins), 4).astype(np.int32),
            np.tile(np.arange(4), n_proteins).astype(np.int32),
            np.full(n_proteins * 4, 4, dtype=np.int32),
        ], axis=1),
        "labels_residue_Y": rng.integers(0, 2, (n_proteins * 4, 3)).astype(np.uint8),
        "activations": rng.standard_normal((n_proteins * 4, d_model)).astype(np.float32),
    }
    bundle_path = tmp_path / "bio_bundle.safetensors"
    save_file(bundle, str(bundle_path))
    seqs = pd.DataFrame({"sequence": ["MAKVITDR" + "G" * (i + 1) for i in range(n_proteins)]})
    seqs_path = tmp_path / "sequences.parquet"
    seqs.to_parquet(seqs_path)
    return run_dir, bundle_path, seqs_path


def _build_bio_sae_residue_fixture(
    tmp_path: Path, *, sequences: list[str], d_model=32, sae_width=32,
):
    """Variant of _build_bio_sae_fixture that builds labels_residue_Y
    sized to the actual residue count produced by re-extracting the
    given sequences through ESM-2 t6_8M's tokenizer (minus CLS/EOS).

    The residue feed requires bundle's labels_residue_Y to row-align
    with re-extracted activations. Real bio-sae bundles have this by
    construction (max_seq_len truncation matches build vs read);
    the synthetic fixture replicates that contract."""
    import pandas as pd
    from safetensors.numpy import save_file
    from transformers import AutoTokenizer

    rng = np.random.default_rng(0)
    run_dir = tmp_path / "run_residue"
    run_dir.mkdir()
    torch.save({
        "encoder.weight": torch.from_numpy(
            rng.standard_normal((sae_width, d_model)).astype(np.float32)
        ),
        "encoder.bias": torch.zeros(sae_width),
        "decoder.weight": torch.from_numpy(
            rng.standard_normal((d_model, sae_width)).astype(np.float32)
        ),
        "decoder.bias": torch.zeros(d_model),
    }, run_dir / "sae.pt")

    # Compute residue counts via the same tokenizer the sweep will use.
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    residue_counts = [
        len(tok(s, return_tensors="pt")["input_ids"][0]) - 2  # strip CLS + EOS
        for s in sequences
    ]
    n_total_residues = sum(residue_counts)
    n_proteins = len(sequences)

    bundle = {
        "pooled": rng.standard_normal((n_proteins, d_model)).astype(np.float32),
        "labels_protein_Y": rng.integers(0, 2, (n_proteins, 3)).astype(np.uint8),
        "residue_index": np.stack([
            np.concatenate([
                np.full(L, i, dtype=np.int32)
                for i, L in enumerate(residue_counts)
            ]),
            np.concatenate([
                np.arange(L, dtype=np.int32) for L in residue_counts
            ]),
            np.concatenate([
                np.full(L, L, dtype=np.int32) for L in residue_counts
            ]),
        ], axis=1),
        "labels_residue_Y": rng.integers(
            0, 2, (n_total_residues, 4),
        ).astype(np.uint8),
        "activations": rng.standard_normal(
            (n_total_residues, d_model),
        ).astype(np.float32),
    }
    bundle_path = tmp_path / "bio_bundle_residue.safetensors"
    save_file(bundle, str(bundle_path))
    seqs_df = pd.DataFrame({"sequence": sequences})
    seqs_path = tmp_path / "sequences_residue.parquet"
    seqs_df.to_parquet(seqs_path)
    return run_dir, bundle_path, seqs_path, n_total_residues


@pytest.fixture
def _tiny_host_model_id(tmp_path: Path):
    """Save a tiny ESM-2-shape host to a temp dir so the host loader
    can find it via from_pretrained. The d_model here MUST match the
    fixture SAE's d_model so the basis construction is shape-compatible."""
    pytest.importorskip("transformers")
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
    # Also save the tokenizer (load from the real esm2_t6_8M_UR50D
    # tokenizer would require network; skip via pytest if unreachable).
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        tok.save_pretrained(host_dir)
    except Exception as exc:
        pytest.skip(f"can't fetch ESM tokenizer: {exc}")
    return str(host_dir)


def test_sweep_emits_frontier_with_capability_fields(tmp_path: Path, _tiny_host_model_id):
    """End-to-end smoke: sweep over 2 widths against the tiny host;
    assert frontier.jsonl carries the capability fields populated."""
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=4, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        widths=[8, 16],
        output_dir=tmp_path / "sweep_out",
        cache_host=True,
        device="cpu",
    )
    assert len(rows) == 2
    for row in rows:
        if row.error_message is not None:
            pytest.fail(f"sweep cell failed: {row.error_message}")
        assert row.host_baseline_mauc is not None, "capability field missing"
        assert row.forge_mauc is not None
        assert row.retained_mauc_vs_host is not None
        assert row.capability_aggregator == "pool_then_encode"
        assert row.capability_min_prevalence == 0

    # Frontier file written + parses.
    frontier_path = tmp_path / "sweep_out" / "frontier.jsonl"
    assert frontier_path.exists()
    parsed = [json.loads(line) for line in frontier_path.read_text().splitlines()]
    assert len(parsed) == 2
    for entry in parsed:
        assert "host_baseline_mauc" in entry, (
            "frontier.jsonl entries must carry capability fields"
        )


def test_sweep_cache_hits_on_second_cell(tmp_path: Path, _tiny_host_model_id):
    """First sweep populates the host cache; second sweep with same
    inputs reads from cache (verified by checking the cache file
    exists + is reused on rerun)."""
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=4, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    output_dir = tmp_path / "sweep_cached"
    # First sweep - populates cache.
    sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        widths=[8],
        output_dir=output_dir,
        cache_host=True,
        device="cpu",
    )
    cache_files = list((output_dir / "host_cache").glob("host_*.safetensors"))
    assert len(cache_files) == 1, f"expected 1 cache file, got {len(cache_files)}"
    meta_files = list((output_dir / "host_cache").glob("host_*.meta.json"))
    assert len(meta_files) == 1
    cache_mtime = cache_files[0].stat().st_mtime

    # Second sweep — cache hit; the .safetensors mtime SHOULD NOT
    # update (we read, not write).
    sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        widths=[8],
        output_dir=output_dir,
        cache_host=True,
        device="cpu",
    )
    new_mtime = cache_files[0].stat().st_mtime
    assert new_mtime == cache_mtime, (
        "host cache was overwritten on second sweep — cache logic broken"
    )


# ---------------------------------------------------------------------------
# Suite 4: residue-feed sweep (added by residue-feed slice 4/N)
# ---------------------------------------------------------------------------


def test_sweep_residue_feed_extracts_per_residue(tmp_path: Path, _tiny_host_model_id):
    """End-to-end residue feed: sweep extracts per-residue host
    activations (not mean-pooled) and scores against the dataset's
    per-residue labels. ParetoFrontierRow rows carry the capability
    fields populated; nothing crashes on label-alignment."""
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    # Synthesize a residue-aligned fixture (labels_residue_Y rows ==
    # actual ESM-2-tokenizer-produced residue count).
    sequences = ["MAKVITDR", "GLEPVAGR" + "G" * 3, "TKMRSEW" + "K" * 5]
    run_dir, bundle_path, seqs_path, n_total = _build_bio_sae_residue_fixture(
        tmp_path, sequences=sequences,
    )
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="residue", n_proteins=len(sequences), sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    assert dataset.feed == "residue"
    assert dataset.labels.shape[0] == n_total

    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        widths=[8, 16],
        output_dir=tmp_path / "sweep_residue",
        cache_host=True,
        device="cpu",
    )
    assert len(rows) == 2
    for row in rows:
        if row.error_message is not None:
            pytest.fail(f"residue-feed sweep cell failed: {row.error_message}")
        assert row.host_baseline_mauc is not None
        assert row.forge_mauc is not None
        assert row.retained_mauc_vs_host is not None


def test_pooled_and_residue_caches_dont_collide(tmp_path: Path, _tiny_host_model_id):
    """Same sequences under different feeds MUST produce different
    cache files (cache key includes feed). Otherwise pooled and
    residue runs would silently share corrupt data."""
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    # Build separate fixtures because pooled/residue need different
    # label shapes; but use IDENTICAL sequences across them so the
    # only differing cache-key component is feed.
    pooled_run, pooled_bundle, pooled_seqs = _build_bio_sae_fixture(tmp_path)
    sequences = ["MAKVITDR", "GLEPVAGR"]
    residue_base = tmp_path / "residue_only"
    residue_base.mkdir()
    residue_run, residue_bundle, residue_seqs, _ = _build_bio_sae_residue_fixture(
        residue_base, sequences=sequences,
    )
    # Force same sequences on both datasets.
    import pandas as pd
    pd.DataFrame({"sequence": sequences}).to_parquet(pooled_seqs)

    pooled_ds = CapabilityDataset.from_bio_sae(
        pooled_run, pooled_bundle, pooled_seqs,
        feed="pooled", n_proteins=2, sae_k=4,
        tokenizer_id=_tiny_host_model_id,
    )
    residue_ds = CapabilityDataset.from_bio_sae(
        residue_run, residue_bundle, residue_seqs,
        feed="residue", n_proteins=2, sae_k=4,
        tokenizer_id=_tiny_host_model_id,
    )

    output_dir = tmp_path / "sweep_both_feeds"
    sweep_pareto_capability(
        sae_checkpoint=pooled_run / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=pooled_ds, widths=[4],
        output_dir=output_dir, cache_host=True, device="cpu",
    )
    sweep_pareto_capability(
        sae_checkpoint=residue_run / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=residue_ds, widths=[4],
        output_dir=output_dir, cache_host=True, device="cpu",
    )
    all_caches = sorted((output_dir / "host_cache").glob("host_*.safetensors"))
    # Two distinct cache files SHALL exist — one per feed.
    assert len(all_caches) >= 2, (
        f"expected ≥2 cache files (pooled + residue); got {len(all_caches)}. "
        f"Cache key likely doesn't include feed."
    )


# ---------------------------------------------------------------------------
# Suite 5: partition-aware basis slicing
# (added by add-partition-encoding-capability-validation)
# ---------------------------------------------------------------------------


def test_partition_aware_basis_slicing_proportional():
    """Per spec: tiers get allocated proportionally to their feature
    count."""
    import numpy as np

    from saeforge.sweep_capability import _slice_partition_aware

    # 8 features across 4 tiers [0, 0, 0, 0, 1, 1, 2, 3] →
    # tier_sizes = [4, 2, 1, 1]. target=4: proportional = [2.0, 1.0, 0.5, 0.5].
    # Floor = [2, 1, 0, 0] = 3 allocated; 1 remainder.
    # Residuals = [0.0, 0.0, 0.5, 0.5]; largest residual ties at
    # tiers 2+3. Lowest tier id (2) wins by stable sort.
    row_norms = np.array([5, 4, 3, 2, 9, 8, 7, 6], dtype=np.float64)
    partition_block_ids = np.array([0, 0, 0, 0, 1, 1, 2, 3], dtype=np.int64)
    kept = _slice_partition_aware(
        row_norms=row_norms, partition_block_ids=partition_block_ids,
        target_n_features_kept=4,
    )
    # Tier 0: 2 features (top by norm = indices 0, 1; norms 5, 4).
    # Tier 1: 1 feature (top = index 4; norm 9).
    # Tier 2: 1 feature (only feature = index 6; norm 7).
    # Tier 3: 0 features.
    assert kept.tolist() == sorted([0, 1, 4, 6])


def test_partition_aware_basis_slicing_largest_residual_wins():
    """Edge case: proportional rounding yields off-by-one. Largest
    fractional remainder wins the last slot."""
    import numpy as np

    from saeforge.sweep_capability import _slice_partition_aware

    # 5 features, 2 tiers [0, 0, 0, 1, 1]. target=2: proportional =
    # [1.2, 0.8]. Floor = [1, 0] = 1 allocated; 1 remainder.
    # Residuals = [0.2, 0.8]; largest is tier 1.
    row_norms = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    partition_block_ids = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    kept = _slice_partition_aware(
        row_norms=row_norms, partition_block_ids=partition_block_ids,
        target_n_features_kept=2,
    )
    # Tier 0 gets 1 (top by norm = index 0).
    # Tier 1 gets 1 (top by norm = index 3).
    assert kept.tolist() == [0, 3]


def test_partition_aware_basis_slicing_within_tier_topk():
    """Within each tier, the kept feature ids are the top-K by row
    norm — NOT first-K or last-K by feature id."""
    import numpy as np

    from saeforge.sweep_capability import _slice_partition_aware

    # 6 features, 1 tier [0, 0, 0, 0, 0, 0]. target=3.
    row_norms = np.array([1, 5, 3, 6, 2, 4], dtype=np.float64)
    partition_block_ids = np.zeros(6, dtype=np.int64)
    kept = _slice_partition_aware(
        row_norms=row_norms, partition_block_ids=partition_block_ids,
        target_n_features_kept=3,
    )
    # Top-3 by norm: indices 3 (6), 1 (5), 5 (4). Sorted: [1, 3, 5].
    assert kept.tolist() == [1, 3, 5]


def test_partition_aware_basis_slicing_exceeds_total():
    """target_n_features_kept > sum(tier_sizes) → ValueError."""
    import numpy as np

    from saeforge.sweep_capability import _slice_partition_aware

    with pytest.raises(ValueError, match="exceeds sum"):
        _slice_partition_aware(
            row_norms=np.array([1, 2, 3], dtype=np.float64),
            partition_block_ids=np.zeros(3, dtype=np.int64),
            target_n_features_kept=10,
        )


def test_partition_aware_basis_falls_back_to_row_norm_when_absent(
    tmp_path: Path, _tiny_host_model_id,
):
    """SAE state dict without partition_block_ids SHALL use the
    current row-norm slicing path (back-compat byte-equivalent)."""
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=4, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        widths=[8, 16],
        output_dir=tmp_path / "sweep_no_partition",
        cache_host=False,
        device="cpu",
    )
    assert len(rows) == 2
    for row in rows:
        assert row.error_message is None, row.error_message


def test_partition_aware_basis_used_when_present(
    tmp_path: Path, _tiny_host_model_id,
):
    """SAE state dict WITH partition_block_ids SHALL use the
    partition-aware path. Verified end-to-end with an explicit
    4-tier partition."""
    import torch

    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    sae_state = torch.load(
        run_dir / "sae.pt", map_location="cpu", weights_only=True,
    )
    sae_width = sae_state["decoder.weight"].shape[1]
    quarter = sae_width // 4
    partition = (
        [0] * quarter
        + [1] * quarter
        + [2] * quarter
        + [3] * (sae_width - 3 * quarter)
    )
    sae_state["partition_block_ids"] = torch.tensor(partition, dtype=torch.int64)
    torch.save(sae_state, run_dir / "sae.pt")

    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=4, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id=_tiny_host_model_id,
        dataset=dataset,
        widths=[8, 16],
        output_dir=tmp_path / "sweep_with_partition",
        cache_host=False,
        device="cpu",
    )
    assert len(rows) == 2
    for row in rows:
        assert row.error_message is None, row.error_message


def test_partition_block_ids_shape_mismatch_raises(
    tmp_path: Path, _tiny_host_model_id,
):
    """A partition_block_ids tensor whose shape doesn't match the
    decoder rows SHALL raise a clear ValueError before any forge
    cost is paid."""
    import torch

    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle_path, seqs_path = _build_bio_sae_fixture(tmp_path)
    sae_state = torch.load(
        run_dir / "sae.pt", map_location="cpu", weights_only=True,
    )
    # Wrong shape: too short.
    sae_state["partition_block_ids"] = torch.zeros(3, dtype=torch.int64)
    torch.save(sae_state, run_dir / "sae.pt")
    dataset = CapabilityDataset.from_bio_sae(
        run_dir, bundle_path, seqs_path,
        feed="pooled", n_proteins=4, sae_k=8,
        tokenizer_id=_tiny_host_model_id,
    )
    with pytest.raises(ValueError, match="partition_block_ids has shape"):
        sweep_pareto_capability(
            sae_checkpoint=run_dir / "sae.pt",
            host_model_id=_tiny_host_model_id,
            dataset=dataset,
            widths=[8],
            output_dir=tmp_path / "sweep_bad_partition",
            cache_host=False,
            device="cpu",
        )


def test_residue_feed_label_misalignment_raises(tmp_path: Path, _tiny_host_model_id):
    """If sequences and labels disagree on residue count (e.g. user
    passes a bundle built with a different max_seq_len), the sweep
    SHALL raise loudly at the host-extraction step — not produce
    nonsense AUCs."""
    import numpy as np

    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    sequences = ["MAKVITDR", "GLEPVAGR"]
    run_dir, _, _, n_total = _build_bio_sae_residue_fixture(
        tmp_path, sequences=sequences,
    )
    # Construct a dataset whose labels row count is correct per the
    # frozen-dataclass check but wrong vs the actual sequences. Pad
    # labels with extra dummy rows.
    dataset = CapabilityDataset(
        sequences=sequences,
        labels=np.zeros((n_total + 5, 3), dtype=np.uint8),  # 5 too many
        encoder=lambda x: x[:, :4],  # any callable; not exercised
        tokenizer_id=_tiny_host_model_id,
        feed="residue",
    )
    with pytest.raises(RuntimeError, match="residue rows"):
        sweep_pareto_capability(
            sae_checkpoint=run_dir / "sae.pt",
            host_model_id=_tiny_host_model_id,
            dataset=dataset, widths=[4],
            output_dir=tmp_path / "sweep_misaligned",
            cache_host=False, device="cpu",
        )
