"""Shared fixtures for the test suite."""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def synthetic_compressed_sae(tmp_path):
    """Build a fake Polygram-compressed checkpoint + companion report.

    8-feature SAE with 16-dim residual stream. Cluster {2, 5} is collapsed
    onto representative 2 with merged_norm 1.5; row 5 is zeroed. Cluster
    {3, 7} is zeroed onto representative 3 with no merged_norm (zero
    strategy); row 7 is zeroed. Kept ids: [0, 1, 2, 3, 4, 6].
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(42)
    n_total = 8
    d_model = 16
    W_dec = rng.standard_normal((n_total, d_model)).astype(np.float32)
    # Zero the non-representative rows so loader's bookkeeping matches polygram's contract.
    W_dec[5] = 0.0
    W_dec[7] = 0.0
    # Rescale rep row 2 to merged_norm 1.5.
    rep_norm = np.linalg.norm(W_dec[2])
    if rep_norm > 0:
        W_dec[2] *= 1.5 / rep_norm

    checkpoint = tmp_path / "sae.compressed.safetensors"
    save_file({"W_dec": W_dec}, str(checkpoint))

    report = {
        "schema_version": 1,
        "source_checkpoint": "sae.safetensors",
        "source_checkpoint_sha256": "deadbeef",
        "output_checkpoint": str(checkpoint),
        "output_checkpoint_sha256": "feedface",
        "validation_report_dictionary_name": "synthetic",
        "validation_report_schema_version": 1,
        "strategy": "merge",
        "feature_ids": [2, 3, 5, 7],
        "clusters": [
            {
                "cluster_id": 0,
                "members": [2, 5],
                "representative": 2,
                "zeroed": [5],
                "cluster_norm_mean": 1.4,
                "cluster_norm_std": 0.1,
                "merged_norm": 1.5,
            },
            {
                "cluster_id": 1,
                "members": [3, 7],
                "representative": 3,
                "zeroed": [7],
                "cluster_norm_mean": 1.0,
                "cluster_norm_std": 0.05,
                "merged_norm": None,
            },
        ],
        "n_features_zeroed": 2,
        "n_features_kept": 6,
        "n_clusters": 2,
        "scale_compression_ratio": 0.92,
    }
    report_path = tmp_path / "sae.compressed_compression_report.json"
    report_path.write_text(json.dumps(report))

    return {
        "checkpoint": checkpoint,
        "report_path": report_path,
        "report": report,
        "W_dec_full": W_dec,
        "expected_kept_ids": [0, 1, 2, 3, 4, 6],
    }


@pytest.fixture
def tiny_synthetic_basis():
    """In-memory FeatureBasis for projector / model unit tests."""
    from saeforge import FeatureBasis

    rng = np.random.default_rng(0)
    n_kept = 8
    d_model = 16
    W_dec = rng.standard_normal((n_kept, d_model)).astype(np.float64)
    norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n_kept),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )


@pytest.fixture
def synthetic_validation_report(tmp_path):
    """Build a minimal polygram ValidationReport JSON for compress action tests.

    Two clusters: features (0,1) confirmed and (4,5) confirmed against a
    synthetic 8-feature SAE. The actual numerical fields are placeholders;
    only the schema and the confirmed-pair list matter for downstream
    Compressor wiring.
    """
    pytest.importorskip("polygram")
    from polygram import (
        BucketStats,
        CandidatePair,
        ValidationReport,
        ValidationSummary,
    )

    pair_a = CandidatePair(
        i=0, j=1, polygram_overlap=0.9, decoder_overlap=0.95, jaccard=0.5,
        pearson_activation=0.4, kl_ablate_i=0.01, kl_ablate_j=0.01,
        kl_ratio_paired=0.0, kl_log_ratio_abs=0.0,
        n_fires_i=10, n_fires_j=10, n_both_fire=8, n_either_fire=12, gate_pass=True,
    )
    pair_b = CandidatePair(
        i=4, j=5, polygram_overlap=0.85, decoder_overlap=0.9, jaccard=0.5,
        pearson_activation=0.4, kl_ablate_i=0.01, kl_ablate_j=0.01,
        kl_ratio_paired=0.0, kl_log_ratio_abs=0.0,
        n_fires_i=10, n_fires_j=10, n_both_fire=8, n_either_fire=12, gate_pass=True,
    )
    buckets = {"all": BucketStats(
        polygram_range="0-1", n_pairs=2, jaccard_mean=0.5, jaccard_ci_95=(0.4, 0.6),
    )}
    summary = ValidationSummary(
        spearman_polygram_jaccard=0.0, spearman_decoder_jaccard=0.0,
        spearman_polygram_log_kl_abs=0.0, pearson_polygram_jaccard=0.0,
        pearson_decoder_jaccard=0.0, buckets=buckets, outcome="confirmed",
    )
    report = ValidationReport(
        schema_version=1, dictionary_name="synthetic", model_name="gpt2", layer=10,
        n_prompts=4, n_tokens=64, polygram_overlap_threshold=0.7, jaccard_threshold=0.3,
        min_firing_rate=0.0, min_both_fire=0,
        feature_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        pairs=(pair_a, pair_b),
        summary=summary,
        confirmed=((0, 1), (4, 5)),
    )
    path = tmp_path / "validation_report.json"
    report.to_json(path)
    return {"path": path, "report": report}


@pytest.fixture
def tiny_gpt2(monkeypatch):
    """A tiny torch GPT-2 — 16-dim residual, 2 layers, 4 heads, 100 vocab."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_inner=32,
    )
    model = GPT2LMHeadModel(config).eval()
    return model


@pytest.fixture
def tiny_llama():
    """A tiny torch Llama — 128-dim residual, 4 layers, 4 heads, 2 KV heads (GQA), 1024 vocab.

    Bumped to 4 layers (was 2) per the ``hybrid-bridge-llama-family`` change
    so the same fixture can exercise both single-basis and hybrid (n_layer
    >= 3) paths. Architecture is structurally identical to the 2-layer
    version; existing single-basis tests are unaffected.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    return model


@pytest.fixture
def tiny_llama_tied():
    """A tiny torch Llama with tied lm_head weights."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).eval()
    return model


@pytest.fixture
def tiny_gemma2():
    """A tiny torch Gemma-2 with logit soft-capping enabled, GQA, 4-norms-per-block."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Gemma2Config, Gemma2ForCausalLM

    config = Gemma2Config(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
    )
    model = Gemma2ForCausalLM(config).eval()
    return model


@pytest.fixture
def tiny_qwen2():
    """A tiny torch Qwen2 — 128-dim residual, 2 layers, 4 heads, 2 KV heads (GQA), 1024 vocab.

    Q/K/V biases ON (Qwen2 default), untied embeddings.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def tiny_qwen2_untied_4layer():
    """A 4-layer untied Qwen2 — minimum for hybrid (n_layer >= 3).

    Mirrors ``tiny_gpt2_untied_4layer`` so the GPT-2 / Llama / Qwen2
    integration tests share fixture shape. Q/K/V biases ON.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def tiny_qwen3_untied_4layer():
    """A 4-layer untied Qwen3 — minimum for hybrid (n_layer >= 3).

    Qwen3 dense is Llama-shaped + per-head Q/K RMSNorm. Q/K/V biases OFF
    (Qwen2 had them; Qwen3 dropped them — auto-detection handles this).
    Requires ``transformers >= 4.51``; older installs (``[intel]`` extra)
    skip via ``importorskip``.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="4.51")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1024,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture
def feature_basis_128_to_32():
    """A 32-feature FeatureBasis over a 128-d residual (matches the
    tiny_llama / tiny_gemma2 fixtures)."""
    import numpy as np

    from saeforge.basis import FeatureBasis

    n = 32
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n, 128)).astype(np.float32)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=W,
        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
        original_norms=np.linalg.norm(W, axis=1).astype(np.float32),
    )
