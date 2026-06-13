"""Tests for train_encoder(objective="forge_distill") — full-forge training (task 2.3)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saeforge.basis import FeatureBasis  # noqa: E402
from saeforge.forge_diff import DifferentiableEsm2Forge  # noqa: E402
from saeforge.training import EncoderCalibrationReport, train_encoder  # noqa: E402


def _tiny_esm(d=32):
    from transformers import EsmConfig, EsmForMaskedLM
    cfg = EsmConfig(vocab_size=33, hidden_size=d, num_hidden_layers=2, num_attention_heads=4,
                    intermediate_size=2 * d, max_position_embeddings=128, position_embedding_type="rotary",
                    emb_layer_norm_before=False, token_dropout=False, mask_token_id=32, pad_token_id=1)
    torch.manual_seed(0)
    return EsmForMaskedLM(cfg).eval()


def _basis(n=16, d=32):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float64)
    return FeatureBasis(kept_ids=np.arange(n), W_dec=W,
                        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
                        original_norms=np.linalg.norm(W, axis=1).astype(np.float32))


def test_forge_distill_requires_forge_context():
    basis = _basis()
    X = np.zeros((6, 32))
    Y = (np.random.default_rng(0).random((6, 4)) > 0.5).astype(float)

    def enc(x):
        return torch.relu(x)
    with pytest.raises(ValueError, match="forge"):
        train_encoder(basis=basis, host_acts=X, host_encoder=enc, labels=Y, objective="forge_distill")


def test_forge_distill_runs_through_full_forge():
    host = _tiny_esm()
    basis = _basis()
    try:
        forge = DifferentiableEsm2Forge(host, basis, scale_boost=0.5)
        _ = forge.tokenize(["MKL"])  # triggers the tokenizer; skip if unavailable offline
    except Exception as exc:
        pytest.skip(f"ESM tokenizer unavailable: {exc}")

    rng = np.random.default_rng(0)
    n_prot, d, V = 12, 32, 5
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=int(s))) for s in rng.integers(6, 14, n_prot)]
    X = rng.standard_normal((n_prot, d)).astype(np.float64)            # synthetic host pooled acts (target)
    Wt = torch.tensor(rng.standard_normal((d, V)) / np.sqrt(d), dtype=torch.float32)

    def enc(x):
        return torch.relu(x @ Wt)
    Y = (rng.random((n_prot, V)) > 0.5).astype(np.float64)

    E, rep = train_encoder(
        basis=basis, host_acts=X, host_encoder=enc, labels=Y, objective="forge_distill",
        forge=forge, sequences=seqs, feed="pooled", steps=10, minibatch=6, holdout_frac=0.34, seed=0,
    )
    assert isinstance(rep, EncoderCalibrationReport)
    assert rep.objective == "forge_distill"
    assert E.shape == (d, 16)                                          # matched capacity (d_model, n_features)
    assert rep.n_fit + rep.n_heldout == n_prot
    assert rep.retained_mauc_trained >= rep.retained_mauc_pinv_baseline - 1e-6   # early-stop protection holds
