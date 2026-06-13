"""Tests for the differentiable forge forward (change add-full-forge-encoder-training, task 1.3)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saeforge.basis import FeatureBasis  # noqa: E402
from saeforge.forge_diff import DifferentiableEsm2Forge, differentiable_forge_h  # noqa: E402


def _tiny_esm(d=32, layers=2):
    from transformers import EsmConfig, EsmForMaskedLM
    cfg = EsmConfig(vocab_size=33, hidden_size=d, num_hidden_layers=layers, num_attention_heads=4,
                    intermediate_size=2 * d, max_position_embeddings=128, position_embedding_type="rotary",
                    emb_layer_norm_before=False, token_dropout=False, mask_token_id=32, pad_token_id=1)
    torch.manual_seed(0)
    return EsmForMaskedLM(cfg).eval()


def _basis(n=16, d=32, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float64)
    return FeatureBasis(kept_ids=np.arange(n), W_dec=W,
                        merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
                        original_norms=np.linalg.norm(W, axis=1).astype(np.float32))


def test_baseline_E_reproduces_inference_forge():
    """At E = pinv(W_dec)*scale, forge_d reproduces the numpy forge's decoded activations (float32 tol)."""
    host = _tiny_esm()
    basis = _basis()
    forge = DifferentiableEsm2Forge(host, basis, scale_boost=0.5)
    E0 = torch.tensor(np.linalg.pinv(basis.W_dec) * 0.5, dtype=torch.float32)
    ids = [torch.tensor([[0, 5, 7, 9, 11, 2]]), torch.tensor([[0, 3, 8, 2]])]
    # numpy forge: module forward → strip CLS/EOS → pool → decode
    with torch.no_grad():
        ref = torch.cat([forge.module(i)[0, 1:-1, :].mean(0, keepdim=True) for i in ids], 0) @ forge.W_dec
        got = forge.forge_d(E0, ids, feed="pooled")
    assert got.shape == (2, basis.d_model)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), (got - ref).abs().max().item()


def test_autograd_reaches_E():
    """A scalar loss on forge_d backpropagates to E (finite, nonzero gradient)."""
    host = _tiny_esm()
    basis = _basis()
    forge = DifferentiableEsm2Forge(host, basis, scale_boost=0.5)
    E = torch.tensor(np.linalg.pinv(basis.W_dec) * 0.5, dtype=torch.float32, requires_grad=True)
    ids = [torch.tensor([[0, 5, 7, 9, 11, 2]])]
    (forge.forge_d(E, ids, feed="pooled") ** 2).mean().backward()
    assert E.grad is not None and E.grad.shape == (basis.d_model, basis.n_features)
    assert bool(torch.isfinite(E.grad).all()) and float(E.grad.abs().sum()) > 0


def test_residue_feed_keeps_per_residue_rows():
    host = _tiny_esm()
    basis = _basis()
    forge = DifferentiableEsm2Forge(host, basis, scale_boost=0.5)
    E0 = torch.tensor(np.linalg.pinv(basis.W_dec) * 0.5, dtype=torch.float32)
    ids = [torch.tensor([[0, 5, 7, 9, 11, 2]])]  # 6 tokens → 4 residues after CLS/EOS strip
    out = forge.forge_d(E0, ids, feed="residue")
    assert out.shape == (4, basis.d_model)


def test_non_esm2_family_raises():
    """v1 is esm2-only; a non-esm host raises NotImplementedError (no silent fallback)."""
    from types import SimpleNamespace
    fake_gpt2 = SimpleNamespace(config=SimpleNamespace(model_type="gpt2", _name_or_path="gpt2"))
    with pytest.raises(NotImplementedError, match="esm2"):
        DifferentiableEsm2Forge(fake_gpt2, _basis())


def test_one_shot_wrapper_runs():
    """differentiable_forge_h convenience wrapper builds + tokenizes + forges (uses the lazy tokenizer)."""
    host = _tiny_esm()
    basis = _basis()
    try:
        E = torch.tensor(np.linalg.pinv(basis.W_dec) * 0.5, dtype=torch.float32, requires_grad=True)
        out = differentiable_forge_h(host, basis, E, ["MKL", "AAAA"], feed="pooled", scale_boost=0.5)
    except Exception as exc:  # tokenizer fetch may need network in CI-less envs
        pytest.skip(f"tokenizer unavailable: {exc}")
    assert out.shape == (2, basis.d_model)
