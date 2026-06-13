#!/usr/bin/env python3
"""Task 0.1 differentiability spike (change add-full-forge-encoder-training, the make-or-break gate).

Question: can autograd reach the encoder `E` end-to-end through the ESM-2 forged forward, so `E` can be
trained against the FULL forge (not the activation proxy that returned a null in #115)?

Approach (the core of the eventual `differentiable_forge_h`): every E-dependent forged weight is
`host_source @ E` (the `encode` projections all route through `· @ E`); the E-independent ones use
`W_dec @ ·` and are constant. So we reparametrize the forged module's params as torch functions of a
grad-enabled `E` and run the *existing* forged forward via `torch.func.functional_call`. The E-dependent
params auto-detect: `host.shape[-1] == d_model AND forged.shape[-1] == n_features`.

Confirms (a) `E.grad` is finite/nonzero after a dummy loss, and (b) at `E = pinv(W_dec)*scale` the
differentiable forge reproduces the numpy `project_module → NativeModel → forward` to float32 tolerance.
"""
import numpy as np
import torch
from torch.func import functional_call

from saeforge.adapters import adapter_for
from saeforge.basis import FeatureBasis
from saeforge.model import NativeModel
from saeforge.projector import SubspaceProjector


def tiny_esm():
    from transformers import EsmConfig, EsmForMaskedLM
    cfg = EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                    intermediate_size=64, max_position_embeddings=128, position_embedding_type="rotary",
                    emb_layer_norm_before=False, token_dropout=False, mask_token_id=32, pad_token_id=1)
    torch.manual_seed(0)
    return EsmForMaskedLM(cfg).eval()


def build_forge(host, scale_boost=0.5, n_features=16, seed=0):
    d = host.config.hidden_size
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((n_features, d)) / np.sqrt(d)).astype(np.float64)
    basis = FeatureBasis(kept_ids=np.arange(n_features), W_dec=W,
                         merged_norms=np.linalg.norm(W, axis=1).astype(np.float32),
                         original_norms=np.linalg.norm(W, axis=1).astype(np.float32))
    proj = SubspaceProjector(basis=basis, scale_boost=scale_boost)
    adapter = adapter_for(host)
    weights = proj.project_module(host, attention_width="host")
    cfgn = adapter.build_native_config(host, basis.n_features)
    cfgn.forward_mode = "native_in_basis"
    fm = NativeModel.from_projected_weights(cfgn, weights)
    fm._move(dtype="float32", device="cpu")
    return proj, basis, adapter, fm.torch_module


def differentiable_forge_h(host, adapter, mod, basis, E, input_ids):
    """Run the forged forward with E-dependent params reparametrized as `host_source @ E` (grad to E)."""
    d, n = basis.d_model, basis.n_features
    root = adapter._extract_encoder_root(host)
    host_sd = {k: v.detach().float() for k, v in root.state_dict().items()}
    call_params = {}
    for name, p in mod.named_parameters():
        hs = host_sd.get(name)
        if hs is not None and hs.shape[-1] == d and p.shape[-1] == n:
            call_params[name] = hs.to(E.dtype) @ E          # E-dependent: host_source @ E (grad flows)
        else:
            call_params[name] = p.detach().to(E.dtype)      # const (E-independent / pass-through)
    return functional_call(mod, call_params, (input_ids,))


def main():
    host = tiny_esm()
    proj, basis, adapter, mod = build_forge(host)
    ids = torch.tensor([[0, 5, 7, 9, 11, 2]])
    E0 = (np.linalg.pinv(basis.W_dec) * proj.scale_boost).astype(np.float32)  # (d, n)

    # (b) baseline match: differentiable forge at E0 vs the numpy forge (the existing module forward)
    with torch.no_grad():
        numpy_h = mod(ids).float()
    E_base = torch.tensor(E0, dtype=torch.float32)
    diff_h = differentiable_forge_h(host, adapter, mod, basis, E_base, ids).float()
    match = torch.allclose(diff_h, numpy_h, atol=1e-5, rtol=1e-4)
    max_abs = (diff_h - numpy_h).abs().max().item()
    print(f"(b) baseline match (E=pinv·scale): allclose(atol=1e-5,rtol=1e-4)={match}  max|Δ|={max_abs:.2e}")

    # (a) autograd: E.grad finite/nonzero after a dummy loss through the full forge
    E = torch.tensor(E0, dtype=torch.float32, requires_grad=True)
    h = differentiable_forge_h(host, adapter, mod, basis, E, ids)
    loss = (h ** 2).mean()
    loss.backward()
    g = E.grad
    finite = bool(torch.isfinite(g).all())
    nonzero = float(g.abs().sum())
    print(f"(a) autograd to E: grad shape {tuple(g.shape)} (= (d,n)=({basis.d_model},{basis.n_features}))  "
          f"finite={finite}  sum|grad|={nonzero:.4f}  nonzero={nonzero > 0}")

    ok = match and finite and nonzero > 0
    verdict = (
        "PASS — autograd flows to E end-to-end through the ESM-2 forge, and the differentiable path "
        "reproduces the inference forge at the baseline E. differentiable_forge_h is feasible."
        if ok else "FAIL — see above."
    )
    print(f"\nSPIKE VERDICT: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
