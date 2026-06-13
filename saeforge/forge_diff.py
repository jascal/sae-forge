"""Differentiable forge forward — train the encoder `E` against the FULL forge (change
``add-full-forge-encoder-training``), not the activation proxy that returned a null in #115.

Every E-dependent forged weight is ``host_source @ E`` (the ``encode`` projections all route through
``· @ E``); the E-independent ones use ``W_dec @ ·`` and are constant. So we reparametrize the forged
module's E-dependent params as torch functions of a grad-enabled ``E`` (auto-detected by shape:
``host.shape[-1] == d_model`` and ``forged.shape[-1] == n_features``) and run the *existing* forged forward
via ``torch.func.functional_call``. Autograd then reaches ``E`` end-to-end (verified by the task 0.1 spike,
``scripts/spike_forge_diff_autograd.py``). The numpy ``SubspaceProjector.project_module`` /
``NativeModel`` inference path is untouched.

v1 implements the ``esm2`` host family (the gate host). Other families raise ``NotImplementedError`` — no
silent fallback to the activation proxy.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from saeforge.basis import FeatureBasis


class DifferentiableEsm2Forge:
    """Builds the ESM-2 forged module once and exposes a differentiable forward in ``E``.

    Construct once per (host, basis, scale_boost); call :meth:`forge_d` per training step with the current
    ``E`` and a minibatch of pre-tokenized sequences. Only ``E`` carries grad; host weights, ``W_dec`` and
    the const (E-independent) forged params are fixed buffers.
    """

    def __init__(self, host, basis: FeatureBasis, scale_boost: float = 1.0, device: str = "cpu"):
        import torch
        from saeforge.adapters import adapter_for
        from saeforge.model import NativeModel
        from saeforge.projector import SubspaceProjector

        family = _host_family(host)
        if family != "esm2":
            raise NotImplementedError(
                f"DifferentiableEsm2Forge: full-forge training is implemented for the 'esm2' host family "
                f"only in v1; got '{family}'. The differentiable forward for LM / whisper families is a "
                f"follow-up — there is no silent fallback to the activation proxy."
            )
        self.device = device
        self.d_model = basis.d_model
        self.n_features = basis.n_features
        self.W_dec = torch.tensor(np.asarray(basis.W_dec), dtype=torch.float32, device=device)  # (n, d)

        adapter = adapter_for(host)
        proj = SubspaceProjector(basis=basis, scale_boost=scale_boost)
        weights = proj.project_module(host, attention_width="host")
        cfgn = adapter.build_native_config(host, basis.n_features)
        cfgn.forward_mode = "native_in_basis"
        fm = NativeModel.from_projected_weights(cfgn, weights)
        fm._move(dtype="float32", device=device)
        self.module = fm.torch_module.eval()

        # Map each E-dependent forged param name -> its host source tensor (forged = host_source @ E).
        # Const params (E-independent: project_residual_input = W_dec@·, pass-through biases) are detached.
        root = adapter._extract_encoder_root(host)
        host_sd = {k: v.detach().float().to(device) for k, v in root.state_dict().items()}
        self._e_dep: dict[str, "torch.Tensor"] = {}
        self._const: dict[str, "torch.Tensor"] = {}
        for name, p in self.module.named_parameters():
            hs = host_sd.get(name)
            if hs is not None and hs.shape[-1] == self.d_model and p.shape[-1] == self.n_features:
                self._e_dep[name] = hs                 # forged = host_source @ E (grad flows to E)
            else:
                self._const[name] = p.detach()         # E-independent / pass-through

        self._tok_id = getattr(host.config, "_name_or_path", None) or "facebook/esm2_t6_8M_UR50D"
        self._tokenizer = None  # lazy — only needed by tokenize(), not forge_d(pre-tokenized ids)

    def _call_params(self, E):
        """Param dict for functional_call: E-dependent = host_source @ E (grad), const = detached."""
        params = {name: hs.to(E.dtype) @ E for name, hs in self._e_dep.items()}
        params.update(self._const)
        return params

    def forge_d(self, E, input_ids_list, feed: str = "pooled"):
        """Differentiable forged activations decoded to host coords. Returns ``(N, d_model)`` with grad to
        ``E``. ``input_ids_list``: a list of ``(1, L)`` token-id tensors (one per sequence in the minibatch)."""
        import torch
        from torch.func import functional_call

        call_params = self._call_params(E)
        chunks = []
        for ids in input_ids_list:
            h = functional_call(self.module, call_params, (ids.to(self.device),))[0, 1:-1, :]  # (L-2, n)
            chunks.append(h.mean(dim=0, keepdim=True) if feed == "pooled" else h)
        forged_h = torch.cat(chunks, dim=0)            # (N, n_features)
        return forged_h @ self.W_dec.to(E.dtype)       # (N, d_model)

    def tokenize(self, sequences, max_seq_len: int = 512):
        """Pre-tokenize sequences to a list of ``(1, L)`` id tensors (call once; reuse across steps)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tok_id)
        return [self._tokenizer(s[:max_seq_len], return_tensors="pt")["input_ids"] for s in sequences]


def differentiable_forge_h(host, basis: FeatureBasis, E, sequences, *, feed: str = "pooled",
                           aggregator: Any = "pool_then_encode", scale_boost: float = 1.0,
                           device: str = "cpu", max_seq_len: int = 512):
    """One-shot differentiable forged activations ``(N, d_model)`` (grad to ``E``). For repeated training
    steps, construct :class:`DifferentiableEsm2Forge` once and call :meth:`forge_d` (avoids rebuilding)."""
    forge = DifferentiableEsm2Forge(host, basis, scale_boost=scale_boost, device=device)
    ids = forge.tokenize(sequences, max_seq_len=max_seq_len)
    return forge.forge_d(E, ids, feed=feed)


def _host_family(host) -> str:
    """Best-effort host-family tag (mirrors the adapter registry's dispatch)."""
    name = type(host).__name__.lower()
    cfg = getattr(host, "config", None)
    arch = (getattr(cfg, "model_type", "") or "").lower()
    if "esm" in name or "esm" in arch:
        return "esm2"
    if "whisper" in name or "whisper" in arch:
        return "whisper"
    if "gpt2" in name or "gpt2" in arch:
        return "gpt2"
    if "llama" in name or "llama" in arch:
        return "llama"
    if "gemma" in name or "gemma" in arch:
        return "gemma2"
    return arch or name
