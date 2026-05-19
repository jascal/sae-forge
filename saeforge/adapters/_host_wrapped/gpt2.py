"""GPT-2 host-wrapped forward module.

Wraps a loaded ``GPT2LMHeadModel`` with decode/encode at every block
boundary. Residual stream lives in basis coordinates between blocks;
each block runs host-native (host's exact weights, host LayerNorm,
host softmax, host activation) on the decoded residual.

See ``openspec/changes/add-host-wrapped-forge-fallback`` for the
falsifiable acceptance gate and the per-layer diagnosis that motivated
this path.
"""

from __future__ import annotations

import numpy as np

from saeforge.utils.lazy import require_extra


def build_host_wrapped_gpt2(host_model, basis, scale_boost: float = 1.0):
    """Construct a ``HostWrappedGPT2`` instance.

    ``host_model`` SHALL be a loaded ``GPT2LMHeadModel`` (or compatible
    subclass exposing ``transformer.wte`` / ``transformer.wpe`` /
    ``transformer.h`` / ``transformer.ln_f`` / ``lm_head``).
    """
    cls = _get_host_wrapped_gpt2_class()
    return cls(host_model, basis, scale_boost=scale_boost)


_CLASS_CACHE = None


def _get_host_wrapped_gpt2_class():
    global _CLASS_CACHE
    if _CLASS_CACHE is not None:
        return _CLASS_CACHE

    torch = require_extra("torch", "torch")
    import torch.nn as nn

    class HostWrappedGPT2(nn.Module):
        """GPT-2 forged module that runs host-native inside the stream.

        The residual stream ``z`` lives in basis coordinates at every
        block boundary. Per block: decode ``z`` to ``d_model`` via
        ``W_dec``, run the host's transformer block, encode the result
        via ``pinv`` back to ``n_features``. Entry and exit use the
        host's wte/wpe/ln_f/lm_head directly.

        ``W_dec`` (shape ``(n_features, d_model)``) and ``pinv``
        (shape ``(d_model, n_features)``) are registered as buffers,
        not parameters. ``scale_boost`` is a python float. The host
        model's parameters are held under ``self.host`` and frozen
        (``requires_grad=False``).
        """

        def __init__(self, host_model, basis, scale_boost: float = 1.0):
            super().__init__()
            self.host = host_model
            for p in self.host.parameters():
                p.requires_grad = False
            W_dec = np.asarray(basis.W_dec, dtype=np.float32)
            pinv = np.asarray(basis.pseudoinverse(), dtype=np.float32)
            self.register_buffer("W_dec", torch.from_numpy(W_dec))
            self.register_buffer("pinv", torch.from_numpy(pinv))
            self.scale_boost = float(scale_boost)
            self.n_features = int(W_dec.shape[0])
            self.d_model = int(W_dec.shape[1])

        def forward(self, input_ids):
            t = self.host.transformer
            seq = input_ids.size(-1)
            pos = torch.arange(
                seq, device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)
            x_host = t.wte(input_ids) + t.wpe(pos)
            z = x_host @ self.pinv * self.scale_boost
            for block in t.h:
                x_host = z @ self.W_dec
                block_out = block(x_host)
                # HF GPT2Block returns a tuple (hidden_states, ...);
                # other shapes (bare tensor) tolerated for forward-compat.
                x_host = block_out[0] if isinstance(block_out, tuple) else block_out
                z = x_host @ self.pinv * self.scale_boost
            x_host = z @ self.W_dec
            x_host = t.ln_f(x_host)
            return self.host.lm_head(x_host)

    _CLASS_CACHE = HostWrappedGPT2
    return HostWrappedGPT2
