"""Behavioral identification of circuit WRITER heads — the load-bearing input to writer-output U_C.

Circuit preservation needs the heads that WRITE the circuit's signal (the predecessor-write for induction),
and there is no functional/gradient shortcut (loss-sensitivity != circuit-mechanism; the attribution
subspace is ~orthogonal to the writer subspace). So we identify writers behaviorally, by their attention
signature on a calibration corpus:

  prev_token_heads      heads with high Δ=1 (previous-token) attention — the induction feeders
  duplicate_token_heads heads attending to an earlier occurrence of the SAME token

Each returns up to ``top_k`` heads ABOVE ``min_attention`` (with their score), so a model with few/no strong
movers returns fewer heads rather than noise. GPT-2 in v1 (eager attention). See
``openspec/changes/two-basis-uc-writer-output``.
"""

from __future__ import annotations

import numpy as np


def _attn_scores(host, corpus, kind, ctx):
    """Per-head mean attention of `kind` ('prev'|'dup') over the corpus. Returns (n_layer, n_head)."""
    import torch

    cfg = host.config
    if getattr(cfg, "model_type", "") not in ("gpt2",):
        raise NotImplementedError(
            f"circuit_heads supports gpt2 hosts in v1; got model_type={getattr(cfg, 'model_type', '')!r}."
        )
    nL, H = cfg.n_layer, cfg.n_head
    tr = host.transformer
    chunks = [corpus[i:i + ctx] for i in range(0, len(corpus), ctx) if len(corpus[i:i + ctx]) >= 8]
    acc = np.zeros((nL, H))
    n = 0
    # output_attentions needs eager attention — the default sdpa/flash kernels return no maps
    # (transformers >= 5 returns an empty attentions tuple). Force it, then restore.
    prev_impl = getattr(cfg, "_attn_implementation", None)
    if prev_impl is not None and prev_impl != "eager":
        cfg._attn_implementation = "eager"
    try:
        with torch.no_grad():
            for c in chunks:
                o = tr(input_ids=torch.tensor([c]), output_attentions=True)
                if not o.attentions:
                    raise RuntimeError(
                        "circuit_heads: host returned no attention maps even under eager attention; "
                        "cannot detect writer heads."
                    )
                ca = np.array(c)
                Lc = len(c)
                qi = np.arange(Lc)
                if kind == "prev":
                    for L in range(nL):
                        a = o.attentions[L][0].float().cpu().numpy()
                        acc[L] += np.diagonal(a, offset=-1, axis1=1, axis2=2).sum(1)
                    n += Lc - 1
                else:  # dup: same-token earlier (minus base rate handled by the caller's threshold)
                    DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
                    has = DM.any(1)
                    for L in range(nL):
                        a = o.attentions[L][0].float().cpu().numpy()
                        acc[L] += (a * DM[None]).sum((1, 2))
                    n += int(has.sum())
    finally:
        if prev_impl is not None and prev_impl != "eager":
            cfg._attn_implementation = prev_impl
    return acc / max(n, 1)


def _top(scores, top_k, min_attention):
    flat = [(int(L), int(h), float(scores[L, h])) for L in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(key=lambda r: -r[2])
    return [t for t in flat[:top_k] if t[2] >= min_attention]


def prev_token_heads(host, corpus, *, top_k=4, ctx=96, min_attention=0.15):
    """Up to ``top_k`` previous-token (Δ=1) heads above ``min_attention``, as ``(layer, head, score)``."""
    return _top(_attn_scores(host, corpus, "prev", ctx), top_k, min_attention)


def duplicate_token_heads(host, corpus, *, top_k=4, ctx=96, min_attention=0.05):
    """Up to ``top_k`` duplicate-token (same-token-earlier) heads, as ``(layer, head, score)``."""
    return _top(_attn_scores(host, corpus, "dup", ctx), top_k, min_attention)


def identify(host, corpus, preset, *, top_k=4, ctx=96):
    """Resolve a writer-head preset to ``(layer, head, score)`` triples."""
    if preset == "prev-token":
        return prev_token_heads(host, corpus, top_k=top_k, ctx=ctx)
    if preset == "duplicate-token":
        return duplicate_token_heads(host, corpus, top_k=top_k, ctx=ctx)
    raise ValueError(f"unknown writer preset {preset!r}; supported: 'prev-token', 'duplicate-token'")
