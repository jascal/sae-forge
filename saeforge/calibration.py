"""Forge magnitude/anomaly diagnostics.

Loads a calibration corpus (host residual activations + lm_head weight)
and exposes pure-numpy helpers for two per-row diagnostics that the
sweep populates on :class:`~saeforge.sweep.ParetoFrontierRow`:

- ``logit_std_ratio``: forged-logit std vs host-logit std on the
  calibration corpus, computed via a layer-L shortcut
  (``host_residual @ host_unembed``). Catches the magnitude-collapse
  / blow-up failure modes.
- ``top1_anomalous``: mode top-1 prediction in a curated glitch-token
  set per tokenizer (the GPT-2 SolidGoldMagikarp family + unicode-
  fragment BPE artifacts). Catches the SolidGoldMagikarp-style blow-up
  signature.

Originally this module also housed a ``scale_boost="calibrate"``
auto-picking mechanism. The 2026-05-16 smoke gate
([[project_fix_scale_boost_smoke]]) found that three successive proxies
for forge KL all diverged from the real target — the calibrate mode
was dropped. The diagnostic surface survives because it's still useful
post-mortem when a sweep produces poor faithfulness_kl.

Module surface
--------------

- ``ANOMALOUS_TOKEN_IDS``: curated glitch-token IDs per tokenizer.
- ``compute_host_logit_std`` / ``compute_forged_logit_std``: per-position
  logit-std diagnostics that populate ``ParetoFrontierRow.logit_std_ratio``.
- ``top1_is_anomalous``: mode top-1 check against the anomalous set;
  populates ``ParetoFrontierRow.top1_anomalous``.
- ``load_calibration_corpus`` / ``load_host_unembed``: ``transformers``-
  dependent loaders, lazy-imported.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — type-only import to avoid cycles
    from saeforge.projector import SubspaceProjector


# GPT-2 tokenizer (shared by gpt2, gpt2-medium, gpt2-large, gpt2-xl).
# Starter set of well-documented glitch tokens: the SolidGoldMagikarp
# family that the broken-projector failure mode resolves to as top-1
# under blow-up. IDs are gpt2-tokenizer constants. The set is
# intentionally non-exhaustive — additions are gated on observing new
# failure-mode signatures in the wild.
_GPT2_ANOMALOUS_TOKEN_IDS: frozenset[int] = frozenset(
    {
        36174,  # 'SolidGoldMagikarp'
        30898,  # 'cloneembedreportprint'
        30899,  # 'rawdownloadcloneembedreportprint'
        30212,  # 'guiActiveUn'
        42089,  # 'TheNitromeFan'
        37444,  # 'StreamerBot'
        37574,  # 'TPPStreamerBot'
        40240,  # 'externalToEVAOnly'
        45544,  # ' Mechdragon'
    }
)


ANOMALOUS_TOKEN_IDS: dict[str, frozenset[int]] = {
    "gpt2": _GPT2_ANOMALOUS_TOKEN_IDS,
    "gpt2-medium": _GPT2_ANOMALOUS_TOKEN_IDS,
    "gpt2-large": _GPT2_ANOMALOUS_TOKEN_IDS,
    "gpt2-xl": _GPT2_ANOMALOUS_TOKEN_IDS,
}


_BUILTIN_CALIBRATION_TEXT = """\
The atmosphere of Earth is the layer of gases retained by gravity, surrounding the planet and forming its planetary atmosphere. It contains roughly seventy-eight percent nitrogen, twenty-one percent oxygen, and small amounts of argon, carbon dioxide, neon, helium and other gases. The atmosphere protects life on Earth by absorbing ultraviolet solar radiation, warming the surface through heat retention, and reducing temperature extremes between day and night.

Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. In most cases, oxygen is also released as a waste product. Most plants, most algae, and cyanobacteria perform photosynthesis; such organisms are called photoautotrophs.

The Roman Empire was the post-Republican period of ancient Rome. As a polity it included large territorial holdings around the Mediterranean Sea in Europe, North Africa, and West Asia. The empire was ruled by emperors. From the accession of Caesar Augustus in 27 BC to the reign of Marcus Aurelius, the empire experienced a period of relative stability that historians call the Pax Romana. Trade flourished along well-maintained roads and shipping lanes, allowing goods and ideas to move efficiently between provinces.

A river is a natural flowing watercourse, usually freshwater, flowing towards an ocean, sea, lake or another river. In some cases a river flows into the ground and becomes dry at the end of its course without reaching another body of water. Small rivers can be referred to using names such as creek, brook, rivulet, and rill. Rivers are part of the hydrological cycle. Water generally collects in a river from precipitation through a drainage basin from surface runoff and other sources such as groundwater recharge, springs, and the release of stored water in natural ice and snowpacks.

Music is the art of arranging sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content. Definitions of music vary depending on culture, though it is an aspect of all human societies and a cultural universal. While scholars agree that music is defined by a few specific elements, there is no consensus on their precise definitions. The creation of music is commonly divided into musical composition, musical improvisation, and musical performance, though the topic itself extends into academic disciplines, criticism, philosophy, and psychology.

Computer programming is the process of designing and building an executable computer program to accomplish a specific computing result or to perform a specific task. Programming involves tasks such as analysis, generating algorithms, profiling algorithms' accuracy and resource consumption, and the implementation of algorithms in a chosen programming language, commonly referred to as coding. The source code of a program is written in one or more languages that are intelligible to programmers, rather than machine code, which is directly executed by the central processing unit.
"""


def load_calibration_corpus(
    host_model_id: str,
    layer: int,
    *,
    n_tokens: int = 1024,
    prompts_path: Path | None = None,
) -> np.ndarray:
    """Load ``(n_tokens, d_model)`` residual-stream activations from ``host_model_id``.

    Lazy-imports ``torch`` and ``transformers``. Runs one forward pass over
    the built-in calibration corpus (or ``prompts_path`` JSONL if provided),
    truncates to ``n_tokens``, and returns the ``layer``-indexed hidden
    state.

    The built-in corpus is deterministic (fixed text, no randomness). The
    JSONL override expects one ``{"text": "..."}`` object per line; lines
    are concatenated with blank-line separators before tokenisation.
    """
    import torch
    import transformers

    if prompts_path is None:
        text = _BUILTIN_CALIBRATION_TEXT
    else:
        import json

        path = Path(prompts_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"load_calibration_corpus: prompts_path {path} not found"
            )
        chunks: list[str] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            chunks.append(str(json.loads(line)["text"]))
        if not chunks:
            raise ValueError(
                f"load_calibration_corpus: prompts_path {path} has no non-empty lines"
            )
        text = "\n\n".join(chunks)

    tokenizer = transformers.AutoTokenizer.from_pretrained(host_model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(n_tokens),
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        host_model_id, torch_dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        out = model(enc.input_ids, output_hidden_states=True)
    # output_hidden_states is a tuple of (batch, T, d_model) tensors,
    # length n_layer+1 — index 0 is the embedding output, index n is
    # after layer n. Mirror the indexing convention used elsewhere in
    # sae-forge / the bundled SAE checkpoints.
    n_layers = len(out.hidden_states) - 1
    if not 0 <= int(layer) <= n_layers:
        raise ValueError(
            f"load_calibration_corpus: layer={layer} out of range "
            f"[0, {n_layers}] for {host_model_id}"
        )
    hidden = out.hidden_states[int(layer)][0]  # (T, d_model)
    return (
        hidden.detach().cpu().float().numpy().astype(np.float64, copy=False)
    )


def load_host_unembed(host_model_id: str) -> np.ndarray:
    """Return ``(vocab, d_model)`` lm_head weight as float64 numpy.

    Lazy-imports ``torch`` and ``transformers``. Used by the row-field
    logit-std diagnostics; never read by the grid-sweep itself.
    """
    import torch  # noqa: F401  — required for the from_pretrained dtype
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        host_model_id, torch_dtype=torch.float32
    )
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError(
            f"load_host_unembed: {host_model_id} has no output embeddings "
            f"(get_output_embeddings() returned None)"
        )
    return (
        head.weight.detach()
        .cpu()
        .float()
        .numpy()
        .astype(np.float64, copy=False)
    )


def compute_host_logit_std(
    host_acts: np.ndarray, host_unembed: np.ndarray
) -> float:
    """Per-position logit std, averaged across positions.

    ``host_acts`` is ``(n_tokens, d_model)``; ``host_unembed`` is
    ``(vocab, d_model)``. Returns the mean of ``std(logits, axis=-1)``,
    a magnitude diagnostic that's invariant to constant logit shifts
    (softmax-equivalent transforms) but tracks the blow-up failure mode
    by construction.
    """
    acts = np.asarray(host_acts, dtype=np.float64)
    unembed = np.asarray(host_unembed, dtype=np.float64)
    if acts.ndim != 2 or unembed.ndim != 2 or acts.shape[1] != unembed.shape[1]:
        raise ValueError(
            f"compute_host_logit_std: shape mismatch — "
            f"host_acts={acts.shape}, host_unembed={unembed.shape}"
        )
    logits = acts @ unembed.T
    return float(logits.std(axis=-1).mean())


def compute_forged_logit_std(
    host_acts: np.ndarray,
    projector: "SubspaceProjector",
    host_unembed: np.ndarray,
) -> float:
    """Per-position logit std after one ``decode(encode(...))`` round-trip.

    Mirror of :func:`compute_host_logit_std` post-projection. The ratio
    of these two quantities is the ``logit_std_ratio`` row diagnostic.
    """
    acts = np.asarray(host_acts, dtype=np.float64)
    unembed = np.asarray(host_unembed, dtype=np.float64)
    z = projector.encode(acts)
    x_recon = projector.decode(z)
    logits = x_recon @ unembed.T
    return float(logits.std(axis=-1).mean())


def top1_is_anomalous(
    host_acts: np.ndarray,
    projector: "SubspaceProjector",
    host_unembed: np.ndarray,
    anomalous_set: "frozenset[int] | set[int]",
) -> bool:
    """``True`` if the mode (most-common) top-1 prediction across
    calibration positions lands in ``anomalous_set``.

    Mode rather than any-position because under blow-up the broken
    projector consistently emits the same anomalous token across many
    positions — a single anomalous argmax would be noise.
    """
    acts = np.asarray(host_acts, dtype=np.float64)
    unembed = np.asarray(host_unembed, dtype=np.float64)
    z = projector.encode(acts)
    x_recon = projector.decode(z)
    logits = x_recon @ unembed.T
    top1 = np.asarray(logits).argmax(axis=-1)
    if top1.size == 0:
        return False
    mode = Counter(int(t) for t in top1.tolist()).most_common(1)[0][0]
    return mode in anomalous_set
