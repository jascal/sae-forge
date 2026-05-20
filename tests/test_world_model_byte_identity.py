"""Byte-identity guard for the world-model-protocol refactor.

For each bundled architecture family with a tiny fixture, run
``ForgePipeline.run_synthetic(...)`` and hash a stable digest of the
result. Pin the digest per-family; assert it across re-runs.

The digest combines:

- ``n_params`` (forged model parameter count)
- ``round(faithfulness, 8)`` (FP comparison tolerance)
- ``faithfulness_target_name``
- ``basis.W_dec.tobytes()``

The spec's load-bearing invariant: the registry-driven dispatch in
``_build_torch_module`` and ``_default_target_for`` produces
byte-identical results to the pre-refactor hardcoded tables.

When the digests change intentionally (architecture changes, FP
drift across torch versions, fixture rebuild), update the
``_PINNED_DIGESTS`` table and document why in the PR description.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


# Pinned digests per (family, fixture). Captured on the first CI run
# that installs `[torch,orca]` (PR #69's `ci: install [torch] extra ...`
# change) — prior CI was `[dev]`-only, so this test was silently
# skipping via `pytest.importorskip("torch")` at the helper level.
#
# The earlier values were locally captured by individual contributors
# under varying transformers / torch combinations and never matched
# any CI-resolved environment. They are preserved in commit history
# (search test_world_model_byte_identity.py blame).
#
# Going forward: update only when the forge's observable output
# changes for a documented reason. The digest hashes
# (n_params, faithfulness, target_name, basis bytes). The basis is
# deterministic via rng(0); n_params is invariant; the variable is
# `faithfulness`, which depends on the host model's logits — i.e. on
# the transformers version. Pin transformers tightly in the workflow
# if you need stronger reproducibility than "matches current CI env."
_PINNED_DIGESTS: dict[str, str] = {
    "gpt2":   "3fa7f09c4cd5427230e5ade39f37377bd7cbd5ba34716e8aa8dbd5cb6a7426c0",
    "llama":  "57bb9fdf3d175308da0ee15c7f14663888d1e859fea9ce34ed9a48f7bc58593a",
    "gemma2": "7fac2207977d7770a7c86a3238ec79d3ce9cf7fae164fe15283e3dc264cfc409",
    "qwen2":  "2c32e1039497421d1402e64ee620dd2a248520265a7b49dcb472316e1320f86a",
}


def _digest_result(result, basis) -> str:
    """Stable hash of (n_params, faithfulness, target_name, basis bytes)."""
    h = hashlib.sha256()
    h.update(str(int(result.n_params)).encode())
    h.update(b"|")
    h.update(f"{float(result.faithfulness):.8f}".encode())
    h.update(b"|")
    h.update(result.faithfulness_target_name.encode())
    h.update(b"|")
    h.update(basis.W_dec.astype(np.float64).tobytes())
    return h.hexdigest()


def _run_forge_with(host_fixture, tmp_path):
    """Run ForgePipeline.run_synthetic on a tiny host with a small
    synthetic basis. Returns (result, basis)."""
    pytest.importorskip("torch")
    import torch

    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    # Tiny basis sized to the host's hidden dimension.
    n_embd = getattr(host_fixture.config, "n_embd",
                     getattr(host_fixture.config, "hidden_size", None))
    assert n_embd is not None, "host fixture must expose n_embd or hidden_size"
    rng = np.random.default_rng(0)
    n_features = max(4, n_embd // 2)
    W_dec = rng.standard_normal((n_features, n_embd)).astype(np.float64)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-9
    norms = np.linalg.norm(W_dec, axis=1)
    basis = FeatureBasis(
        kept_ids=np.arange(n_features),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        scale_compression_ratio=1.0,
    )

    projector = SubspaceProjector(basis)
    pipeline = ForgePipeline(basis=basis, projector=projector)
    input_ids = torch.randint(0, host_fixture.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(
        host_fixture, tmp_path / "byte_identity", eval_input_ids=input_ids,
    )
    return result, basis


@pytest.mark.parametrize("family,fixture_name", [
    ("gpt2", "tiny_gpt2"),
    ("llama", "tiny_llama"),
    ("gemma2", "tiny_gemma2"),
    ("qwen2", "tiny_qwen2"),
    # qwen3 / qwen3_moe adapters are gated on transformers>=4.51; the
    # registered_families() skip below covers older envs. Digests will
    # populate on first run wherever those adapters DO register.
    ("qwen3", "tiny_qwen3_untied_4layer"),
    ("qwen3_moe", "tiny_qwen3_moe_untied"),
])
def test_forge_result_digest_per_family(
    family, fixture_name, request, tmp_path,
):
    """Run ForgePipeline on a tiny bundled host; assert the result
    digest matches the pinned value.

    On first run with an empty ``_PINNED_DIGESTS`` table, this test
    captures-and-prints the digest; the developer is expected to
    paste it into the table and re-run. Once pinned, subsequent
    runs assert equality.
    """
    from saeforge.adapters import registered_families
    if family not in registered_families():
        pytest.skip(f"adapter for {family!r} not registered "
                    f"(transformers version may be too old)")

    # Seed torch BEFORE materialising the host fixture so the host's
    # random init is reproducible across test runs. The bundled
    # tiny_* fixtures don't pin their own seed.
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    host = request.getfixturevalue(fixture_name)
    torch.manual_seed(0)

    result, basis = _run_forge_with(host, tmp_path)
    digest = _digest_result(result, basis)

    pinned = _PINNED_DIGESTS.get(family)
    if pinned is None:
        pytest.skip(
            f"no pinned digest for family={family!r}; observed "
            f"digest={digest!r}. Paste this into _PINNED_DIGESTS in "
            f"tests/test_world_model_byte_identity.py and re-run to "
            f"activate the guard."
        )
    assert digest == pinned, (
        f"forge result digest drift for family={family!r}: "
        f"pinned={pinned!r}, observed={digest!r}. If the change is "
        f"intentional, update _PINNED_DIGESTS with the new value and "
        f"document why in the PR description."
    )
