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


# Pinned digests per (family, fixture) — captured on this change's
# first CI run. Update only when the change is intentional and
# documented.
_PINNED_DIGESTS: dict[str, str] = {
    # Captured on the world-model-protocol PR with explicit pre-fixture
    # torch.manual_seed(0). Update only when the change is intentional
    # and documented in the PR description.
    "gpt2":   "16ef3051e219dd6f4af4ace0306e43ed31b2976c3326932e3475129feb3aeae8",
    # llama / qwen2 digests refreshed by add-llama-family-rope: the
    # forge now applies RoPE in Llama-family attention by default
    # (rope_mode="standard"), which is an intentional behaviour change
    # vs. the pre-fix no-RoPE forge. gemma2 / qwen3* digests below
    # happened to be invariant to the RoPE addition at this fixture
    # size (likely because LN-pinv drift + small synthetic basis
    # already dominate the faithfulness scalar to ~noise floor).
    "llama":  "aea996d999cca3321c800d0bd180a77ea459fffd0f4280d4c61e69f2ec632564",
    "gemma2": "8a763784cb28cff20026827078646ab2c13aab72b7cd2001a0e01c2802dd770b",
    "qwen2":  "a5ccb3a746423fdc05fbe6bc3a4163f54533f895140f8211a6d28c2263304c7b",
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
