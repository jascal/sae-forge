"""Conformance tests for the WorldModel protocol.

Pins:
- ``WorldModel`` is ``@runtime_checkable``.
- Every bundled adapter satisfies ``WorldModel`` structurally.
- Every adapter's ``default_faithfulness_target()`` returns an instance
  satisfying ``FaithfulnessTarget``.
- LM-family adapters return ``KLTarget``; the Whisper-encoder adapter
  returns ``CosineTarget``.
- ``_default_target_for`` routes through the registry post-refactor.
- ``_default_target_for("fictional")`` raises ``ValueError`` with the
  registered-families list in the message.
- ``NativeModelConfig.__post_init__`` rejects unknown families via
  the registry (post-refactor) and the error message names the
  ``adapter_for_family`` lookup.
"""

from __future__ import annotations

import pytest


def _registered():
    from saeforge.adapters import _REGISTRY
    return sorted({a.family for _, a in _REGISTRY})


def test_world_model_is_runtime_checkable():
    from saeforge.world_model import WorldModel
    # @runtime_checkable Protocols expose this internal attribute.
    assert getattr(WorldModel, "_is_runtime_protocol", False), (
        "WorldModel must be decorated with @runtime_checkable so "
        "isinstance(adapter, WorldModel) works for third-party "
        "structural conformers")


def test_world_model_re_exported_from_top_level():
    import saeforge
    from saeforge.adapters import WorldModel as WMA
    assert saeforge.WorldModel is WMA


def test_every_bundled_adapter_satisfies_world_model():
    from saeforge.adapters import WorldModel, adapter_for_family
    for family in _registered():
        adapter = adapter_for_family(family)
        assert isinstance(adapter, WorldModel), (
            f"{type(adapter).__name__} (family={family!r}) does not "
            f"satisfy WorldModel structurally")


def test_isinstance_world_model_rejects_random_object():
    from saeforge.adapters import WorldModel
    assert not isinstance(object(), WorldModel)
    assert not isinstance(42, WorldModel)


def test_every_adapter_default_target_is_faithfulness_target():
    from saeforge.adapters import adapter_for_family
    from saeforge.eval.faithfulness import FaithfulnessTarget
    for family in _registered():
        target = adapter_for_family(family).default_faithfulness_target()
        assert isinstance(target, FaithfulnessTarget), (
            f"{family!r}.default_faithfulness_target() returned "
            f"{type(target).__name__}, not a FaithfulnessTarget")


@pytest.mark.parametrize("family,target_name", [
    ("gpt2", "kl"),
    ("llama", "kl"),
    ("gemma2", "kl"),
    ("qwen2", "kl"),
    ("qwen3", "kl"),
    ("qwen3_moe", "kl"),
    ("whisper_encoder", "cosine"),
])
def test_default_target_names_match_pre_refactor_dispatch(family, target_name):
    """Byte-identity on the family→target-type mapping: every bundled
    family returns the same target type the pre-refactor
    ``_LM_FAMILIES``-based dispatcher returned (with ``qwen3_moe``
    being the one intentional widening — previously raised, now
    inherits KLTarget like the rest of the LM families)."""
    from saeforge.eval.targets import _default_target_for
    if family not in _registered():
        pytest.skip(f"adapter for {family!r} not registered in this env")
    assert _default_target_for(family).name == target_name


@pytest.mark.parametrize("family", ["fictional", None, ""])
def test_default_target_for_unknown_family_raises(family):
    from saeforge.eval.targets import _default_target_for
    with pytest.raises(ValueError, match="unsupported family"):
        _default_target_for(family)


def test_unknown_family_error_lists_registered_families():
    """Error message names the registered set so the user can recover
    without code-spelunking."""
    from saeforge.eval.targets import _default_target_for
    if not _registered():
        pytest.skip(
            "no bundled adapters registered (base install without "
            "transformers); the error-message-shape pin requires at "
            "least one registered family to compare against")
    with pytest.raises(ValueError) as excinfo:
        _default_target_for("not_a_real_family")
    msg = str(excinfo.value)
    # At least one bundled family must be present in the error.
    for family in ("gpt2", "llama"):
        if family in _registered():
            assert family in msg, (
                f"error should list registered families "
                f"(missing {family!r} in: {msg})")
            return


def test_native_model_config_rejects_unknown_family():
    """``NativeModelConfig.__post_init__`` now validates via the
    registry. Unknown families raise with the registered set in the
    message."""
    from saeforge.model import NativeModelConfig
    with pytest.raises(ValueError, match="family must be one of"):
        NativeModelConfig(
            family="fictional",
            hidden_size=8, qkv_inner_size=8, num_layers=1,
            num_heads=1, head_dim=8, intermediate_size=16,
            vocab_size=10,
        )
