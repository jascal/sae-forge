"""Tests for ForgePipeline + faithfulness_kl + the toy example."""

from __future__ import annotations

import json

import pytest

from saeforge import ForgePipeline, NativeModel, SubspaceProjector


def test_run_synthetic_end_to_end(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "toy", eval_input_ids=eval_input_ids)

    assert isinstance(result.model, NativeModel)
    assert result.n_params > 0
    assert result.faithfulness_kl is not None
    assert result.faithfulness_kl >= 0.0
    assert (tmp_path / "toy" / "forged" / "config.json").is_file()
    assert (tmp_path / "toy" / "forged" / "model.safetensors").is_file()
    payload = json.loads((tmp_path / "toy" / "forge_result.json").read_text())
    assert payload["n_params"] == result.n_params


def test_run_requires_host_model_id_when_called_directly(tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        host_model_id=None,
    )
    with pytest.raises(ValueError, match="host_model_id"):
        pipeline.run(tmp_path / "out")


def test_faithfulness_kl_matches_when_forged_equals_host(tiny_gpt2):
    """Sanity check: KL(host || host) == 0. Constructs a forged model whose
    forward pass exactly matches the host by using an identity-like basis.
    """
    pytest.importorskip("torch")
    import numpy as np
    import torch

    from saeforge import FeatureBasis

    d_model = tiny_gpt2.config.n_embd
    identity_basis = FeatureBasis(
        kept_ids=np.arange(d_model),
        W_dec=np.eye(d_model, dtype=np.float64),
        merged_norms=np.ones(d_model),
        original_norms=np.ones(d_model),
        scale_compression_ratio=1.0,
    )
    projector = SubspaceProjector(identity_basis)
    pipeline = ForgePipeline(basis=identity_basis, projector=projector)
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, "/tmp/sae-forge-identity", eval_input_ids=input_ids)
    assert result.faithfulness_kl < 1e-3, f"identity-basis forge should be ~zero KL, got {result.faithfulness_kl}"


def test_faithfulness_kl_signature(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    import torch

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(basis=tiny_synthetic_basis, projector=projector)
    input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    forged = pipeline.run_synthetic(tiny_gpt2, tmp_path / "sig", eval_input_ids=input_ids)
    assert isinstance(forged.faithfulness_kl, float)
    assert forged.faithfulness_kl >= 0.0


def test_toy_example_runs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from examples.forge_gpt2_toy import main

    summary = main(output_dir=tmp_path / "toy")
    assert summary["n_features"] == 8
    assert summary["n_params"] > 0
    assert summary["faithfulness_kl"] is not None


# ---------------------------------------------------------------------------
# Regression tests for the v0.3 fine-tune-recipe wiring fix.
#
# Before this fix, ForgePipeline.run() against a real HF host always
# took the imperative path — fine-tune fields silently dropped on the
# floor regardless of orchestrator. The example forge_gemma2_2b.py
# (1k-step recipe documented in the script header) had never actually
# trained anything end-to-end. The fix routes orchestrator="fsm" through
# a new _run_real_fsm dispatcher that mirrors _run_synthetic_fsm, and
# the imperative path now warns when finetune_corpus is set so the
# silent skip never recurs.
# ---------------------------------------------------------------------------


def test_run_fsm_dispatches_through_recipe(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """orchestrator='fsm' on the real-host run() routes through
    _run_real_fsm and the FSM's fine_tune_model action picks the recipe
    path when a pre-built iterator is supplied. Mocks
    AutoModelForCausalLM/AutoTokenizer so the test doesn't hit HF.
    """
    pytest.importorskip("orca_runtime_python")
    from unittest.mock import patch

    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        host_model_id="gpt2-stub",
        orchestrator="fsm",
        finetune_total_steps=4,
        finetune_warmup_steps=1,
        finetune_peak_lr=1e-3,
        finetune_batch_size=2,
        finetune_seq_len=8,
        finetune_log_every=1,
        finetune_eval_every=10000,
        finetune_save_every=10000,
        eval_prompts=["smoke prompt"],
    )

    def gen():
        while True:
            yield torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))

    class _StubTokenizer:
        # Minimal stand-in: emits a tiny tensor for any prompt list.
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
            return {"input_ids": torch.tensor([[1, 2, 3, 4]] * len(prompts))}

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=tiny_gpt2,
    ), patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=_StubTokenizer(),
    ):
        result = pipeline.run(
            tmp_path / "fsm_run", finetune_iterator=gen()
        )

    # The recipe ran end-to-end: transitions log carries an action whose
    # mode is "recipe" (vs "passthrough" or "v01_smoke").
    log = result.extras["transitions_log"]
    finetune_entry = next(
        e for e in log if e.get("action") == "fine_tune_model"
    )
    assert finetune_entry["mode"] == "recipe"
    assert finetune_entry["n_steps"] == 4
    assert "final_loss" in finetune_entry

    # Faithfulness was computed on the post-tune model (eval_prompts
    # supplied → input_ids tokenised → FSM evaluate_faithfulness ran).
    assert result.faithfulness_kl is not None


def test_run_imperative_warns_when_finetune_corpus_set(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """Setting finetune_corpus on the imperative path is a silent no-op
    (recipe only runs on the FSM path). The fix surfaces a UserWarning
    so callers see the mismatch instead of getting a forge that looks
    like it ran but didn't.
    """
    import warnings
    from unittest.mock import patch

    from saeforge import ForgePipeline, SubspaceProjector

    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        host_model_id="gpt2-stub",
        # orchestrator defaults to "imperative" — that's the silent-skip path.
        finetune_corpus="HuggingFaceFW/fineweb-edu",
        finetune_total_steps=1000,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=tiny_gpt2,
        ):
            pipeline.run(tmp_path / "imperative_run")

    finetune_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "fine-tune recipe" in str(w.message)
    ]
    assert len(finetune_warnings) == 1, [str(w.message) for w in caught]
    msg = str(finetune_warnings[0].message)
    assert "imperative" in msg
    assert "orchestrator='fsm'" in msg
    assert "fineweb-edu" in msg


# ---------------------------------------------------------------------------
# Regression tests for the FSM-failure-surfacing fix.
#
# Before the fix, an action raising inside the FSM (e.g. AttributeError
# from the GPT-2-only grad-checkpointing path running against a
# ForgedLlama) was swallowed into final_state: failed and returned to
# the caller as a ForgeResult with n_params=0, faithfulness_kl=0.0,
# exit code 0. The fix raises ForgeFailed instead so the caller sees
# the recorded error_message.
# ---------------------------------------------------------------------------


def _minimal_regrow_config():
    """Cheap RegrowConfig stand-in so the adaptive validation matrix can
    exercise the new ``__post_init__`` branch without tripping the older
    ``regrow_count > 0 requires regrow`` check.
    """
    pytest.importorskip("polygram")
    from polygram import RegrowConfig

    return RegrowConfig(model_name="gpt2", layer=0)


def test_adaptive_regrow_without_regrow_max_raises_value_error(tiny_synthetic_basis):
    """``adaptive_regrow=True`` requires ``regrow_max > regrow_count``."""
    projector = SubspaceProjector(tiny_synthetic_basis)
    with pytest.raises(ValueError, match=r"regrow_max"):
        ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            regrow=_minimal_regrow_config(),
            adaptive_regrow=True,
            regrow_count=5,
            regrow_max=0,
            n_features_target=128,
        )


def test_adaptive_regrow_with_regrow_max_below_regrow_count_raises(tiny_synthetic_basis):
    """``regrow_max <= regrow_count`` is incoherent under adaptation."""
    projector = SubspaceProjector(tiny_synthetic_basis)
    with pytest.raises(ValueError, match=r"regrow_max"):
        ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            regrow=_minimal_regrow_config(),
            adaptive_regrow=True,
            regrow_count=10,
            regrow_max=10,
            n_features_target=128,
        )


def test_adaptive_regrow_without_n_features_target_raises(tiny_synthetic_basis):
    """``adaptive_regrow=True`` requires ``n_features_target > 0``."""
    projector = SubspaceProjector(tiny_synthetic_basis)
    with pytest.raises(ValueError, match=r"n_features_target"):
        ForgePipeline(
            basis=tiny_synthetic_basis,
            projector=projector,
            regrow=_minimal_regrow_config(),
            adaptive_regrow=True,
            regrow_count=5,
            regrow_max=32,
            n_features_target=0,
        )


def test_adaptive_regrow_disabled_silently_accepts_other_knobs(tiny_synthetic_basis):
    """When master toggle is off, the dependent knobs are inert (no validation)."""
    projector = SubspaceProjector(tiny_synthetic_basis)
    # Should construct without error even with ostensibly-incoherent knobs;
    # no ``regrow=`` is needed because ``regrow_count`` is the default 0.
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        adaptive_regrow=False,
        regrow_max=99,
        n_features_target=999,
        regrow_damping=0.7,
    )
    assert pipeline.adaptive_regrow is False
    assert pipeline.regrow_max == 99
    assert pipeline.n_features_target == 999


def test_run_fsm_raises_forge_failed_on_action_error(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """Simulate an action failure: monkey-patch project_to_subspace to
    raise. The FSM should record a log_error transition and the
    pipeline should raise ForgeFailed (not return a successful-looking
    result with n_params=0 / KL=0.0)."""
    pytest.importorskip("orca_runtime_python")
    from unittest.mock import patch

    from saeforge import ForgeFailed, ForgePipeline, SubspaceProjector

    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=SubspaceProjector(tiny_synthetic_basis),
        host_model_id="gpt2-stub",
        orchestrator="fsm",
    )

    def _boom(ctx, payload=None):
        raise AttributeError(
            "'ForgedLlama' object has no attribute 'transformer'"
        )

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=tiny_gpt2,
    ), patch.dict(
        "saeforge.actions.ACTION_TABLE",
        {"project_to_subspace": _boom},
    ):
        with pytest.raises(ForgeFailed) as excinfo:
            pipeline.run(tmp_path / "fsm_failed")

    # Error message surfaces from the action's exception text.
    msg = str(excinfo.value)
    assert "ForgedLlama" in msg or "transformer" in msg or "FSM ended" in msg
    # Diagnostics are attached so callers can inspect what got far.
    assert hasattr(excinfo.value, "transitions_log")
    assert hasattr(excinfo.value, "extras")
    log = excinfo.value.transitions_log
    assert any(e.get("action") == "log_error" for e in log)
