"""Tests for v0.3 FSM recipe-delegation: action picks recipe path when a
corpus or pre-built iterator is supplied, falls back to v0.1 smoke
behaviour otherwise (preserving the byte-equivalence safety net).
"""

from __future__ import annotations

import hashlib

import pytest


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fsm_with_pretokenized_iterator_uses_recipe_path(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """Pre-tokenized iterator routes to the recipe path. This is the realistic
    test path: tiny_gpt2's vocab is 100 so we can't use a real tokenizer.
    """
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        finetune_total_steps=4,
        finetune_warmup_steps=1,
        finetune_peak_lr=1e-3,
        finetune_batch_size=2,
        finetune_seq_len=8,
        finetune_log_every=1,
        finetune_eval_every=10000,
        finetune_save_every=10000,
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    def gen():
        while True:
            yield torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))

    result = pipeline.run_synthetic(
        tiny_gpt2,
        tmp_path / "out",
        eval_input_ids=eval_input_ids,
        finetune_iterator=gen(),
    )
    finetune_entry = next(
        e for e in result.extras["transitions_log"] if e.get("action") == "fine_tune_model"
    )
    assert finetune_entry["mode"] == "recipe"
    assert finetune_entry["n_steps"] == 4
    assert "final_loss" in finetune_entry


def test_fsm_no_corpus_no_input_ids_passthrough(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """Without finetune_corpus and without _finetune_input_ids, action passes through."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis, projector=projector, orchestrator="fsm",
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    result = pipeline.run_synthetic(tiny_gpt2, tmp_path / "noft", eval_input_ids=eval_input_ids)
    finetune_entry = next(
        e for e in result.extras["transitions_log"] if e.get("action") == "fine_tune_model"
    )
    assert finetune_entry["mode"] == "passthrough"


def test_fsm_with_only_input_ids_uses_v01_smoke_path(
    tiny_gpt2, tiny_synthetic_basis, tmp_path
):
    """v0.1 byte-equivalence safety net — input_ids without corpus → v0.1 4-step path."""
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        finetune_steps=4,
        finetune_lr=1e-2,
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))
    finetune_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))
    result = pipeline.run_synthetic(
        tiny_gpt2,
        tmp_path / "smoke",
        eval_input_ids=eval_input_ids,
        finetune_input_ids=finetune_input_ids,
    )
    finetune_entry = next(
        e for e in result.extras["transitions_log"] if e.get("action") == "fine_tune_model"
    )
    assert finetune_entry["mode"] == "trained"  # v0.1 mode label
    assert finetune_entry["n_steps"] == 4


def test_recipe_disabled_path_byte_equivalent_to_v01(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    """The byte-equivalence safety net from forge-outer-loop-fsm continues to hold:
    no corpus + no finetune iterator → forged model identical to imperative path.
    """
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    torch.manual_seed(0)
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    imp = ForgePipeline(
        basis=tiny_synthetic_basis, projector=projector, orchestrator="imperative"
    )
    imp_result = imp.run_synthetic(tiny_gpt2, tmp_path / "imp", eval_input_ids=eval_input_ids)

    fsm = ForgePipeline(
        basis=tiny_synthetic_basis, projector=projector, orchestrator="fsm"
    )
    fsm_result = fsm.run_synthetic(tiny_gpt2, tmp_path / "fsm", eval_input_ids=eval_input_ids)

    imp_w = tmp_path / "imp" / "forged" / "model.safetensors"
    fsm_w = tmp_path / "fsm" / "forged" / "model.safetensors"
    assert _file_sha256(imp_w) == _file_sha256(fsm_w)
    assert imp_result.n_params == fsm_result.n_params


def test_pretokenized_iterator_does_not_import_datasets(
    tiny_gpt2, tiny_synthetic_basis, tmp_path, monkeypatch
):
    """Pre-tokenized iterator path triggers no `datasets` import.
    (Local-file path with a real tokenizer is exercised by the example
    script and the corpus-iterator unit tests.)
    """
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import sys

    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        finetune_total_steps=2,
        finetune_warmup_steps=1,
        finetune_peak_lr=1e-3,
        finetune_batch_size=2,
        finetune_seq_len=8,
        finetune_log_every=1,
        finetune_eval_every=10000,
        finetune_save_every=10000,
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    def gen():
        while True:
            yield torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))

    pipeline.run_synthetic(
        tiny_gpt2,
        tmp_path / "local-only",
        eval_input_ids=eval_input_ids,
        finetune_iterator=gen(),
    )
    assert "datasets" not in sys.modules


def test_recipe_save_paths_under_output_dir(tiny_gpt2, tiny_synthetic_basis, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("orca_runtime_python")
    import torch

    from saeforge import ForgePipeline, SubspaceProjector

    projector = SubspaceProjector(tiny_synthetic_basis)
    pipeline = ForgePipeline(
        basis=tiny_synthetic_basis,
        projector=projector,
        orchestrator="fsm",
        finetune_total_steps=10,
        finetune_warmup_steps=2,
        finetune_peak_lr=1e-3,
        finetune_batch_size=2,
        finetune_seq_len=8,
        finetune_log_every=1,
        finetune_eval_every=10000,
        finetune_save_every=5,
    )
    eval_input_ids = torch.randint(0, tiny_gpt2.config.vocab_size, (1, 4))

    def gen():
        while True:
            yield torch.randint(0, tiny_gpt2.config.vocab_size, (2, 8))

    out = tmp_path / "saves"
    pipeline.run_synthetic(
        tiny_gpt2, out, eval_input_ids=eval_input_ids, finetune_iterator=gen()
    )
    save_root = out / "finetuned" / "checkpoints"
    assert save_root.is_dir()
    saved = sorted(p.name for p in save_root.iterdir())
    assert saved == ["step-000005"]  # only step 5 within 10-step run; step 10 is past end
