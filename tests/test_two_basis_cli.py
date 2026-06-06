"""CLI flag parsing for two-basis-forge (task 6)."""

from __future__ import annotations

from saeforge.cli import _build_parser

_BASE = ["forge", "ckpt.safetensors", "--host-model", "gpt2", "--output-dir", "out"]


def test_two_basis_flags_default_off():
    args = _build_parser().parse_args(_BASE)
    assert args.composition_preserve is False
    assert args.assertion_preserve is False
    assert args.composition_rank is None
    assert args.composition_heads == "all"
    assert args.assertion_k == 0
    assert args.circuit_faithfulness is False


def test_two_basis_flags_parse():
    args = _build_parser().parse_args(
        _BASE
        + [
            "--composition-preserve",
            "--composition-rank", "16",
            "--composition-heads", "4,11",
            "--assertion-preserve",
            "--assertion-k", "8",
            "--circuit-faithfulness",
        ]
    )
    assert args.composition_preserve is True
    assert args.composition_rank == 16
    assert args.composition_heads == "4,11"
    assert args.assertion_preserve is True
    assert args.assertion_k == 8
    assert args.circuit_faithfulness is True
