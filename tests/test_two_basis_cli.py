"""CLI flag parsing for two-basis-forge (task 6)."""

from __future__ import annotations

import pytest

from saeforge.cli import _build_parser, _parse_composition_heads

_BASE = ["forge", "ckpt.safetensors", "--host-model", "gpt2", "--output-dir", "out"]


def test_two_basis_flags_default_off():
    args = _build_parser().parse_args(_BASE)
    assert args.composition_preserve is False
    assert args.assertion_preserve is False
    assert args.composition_rank is None
    assert args.composition_heads == "prev-token"
    assert args.composition_mode == "writer-output"
    assert args.assertion_k == 0
    assert args.circuit_faithfulness is False


def test_two_basis_flags_parse():
    args = _build_parser().parse_args(
        _BASE
        + [
            "--composition-preserve",
            "--composition-rank", "16",
            "--composition-heads", "4.11,2.2",
            "--composition-mode", "reader-geometry",
            "--assertion-preserve",
            "--assertion-k", "8",
            "--circuit-faithfulness",
        ]
    )
    assert args.composition_preserve is True
    assert args.composition_rank == 16
    assert args.composition_heads == "4.11,2.2"
    assert args.composition_mode == "reader-geometry"
    assert args.assertion_preserve is True
    assert args.assertion_k == 8
    assert args.circuit_faithfulness is True


def test_parse_composition_heads_presets_passthrough():
    assert _parse_composition_heads("prev-token") == "prev-token"
    assert _parse_composition_heads("duplicate-token") == "duplicate-token"
    assert _parse_composition_heads("all") == "all"


def test_parse_composition_heads_explicit_list():
    assert _parse_composition_heads("4.11,2.2") == [(4, 11), (2, 2)]
    assert _parse_composition_heads("0.3") == [(0, 3)]


def test_parse_composition_heads_rejects_bad_token():
    with pytest.raises(Exception, match="L.H"):
        _parse_composition_heads("4,11")  # legacy comma-list of indices no longer valid


def test_composition_mode_choices_enforced():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(_BASE + ["--composition-mode", "bogus"])
