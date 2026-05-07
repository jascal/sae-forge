"""``ForgePipeline.run`` raises ``NotImplementedError`` from the
adapter dispatcher — not from a downstream shape mismatch — when the
host model's class has no registered adapter.

Covers tasks.md §9.2 of multi-architecture-support.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _basis(n: int = 4):
    from saeforge.basis import FeatureBasis

    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=np.eye(n, dtype=np.float32),
        merged_norms=np.ones(n, dtype=np.float32),
        original_norms=np.ones(n, dtype=np.float32),
    )


def test_unregistered_architecture_raises_at_dispatch(tmp_path: Path):
    """A host model whose class isn't registered raises a clean
    ``NotImplementedError`` from ``adapter_for`` — naming the offending
    type and the registered class set — before any random-init
    NativeModel is constructed and saved.
    """
    from saeforge import ForgePipeline, SubspaceProjector

    basis = _basis()
    projector = SubspaceProjector(basis)

    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id="not-a-real-id",
    )

    class FakeBert:
        """Stand-in for an unregistered HF architecture."""
        def eval(self):
            return self

    # Mock AutoModelForCausalLM.from_pretrained to return our fake host
    # so we can exercise the dispatcher without an actual download.
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=FakeBert(),
    ):
        with pytest.raises(NotImplementedError) as excinfo:
            pipeline.run(tmp_path / "out")

    msg = str(excinfo.value)
    assert "FakeBert" in msg
    assert "Registered" in msg
    assert "GPT2LMHeadModel" in msg

    # No model file got written — failure is at dispatch, not
    # post-projection.
    assert not (tmp_path / "out" / "forged").exists()
