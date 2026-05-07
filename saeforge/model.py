"""NativeModel — small transformer whose residual width equals the feature-basis size."""

from __future__ import annotations

from dataclasses import dataclass

from saeforge.projector import SubspaceProjector


@dataclass
class NativeModelConfig:
    """Architecture knobs for a forged native model.

    ``hidden_size`` is fixed by the feature basis (``basis.n_features``);
    other shapes default to the host's per-layer dimensions when constructed
    via ``NativeModel.from_host``.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int = 2048
    activation: str = "gelu"
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.hidden_size:
            # Native models do not require attention to span the residual.
            # This is a soft check; the projector decides per-layer geometry.
            pass


class NativeModel:
    """HF-compatible small transformer skeleton with a feature-basis residual stream."""

    def __init__(self, config: NativeModelConfig) -> None:
        self.config = config
        self._torch_module = None

    @classmethod
    def from_host(
        cls,
        host_model_id: str,
        projector: SubspaceProjector,
        *,
        dtype: str = "float32",
        device: str = "cpu",
    ) -> NativeModel:
        """Construct a native model by projecting ``host_model_id``'s weights through ``projector``.

        Lazy-imports torch + transformers; requires the ``[torch]`` extra.
        """
        raise NotImplementedError(
            "NativeModel.from_host is the change-4 deliverable; "
            "see openspec/changes/native-model/proposal.md."
        )

    @classmethod
    def from_projected_weights(cls, config: NativeModelConfig, weights: dict) -> NativeModel:
        """Assemble a native model from a dict of pre-projected ``np.ndarray`` weights."""
        raise NotImplementedError(
            "NativeModel.from_projected_weights is the change-4 deliverable."
        )

    def forward(self, input_ids):
        raise NotImplementedError("NativeModel.forward is the change-4 deliverable.")

    def save_pretrained(self, output_dir):
        raise NotImplementedError("NativeModel.save_pretrained is the change-4 deliverable.")
