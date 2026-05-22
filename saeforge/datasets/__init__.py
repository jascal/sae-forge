"""Dataset surfaces for capability-aware forge tuning.

Currently houses one dataset abstraction:

- :class:`CapabilityDataset` — sequences + labels + downstream encoder
  bundled for :class:`saeforge.eval.targets.DownstreamCapabilityTarget`.
  Constructor :func:`CapabilityDataset.from_bio_sae` parses a bio-sae
  bundle without importing the ``biosae`` package.

Future fixture formats (sm-sae, econ-sae) get their own
``from_<repo>`` constructors in the same module.
"""

from __future__ import annotations

from saeforge.datasets.capability import CapabilityDataset

__all__ = ["CapabilityDataset"]
