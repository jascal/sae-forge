"""Internal building blocks for ``saeforge.forge_to_moe``.

The public surface is ``saeforge.ForgedMoE`` / ``saeforge.forge_to_moe``
(see ``saeforge/moe.py``); this private namespace holds the v1 expert
and router implementations. Both modules import torch at load time, so
they are only imported on the torch-backed forge path — never from the
torch-free ``import saeforge`` graph.
"""
