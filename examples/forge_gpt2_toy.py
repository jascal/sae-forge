"""Toy GPT-2-small forge against a synthetic 64-feature compressed SAE.

This example is the smoke target for the v0 milestone. It exercises the
full pipeline end-to-end on CPU with a synthetic basis, so contributors
can confirm the four core components compose without needing real SAE
artifacts on disk.

Status: stub. The bodies land with the `forge-pipeline` change.
"""

from __future__ import annotations

from pathlib import Path


def main(output_dir: str | Path = "examples/output/") -> None:
    raise NotImplementedError(
        "examples/forge_gpt2_toy.py is the change-5 deliverable; "
        "see openspec/changes/forge-pipeline/proposal.md."
    )


if __name__ == "__main__":
    main()
