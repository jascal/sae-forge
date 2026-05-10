## Why

sae-forge has a working v0 implementation but no canonical
mathematical reference describing how the projection works, why it is
expected to succeed, or what the error budget looks like. New
contributors and external readers currently have to reverse-engineer
the algebra from `saeforge/projector.py` plus the Polygram
compression docs. The README explains *what* sae-forge does but not
*how* — and certainly not *why fine-tuning is the right corrective*
for the projection's known error sources.

This change ships the canonical algorithmic foundation as
`docs/algorithm.md` — readable enough to live alongside the README,
formal enough for future academic writing — and links it from the
README's "How it works" section.

The doc deliberately documents the spec as written, then has a final
"v0 implementation notes" section flagging the two places the shipped
code makes deliberate engineering tradeoffs (pinv-of-W_dec instead of
SAE encoder Eᵀ; host-inherited attention internal widths instead of
full k-wide attention). That keeps the spec canonical while not
misleading readers about what the production code does today.

## What Changes

- Add `docs/algorithm.md` covering: core thesis, notation,
  high-level algorithm pseudocode, projection rules with concrete
  examples, error sources and why fine-tuning works, FSM
  orchestration (with a Mermaid state diagram), informal theoretical
  guarantees, limitations, v0 acceptance criteria, and a final
  section enumerating the two v0 implementation deviations.
- Update `README.md`'s "How it works" section to link to
  `docs/algorithm.md` for the full math.
- Add `openspec/changes/algorithm-foundation/` (proposal, tasks, and
  an `algorithm-foundation` capability spec covering: the document
  exists at the canonical path, the README links it, the v0
  deviations section is non-empty and references both
  `subspace-projector` and `forge-outer-loop-fsm` OpenSpec changes
  for cross-referenceability).

## Capabilities

### New Capabilities

- `algorithm-foundation`: A canonical, prominently-linked Markdown
  reference explaining sae-forge's projection math, error model, and
  iteration story. Includes a v0 implementation notes section that
  cross-references the OpenSpec changes pinning the shipped code's
  deviations from the spec.

### Modified Capabilities

None.

## Impact

- New file: `docs/algorithm.md` (~150 lines).
- One README edit: the "How it works" section gets a one-line link
  to `docs/algorithm.md`.
- No code changes. No test changes.
- The "v0 implementation notes" section is the deliberate hook that
  prevents this doc from drifting silently away from the shipped
  code: when a v1 change converges the implementation onto the
  spec's full Eᵀ / both-sides-projected form, the notes section
  shrinks accordingly.
