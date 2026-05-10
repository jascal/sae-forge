# algorithm-foundation Specification

## Purpose

Defines the canonical mathematical reference at `docs/algorithm.md` —
the source of truth for sae-forge's projection algebra, error model,
and iteration story.

## Requirements

### Requirement: Canonical algorithm document exists at docs/algorithm.md

The repository SHALL ship `docs/algorithm.md` covering, at minimum:

- A core-thesis section naming the basis-restricted forge approach
- A notation section defining `d`, `k`, `B`, `E`, `z`
- A high-level algorithm in pseudocode covering compress → project →
  fine-tune → evaluate → optional iterate
- A projection rules section with concrete examples for embedding,
  unembedding, Q / K / V, output projection, and MLP weights
- An error-sources section enumerating `ε_rare`, `ε_attn`,
  `ε_nonlin`
- A FSM orchestration section with a Mermaid state diagram
- A theoretical-guarantees section (informal)
- A limitations section
- A v0 acceptance-criteria section
- A v0 implementation notes section flagging where the shipped code
  diverges from the spec

#### Scenario: file is present and non-empty

- **WHEN** `docs/algorithm.md` is read
- **THEN** the file exists and is at least 100 lines long

#### Scenario: every required section heading is present

- **WHEN** `docs/algorithm.md` is searched for the section headings
  "Core thesis", "Notation", "High-level algorithm", "Projection
  rules", "Error sources", "FSM orchestration", "Theoretical
  guarantees", "Limitations", "Acceptance criteria", "v0
  implementation notes"
- **THEN** each heading appears at least once

### Requirement: v0 implementation notes section names both deviations

The "v0 implementation notes" section SHALL explicitly call out the
two known places where the shipped code diverges from the spec:

1. The encode direction uses `pinv(W_dec)` rather than the SAE's
   trained encoder slice `Eᵀ`.
2. Attention / MLP internal widths are inherited from the host model
   rather than projected to k-wide.

The section SHALL link back to `saeforge/projector.py`, the
`subspace-projector` capability spec, and the `forge-outer-loop-fsm`
OpenSpec change so a reader auditing the deviations can navigate to
the pinned production behaviour.

#### Scenario: deviation list is exhaustive

- **WHEN** the "v0 implementation notes" section is read
- **THEN** both `pinv(W_dec)` and `host-inherited` (or equivalent
  language) appear as labelled deviations
- **AND** the section contains at least one Markdown link to a path
  under `saeforge/` or `openspec/`

### Requirement: README links the algorithm document

The repository's `README.md` SHALL contain at least one Markdown link
whose target is `docs/algorithm.md`. The link SHALL appear inside or
immediately adjacent to the "How it works" / "How sae-forge works"
section, so a reader following the README's natural reading order
encounters the link before the Quickstart.
