# bootstrap Specification

## Purpose

The `bootstrap` capability defines the install path, public import
surface, CLI entry point, and CI baseline that every other sae-forge
change builds on. It is process-only — no algorithm code lives in this
capability — but it locks the shape of the v0 milestone so subsequent
changes are pure additions.

## Requirements

### Requirement: Package installs with numpy + safetensors only

`pip install -e .` SHALL succeed against a Python 3.10–3.13 interpreter
that has only `numpy>=1.24` and `safetensors>=0.4` available. `import
saeforge` SHALL succeed in that environment without torch, transformers,
or polygram installed.

#### Scenario: bare install imports cleanly

- **GIVEN** a Python 3.11 environment with only `numpy` and
  `safetensors` installed
- **WHEN** `pip install -e .` is run from the repo root
- **AND** `python -c "import saeforge; print(saeforge.__version__)"` is run
- **THEN** the install exits zero and the import prints a non-empty
  PEP-440 version string

#### Scenario: torch and polygram are optional

- **WHEN** `pip install -e .` is run without any optional extras
- **THEN** `import saeforge.basis`, `import saeforge.projector`,
  `import saeforge.model`, `import saeforge.forge`, and `import
  saeforge.cli` all succeed without raising

### Requirement: Public surface is frozen

`saeforge/__init__.py` SHALL re-export exactly `FeatureBasis`,
`SubspaceProjector`, `NativeModel`, `ForgePipeline`, `ForgeResult`, and
`__version__`. Adding a new top-level name is a v1 concern and SHALL go
through its own OpenSpec change.

#### Scenario: __all__ is exhaustive

- **WHEN** `saeforge.__all__` is inspected
- **THEN** it equals `["FeatureBasis", "ForgePipeline", "ForgeResult",
  "NativeModel", "SubspaceProjector", "__version__"]` in alphabetical order

### Requirement: Stubs raise NotImplementedError pointing to their change

Every classmethod or function on the four core classes whose
implementation is deferred to a later v0 change SHALL raise
`NotImplementedError` whose message names the deferring change's
OpenSpec proposal path.

#### Scenario: FeatureBasis.from_polygram_checkpoint stub (superseded by feature-basis)

This scenario applied while the loader was a stub. As of the
`feature-basis` change, the implementation raises `FileNotFoundError`
on a missing checkpoint instead of `NotImplementedError`.

#### Scenario: ForgePipeline.run stub (superseded by forge-pipeline)

This scenario applied while `ForgePipeline.run` was a stub. As of the
`forge-pipeline` change, the method runs the full pipeline (load host
→ project → assemble → optional KL eval → save) and raises
`ValueError` only when `host_model_id is None`.

### Requirement: CLI uses verb-first style

The `sae-forge` console script SHALL accept the polygram-style verb-first
grammar: `sae-forge <verb> <positional-checkpoint> [flags]`. v0 ships
exactly two verbs: `forge` and `inspect`. A `--version` flag SHALL print
`sae-forge <version>` and exit zero.

#### Scenario: --version exits zero

- **WHEN** `sae-forge --version` is run
- **THEN** stdout contains `sae-forge` followed by the package version
- **AND** the exit code is zero

#### Scenario: forge subcommand parses

- **WHEN** `sae-forge forge sae.safetensors --host-model gpt2 --output-dir
  out/` is invoked through `cli.main`
- **THEN** parsing succeeds and the dispatcher reaches
  `FeatureBasis.from_polygram_checkpoint`, which raises
  `NotImplementedError` (the change-2 deliverable)

### Requirement: FeatureBasis dataclass shape contract

`FeatureBasis` SHALL be a dataclass exposing `kept_ids`, `W_dec`,
`merged_norms`, `original_norms`, `scale_compression_ratio`, and
`metadata`. `__post_init__` SHALL validate that `W_dec` is 2-D and that
`kept_ids`, `merged_norms`, and `original_norms` all have length equal
to `W_dec.shape[0]`. Mismatches SHALL raise `ValueError` whose message
names the offending field.

#### Scenario: mismatched merged_norms is rejected

- **WHEN** `FeatureBasis` is constructed with `W_dec.shape = (8, 64)`
  and `merged_norms.shape = (7,)`
- **THEN** `__post_init__` raises `ValueError` whose message contains
  `merged_norms` and the lengths `7` and `8`

#### Scenario: pseudoinverse is cached

- **GIVEN** a `FeatureBasis` constructed with random `W_dec`
- **WHEN** `pseudoinverse()` is called twice
- **THEN** the second call returns the same `np.ndarray` instance as
  the first (object identity, not just equality)

### Requirement: CI runs ruff + pytest on the dev extra

The repo SHALL ship a GitHub Actions workflow at
`.github/workflows/test.yml` that, on push to main and on pull requests,
installs the package with the `[dev]` extra on Python 3.11 and 3.12,
runs `ruff check saeforge tests examples`, and runs `pytest --tb=short
-v`. Torch-bound jobs are deferred to the `native-model` change.

#### Scenario: workflow exists and matches

- **WHEN** `.github/workflows/test.yml` is read
- **THEN** it declares `runs-on: ubuntu-latest`, a Python matrix
  including `"3.11"` and `"3.12"`, an install step using `pip install
  -e ".[dev]"`, and separate ruff and pytest steps
