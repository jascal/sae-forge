## Why

sae-forge is a new repository. Before any of the four core components
(`FeatureBasis`, `SubspaceProjector`, `NativeModel`, `ForgePipeline`) can
land, the package needs a working install path, a CI baseline, the
canonical OpenSpec layout copied from polygram, and a public
import-surface that downstream changes can extend without churning the
top-level `__init__.py`.

This change is process-only. No real algorithm code lands here — every
public class is a documented stub that raises `NotImplementedError` with
a pointer to the change that will fill it in. The point is to lock the
shape of the v0 milestone and the conventions (`numpy + safetensors`-only
runtime deps, lazy-imported torch / transformers / polygram, polygram-style
verb-first CLI) so subsequent changes are pure additions, not redesigns.

## What Changes

- Add `pyproject.toml` declaring the package, the runtime deps
  (`numpy`, `safetensors`), and the optional extras (`dev`, `plot`,
  `notebook`, `torch`, `polygram`, `all`).
- Add the `saeforge/` package skeleton: `__init__.py` re-exports,
  `basis.py` (`FeatureBasis`), `projector.py` (`SubspaceProjector`),
  `model.py` (`NativeModel`, `NativeModelConfig`), `forge.py`
  (`ForgePipeline`, `ForgeResult`), `cli.py` (`sae-forge` console
  script), `eval/faithfulness.py`, `utils/lazy.py`.
- Add `tests/test_smoke.py` covering:
  the package imports without torch / polygram installed; `__version__`
  is a valid PEP-440 string; the four core stubs construct against
  in-memory inputs where applicable; CLI parser builds and `--version`
  exits zero; every public class has a non-empty docstring.
- Add `.github/workflows/test.yml` running `ruff` + `pytest` on Python
  3.11 / 3.12 with the `[dev]` extra only — torch-bound tests come
  online in change 4.
- Add `LICENSE` (Apache-2.0), `CHANGELOG.md`, `CONTRIBUTING.md`,
  `AGENTS.md`, `README.md`, `.gitignore`.
- Stage the v0 milestone in `AGENTS.md` as five OpenSpec changes:
  `bootstrap-package`, `feature-basis`, `subspace-projector`,
  `native-model`, `forge-pipeline`.

## Capabilities

### New Capabilities

- `bootstrap`: Installable package with a frozen public surface
  (`FeatureBasis`, `SubspaceProjector`, `NativeModel`, `ForgePipeline`,
  `ForgeResult`, `__version__`), a working CLI entry point with the
  polygram verb-first style (`sae-forge forge`, `sae-forge inspect`,
  `sae-forge --version`), and a CI baseline that runs ruff + pytest
  against the `[dev]` extra.

### Modified Capabilities

None — this is the first change.

## Impact

- New files only. No existing code paths are altered.
- Downstream changes (`feature-basis`, `subspace-projector`,
  `native-model`, `forge-pipeline`) extend the stubs in place rather than
  introducing new top-level modules. This change locks that shape.
- Polygram is **not** a runtime dep at the import-surface level. Adding
  it to `dependencies` would force every sae-forge user to also install
  Polygram even when they only want to inspect a basis loaded from a
  hand-rolled dict; instead it lives behind the `[polygram]` extra.
- Torch is similarly behind `[torch]` so `import saeforge` works for
  Polygram-only users triaging a compression before forging.
