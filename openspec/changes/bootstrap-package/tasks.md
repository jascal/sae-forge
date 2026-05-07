## 1. Repository scaffolding

- [x] 1.1 Add `pyproject.toml` with hatchling build backend, dynamic version pulled from `saeforge/__init__.py`, runtime deps `numpy>=1.24` and `safetensors>=0.4`, optional extras `dev`, `plot`, `notebook`, `torch`, `polygram`, `all`, and the `sae-forge = "saeforge.cli:main"` console script
- [x] 1.2 Add `LICENSE` (Apache-2.0), `CHANGELOG.md` with a `[Unreleased]` section, `CONTRIBUTING.md` describing the OpenSpec flow and house rules, `.gitignore` matching polygram
- [x] 1.3 Add `AGENTS.md` listing the five v0 changes (`bootstrap-package`, `feature-basis`, `subspace-projector`, `native-model`, `forge-pipeline`), the polygram dep contract, and the torch dep contract
- [x] 1.4 Add `README.md` with the canonical thesis, Status / Install / Layout / How it works / Quickstart / CLI / Hardware notes / Components / Examples / Integration with Polygram / Development / License sections, in that order

## 2. Package skeleton

- [x] 2.1 Create `saeforge/__init__.py` exposing `FeatureBasis`, `SubspaceProjector`, `NativeModel`, `ForgePipeline`, `ForgeResult`, and `__version__ = "0.0.1"`
- [x] 2.2 Create `saeforge/basis.py` with the `FeatureBasis` dataclass (kept_ids, W_dec, merged_norms, original_norms, scale_compression_ratio, metadata), shape validation in `__post_init__`, `n_features` / `d_model` properties, a cached `pseudoinverse()`, a `to_summary()` method, and a `from_polygram_checkpoint` classmethod stub that raises `NotImplementedError` pointing to change 2
- [x] 2.3 Create `saeforge/projector.py` with the `SubspaceProjector` dataclass (basis, scale_boost), `encode` / `decode` numpy methods, per-module projection helpers (embed, unembed, qkv, mlp_in, mlp_out), and a `project_module` stub raising `NotImplementedError` pointing to change 3
- [x] 2.4 Create `saeforge/model.py` with `NativeModelConfig`, `NativeModel`, `from_host` and `from_projected_weights` stubs, plus `forward` / `save_pretrained` stubs; all torch-bound entry points raise `NotImplementedError` pointing to change 4
- [x] 2.5 Create `saeforge/forge.py` with `ForgeResult` and `ForgePipeline`; the `run` method raises `NotImplementedError` pointing to change 5
- [x] 2.6 Create `saeforge/cli.py` exposing `main(argv=None)` with `forge` and `inspect` subcommands and a `--version` flag; document the verb-first style in the docstring
- [x] 2.7 Create `saeforge/eval/__init__.py` re-exporting `faithfulness_kl` and `saeforge/eval/faithfulness.py` with a stub function
- [x] 2.8 Create `saeforge/utils/__init__.py` and `saeforge/utils/lazy.py` exposing `require_extra(module_name, extra)` that raises an actionable `ImportError`

## 3. Tests

- [x] 3.1 Add `tests/__init__.py` (empty)
- [x] 3.2 Add `tests/test_smoke.py` covering: `import saeforge` succeeds without torch / transformers / polygram, `saeforge.__version__` is a valid PEP-440 string, `FeatureBasis(...)` constructs against synthetic numpy inputs and rejects mismatched shapes, `FeatureBasis.pseudoinverse()` returns the right shape, `SubspaceProjector.encode` round-trips a unit vector through `decode`, `NativeModelConfig(...)` constructs, `cli._build_parser()` builds without error, `cli.main(["--version"])` prints the version and exits zero
- [ ] 3.3 Add `tests/test_cli.py` covering: `sae-forge --version` exits zero with the version on stdout, `sae-forge forge --help` and `sae-forge inspect --help` exit zero, `sae-forge inspect` on a missing checkpoint raises `NotImplementedError` (the change-2 deliverable; the test pins the contract)

## 4. CI

- [x] 4.1 Add `.github/workflows/test.yml` running ruff + pytest on Python 3.11 and 3.12 with the `[dev]` extra; matrix `fail-fast: false`
- [x] 4.2 Verify `ruff check saeforge tests examples` passes locally before opening the change

## 5. OpenSpec scaffolding

- [x] 5.1 Add `openspec/changes/bootstrap-package/proposal.md`
- [x] 5.2 Add `openspec/changes/bootstrap-package/tasks.md` (this file)
- [x] 5.3 Add `openspec/changes/bootstrap-package/specs/bootstrap/spec.md` defining the bootstrap capability
- [ ] 5.4 Run `openspec validate bootstrap-package` and fix any reported issues
