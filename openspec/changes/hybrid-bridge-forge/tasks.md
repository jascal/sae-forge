## 1. BridgeModule (`saeforge/bridges.py`)

- [ ] 1.1 New module `saeforge/bridges.py`. `BridgeConfig` dataclass with `init: Literal["orthogonal","identity","zero"] = "orthogonal"`, `nonlin: Literal["none","relu","gelu"] = "none"`, `pre_layernorm: bool = True`, `train: bool = True`
- [ ] 1.2 `BridgeModule(nn.Module)` class. `__init__(n_features, config)` constructs `nn.LayerNorm` (if `pre_layernorm`), `nn.Linear(n_features, n_features, bias=False)`, and the configured activation. Linear is initialized per `config.init` (orthogonal via `torch.nn.init.orthogonal_`; identity via `torch.eye`; zero via `torch.zeros_`)
- [ ] 1.3 `BridgeModule.forward(x)` implements the order documented in design.md (LN → linear → nonlin). Returns same shape as input
- [ ] 1.4 `BridgeModule.requires_grad_(config.train)` honored in `__init__`
- [ ] 1.5 Tests in `tests/test_hybrid_bridge.py::TestBridgeModule`: shape preservation; orthogonal init `||W||_F == sqrt(n)`; identity init reproduces input under `nonlin="none", pre_layernorm=False`; zero init outputs zero; `requires_grad` honored

## 2. HybridBasisBundle (`saeforge/hybrid_basis.py`)

- [ ] 2.1 New module `saeforge/hybrid_basis.py`. `HybridBasisBundle` dataclass: `basis_embed: FeatureBasis`, `basis_mid: FeatureBasis`, `basis_lm_head: FeatureBasis`, `n_layer: int`
- [ ] 2.2 `__post_init__` validates: all three bases share `d_model`; all three share `n_features`; `n_layer >= 3` (must have room for embed + at least 1 mid + lm-head regions)
- [ ] 2.3 Method `basis_for_layer(idx: int) -> FeatureBasis`. Returns `basis_embed` when `idx == 0`, `basis_lm_head` when `idx == n_layer - 1`, `basis_mid` otherwise. Raises `IndexError` for out-of-range
- [ ] 2.4 Property `boundaries -> tuple[int, int]` returning `(emb_to_mid, mid_to_lm)` layer indices (`(0, n_layer-1)` in v1; here for forward-compat with `multi-anchor-forge`)
- [ ] 2.5 Tests in `tests/test_hybrid_bridge.py::TestBundle`: shape-mismatch detection; layer routing matrix for `n_layer=12` (GPT-2); layer routing for `n_layer=26` (Gemma-2-2B); out-of-range raises

## 3. Hybrid routing in adapters (`saeforge/adapters/_hybrid.py`)

- [ ] 3.1 New module `saeforge/adapters/_hybrid.py`. Function `walk_hybrid(host, adapter, projector, *, bundle: HybridBasisBundle, attention_width="host") -> dict[str, np.ndarray]`. Internally constructs three `SubspaceProjector` instances (one per basis, sharing `scale_boost` per `projector`) and dispatches each adapter-emitted key through the basis whose region matches the key's layer index
- [ ] 3.2 Layer-index extraction: regex `r"\.h\.(\d+)\."` for GPT-2-style keys; the adapter exposes a `layer_index_for(key)` classmethod that the helper consults so non-GPT-2 hosts plug in cleanly
- [ ] 3.3 Non-block keys: `wte.weight` and `wpe.weight` go through `basis_embed`; `lm_head.weight` and `ln_f.*` go through `basis_lm_head`. Explicit table in the module docstring
- [ ] 3.4 Tied-embedding refusal: if `host.config.tie_word_embeddings`, raise `ValueError` with the design.md-pinned message
- [ ] 3.5 Tests in `tests/test_hybrid_bridge.py::TestRouting`: every key in the GPT-2 fixture's `project_module` output is attributed to exactly one basis; the attribution matches the design.md region table

## 4. SubspaceProjector dispatch (`saeforge/projector.py`)

- [ ] 4.1 Add optional `hybrid: HybridBasisBundle | None = None` kwarg to `SubspaceProjector.project_module`
- [ ] 4.2 When `hybrid is None`: existing dispatch unchanged (byte-equivalence preserved)
- [ ] 4.3 When `hybrid is not None`: call `walk_hybrid(host, adapter, self, bundle=hybrid, attention_width=...)`
- [ ] 4.4 Tests: the single-basis path under `hybrid=None` produces the same output dict as before the change (regression test, compared against a committed reference dict for `tiny_gpt2`)

## 5. NativeModel bridges (`saeforge/model.py`)

- [ ] 5.1 Add optional `bridges: dict[str, BridgeModule] | None = None` kwarg to `NativeModel.from_projected_weights`
- [ ] 5.2 When provided, the constructor registers `self.bridge_emb_mid = bridges["emb_mid"]` and `self.bridge_mid_lm = bridges["mid_lm"]` as submodules
- [ ] 5.3 `NativeModel.forward` calls the bridges at the two boundaries per design.md (after block 0, after block L-2). When the attribute is absent (single-basis case), forward is byte-identical to today
- [ ] 5.4 `NativeModel.state_dict()` includes the bridge parameters when present; `load_state_dict` round-trips
- [ ] 5.5 Tests: bridge insertion at correct positions; forward shape preserved; safetensors round-trip with bridges

## 6. ForgePipeline knobs + ctx wiring

- [ ] 6.1 Add four new fields to `ForgePipeline` (`saeforge/forge.py`): `hybrid_bridge: bool = False`, `basis_embed: FeatureBasis | None = None`, `basis_lm_head: FeatureBasis | None = None`, `bridge_config: BridgeConfig = field(default_factory=BridgeConfig)`
- [ ] 6.2 `__post_init__` validation: `hybrid_bridge=True` requires `basis_embed is not None` AND `basis_lm_head is not None` AND `basis_embed.n_features == self.basis.n_features == basis_lm_head.n_features` AND `basis_embed.d_model == self.basis.d_model == basis_lm_head.d_model`. Clear `ValueError` naming the failing pair
- [ ] 6.3 Construct `HybridBasisBundle` in `_build_fsm_ctx` when `hybrid_bridge=True` and write to ctx as `ctx["hybrid_basis_bundle"]`
- [ ] 6.4 `project_to_subspace` action reads `ctx.get("hybrid_basis_bundle")` and forwards it via `projector.project_module(host, hybrid=bundle)`
- [ ] 6.5 `fine_tune_model` action picks up `ctx["bridges"]` and adds their parameters to the optimizer's param group
- [ ] 6.6 Tests in `tests/test_forge_pipeline.py`: validation matrix (hybrid=True without basis_embed → raises; hybrid=True with shape mismatch → raises; hybrid=False with extras → silent)

## 7. Tied-embedding refusal

- [ ] 7.1 In `ForgePipeline.__post_init__` (after host_config is loaded) raise `ValueError` with the design.md-pinned message when `host_config.tie_word_embeddings is True and hybrid_bridge is True`
- [ ] 7.2 Tests: GPT-2 with default tied embeddings + `hybrid_bridge=True` → raises; GPT-2 with `tie_word_embeddings=False` + `hybrid_bridge=True` → constructs cleanly
- [ ] 7.3 The Intel/GPT-2 integration test fixture (§9.1) explicitly constructs an untied GPT-2 host

## 8. CLI

- [ ] 8.1 Add flags to `forge` subparser in `saeforge/cli.py`: `--hybrid-bridge`, `--basis-embed PATH`, `--basis-lm-head PATH`, `--bridge-init {orthogonal,identity,zero}` (default `orthogonal`), `--bridge-nonlin {none,relu,gelu}` (default `none`), `--bridge-no-pre-ln` (boolean, inverts default `pre_layernorm=True`)
- [ ] 8.2 Mutually-required: `--hybrid-bridge` requires both `--basis-embed` and `--basis-lm-head`. Argparse-level error if missing
- [ ] 8.3 `_cmd_forge` threads the new flags into `ForgePipeline(...)` constructor — load the two extra bases from the provided paths via `FeatureBasis.load`
- [ ] 8.4 Tests in `tests/test_cli.py`: parser accepts the new flags; mutually-required validation rejects partial use

## 9. Cross-architecture integration tests (T0 + T1)

- [ ] 9.1 New test `tests/integration/test_hybrid_bridge_gpt2.py::test_t0_tiny_gpt2_smoke`: build three 16-feature bases over a `tiny_gpt2` fixture (n_embd=16, n_layer=4, untied embeddings); run a 10-step forge with `hybrid_bridge=True`; assert pre-FT KL is finite, post-FT KL is finite, post-FT KL ≤ pre-FT KL, safetensors round-trip succeeds. Runs on CPU in <30s
- [ ] 9.2 New test `tests/integration/test_hybrid_bridge_gpt2.py::test_t1_gpt2_smoke`: build three 64-feature bases over `gpt2` (untied), run a 50-step forge. Skipped via `pytest.mark.intel_only` when memory < 14GB. Assert post-FT KL is finite and ≤ pre-FT KL
- [ ] 9.3 New test `test_byte_equivalent_when_hybrid_bridge_disabled`: explicitly construct a pipeline with `hybrid_bridge=False, basis_embed=<something>, basis_lm_head=<something>` and assert the forged weights are byte-identical to a pipeline with the v0 minimal config (no hybrid fields set). The byte-equivalence gate
- [ ] 9.4 Add the existing `test_imperative_and_fsm_byte_equivalent` to the post-change tree's CI matrix. Confirm it passes unmodified

## 10. Comparison harness (`scripts/compare_single_vs_hybrid_gpt2.py`)

- [ ] 10.1 New script not part of pytest. Args: `--n-features`, `--basis-layers EMBED,MID,LM`, `--finetune-steps`, `--eval-prompts PATH`, `--seed`, `--output PATH`
- [ ] 10.2 Runs two forges back-to-back with identical seed: single-basis (at the mid layer) and hybrid (all three). Emits a JSON with per-step KL trajectory + final faithfulness for each
- [ ] 10.3 README in `scripts/README.md` documents the contract: when do you run this, what numbers do you expect, where do they get logged (`docs/forge_layer_choice.md` "Hybrid / multi-basis" subsection)
- [ ] 10.4 Run the harness on Intel/`gpt2` (untied) with the default `bridge_config` once the implementation is in place. Paste the resulting JSON into `docs/hybrid_bridge_intel_gpt2.md` (new file). This is the artifact that decides whether the defaults are correct

## 11. External-CUDA validation request (post-merge)

- [ ] 11.1 New doc `docs/hybrid_bridge_cuda_validation_request.md`: clear instructions for an external NVIDIA/CUDA contributor to (a) install sae-forge, (b) train three bases on a target host (Gemma-2-9B, Llama-3-8B untied, etc.), (c) run the comparison harness from §10, (d) submit the JSON output as a PR adding a table row to `docs/hybrid_bridge_external_results.md`
- [ ] 11.2 Open one GitHub issue with the `help-wanted` and `validation` labels linking the doc and inviting contributions. (Post-merge — the issue links the comparison-harness numbers from §10.4 as the baseline pattern to follow)

## 12. Documentation

- [ ] 12.1 New "Hybrid / multi-basis forging" subsection in `docs/forge_layer_choice.md`. Required content: when to use it (cross-boundary distributional shift between embed / mid / lm-head); the three-basis architecture diagram; the `BridgeConfig` knob table; the Intel/GPT-2 baseline numbers from §10.4; pointer to the comparison harness; tied-embeddings caveat
- [ ] 12.2 New row in `CHANGELOG.md` `## [Unreleased]` `### Added`: "Hybrid three-basis forge path with learnable bridges. Opt-in via `--hybrid-bridge`. Defaults preserve byte-equivalence with single-basis v0 path"
- [ ] 12.3 Update `AGENTS.md`: mention `saeforge/bridges.py` and `saeforge/hybrid_basis.py` as new modules
- [ ] 12.4 `docs/forge_layer_choice.md`: tied-embeddings caveat with a one-paragraph explanation of why GPT-2-default-tied is refused and how to construct an untied GPT-2 for testing

## 13. OpenSpec scaffolding

- [x] 13.1 `openspec/changes/hybrid-bridge-forge/proposal.md`
- [x] 13.2 `openspec/changes/hybrid-bridge-forge/design.md`
- [x] 13.3 `openspec/changes/hybrid-bridge-forge/tasks.md` (this file)
- [x] 13.4 `openspec/changes/hybrid-bridge-forge/specs/hybrid-bridge-forge/spec.md` (ADDED capability)
- [x] 13.5 `openspec/changes/hybrid-bridge-forge/specs/subspace-projector/spec.md` (MODIFIED — optional `hybrid` dispatch arm)
- [ ] 13.6 Run `openspec validate hybrid-bridge-forge --strict`; resolve any structural complaints before opening the PR

## 14. Validation matrix (pre-merge gates)

- [ ] 14.1 Full `pytest` suite passes (existing + new) on Python 3.11 + 3.12 with `[dev,intel,polygram,orca]` extras
- [ ] 14.2 The byte-equivalence gate (`test_imperative_and_fsm_byte_equivalent`) passes
- [ ] 14.3 The hybrid-disabled byte-equivalence gate (§9.3) passes
- [ ] 14.4 T0 (`tiny_gpt2`) integration test passes on CI (CPU only)
- [ ] 14.5 T1 (`gpt2` untied) integration test passes on Intel (skipped if RAM insufficient on the CI runner, but **must** pass on the user's Intel Mac)
- [ ] 14.6 Comparison harness output (§10.4) committed to `docs/hybrid_bridge_intel_gpt2.md`. If hybrid is not measurably better than single-basis at equal `n_features` and equal step budget, **document it** — the proposal explicitly accepts this as a valid outcome and keeps the toggle default-off

## 15. Deferred follow-ups

- [ ] 15.1 **`hybrid-bridge-tied-embeddings`** — support tied-embedding hosts (GPT-2 default, Llama). Either share embed/lm_head bases (two-basis hybrid) or impose an equality constraint on the bridge product
- [ ] 15.2 **`multi-anchor-forge`** — open up the embed/mid/lm-head triple to arbitrary `k` bridges at user-configured layer indices
- [ ] 15.3 **`hybrid-bridge-save-time-fold`** — when `nonlin="none"` and `pre_layernorm=False`, fold each linear bridge into the adjacent block's projection matrix at save time. Zero inference overhead, zero extra params on disk
- [ ] 15.4 **`bridges-only-finetune`** — freeze the three bases and train only the bridges. Cheap ablation that isolates the bridge's effect from basis fine-tuning
- [ ] 15.5 **T3 (M4 / Gemma-2-2B reproduction)** — reproduce the prototype's `KL=11.81` headline number on M4 with the shipped mechanism. The user runs this on their M4 box once the mechanism is on `main`
- [ ] 15.6 **T4 (external CUDA validation)** — community validation pass on NVIDIA/CUDA hosts via the request doc in §11
