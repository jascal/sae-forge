## 1. Composition subspace extraction (`saeforge/composition_subspace.py`)

- [x] 1.1 New module `saeforge/composition_subspace.py`. `CompositionSubspace` dataclass: `U: np.ndarray` (`d_model × r`, orthonormal), `layer: int`, `rank: int`, `source_heads: list[int] | str`, `singular_tail: np.ndarray` (logged for the rank choice), `d_model: int`
- [x] 1.2 `__post_init__` validates `U` is `d_model × r` with orthonormal columns (`||UᵀU − I||_F < 1e-5`) and `r <= d_model`
- [x] 1.3 `extract_composition_subspace(host, *, layers, rank=None, heads="all", fold_ln1=True) -> dict[int, CompositionSubspace]`. Per layer: build read geometry `R = [W_Q^h | W_K^h]` over `heads`, write geometry `Wo = [W_V^h W_O^h]`, fold `ln_1.weight` into the residual side when `fold_ln1`, SVD each, take top-`rank` (or singular-value-knee when `None`), orthonormalise the union. Record the `ln_1` mean-subtraction rank-1 approximation (`ln_meansub_approx` + dropped-rank magnitude) in `CompositionSubspace.metadata` for the run report
- [ ] 1.4 Layer-index + head-slice extraction reuses the architecture adapter's `layer_index_for` / head geometry helpers so non-GPT-2 hosts plug in (GPT-2 `c_attn` Conv1D slicing handled in the adapter, not here)
- [ ] 1.5 Budget reporting: `preserved_fraction(d_model)` = `r / d_model`; warn-log when any layer exceeds a configurable budget cap (default 0.25)
- [x] 1.6 Tests `tests/test_composition_subspace.py`: orthonormality; rank honoured; `heads` list restricts the source; the algebraic invariant — for a tiny GPT-2, `Uᵀ M_h U` (QK in the preserved subspace) matches the host `M_h` restricted to `U` to `1e-6`

## 2. Augmented basis (`saeforge/augmented_basis.py`)

- [x] 2.1 New module `saeforge/augmented_basis.py`. `AugmentedBasis` dataclass: `basis: FeatureBasis`, `assertion_atoms: np.ndarray | None` (`U_A`, `K_A × d_model`), `composition: dict[int, CompositionSubspace] | None`
- [x] 2.2 `__post_init__` validates `d_model` agreement across `basis.W_dec`, `U_A`, every `U_C`; raises `ValueError` naming the mismatched source and the two conflicting `d_model` values
- [x] 2.3 `kept_subspace(layer) -> (W_dec_eff, preserve_mask)`: orthonormalise the stack `[U_A ; U_C[layer] ; W_dec_remainder]` (Gram–Schmidt with the verbatim rows first so they are preserved exactly), and return the boolean `preserve_mask` marking which effective rows must be written verbatim vs. Polygram-merged. `W_dec_remainder` = `basis.W_dec` with the components already in `span(U_A ∪ U_C)` removed
- [x] 2.4 `preserved_dimension(layer) -> int` and `preserved_fraction(layer, d_model) -> float` for reporting
- [x] 2.5 When `assertion_atoms is None and composition is None`, `kept_subspace` returns `(basis.W_dec, all-False mask)` — i.e. the single-basis path, byte-identical
- [x] 2.6 Tests `tests/test_augmented_basis.py`: kept-subspace orthonormality; preserve-mask marks exactly the `U_A ∪ U_C` rows; `d_model` mismatch raises; budget fraction; null-augment returns the original `W_dec` unchanged

## 3. Projector dispatch (`saeforge/projector.py`)

- [x] 3.1 Add optional `augmented: AugmentedBasis | None = None` kwarg to `SubspaceProjector.project_module`
- [x] 3.2 When `augmented is None`: existing single-basis dispatch unchanged (byte-equivalence)
- [x] 3.3 When provided: for each emitted weight, look up the weight's layer (adapter `layer_index_for`), project through `augmented.kept_subspace(layer)`, and write the preserve-mask rows verbatim from the host instead of through the merged decode
- [x] 3.4 Non-block weights (`wte`, `wpe`, `lm_head`, `ln_f`) use the basis with no composition augmentation (attention geometry is per-block); assertion atoms still apply
- [x] 3.5 Tests: `augmented=None` reproduces the committed single-basis reference dict for `tiny_gpt2` exactly (regression); augmented path reproduces host QK/OV action on `U_C` to tolerance

## 4. Circuit-faithfulness metric (`saeforge/eval/circuit_faithfulness.py`)

- [x] 4.1 New module. `induction_predictable(token_ids) -> np.ndarray` and `in_context_repeat(token_ids) -> np.ndarray` boolean masks (port from `lm-sae`)
- [x] 4.2 `circuit_kl(host_logits, forged_logits, *, mask) -> dict` returning `{"masked_kl", "complement_kl", "n_masked"}`
- [x] 4.3 `assertion_cov95(forged_residual, oracle) -> dict` reusing the existing oracle-probe to report monosemantic single-atom detector fraction on the forged residual
- [x] 4.4 Export from `saeforge/eval/__init__.py`
- [x] 4.5 Tests: masks match a hand-computed fixture; `circuit_kl` is `0` for `host==forged`; cov95 monotone under a controlled smear

## 5. ForgePipeline knobs + ctx wiring (`saeforge/forge.py`)

- [x] 5.1 Add optional fields: `composition_preserve: bool = False`, `assertion_preserve: bool = False`, `composition_rank: int | None = None`, `composition_heads: list[int] | str = "all"`, `assertion_k: int = 0`
- [x] 5.2 `__post_init__`: when any preserve toggle is on, build the `AugmentedBasis` (extract `U_C` for the capture layers, select top-`assertion_k` sharp atoms for `U_A`); when all off, `self.basis` is used unchanged and no extraction runs
- [x] 5.3 Thread the `AugmentedBasis` through the `project_to_subspace` action ctx (mirror the `hybrid` bundle wiring); no FSM state/guard/target change
- [x] 5.4 Run report includes preserved-dimension budget per layer and the `dim(U_C ∩ S)/dim(U_C)` overlap (how much the basis already covered)
- [x] 5.5 Tests: toggles-off run is byte-equivalent to v0; toggles-on run produces finite weights and a populated budget report

## 6. CLI (`saeforge/cli.py`)

- [x] 6.1 New `forge` flags: `--composition-preserve`, `--composition-rank N`, `--composition-heads {all|comma-list}`, `--assertion-preserve`, `--assertion-k N`, `--circuit-faithfulness`
- [x] 6.2 `--circuit-faithfulness` emits the new metric block in the run report (masked vs complement KL, assertion cov95)
- [x] 6.3 Tests: flag parsing; mutually-consistent defaults (ranks ignored when the matching toggle is off, with a warn-log)

## 7. Integration + comparison harness

- [x] 7.1 `tests/integration/test_two_basis_forge_gpt2.py`: end-to-end GPT-2 forge with `composition_preserve=True` — finite pre/post-FT KL, safetensors round-trip, and **induction-predictable KL(two-basis) ≤ single-basis** on matched bases/seed
- [x] 7.2 `scripts/compare_single_vs_two_basis_gpt2.py`: single / assertion-only / composition-only / two-basis on `gpt2`; emit table of global KL, induction-predictable KL, assertion cov95, preserved-dim budget, `U_C∩S` overlap. Also emit a **Pareto plot** (preserved-dim % vs. {induction-predictable KL, global KL, assertion cov95}) over a `composition_rank` / `assertion_k` sweep, so the budget knee is visible
- [ ] 7.3 Run the harness on Intel/GPT-2; record numbers in `docs/two_basis_forge.md`; this is the defaults-decision artifact

## 8. Docs + changelog

- [x] 8.1 `docs/two_basis_forge.md`: two-kinds-of-content framing, the `U_A`/`U_C`/`S'` algebra, the metric, the `lm-sae` provenance and its caveats
- [x] 8.2 `CHANGELOG.md` `## [Unreleased]` `### Added`

## 9. Validation gate

- [x] 9.1 Byte-equivalence gate (`test_imperative_and_fsm_byte_equivalent`) passes unmodified
- [ ] 9.2 `openspec validate two-basis-forge --strict` passes
- [ ] 9.3 Decision: if two-basis fails to beat single-basis on induction-predictable KL at a non-regressing global KL with conservative defaults, keep both toggles off and record the negative result in `docs/two_basis_forge.md`
