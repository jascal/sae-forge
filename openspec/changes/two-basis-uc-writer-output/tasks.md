## 1. Behavioral writer identification (`saeforge/circuit_heads.py`)

- [ ] 1.1 New module `saeforge/circuit_heads.py`. `prev_token_heads(host, corpus, *, top_k=4, ctx=96, min_attention=0.15) -> list[tuple[int,int,float]]`: one forward pass with `attn_implementation="eager"`, return up to `top_k` heads (with their Δ=1 attention SCORE) above `min_attention` (so a model with few/no strong movers returns fewer, not noise). Caller may trim further
- [ ] 1.2 `duplicate_token_heads(host, corpus, *, top_k=4) -> list[tuple[int,int]]`: top heads by attention to an earlier same-token key (minus base rate)
- [ ] 1.3 `identify(host, corpus, preset) -> list[tuple[int,int]]` dispatching on `preset ∈ {"prev-token","duplicate-token"}`; raise on unknown preset naming the supported set
- [ ] 1.4 Tests `tests/test_circuit_heads.py`: on a tiny GPT-2 fixture trained/seeded so one head is a clear Δ=1 mover, `prev_token_heads` returns it first; unknown preset raises

## 2. Writer subspace (`saeforge/composition_subspace.py`)

- [ ] 2.1 `extract_writer_subspace(host, *, writer_heads, rank) -> CompositionSubspace`: for each `(L,h)` in `writer_heads`, `OV = W_V^h W_O^h`; stack and take the top-`rank` right singular vectors (the OV-output row space); orthonormal `U`, `source_heads=writer_heads`
- [ ] 2.2 `extract_composition_subspace` gains `mode: Literal["writer-output","reader-geometry"] = "writer-output"`. `writer-output` requires `heads` to be an explicit `(layer,head)` list (or resolved upstream from a preset) and dispatches to `extract_writer_subspace`; `reader-geometry` is the legacy aggregate path with a docstring note that it does NOT protect circuits
- [ ] 2.3 Tests `tests/test_writer_subspace.py`: `U` orthonormality; rank ≤ requested; the preserved-subspace invariant — projecting a writer head's OV output onto `span(U)` reproduces it to `1e-6`; `source_heads` recorded

## 3. ForgePipeline rewiring (`saeforge/forge.py`)

- [ ] 3.1 `composition_heads` accepts a `list[tuple[int,int]]` (explicit writers), a preset string (`"prev-token"` / `"duplicate-token"`), or `"all"` (legacy reader-geometry). Validate the type/preset in `__post_init__`
- [ ] 3.2 `_build_augmented_basis`: when a preset, call `circuit_heads.identify(host, eval_corpus, preset)`; build `U_C` via `extract_writer_subspace(host, writer_heads=…, rank=composition_rank)`. When `"all"`, keep the legacy reader-geometry path
- [ ] 3.3 `_augmented_report` records the writer heads used **with their detection scores** + the mode (`writer-output` vs `reader-geometry`) + (when both subspaces are computed) the `writer_overlap` / `attribution_overlap` transparency numbers, so users see WHY heads were chosen and the circuit-vs-global subspace separation
- [ ] 3.4 Tests: preset resolves to writer heads and a writer-output `U_C`; explicit list path; `"all"` reproduces the legacy reader-geometry subspace; toggles-off still byte-equivalent

## 4. CLI (`saeforge/cli.py`)

- [ ] 4.1 `--composition-heads` accepts `prev-token` / `duplicate-token` / a comma list of `L.H` / `all`
- [ ] 4.2 `--composition-mode {writer-output,reader-geometry}` (default `writer-output`)
- [ ] 4.3 Tests: flag parsing for presets, `L.H` lists, and the mode

## 5. Metric + harness + docs

- [ ] 5.1 `scripts/compare_single_vs_two_basis_gpt2.py`: add the writer-output and reader-geometry rows (and, as a documented control, the attribution row) so the circuit-vs-global-fidelity trade is visible
- [ ] 5.2 `docs/two_basis_forge.md`: replace the `U_C` definition with the writer-output mechanism; add the alive-forge evidence table (reader −6% / writer −111% / attribution +14% worse, overlap 0.05) and the "loss-sensitivity ≠ circuit-mechanism" note
- [ ] 5.3 `CHANGELOG.md` `## [Unreleased]` `### Changed`

## 6. Validation gate

- [ ] 6.1 Byte-equivalence gate (`test_imperative_and_fsm_byte_equivalent`) passes unmodified
- [ ] 6.2 Integration: end-to-end GPT-2 forge with `composition_preserve=True, composition_heads="prev-token"` runs, `_augmented_report` lists the identified writers, weights finite + round-trip
- [ ] 6.3 `openspec validate two-basis-uc-writer-output --strict` passes
- [ ] 6.4 Decision: the comparison harness confirms writer-output protects circuit KL where reader-geometry does not, at the documented global-KL cost; otherwise keep the legacy default and record the negative
