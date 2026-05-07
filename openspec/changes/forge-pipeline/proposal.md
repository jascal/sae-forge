## Why

`ForgePipeline` is the v0 imperative orchestrator that ties the four
component changes (basis → projector → native model) together with a
faithfulness eval and an artifact-tree writer. The bootstrap change
shipped a `NotImplementedError` stub; this change implements the
imperative path end-to-end and provides the v0 worked example
(`examples/forge_gpt2_toy.py`).

The strongest validation in this change is the **identity-basis sanity
check**: when the basis is the d_model × d_model identity, the forged
model's forward pass equals the host's forward pass exactly, so KL(host
|| forged) → 0. That's the test that proves the projection algebra in
`SubspaceProjector` plus the in-tree `NativeModel` are wired correctly
across attention, MLP, layer norm, embedding, and unembedding.

## What Changes

- Implement `ForgePipeline.run(output_dir)` with the five stages from
  the bootstrap docstring: load basis (already in `self.basis`), load
  host via `host_model_id`, project weights, assemble `NativeModel`,
  optionally run faithfulness KL on `eval_prompts`, write the artifact
  tree.
- Add `ForgePipeline.run_synthetic(host_model, output_dir,
  eval_input_ids=None)` for tests and the toy example: takes an
  already-loaded host and pre-tokenized input ids, skipping the
  `from_pretrained` and tokenization paths.
- Implement `faithfulness_kl(forged_model, host_model, prompts, *,
  tokenizer=None, max_length=32, device="cpu")`: encodes prompts, runs
  both models, computes mean per-token KL(host || forged) over masked
  positions. Auto-loads the host's tokenizer when none is supplied.
- Implement an internal `_kl_from_input_ids(forged, host, input_ids,
  *, device)` helper used by `run_synthetic` that bypasses tokenization.
- Implement `examples/forge_gpt2_toy.py`: builds a tiny in-memory GPT-2
  (16-embed, 2-layer, 4-head, vocab 100), constructs a synthetic
  8-feature basis, runs the pipeline against it, prints a JSON summary.
  CPU-friendly. No host weights downloaded.
- Artifact tree under `output_dir/`:
  - `forged/config.json` + `forged/model.safetensors` (the saved
    native model, via `NativeModel.save_pretrained`)
  - `forge_result.json` (host id, n_params, faithfulness_kl,
    n_features, scale_compression_ratio)

## Capabilities

### New Capabilities

- `forge-pipeline-imperative`: End-to-end `ForgePipeline.run` orchestrator
  that loads a host, projects through a `SubspaceProjector`, assembles a
  `NativeModel`, evaluates faithfulness, and writes a structured artifact
  tree. Plus a `run_synthetic` variant for in-memory hosts.
- `faithfulness-kl-eval`: `faithfulness_kl(forged, host, prompts, ...)`
  computing mean per-token KL(host || forged) on a held-out prompt set
  with optional tokenizer override.

### Modified Capabilities

- `bootstrap`: the `ForgePipeline.run` stub scenario is superseded —
  the method now runs the pipeline rather than raising
  `NotImplementedError`.

## Impact

- `saeforge/forge.py`: ~120 lines covering `ForgeResult`,
  `ForgePipeline.run`, `ForgePipeline.run_synthetic`, and the internal
  `_kl_from_input_ids` helper.
- `saeforge/eval/faithfulness.py`: ~50-line `faithfulness_kl`.
- `examples/forge_gpt2_toy.py`: the worked example (replaces the stub).
- `tests/test_forge_pipeline.py`: 5 tests covering end-to-end
  `run_synthetic` against `tiny_gpt2`, the host-id-required raise, the
  identity-basis KL ≈ 0 sanity check (the strongest correctness signal
  in v0), `faithfulness_kl` smoke, and the toy example end-to-end.
- One scenario in `bootstrap-package`'s spec is annotated as
  superseded.
