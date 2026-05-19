# Flagship demo: Gemma-2-2B real-SAE forge

The end-to-end run that the structural-fix work in
`add-host-wrapped-forge-fallback` was building toward. Not a pytest
fixture — this runs against the actual `google/gemma-2-2b` host
weights and a published Gemma Scope SAE, on hardware that fits both
(M4 24GB+ or NVIDIA 24GB+). The Intel 16GB MBP cannot run this; it
served as the validation surface for the smaller pieces.

## What this demo proves

1. **The structural KL fix lands at scale.** v0.5.1 + `forward_mode`
   dispatch removes the rank-dependent amplification documented in
   `fix-scale-boost-calibration`. The Gemma Scope L12 SAE compressed
   into a degenerate/undersized basis would, pre-fix, blow up KL
   monotonically with kept-feature count. Post-fix, `forward_mode=
   "auto"` routes those bases through `host_wrapped` and KL stays
   bounded by per-basis approximation error rather than amplification.
2. **The pipeline produces a usable forge.** Real polygram
   compression → real Gemma forge → real faithfulness eval, with
   provenance (resolved forward mode, cluster diagnostics) surfaced
   in the run summary.

## Hardware

| Platform | Status | Notes |
|---|---|---|
| Apple M4 Pro 24GB+ MPS | **required** for the at-scale run | Tested wall-clock ~30–90 min for 1k-step fine-tune (per the script's own docstring) |
| NVIDIA 24GB+ CUDA | works | Faster; ~10–30 min |
| Intel 16GB MBP CPU | **does not fit** | GPT-2 tier only |

The CI suite (`pytest tests/`) runs on Intel via 605 unit tests + 4
smoke scripts and does **not** include the Gemma-2-2B end-to-end run.
The acceptance gate for this demo is qualitative + numerical, not
automated.

## Pre-conditions

1. Accept the Gemma license at
   https://huggingface.co/google/gemma-2-2b and run
   `huggingface-cli login` with a token that has read access.
2. ~10 GB free under `~/.cache/huggingface/` for Gemma weights + SAE
   checkpoints.
3. `pip install -e ".[dev,torch,polygram,orca]"` (polygram >= 0.9.0
   floors this since 2026-05-19).

## The command

The headline at-scale invocation, run from the project root:

```bash
.venv/bin/python examples/forge_gemma2_2b.py \
    runs/flagship_gemma2_2b \
    --device mps \
    --precision bf16 \
    --layer 12 \
    --l0 82 \
    --n-features 256 \
    --n-compress-prompts 32 \
    --steps 1000 \
    --forward-mode auto
```

`--forward-mode auto` is the default and the recommended setting.
It will:

- Read `basis.quality_tier` from the compressed SAE.
- Route to `native_in_basis` when the tier is `good`/`saturated`
  (Gemma Scope's standard 16k-width SAEs typically land here after
  256-feature slicing + polygram compression — but verify per run).
- Route to `host_wrapped` when the tier is `undersized`/`degenerate`
  (the path that removes the documented blow-up).

The example script surfaces the resolved mode in stage 4 and again
in `run_summary.json` (`forge.forward_mode_requested` /
`forge.forward_mode_resolved`).

### For research runs comparing both paths

```bash
# Native-in-basis (the existing v0.5.1 path; reference for comparison).
.venv/bin/python examples/forge_gemma2_2b.py runs/native --forward-mode native_in_basis ...

# Host-wrapped (the new fallback).
.venv/bin/python examples/forge_gemma2_2b.py runs/wrapped --forward-mode host_wrapped ...
```

Note: as of 2026-05-19, `host_wrapped` is **GPT-2 only**. Forcing it
on Gemma-2 will raise `NotImplementedError` from the Gemma-2 adapter
stub pointing at the queued `add-host-wrapped-gemma2` follow-up. Use
`--forward-mode auto` for now; per-family rollout is a tracked
follow-up.

## Acceptance bands

Numbers below are projected from the GPT-2 smoke and the Gemma Scope
L12 quality profile. **Actual numbers should be filled in by the
first M4 run** and committed to this file.

| Metric | Native-in-basis | Host-wrapped (when on a degenerate basis) |
|---|---|---|
| `forge.n_params` | ~ {n_features × scaled host} | full host (~2.6 B) |
| `forge.faithfulness_kl` | depends on basis tier | strictly ≤ native KL |
| Wall clock (forge stage only) | minutes | minutes (host inference overhead) |
| Wall clock (1k-step fine-tune) | 30–90 min on M4 24GB | **n/a** (host_wrapped is inference-only in v1; --steps ignored) |

Red flags to watch for in `run_summary.json`:

- `forge.faithfulness_kl > 50` with `forge.forward_mode_resolved =
  "native_in_basis"` — likely under-complete basis that auto-dispatch
  should have caught. Re-run with `--forward-mode auto` (default)
  and verify the resolved mode in the printed stage-4 output.
- `forge.forward_mode_resolved = "host_wrapped"` and `--steps > 0`
  in the CLI — the example auto-zeros steps in this case and prints
  a warning; if you see fine-tune still running, file an issue.
- Polygram cluster diagnostics absent from `compression.polygram_diagnostics`
  — older polygram outputs without `n_clusters`/`n_zeroed` fields.
  Bump to polygram >= 0.9.0 (the floor since 2026-05-19) and the
  fields populate.

## After the run

The artifacts at `runs/flagship_gemma2_2b/`:

- `forge/forge_result.json` — pipeline-level result (n_params, KL,
  faithfulness target name).
- `forge/forged/` — `config.json` + serialised forged weights.
  Host-wrapped mode writes `host_wrapped_buffers.safetensors`
  (basis matrices only) since the host weights live in the HF
  cache and don't need re-saving.
- `run_summary.json` — the human-readable summary with cluster
  diagnostics, resolved forward mode, and wall clocks.

Commit the run summary back into this doc's "Acceptance bands"
section so subsequent runs have a baseline to regress against.

## Known limitations

- **v1 host_wrapped is GPT-2 only.** Gemma-2/Llama/Qwen/Whisper
  adapters raise `NotImplementedError` from their stub. The auto
  dispatch on a degenerate Gemma-2 basis surfaces this error rather
  than silently falling back to native. The full per-family rollout
  lives behind `add-host-wrapped-{gemma2,llama,…}` follow-up
  proposals.
- **`save_intermediate_reports=True` is not plumbed.** The sm-sae
  LLM-scale recommendation table calls for this; `--llm-scale`
  documents the limitation in its `--help` and surfaces nothing on
  this knob.
- **Multi-GPU is not supported.** The example assumes a single
  device. M4 24GB and NVIDIA 24GB cards both fit Gemma-2-2B at
  bf16 + grad-checkpointing.
