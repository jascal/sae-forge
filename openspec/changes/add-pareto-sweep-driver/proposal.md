## Why

Polygram 0.4.0 (PR polygram#67, `add-pareto-target-compression`) introduces target-K compression and `Compressor.plan_pareto()` — a single-sort, K-times-cut planning primitive that makes feature-count-vs-faithfulness Pareto frontiers cheap to enumerate.

The sae-forge side has no consumer for this yet. Today `ForgePipeline.run()` does one forge per call, and the only way to map a Pareto frontier is to invoke the CLI K times by hand, with K threshold-tuned values that aren't apples-to-apples. The forge pipeline does not currently surface:

- A way to request a target feature count rather than threshold tuning.
- A way to sweep multiple K values from a single validator pass.
- A way to compare multiple SAE checkpoints (e.g. across encodings — MPSRung1 vs Rung3 vs Rung4) on the same frontier coordinate system.

This change adds a sweep driver — `ForgePipeline.sweep_pareto()` plus a `saeforge sweep-pareto` CLI subcommand — that consumes pre-compressed per-K SAE checkpoints produced by `polygram compress --pareto --pareto-materialize`, runs the full StreamMachine forge per K, and emits one JSONL frontier row per `(encoding, K)` with the kept-feature count, downstream KL, perplexity, and faithfulness.

This is the load-bearing primitive for **Axis 4** of the polygram rung-viability methodology (`polygram/docs/research/rung4-viability-spike-v2.md` line 132): end-to-end downstream confirmation that Axis 1's compression-coverage lift cashes out in forged-model KL space. Without target-K, Axis 4 has no apples-to-apples comparison across encodings. With it, Axis 4 is a fixed-K row-by-row contrast.

### Expected workflow

```
# Step 1 (polygram-side, cheap): plan + materialise N SAEs per encoding
polygram compress --sae-checkpoint mps_sae.safetensors --validation-report report.json \
  --pareto 200,500,1000,2000 --pareto-materialize --out runs/mps/

polygram compress --sae-checkpoint rung4_sae.safetensors --validation-report report.json \
  --pareto 200,500,1000,2000 --pareto-materialize --out runs/rung4/

# Step 2 (sae-forge, expensive): forge each materialised SAE
saeforge sweep-pareto \
  --encoding mps:runs/mps/pareto \
  --encoding rung4:runs/rung4/pareto \
  --host-model gpt2 --output-dir runs/axis4/ \
  --eval-prompts data/eval.jsonl
```

The Step 1 cost is dominated by **one BehaviouralValidator pass per encoding**; planning across K is amortised. The Step 2 cost is N×E forge runs. Both steps are checkpointable — a partial sweep is resumable.

## What Changes

### New capability: `pareto-sweep`

- **`ForgePipeline.sweep_pareto(...)` method** that accepts a sequence of `(encoding_label, sae_checkpoint_path)` tuples (or a single encoding's directory of `k_{K}.safetensors` files from `polygram compress --pareto-materialize`), runs the existing forge pipeline once per checkpoint with the same host/eval config, and emits one frontier row per forge to a JSONL file.
- **`ParetoFrontierRow` dataclass** capturing per-forge result: `encoding_label`, `target_n_features_kept`, `n_features_kept_actual`, `pareto_reached_target`, `faithfulness_kl`, `perplexity`, `final_fine_tune_loss`, `sae_checkpoint`, `forged_model_path`, `elapsed_seconds`, `error_message` (None on success). JSONL is the wire format; this is the row schema.
- **Resumability**: the driver checks the output JSONL for already-completed `(encoding_label, target_n_features_kept)` rows and skips them. Single-row failures are recorded (`error_message` populated, other fields null/NaN) and don't abort the sweep — the driver continues to the next row.
- **No FSM changes**: each row's forge call is a vanilla `ForgePipeline.run()`. The sweep driver is pure outer-loop orchestration.

### New CLI subcommand: `saeforge sweep-pareto`

- `--encoding LABEL:PATH` (repeatable) — labelled SAE source. `PATH` is either a single `.safetensors` file or a directory whose `k_{K}.safetensors` files are enumerated.
- `--host-model ID` — host transformer (same as `forge`).
- `--output-dir DIR` — sweep output root; contains `frontier.jsonl` + per-forge subdirectories (`<dir>/<encoding>/k_{K}/`).
- `--eval-prompts PATH` — same JSONL schema as `forge` (already supports dict-shorthand + raw text per CHANGELOG e1f1246).
- Standard forge knobs (`--dtype`, `--device`, `--finetune-steps`, `--seed`, etc.) — passthrough.
- `--frontier-only` — skip forge runs; emit a JSONL with `target_n_features_kept` and `n_features_kept_actual` columns only, populated from each checkpoint's polygram metadata. Cheap exploratory mode. Pairs naturally with `jq` for quick triage, e.g. `jq -r 'select(.error_message == null) | [.encoding_label, .target_n_features_kept, .n_features_kept_actual] | @tsv' frontier.jsonl | sort -t$'\t' -k2 -n` to find candidate K values before committing forge compute.

### Polygram version bump

- `pyproject.toml`: bump `polygram>=0.4.0` (currently `>=0.3.0`).
- `ForgePipeline.compression: CompressionConfig | None` already accepts the new `target_n_features_kept` / `score_field` fields automatically via `_ConfigMixin`. No code change needed in `polygram-tuning-passthrough`; the spec gains scenarios documenting that the new fields are recognised.

### Out of scope, deliberately

- **Polygram orchestration inside sae-forge.** The sweep driver does not invoke `polygram compress --pareto` itself. The caller is responsible for producing the per-K materialised checkpoints first. Reason: it keeps sae-forge a pure consumer of polygram artifacts, the way it already is for any `Compressor`-produced checkpoint. A `--auto-materialise` flag could be added later if it proves a friction point.
- **Frontier plotting / visualisation.** JSONL is the deliverable. Plotting belongs in user notebooks or a separate `saeforge plot-frontier` tool, not this change.
- **Automatic K selection (elbow detection).** Caller chooses K (passed through to the polygram CLI in Step 1).
- **A new outer FSM (e.g. `SweepMachine`).** The sweep is a flat Python loop. There's no orchestration state worth elevating to orca-lang here — each forge is independent, restart semantics are file-presence-based, and adding a machine layer would be ceremony without payoff.
- **Cross-encoding statistical analysis (paired tests, confidence intervals).** Out of scope; emit the data, the analyst computes the verdict.
- **In-process polygram `plan_pareto` consumption.** Possible follow-up: skip the disk roundtrip and have sae-forge call `Compressor.plan_pareto()` directly, feeding each plan to a new in-process Compressor.apply path. Deferred — the disk roundtrip is fine for the K=4–12 sweep sizes Axis 4 needs, and the materialised checkpoints are useful artifacts on their own.

## Capabilities

### New Capabilities

- `pareto-sweep`: `ForgePipeline.sweep_pareto()`, `ParetoFrontierRow` dataclass, `saeforge sweep-pareto` CLI subcommand. Public API, exported from `saeforge`.

### Modified Capabilities

- `polygram-tuning-passthrough`: documents that `CompressionConfig`'s new `target_n_features_kept` and `score_field` fields are recognised and round-trip through ctx via the existing `_ConfigMixin` `to_dict` / `from_dict` machinery — no code change, just a scenario pinning the contract. Pins `polygram>=0.4.0` as the minimum.

## Impact

- **New module**: `saeforge/sweep.py` — `ParetoFrontierRow` dataclass, `sweep_pareto()` driver (the directory/file enumeration, the per-row try/except, the resumability check).
- **Modified**:
  - `saeforge/forge.py` — add `ForgePipeline.sweep_pareto(...)` method (thin wrapper around `saeforge.sweep.sweep_pareto` so the call site is `pipeline.sweep_pareto(...)`).
  - `saeforge/cli.py` — new `sweep-pareto` subcommand (mirrors the `forge` subcommand's argument shape).
  - `saeforge/__init__.py` — export `ParetoFrontierRow`.
  - `pyproject.toml` — bump `polygram>=0.4.0` (lines 20, 66, 89).
- **No breaking changes**: existing `ForgePipeline.run()` is unchanged. The new `compression=CompressionConfig(target_n_features_kept=K, ...)` path already flows through the existing FSM via Polygram's own dispatch — sae-forge doesn't need to know about it at the ctx layer.
- **Dependencies**: bumps `polygram` minimum. No new top-level deps.
