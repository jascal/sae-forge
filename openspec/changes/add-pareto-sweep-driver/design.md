## Context

PR polygram#67 ships target-K and Pareto-path planning in `polygram.Compressor`. The cheap step (planning) and the expensive step (SAE rewrite) are deliberately separated there: `polygram compress --pareto K1,K2,K3` writes a `pareto.json` only; `--pareto-materialize` opts into the per-K SAE rewrite.

The natural consumer is a sae-forge sweep driver that takes those materialised per-K SAEs and runs the full forge pipeline against each, emitting one frontier row per `(encoding, K)` for downstream analysis. This is the load-bearing primitive for Axis 4 of the polygram rung-viability methodology — "does the Axis 1 compression-coverage lift cash out in forged-model KL space?"

The forge pipeline already accepts a SAE checkpoint as input and runs StreamMachine → RefineMachine → BasisMachine to project + fine-tune. So the sweep driver is mostly outer-loop orchestration: enumerate checkpoints, invoke `ForgePipeline.run()` per one, collect results, write JSONL.

## Goals / Non-Goals

**Goals:**
- One JSONL row per `(encoding, target_n_features_kept)` forge result, with all the fields needed to plot two overlaid Pareto frontiers (e.g. MPSRung1 vs Rung4).
- Resumability: a sweep that fails midway can be re-invoked and skips completed rows.
- Per-row failure isolation: one row's `RuntimeError` doesn't abort the sweep.
- A `--frontier-only` mode that skips forge runs and just emits per-checkpoint plan metadata, for cheap exploration.
- Byte-equivalence with `ForgePipeline.run()`: `sweep_pareto([single_ckpt])` produces a single row whose forged-model artifacts are byte-identical to running `ForgePipeline.run(single_ckpt)` directly.

**Non-Goals:**
- Invoking `polygram compress` from sae-forge. Caller handles Step 1.
- Plotting, frontier statistics, encoding-vs-encoding hypothesis tests.
- Adding a `SweepMachine` orca-lang FSM. The sweep loop is flat Python.
- Cross-process parallelism. The sweep is sequential; users with multiple GPUs can run multiple `--encoding` flags in separate processes.
- Automatic target-K selection (elbow detection).

## Decisions

### Decision 1 — Consume materialised SAEs from disk, not in-process `plan_pareto`

The sweep driver accepts directories of `k_{K}.safetensors` files (the `polygram compress --pareto-materialize` output layout) plus a sibling `pareto.json` manifest. It does NOT call `polygram.Compressor.plan_pareto()` in-process.

**Why**: the disk roundtrip is cheap (~10s of MB per checkpoint for typical SAEs, K=4–12 typical sweep size) and the materialised checkpoints are reusable artifacts in their own right — you can re-run sae-forge with different fine-tune knobs against the same per-K SAEs, inspect them with `saeforge inspect`, or hand them to other tools. In-process consumption would couple the sweep driver to polygram's planning lifecycle and force a Python-side cache that doesn't reuse across runs.

**Alternative considered**: import `Compressor` and run `plan_pareto + apply` in-process per row. Rejected — couples sae-forge to polygram's planner internals, doesn't reuse across sweep restarts, complicates resumability (would have to cache plans in memory or to disk anyway).

### Decision 2 — `--encoding LABEL:PATH` flag, repeatable

A sweep over multiple encodings (the Axis 4 cross-encoding comparison) is the primary use case. The CLI takes a repeatable `--encoding mps:runs/mps/pareto/` flag that pairs a human-readable label with a path. The path can be either:

- A directory containing `k_{K}.safetensors` files (the `polygram compress --pareto-materialize --out <dir>` layout — files land under `<dir>/pareto/`).
- A single `.safetensors` file (degenerate single-K sweep — useful for byte-identity testing).

**Why labels at all**: the JSONL frontier rows carry `encoding_label`; without it, downstream analysis can't tell rows apart when SAE checkpoint filenames are ambiguous. Forcing the user to name them at sweep-config time pushes the labelling decision upstream where it belongs.

**Alternative considered**: read the encoding label out of the SAE checkpoint metadata. Rejected — not all SAE checkpoints carry an encoding label, and the polygram side doesn't define a stable metadata key for it.

### Decision 3 — Resumability via output-JSONL scan, not via lockfiles

Before starting a row, the driver reads the existing `frontier.jsonl` (if any) and builds a set of completed `(encoding_label, target_n_features_kept)` tuples. Rows in that set are skipped. There is no separate lockfile or "in-progress" marker.

**Why**: the JSONL is append-only by construction; a crashed mid-row write produces a partial line that the JSON parser will reject on resume. The driver SHALL detect a truncated last line and rewrite it to a `null`-error row (treated as "skip and retry"). This is simpler than lockfiles and gives the analyst a record that a retry happened.

**Alternative considered**: a separate `<dir>/<encoding>/k_{K}/.done` sentinel. Rejected — duplicate state, easy to get out of sync with the JSONL.

### Decision 4 — Per-row failure isolation, not all-or-nothing

A row that raises (OOM, NaN gradient, missing SAE key, etc.) writes a JSONL row with the error message populated and the metric fields as `null`. The sweep continues to the next row. The driver returns a non-zero exit code if any row errored.

**Why**: a 12-row sweep that fails on row 7 should not throw away rows 1–6. The analyst wants to see the partial frontier and the failure mode side-by-side.

**Alternative considered**: abort on first error. Rejected — wastes the cheap rows that already completed.

### Decision 5 — `--frontier-only` reads polygram metadata, no forge

`--frontier-only` is for exploring the Pareto plan space before paying the forge cost. It enumerates the same checkpoints the full sweep would and emits a JSONL with only `encoding_label`, `target_n_features_kept`, `n_features_kept_actual`, `pareto_reached_target`. The other fields (faithfulness, perplexity, ...) are null.

The `n_features_kept_actual` value comes from the per-K entry of the `pareto.json` manifest's `n_features_kept` field (the polygram-side `CompressionReport.n_features_kept` semantic — count of cluster representatives, per polygram#67 Decision 1), and `pareto_reached_target` comes from the per-K `reached_target` field of the same manifest. If the manifest is missing, `--frontier-only` falls back to reading the per-checkpoint SAE metadata directly and counts non-zero feature rows; in that fallback, `pareto_reached_target` is `None` (undeterminable without the manifest).

**Why**: an analyst inspecting a fresh `polygram compress --pareto` output wants to see the shape of the plan family before deciding which K values are worth forging. `--frontier-only` is the cheap-look API.

**Alternative considered**: make this a separate `inspect-pareto` subcommand. Rejected — same enumeration logic; cheaper to expose as a flag.

### Decision 6 — JSONL not CSV, append-only writes

The frontier output is JSONL, one row per forge, written via `f.write(json.dumps(row) + "\n")` with `f.flush()` after each row. Forge failures write a row with `error_message` populated. Successful rows have `error_message: null`.

**Why JSONL**: row schema may evolve (new fields), forward-compatible with column additions, easy to `jq` over, natural for streaming append. CSV would require column-locking and re-emits the header every time the schema changes.

**Why append-only**: enables the Decision-3 resumability scan without a separate state file.

### Decision 7 — `sweep_pareto` is a module-level function; `ForgePipeline.sweep_pareto` is a thin wrapper

The implementation lives in `saeforge/sweep.py::sweep_pareto(...)` as a top-level function taking a `ForgePipeline` instance plus the sweep parameters. `ForgePipeline.sweep_pareto(...)` is a one-line method that delegates to it.

**Why**: keeps the sweep logic testable in isolation (no need to construct a full pipeline for orchestration unit tests; just pass a mock with a `.run()` method), and keeps `forge.py` from growing a hundred lines of orchestration that have nothing to do with the pipeline's core responsibilities.

**Alternative considered**: put it all on `ForgePipeline`. Rejected — bloats the class with orchestration code, harder to test the loop in isolation.

### Decision 8 — `ParetoFrontierRow` is a dataclass, not a TypedDict

Frozen dataclass with explicit `to_json_dict()` / `from_json_dict(cls, data)` methods, matching the polygram convention. `__post_init__` validation: `target_n_features_kept >= 1`, `n_features_kept_actual` either `>= 0` or `None`, `elapsed_seconds >= 0`.

**Why**: typed access, IDE support, validation. TypedDict would lose the validation and the convenience methods.

## Risks / Trade-offs

- **Two-step caller workflow**: users must run `polygram compress --pareto-materialize` first, then `saeforge sweep-pareto`. A `--auto-materialise` flag could collapse this — explicitly deferred. Friction-acceptance check: this is the same two-step pattern callers already use (`polygram compress` → `saeforge forge`); we're just adding a multi-K loop on top.

- **Frontier non-monotonicity is preserved, not papered over**: the [[project_kl_nonmonotonic]] finding (KL got worse going 25→211 features on GPT-2 layer-8) means individual rows may not be monotonic in K. The sweep emits the data as-is. Smoothing or post-hoc Pareto-front filtering is an analyst's job, not the driver's.

- **Sequential by design**: the sweep is single-process. A user with 4 GPUs runs 4 sweeps with different `--encoding` flags. Cross-process coordination (a shared output JSONL) is not provided — encode the parallelism in the output directory layout instead.

- **GPU memory pressure on large sweeps**: every row inside a single sweep loads the host model + a per-K forged model into the same process. Long sweeps (many K, large hosts) accumulate transient state across rows — the driver SHALL release per-row tensors before advancing, but users targeting Gemma-2-2B / 8B-tier hosts on a single GPU should split sweeps by encoding (one process per `--encoding`) rather than packing many encodings into one invocation. Documented in CLI help; not enforced.

- **`--frontier-only` accuracy depends on polygram's manifest**: if `polygram compress --pareto-materialize` doesn't write the `pareto.json` manifest with `n_features_kept_actual` and `pareto_reached_target` per K, the `--frontier-only` mode falls back to reading SAE metadata. The polygram-side spec (PR polygram#67, `pareto-compression`) requires this manifest, so the fallback is defensive.
