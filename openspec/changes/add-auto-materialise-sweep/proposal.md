## Why

The merged `pareto-sweep` capability requires a two-step workflow: `polygram compress --pareto --pareto-materialize` first, then `saeforge sweep-pareto`. That decoupling preserved a clean disk-handoff boundary and made resumability trivial — but it shifts a load-bearing methodological knob (the validator's `polygram_overlap_threshold`) into a different tool than the one the user is driving.

The live N=32 Axis-4 smoke during PR #33 surfaced this friction: 87% of candidate pairs gate-passed the default `polygram_overlap_threshold=0.7` on a small prompt corpus, so every Pareto target collapsed to `kept=1` — a degenerate frontier. The user's feedback loop is "look at sweep output → reach for polygram CLI → retune threshold → re-materialise → retry sweep." Two muscle groups, two tools, easy to mis-tune.

There's a real risk in fixing this naïvely: if sae-forge collapses the two tools into one with a single `--prompts` flag, users will share validation prompts with eval prompts. That's **methodologically broken** — the validator's gate decisions are tuned against the same prompts that score post-forge faithfulness, which is a form of leakage that inflates the KL number for whichever K the validator favoured. The two-tool workflow ducks this by forcing the user to author two corpora.

`--auto-materialise` is the version that owns the workflow at the sae-forge layer while preserving the leakage firewall as a first-class API constraint. Validation prompts and eval prompts are **two separate, required flags**; sharing them is mechanically possible but explicit, surfaced in the row metadata, and not the default.

## What Changes

### `sweep-pareto --auto-materialise` mode

When `--auto-materialise` is passed, the existing `--encoding LABEL:PATH` argument's `PATH` semantics flip: instead of pointing to a directory of pre-materialised `k_{K}.safetensors` files, `PATH` points to a single uncompressed SAE-Lens checkpoint that sae-forge will compress on the fly. The driver runs polygram's `BehaviouralValidator → Compressor.plan_pareto → Compressor.apply` chain once per encoding, materialises per-K SAEs under a deterministic location (`<output-dir>/_materialised/<label>/pareto/k_{K}.safetensors`), and then proceeds with the existing sweep against those artifacts.

New required arguments under `--auto-materialise`:

- `--validation-prompts PATH` — JSONL of prompts for `BehaviouralValidator`. Mutually distinct from `--eval-prompts` at the API layer (different flag names; sweep refuses if they are the same file path unless `--allow-validation-eval-overlap` is set).
- `--pareto K1,K2,K3,...` — target K list, passed through to `Compressor.plan_pareto`.
- `--layer N` — transformer layer whose residual stream the validator hooks.

New optional arguments:

- `--encoding-class LABEL:CLASS_NAME` — repeatable. Maps an encoding label (matching one of the `--encoding` flags) to a polygram encoding class (`MPSRung1`, `Rung3`, `Rung4`, `HEA_Rung2`). Defaults to `MPSRung1`. With `HEA_Rung2`, `--encoding-qubits LABEL:N` selects `n_qubits`.
- `--validation-threshold FLOAT` — `ValidationConfig.polygram_overlap_threshold` (default 0.7, polygram's calibration).
- `--validation-jaccard-threshold FLOAT` — `jaccard_threshold` (default 0.3).
- `--score-field {polygram_overlap,jaccard,decoder_overlap}` — passes through to `CompressionConfig.score_field` for the Pareto sort axis.
- `--rep-selection {n_fires,scale_aware}` — passes through to `CompressionConfig.rep_selection`.
- `--allow-validation-eval-overlap` — disable the same-file-path refusal between `--validation-prompts` and `--eval-prompts`. Off by default; emits a `validation_eval_overlap: true` field in every row of `frontier.jsonl` when set, so downstream analysis can flag the methodological compromise.

### Frontier row schema gains methodological-provenance fields

`ParetoFrontierRow` gains three new fields, populated only when `--auto-materialise` runs:

- `validation_threshold: float | None` — the `polygram_overlap_threshold` that produced the per-K plan. Lets downstream analysis correlate frontier degeneracy with validator gate looseness.
- `encoding_class: str | None` — e.g. `"MPSRung1"`, `"HEA_Rung2"`. Required for cross-encoding sweeps so the row carries its own provenance.
- `validation_eval_overlap: bool` — `True` iff `--validation-prompts` and `--eval-prompts` resolve to the same path AND `--allow-validation-eval-overlap` was set. `False` or `None` otherwise.

When `--auto-materialise` is NOT set (existing flow), the three new fields are `None`.

### Materialisation cache + resumability

`<output-dir>/_materialised/<label>/` is a deterministic location whose existence is the resumability signal: if it contains a `pareto.json` and the expected `pareto/k_{K}.safetensors` files, the driver skips re-validation and re-materialisation for that encoding. The cache key is derived from `(label, sae_checkpoint_path, validation_prompts_path, validation_threshold, jaccard_threshold, encoding_class, encoding_qubits, layer, score_field, rep_selection, targets)` — any change forces a re-materialise. Cache key + manifest go into `<output-dir>/_materialised/<label>/auto_materialise_meta.json` for inspection.

### Out of scope, deliberately

- **Parallelising validators across encodings.** Sequential by design; same single-process posture as the existing sweep.
- **Caching validation reports across runs in different `output-dir`s.** The per-run `_materialised/` directory is the cache; cross-run sharing would require a global cache root and is deferred.
- **Mixed mode** (some `--encoding` flags auto-materialised, others pre-materialised in the same invocation). Refused at CLI parse time — pick one mode per invocation. Mixed sweeps run with two separate invocations.
- **Validator prompt-set autogeneration / stride-sampled-feature-set helpers.** Caller responsibility.
- **Owning polygram's threshold calibration.** This change makes the threshold a sae-forge-side knob in CLI ergonomics but does not change its semantic — polygram still owns what the threshold means.

## Capabilities

### Modified Capabilities

- `pareto-sweep`: `--auto-materialise` flag + `--validation-prompts` / `--validation-threshold` / `--encoding-class` family + the three provenance fields on `ParetoFrontierRow`. Existing two-step flow unchanged when `--auto-materialise` is absent.

## Impact

- **New module**: `saeforge/auto_materialise.py` — wraps polygram's `BehaviouralValidator → Compressor.plan_pareto → Compressor.apply` chain into a single function returning the path to a deterministic `<output-dir>/_materialised/<label>/` directory layout that the existing `sweep_pareto` can consume.
- **Modified**:
  - `saeforge/sweep.py` — `ParetoFrontierRow` gains three new fields with `None` defaults; `_process_row` populates them when an `auto_materialise_meta` is provided.
  - `saeforge/cli.py` — new flags on the `sweep-pareto` subcommand; `_cmd_sweep_pareto` dispatches to the auto-materialise pre-step before the existing sweep loop.
  - `saeforge/forge.py` — `ForgePipeline.sweep_pareto` gains pass-through kwargs for the auto-materialise pre-step.
- **No breaking changes**: existing two-step workflow byte-identical when `--auto-materialise` is absent. Existing `frontier.jsonl` consumers see the three new fields as `null`.
- **Dependencies**: requires `polygram>=0.4.0` (already pinned). Validator + Compressor are part of the existing polygram surface — no new polygram features needed.
