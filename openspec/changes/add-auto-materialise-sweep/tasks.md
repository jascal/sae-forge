## 1. `ParetoFrontierRow` schema extension

- [ ] 1.1 Add three new fields to `ParetoFrontierRow` (`saeforge/sweep.py`), all defaulting to `None`: `validation_threshold: float | None`, `encoding_class: str | None`, `validation_eval_overlap: bool | None`.
- [ ] 1.2 Update `to_json_dict` / `from_json_dict` to round-trip the new fields. `from_json_dict` SHALL tolerate dicts missing the new keys (backwards compat).
- [ ] 1.3 Update the row schema table in `openspec/changes/add-auto-materialise-sweep/specs/pareto-sweep/spec.md` to include the new fields and their nullability per lifecycle state.

## 2. `auto_materialise` module

- [ ] 2.1 Create `saeforge/auto_materialise.py` with `AutoMateriliseSpec` dataclass: `label: str`, `sae_checkpoint: Path`, `encoding_class: str`, `encoding_kwargs: dict` (e.g. `{"n_qubits": 5}` for `HEA_Rung2`).
- [ ] 2.2 Implement `_compute_cache_key(spec, validation_prompts_sha, threshold_kwargs, layer, model_name, targets, score_field, rep_selection) -> dict` returning a serialisable dict suitable for `auto_materialise_meta.json`.
- [ ] 2.3 Implement `_is_cache_hit(materialised_dir: Path, expected_key: dict) -> bool` — checks `auto_materialise_meta.json` content equality and the presence of all `pareto/k_{K}.safetensors` files.
- [ ] 2.4 Implement `materialise(spec, *, validation_prompts, validation_config, layer, model_name, targets, score_field, rep_selection, output_root) -> Path`:
  - Cache check first; return materialised dir on hit.
  - Resolve `encoding_class` string to the polygram class object via a small registry: `{"MPSRung1": MPSRung1, "Rung3": Rung3, "Rung4": Rung4, "HEA_Rung2": HEA_Rung2}`. Reject `ClusteredDictionary` paths (see design Decision 7).
  - Call `from_sae_lens(records, slot_ids, encoding=encoding_class(**encoding_kwargs))`.
  - Run `BehaviouralValidator(...).run()` → write `validation_report.json`.
  - Run `Compressor(...).plan_pareto(targets)` → write `pareto.json` and per-K `pareto/k_{K}.safetensors` via `Compressor.apply(plan=..., output_checkpoint=...)`.
  - Write `auto_materialise_meta.json` with the cache key.
  - Return the materialised dir path.
- [ ] 2.5 Cache-key SHA inputs are computed from file contents (SAE checkpoint, validation prompts), not paths — so renaming or moving a file doesn't accidentally invalidate the cache.

## 3. `sweep_pareto` driver extension

- [ ] 3.1 Add optional `auto_materialise_specs: list[AutoMateriliseSpec] | None = None`, `validation_prompts: Path | None = None`, `validation_config: ValidationConfig | None = None`, `layer: int | None = None`, `targets: list[int] | None = None`, `score_field: str = "polygram_overlap"`, `rep_selection: str = "scale_aware"`, `validation_eval_overlap: bool = False` kwargs to `sweep_pareto(...)`.
- [ ] 3.2 When `auto_materialise_specs` is provided, the function SHALL:
  - For each spec, call `materialise(...)` to get the materialised dir.
  - Override the `encodings` argument's interpretation: each `(label, _)` from `encodings` SHALL match a spec by label; the spec's materialised dir replaces the original path for the subsequent enumeration loop.
  - Propagate `validation_threshold`, `encoding_class`, and `validation_eval_overlap` into every row produced by that label.
- [ ] 3.3 When `auto_materialise_specs is None`, behaviour is byte-identical to today (the three new fields stay `None`).

## 4. CLI surface

- [ ] 4.1 Add `--auto-materialise` (store_true) to the `sweep-pareto` subparser.
- [ ] 4.2 Add the methodological flags: `--validation-prompts PATH`, `--validation-threshold FLOAT` (default 0.7), `--validation-jaccard-threshold FLOAT` (default 0.3), `--score-field {polygram_overlap,jaccard,decoder_overlap}` (default `polygram_overlap`), `--rep-selection {n_fires,scale_aware}` (default `scale_aware`).
- [ ] 4.3 Add the encoding-class plumbing: `--encoding-class LABEL:CLASS` (repeatable), `--encoding-qubits LABEL:N` (repeatable, for HEA_Rung2).
- [ ] 4.4 Add `--pareto K1,K2,K3,...` and `--layer N` (both required iff `--auto-materialise`).
- [ ] 4.5 Add `--allow-validation-eval-overlap` (store_true).
- [ ] 4.5a Add `--force-rematerialise` (store_true). When set, the driver SHALL skip the cache-hit check and re-run validator + plan_pareto + apply for every encoding regardless of `auto_materialise_meta.json` content. Existing files are overwritten in place (no pre-clean step). Has no effect without `--auto-materialise` — refused at parse time if combined alone.
- [ ] 4.5b Add `--plan-only` (store_true). When set, the driver SHALL print per-encoding cache decisions (`HIT` / `MISS` with the diffing field list) plus the resolved target K list and the SHA-256 fingerprints of the SAE checkpoint and validation prompts to stderr, then exit 0. No validator, no Compressor, no forge calls. Mutually exclusive with `--frontier-only` and refused unless `--auto-materialise` is set.
- [ ] 4.6 Validation in `_cmd_sweep_pareto`:
  - Refuse if `--validation-threshold` / `--validation-prompts` / `--pareto` / `--layer` / `--force-rematerialise` / `--plan-only` are passed without `--auto-materialise` (Decision 6 + Decision 7).
  - Refuse if `--validation-prompts` and `--eval-prompts` resolve to the same path without `--allow-validation-eval-overlap` (Decision 1).
  - Refuse if both `--frontier-only` and `--plan-only` are set (mutually exclusive, Decision 7).
  - Refuse mixed mode: if `--auto-materialise`, every `--encoding LABEL:PATH` PATH must be a single `.safetensors` file, not a directory.
- [ ] 4.7 Build per-encoding `AutoMateriliseSpec` instances from the encoding flags + class flags; hand off to `sweep_pareto` via the new kwargs.

## 5. `ForgePipeline.sweep_pareto` pass-through

- [ ] 5.1 Extend `ForgePipeline.sweep_pareto(...)` with the same new kwargs as `sweep_pareto(...)`; delegate unchanged.

## 6. Tests

### 6.1 Schema extension

- [ ] 6.1.1 `ParetoFrontierRow` with all three new fields populated round-trips via `to_json_dict` / `from_json_dict`.
- [ ] 6.1.2 `ParetoFrontierRow.from_json_dict` on a dict missing the three new keys returns an instance with `None` for those fields (backwards compat).

### 6.2 Cache key

- [ ] 6.2.1 Same `(sae_checkpoint, prompts, threshold, encoding_class, encoding_kwargs, layer, targets, score_field, rep_selection)` → identical cache key.
- [ ] 6.2.2 Changing any input (e.g. flipping `score_field`) yields a different cache key.
- [ ] 6.2.3 SHA inputs use file contents not paths: renaming the SAE file produces the same key.

### 6.3 Cache hit / miss

- [ ] 6.3.1 First call materialises; second call with same inputs SHALL skip the validator and Compressor calls (mock them and assert call counts are 0 on the second pass).
- [ ] 6.3.2 Cache miss when `auto_materialise_meta.json` is missing or its content differs from the expected key.
- [ ] 6.3.3 Cache miss when any expected `pareto/k_{K}.safetensors` file is absent.

### 6.4 CLI validation

- [ ] 6.4.1 `--validation-threshold` without `--auto-materialise` → non-zero exit, error message names polygram CLI.
- [ ] 6.4.2 `--validation-prompts` and `--eval-prompts` same file path → non-zero exit with leakage warning; passes with `--allow-validation-eval-overlap`.
- [ ] 6.4.3 `--auto-materialise --encoding LABEL:DIR` (directory, not file) → non-zero exit with mixed-mode error.
- [ ] 6.4.4 No `--auto-materialise` and `--encoding LABEL:FILE` (file, not dir) continues to work via the existing single-file path (regression check).
- [ ] 6.4.5 `--force-rematerialise` or `--plan-only` without `--auto-materialise` → non-zero exit naming the conflict.
- [ ] 6.4.6 `--frontier-only --plan-only` together → non-zero exit with the mutually-exclusive error message.

### 6.4a `--force-rematerialise` behaviour

- [ ] 6.4a.1 With a populated cache that would normally hit, `--force-rematerialise` makes the driver re-invoke validator + Compressor (mocked, count their calls). The new `auto_materialise_meta.json` content is identical to the previous one (same cache key inputs); existing `pareto/k_<K>.safetensors` files are overwritten in place.
- [ ] 6.4a.2 `--force-rematerialise` on a cold cache behaves identically to no flag (first run still materialises).

### 6.4b `--plan-only` behaviour

- [ ] 6.4b.1 With cold cache: stderr contains `MISS` for every encoding, the target K list, and SHA-256 fingerprints; stdout is empty; exit 0. Validator + Compressor + forge are NOT invoked.
- [ ] 6.4b.2 With warm cache: stderr contains `HIT` for matching encodings.
- [ ] 6.4b.3 With cache-key mismatch (e.g. threshold differs): stderr contains `MISS: validation_threshold` (the diffing fields).
- [ ] 6.4b.4 No `frontier.jsonl` is written under `--plan-only`.

### 6.5 Row provenance population

- [ ] 6.5.1 With `--auto-materialise --validation-threshold=0.95 --encoding-class mps:MPSRung1`, every row in `frontier.jsonl` has `validation_threshold == 0.95`, `encoding_class == "MPSRung1"`, `validation_eval_overlap == False`.
- [ ] 6.5.2 With `--allow-validation-eval-overlap` and same-path prompts, `validation_eval_overlap == True` on every row.
- [ ] 6.5.3 Without `--auto-materialise`, all three provenance fields are `None`.

### 6.6 Encoding-class dispatch

- [ ] 6.6.1 `--encoding-class mps:HEA_Rung2 --encoding-qubits mps:5` builds an `HEA_Rung2(n_qubits=5)` instance for that encoding's `from_sae_lens` call.
- [ ] 6.6.2 Unknown encoding class name (e.g. `--encoding-class mps:Bogus`) raises at CLI parse time with a clear error listing the supported classes.
- [ ] 6.6.3 `HEA_Rung2` without `--encoding-qubits` defaults `n_qubits=3` (polygram default).

### 6.7 End-to-end smoke (optional, gated on polygram + torch)

- [ ] 6.7.1 Toy SAE fixture → `--auto-materialise --pareto 2,4 --layer 0 --validation-prompts <fixture> --eval-prompts <other-fixture>` → assert `frontier.jsonl` has 2 rows, both with finite `n_features_kept_actual`, both with `validation_threshold` populated. Mocks the polygram heavy lifts when running without the `[polygram]` extra; uses real polygram when available.

## 7. Spec update

- [ ] 7.1 Author the `specs/pareto-sweep/spec.md` delta (MODIFIED + ADDED requirements) per the proposal scope: extend `ParetoFrontierRow` schema, add `--auto-materialise` CLI surface requirements, add cache-resumability requirement.

## 8. Docs

- [ ] 8.1 Extend the `#### Pareto sweep (Axis 4)` section in README to describe the `--auto-materialise` one-tool workflow alongside the existing two-tool workflow. Lead with the validation-vs-eval-prompts distinction. **Prominently include the `HEA_Rung2(n_qubits=N)` example** for SAEs with >8 features (live finding from PR #33's N=32 smoke: MPSRung1's 8-feature cap is the dominant friction beyond the validation threshold one). Document `--plan-only` as the recommended pre-flight check before a long sweep, and `--force-rematerialise` as the escape hatch when the user knows the cache is stale.
- [ ] 8.2 CHANGELOG entry under `[Unreleased]` → `### Added (add-auto-materialise-sweep)`.

## 9. Validation

- [ ] 9.1 `openspec validate add-auto-materialise-sweep --strict` is green.
- [ ] 9.2 Full `pytest` suite passes; new tests at least cover sections §6.1 through §6.6.
- [ ] 9.3 `ruff check` clean on touched files.
- [ ] 9.4 Live MBP smoke with auto-materialise — confirm the validator threshold knob shows up in row provenance and that bumping it from 0.7 to 0.95 produces a less-degenerate frontier on the N=32 stride-sampled GPT-2 fixture used in PR #33.
- [ ] 9.5 `openspec archive add-auto-materialise-sweep` after merge.

## 10. What this change explicitly defers

- [ ] 10.1 Parallelising the per-encoding validator pass.
- [ ] 10.2 Cross-run / global materialisation cache outside `output-dir`.
- [ ] 10.3 Mixed mode in one invocation (auto + pre-materialised encodings).
- [ ] 10.4 Validator prompt-set autogeneration; stride-sampled feature selection helpers.
- [ ] 10.5 Content-hash-based prompt-overlap detection (catching users who copy the same prompts into two differently-named files).
- [ ] 10.6 The full polygram tuning surface (`min_firing_rate`, `min_both_fire`, `allow_layer_zero`, custom `confirmer`). Power users keep the two-tool flow.
- [ ] 10.7 In-process consumption of `Compressor.plan_pareto` results (still disk-roundtrip via `_materialised/`).
- [ ] 10.8 A `--validation-config FILE` YAML/JSON loader for the long-tail validator knobs. Deferred until the CLI flag set is felt to be insufficient by real users.
- [ ] 10.9 Surfacing `scale_compression_ratio` and average merged-norm in frontier rows when `--rep-selection scale_aware` (or any merge-strategy run) emits them. Suggested in PR #34 review as ties-into-scale-preservation-work. Deferred because (a) the values originate in polygram's `CompressionReport`, not the per-K plan, and (b) it's a schema decision that affects all sweeps, not just auto-materialise — best handled as its own openspec change with explicit scope over the existing `pareto-sweep` capability.
