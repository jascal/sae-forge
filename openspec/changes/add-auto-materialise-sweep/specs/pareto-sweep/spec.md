# pareto-sweep Specification (delta)

## MODIFIED Requirements

### Requirement: ParetoFrontierRow dataclass

The `saeforge.sweep.ParetoFrontierRow` SHALL retain all existing fields and gain three new optional fields capturing methodological provenance when the sweep runs under `--auto-materialise`. The class SHALL continue to expose `.to_json_dict()` and `.from_json_dict(cls, data)`; `from_json_dict` SHALL accept dicts missing the new keys and default them to `None` (backwards compat with frontier.jsonl emitted by sae-forge prior to this change).

The full row schema (in declaration order):

| Field | Type | Pre-materialised | Auto-materialised | Frontier-only | Row failure | Description |
|-------|------|------------------|-------------------|---------------|-------------|-------------|
| `encoding_label` | `str` | populated | populated | populated | populated | Caller-supplied label from `--encoding LABEL:PATH` |
| `target_n_features_kept` | `int` | populated | populated | populated | populated | K requested upstream of polygram |
| `n_features_kept_actual` | `int \| None` | populated | populated | populated | `None` if pre-forge failure; populated if forge ran | Per-K `n_features_kept` from manifest |
| `pareto_reached_target` | `bool \| None` | populated | populated | populated if manifest present, else `None` | `None` if pre-forge failure | Per-K `reached_target` from manifest |
| `faithfulness_kl` | `float \| None` | populated | populated | `None` | `None` | Post-forge KL(host ‖ forged) |
| `perplexity` | `float \| None` | populated | populated | `None` | `None` | Post-forge perplexity on eval set |
| `final_fine_tune_loss` | `float \| None` | populated | populated | `None` | `None` | Last training-loss value |
| `sae_checkpoint` | `str` | populated | populated | populated | populated | Absolute path to the per-K SAE consumed |
| `forged_model_path` | `str \| None` | populated | populated | `None` | `None` | Absolute path to forged-model output dir |
| `elapsed_seconds` | `float` | populated | populated | `0.0` | populated or `0.0` | Wall-clock; `>= 0` |
| `error_message` | `str \| None` | `None` on success | `None` on success | `None` | populated | Set iff the forge raised |
| **`validation_threshold`** | **`float \| None`** | **`None`** | **populated** | **populated when from same auto-materialise run** | **`None` or populated** | **The `polygram_overlap_threshold` that produced this row's per-K plan** |
| **`encoding_class`** | **`str \| None`** | **`None`** | **populated** (e.g. `"MPSRung1"`, `"HEA_Rung2"`) | **populated when from same auto-materialise run** | **`None` or populated** | **Polygram encoding class name used at validation time** |
| **`validation_eval_overlap`** | **`bool \| None`** | **`None`** | **populated** | **populated when from same auto-materialise run** | **`None` or populated** | **`True` iff `--validation-prompts` and `--eval-prompts` resolved to the same path AND `--allow-validation-eval-overlap` was set** |

Three lifecycle states remain normative: success, frontier-only, row failure. Two run modes are normative: pre-materialised (existing) and auto-materialised (new). Pre-materialised rows MAY have the new fields as `None`; auto-materialised rows MUST populate all three.

#### Scenario: ParetoFrontierRow round-trips with provenance fields populated

- **WHEN** a row with `validation_threshold=0.95`, `encoding_class="HEA_Rung2"`, `validation_eval_overlap=False` is serialised via `.to_json_dict()` and reconstructed via `.from_json_dict(...)`
- **THEN** the reconstructed instance equals the original

#### Scenario: ParetoFrontierRow.from_json_dict tolerates missing provenance keys

- **WHEN** a dict missing `validation_threshold`, `encoding_class`, and `validation_eval_overlap` is passed to `from_json_dict`
- **THEN** the resulting instance has those three fields set to `None` without raising (backwards compat with pre-change frontier.jsonl files)

### Requirement: CLI subcommand `saeforge sweep-pareto`

The existing `sweep-pareto` subcommand SHALL retain all its current flags. A new mode `--auto-materialise` SHALL be added that bundles polygram-side validation + Pareto planning + per-K materialisation into the same invocation, with these additional flags:

- `--auto-materialise` (`store_true`) — opt-in to the new mode. Flips `--encoding LABEL:PATH`'s `PATH` semantic from "directory of pre-materialised per-K SAEs" to "single SAE-Lens checkpoint to compress on the fly".
- `--validation-prompts PATH` — required iff `--auto-materialise`; JSONL of prompts fed to `BehaviouralValidator`.
- `--pareto K1,K2,K3,...` — required iff `--auto-materialise`; the target K list for `Compressor.plan_pareto`.
- `--layer N` — required iff `--auto-materialise`; transformer layer for the validator hook.
- `--validation-threshold FLOAT` (default 0.7) — `ValidationConfig.polygram_overlap_threshold`.
- `--validation-jaccard-threshold FLOAT` (default 0.3) — `ValidationConfig.jaccard_threshold`.
- `--score-field {polygram_overlap,jaccard,decoder_overlap}` (default `polygram_overlap`) — Pareto sort axis.
- `--rep-selection {n_fires,scale_aware}` (default `scale_aware`) — Compressor representative-selection mode.
- `--encoding-class LABEL:CLASS` (repeatable, default per encoding `MPSRung1`) — selects polygram encoding class.
- `--encoding-qubits LABEL:N` (repeatable, for `HEA_Rung2` only) — selects `n_qubits`.
- `--allow-validation-eval-overlap` (`store_true`) — opt-in to passing identical paths for `--validation-prompts` and `--eval-prompts`. Without this flag, the CLI refuses identical paths at parse time.
- `--force-rematerialise` (`store_true`) — bypass the materialisation cache. Skips the cache-hit check; re-runs validator + `plan_pareto` + `apply` for every encoding; overwrites existing files in place (no pre-clean).
- `--plan-only` (`store_true`) — print per-encoding cache decisions (`HIT` / `MISS` plus the diffing field list when MISS), target K list, and SHA-256 fingerprints of the SAE checkpoint and validation prompts to stderr; exit 0 without invoking validator, Compressor, or forge. Mutually exclusive with `--frontier-only`.

The CLI SHALL refuse the following invocations with non-zero exit and a clear error message:

1. Any of `--validation-threshold`, `--validation-prompts`, `--validation-jaccard-threshold`, `--layer`, `--pareto`, `--encoding-class`, `--encoding-qubits`, `--allow-validation-eval-overlap` passed WITHOUT `--auto-materialise` (the validator-tuning surface is auto-materialise-only).
2. `--auto-materialise` set with `--encoding LABEL:PATH` where PATH is a directory (mixed mode is disallowed; pre-materialised flow requires absence of `--auto-materialise`).
3. `--auto-materialise` set without one of the required flags (`--validation-prompts`, `--pareto`, `--layer`).
4. `--validation-prompts` and `--eval-prompts` resolving to identical paths without `--allow-validation-eval-overlap`. Error message SHALL name the methodological leakage risk.
5. `--encoding-class LABEL:UNKNOWN` for any class name outside the supported set (`MPSRung1`, `Rung3`, `Rung4`, `HEA_Rung2`).
6. `--force-rematerialise` or `--plan-only` without `--auto-materialise`. Both flags are auto-materialise-only.
7. `--frontier-only` AND `--plan-only` set together (mutually exclusive — different lifecycle stages).

#### Scenario: --auto-materialise drives the full pipeline end-to-end

- **WHEN** `saeforge sweep-pareto --auto-materialise --encoding mps:sae.safetensors --validation-prompts vp.jsonl --eval-prompts ep.jsonl --pareto 8,16,32 --layer 8 --host-model gpt2 --output-dir out/` is invoked
- **THEN** the CLI runs polygram's `BehaviouralValidator → Compressor.plan_pareto → Compressor.apply` chain once (caching artifacts under `out/_materialised/mps/`), then runs the existing sweep loop against those artifacts, writing `out/frontier.jsonl` with 3 rows whose `validation_threshold`, `encoding_class`, and `validation_eval_overlap` fields are populated

#### Scenario: same path for validation and eval prompts is refused

- **WHEN** `--validation-prompts shared.jsonl --eval-prompts shared.jsonl` is passed without `--allow-validation-eval-overlap`
- **THEN** the CLI exits non-zero before any polygram work happens; stderr contains the words "validation" and "eval" and "leakage"

#### Scenario: same path for validation and eval prompts is permitted with the override

- **WHEN** the same invocation is repeated with `--allow-validation-eval-overlap`
- **THEN** the CLI proceeds and every row in `frontier.jsonl` has `validation_eval_overlap == True`

#### Scenario: validator-tuning flag without --auto-materialise is refused

- **WHEN** `saeforge sweep-pareto --validation-threshold 0.9 --encoding mps:/some/dir --host-model gpt2 --output-dir out/` is invoked (no `--auto-materialise`)
- **THEN** the CLI exits non-zero; stderr names `polygram compress` as the place to tune thresholds for the pre-materialised flow

#### Scenario: mixed-mode invocation is refused

- **WHEN** `--auto-materialise --encoding mps:/dir/with/k_files` is invoked (PATH is a directory)
- **THEN** the CLI exits non-zero; stderr says auto-materialise requires a `.safetensors` file path per encoding

#### Scenario: byte-identity preserved when --auto-materialise is absent

- **WHEN** a sweep is run with all existing pre-materialised flow flags and `--auto-materialise` absent
- **THEN** the resulting `frontier.jsonl` is byte-identical to the pre-change reference (modulo timestamps in `elapsed_seconds`), and every row's `validation_threshold`, `encoding_class`, `validation_eval_overlap` fields are `null`

## ADDED Requirements

### Requirement: Auto-materialise cache under `<output-dir>/_materialised/`

When `--auto-materialise` is set, the sweep driver SHALL write all polygram-produced artifacts under `<output-dir>/_materialised/<label>/`. The directory SHALL contain:

- `validation_report.json` — output of `BehaviouralValidator.run()`.
- `pareto.json` — output of `Compressor.plan_pareto(targets).to_json()`.
- `pareto/k_<K>.safetensors` — one materialised SAE per requested K (output of `Compressor.apply(plan=outcome.plan, output_checkpoint=...)`).
- `auto_materialise_meta.json` — a JSON object recording the cache key inputs: SHA-256 of the SAE checkpoint content, SHA-256 of the validation-prompts file content, validator thresholds, encoding class + kwargs, layer, model_name, targets, score_field, rep_selection.

On rerun, the driver SHALL compute the cache key from the current invocation and compare to `auto_materialise_meta.json`. If they match AND all expected `pareto/k_<K>.safetensors` files are present, the driver SHALL skip the validator + Compressor pass entirely and use the cached artifacts.

#### Scenario: cache hit skips validator + Compressor

- **GIVEN** a previous `--auto-materialise` run wrote `<output-dir>/_materialised/mps/` with `auto_materialise_meta.json` matching the current invocation's cache key
- **WHEN** the same invocation is re-issued
- **THEN** polygram's `BehaviouralValidator.run()` and `Compressor.plan_pareto()` are NOT called (instrumented call counters return 0); the sweep proceeds directly with the cached `pareto/k_<K>.safetensors` files

#### Scenario: cache miss when validator threshold changes

- **GIVEN** a previous run cached at `validation_threshold=0.7`
- **WHEN** rerun with `--validation-threshold 0.9` and otherwise identical args
- **THEN** the cache is invalidated; the validator + Compressor run again; the new `auto_materialise_meta.json` records `validation_threshold=0.9`

#### Scenario: cache key is content-addressed, not path-addressed

- **GIVEN** a previous run cached against `/path/A/sae.safetensors`
- **WHEN** rerun against `/path/B/sae.safetensors` whose file content is byte-identical to A's
- **THEN** the cache hits (the cache key SHA-256s the file content, not the path)

#### Scenario: cache miss when a per-K safetensors is deleted

- **GIVEN** a cached materialised dir whose `pareto/k_16.safetensors` was manually removed
- **WHEN** the sweep is rerun
- **THEN** the cache is treated as invalid (any missing per-K file → re-materialise the whole encoding's directory)

### Requirement: Validation-eval prompt leakage firewall

The driver SHALL enforce that `--validation-prompts` and `--eval-prompts` resolve to distinct file system paths by default. The check uses `Path.resolve()` so symbolic links and relative paths are normalised before comparison. If the resolved paths are identical, the driver refuses with a non-zero exit unless `--allow-validation-eval-overlap` is set.

When `--allow-validation-eval-overlap` is set AND the paths resolve identically, every emitted frontier row SHALL have `validation_eval_overlap == True`. In every other case (different paths, or different paths with the override set), the field SHALL be `False`.

The driver does NOT compare prompt-file contents. Users who manually copy identical prompts into two differently-named files defeat the check; this is an accepted limitation (see design.md Risks). The check catches the dominant accidental case.

#### Scenario: distinct paths produce overlap=False

- **WHEN** `--validation-prompts vp.jsonl --eval-prompts ep.jsonl` is passed (distinct files)
- **THEN** every row has `validation_eval_overlap == False`

#### Scenario: identical paths via symlink are caught after resolve()

- **GIVEN** `eval.jsonl` is a symbolic link to `validation.jsonl`
- **WHEN** the sweep is invoked with these two paths without `--allow-validation-eval-overlap`
- **THEN** the CLI exits non-zero; the resolved paths are equal so the firewall triggers

### Requirement: Auto-materialise refuses ClusteredDictionary encodings

The driver SHALL only accept polygram encoding classes whose `from_sae_lens` path returns a plain `Dictionary` instance. The supported set is `MPSRung1`, `Rung3`, `Rung4`, `HEA_Rung2`. `clustered=True` paths returning `ClusteredDictionary` are NOT exposed because `BehaviouralValidator.__post_init__` requires `.features` access that `ClusteredDictionary` does not satisfy.

For feature counts exceeding `MPSRung1`'s cap of 8, the user SHALL pass `--encoding-class LABEL:HEA_Rung2 --encoding-qubits LABEL:N` where `2^N >= N_FEATURES`.

#### Scenario: unknown encoding class is refused at CLI parse time

- **WHEN** `--encoding-class mps:Bogus` is passed
- **THEN** the CLI exits non-zero; stderr lists the supported classes (`MPSRung1`, `Rung3`, `Rung4`, `HEA_Rung2`)

#### Scenario: HEA_Rung2 with n_qubits=5 supports 32-feature SAEs

- **WHEN** `--encoding-class mps:HEA_Rung2 --encoding-qubits mps:5` is passed against a 32-feature sliced SAE
- **THEN** the auto-materialise step builds `HEA_Rung2(n_qubits=5)` and `from_sae_lens` returns a `Dictionary` (not `ClusteredDictionary`) that `BehaviouralValidator` accepts

### Requirement: `--force-rematerialise` bypasses the cache

When `--force-rematerialise` is set under `--auto-materialise`, the driver SHALL bypass the cache-hit check entirely and invoke the validator + `Compressor.plan_pareto` + `Compressor.apply` chain for every encoding regardless of `auto_materialise_meta.json` content. Existing files in the materialised directory SHALL be overwritten in place; the driver SHALL NOT pre-clean the directory.

The cache key recorded in `auto_materialise_meta.json` after a `--force-rematerialise` run is identical to what a cache-miss run with the same inputs would produce.

#### Scenario: --force-rematerialise re-runs validator on a populated cache

- **GIVEN** a cached materialised directory whose `auto_materialise_meta.json` matches the current invocation's cache key
- **WHEN** the sweep is rerun with `--force-rematerialise`
- **THEN** polygram's `BehaviouralValidator.run()` is invoked (instrumented call count > 0); the resulting `pareto/k_<K>.safetensors` files overwrite the previous ones; `auto_materialise_meta.json` content is identical to the pre-rerun version

#### Scenario: --force-rematerialise without --auto-materialise is refused

- **WHEN** `--force-rematerialise` is passed without `--auto-materialise`
- **THEN** the CLI exits non-zero with an error message naming the conflict

### Requirement: `--plan-only` prints the plan and exits

When `--plan-only` is set under `--auto-materialise`, the driver SHALL print to stderr (one block per encoding):

- `label`: the encoding label
- `cache_status`: `HIT` or `MISS`. On `MISS`, the list of cache-key fields that differ from the cached `auto_materialise_meta.json` (or `cold` if no cache exists)
- `sae_sha256`: hex-encoded SHA-256 of the SAE checkpoint file content
- `validation_prompts_sha256`: hex-encoded SHA-256 of the validation prompts file content
- `targets`: the requested K list
- `encoding_class` + `encoding_kwargs`
- `validator_forward_count_estimate`: an integer estimate of the number of host-model forward passes the validator will run (`len(prompts) * average_token_count`)

The driver SHALL exit 0 after printing all blocks. NO validator, Compressor, or forge calls are made. NO files are written under the output directory (including `frontier.jsonl`).

`--plan-only` is mutually exclusive with `--frontier-only`.

#### Scenario: --plan-only on cold cache prints MISS for every encoding

- **GIVEN** a sweep against two encodings with no `_materialised/` directories present
- **WHEN** the invocation is run with `--plan-only --auto-materialise`
- **THEN** stderr contains two blocks, each with `cache_status: MISS (cold)`; stdout is empty; exit code 0; `_materialised/` is NOT created; `frontier.jsonl` is NOT written

#### Scenario: --plan-only on warm cache prints HIT

- **GIVEN** a previous successful `--auto-materialise` run wrote `_materialised/mps/`
- **WHEN** the same invocation is rerun with `--plan-only`
- **THEN** the `mps` block reports `cache_status: HIT`; no Compressor / validator / forge calls happen; exit 0

#### Scenario: --plan-only with cache-key mismatch lists diffing fields

- **GIVEN** a previous run cached at `--validation-threshold 0.7`
- **WHEN** rerun with `--plan-only --validation-threshold 0.95` (otherwise identical)
- **THEN** the encoding's block reports `cache_status: MISS (validation_threshold)`; exit 0

#### Scenario: --plan-only and --frontier-only are mutually exclusive

- **WHEN** both `--plan-only` and `--frontier-only` are passed
- **THEN** the CLI exits non-zero naming the conflict; neither mode's effects occur
