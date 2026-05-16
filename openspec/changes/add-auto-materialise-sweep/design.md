## Context

The merged `pareto-sweep` capability requires the caller to run `polygram compress --pareto --pareto-materialize` per encoding before invoking `saeforge sweep-pareto`. This forces a two-tool workflow where polygram owns the validator + Pareto planner and sae-forge owns the per-K forge sweep.

Live experience from PR #33's N=32 smoke (10 prompts, GPT-2 layer 8, stride-sampled features, `polygram_overlap_threshold=0.7`): 87% of candidate pairs gate-passed and every K collapsed to `kept=1`. The user's reaction was "the validator threshold is too loose for this prompt set" — a sae-forge-frontier-driven judgment that has to be acted on via the polygram CLI.

The two-tool friction has a methodological upside that's load-bearing: the validator's prompts and the eval's prompts are kept lexically separate by the workflow. Collapsing them into one tool with a single `--prompts` flag would invite users to share corpora, which is leakage: the validator gates pairs against prompts that later score post-forge KL, and the gate-favoured features perform artificially well on the eval. The current two-tool flow ducks the problem by forcing the user to author two prompt files.

## Goals / Non-Goals

**Goals:**
- One sae-forge invocation drives the whole Axis-4 workflow (validate → plan_pareto → apply → sweep → frontier).
- Validation prompts and eval prompts are **two separate, required, distinctly-named flags** under `--auto-materialise`. The CLI refuses when their paths resolve identically unless an explicit opt-in flag is set.
- Frontier row metadata carries enough methodological provenance (`validation_threshold`, `encoding_class`, `validation_eval_overlap`) that downstream analysis can correlate frontier shape with validator tuning.
- Materialisation is cached on disk under the run's output directory so reruns don't redo the expensive validator pass.
- Existing two-step flow byte-identical when `--auto-materialise` is absent.

**Non-Goals:**
- Parallelising validators across encodings.
- Cross-run / global materialisation cache.
- Mixed mode (auto + pre-materialised encodings in one invocation).
- Validator prompt-set autogeneration.
- Owning polygram's threshold semantics or shipping a sae-forge-side validator.

## Decisions

### Decision 1 — Two separate prompt flags, refuse same-path by default

`--validation-prompts` and `--eval-prompts` are distinct CLI flags. When `--auto-materialise` is set, both are required (the no-eval-prompts path is allowed in the existing sweep, but auto-materialise without eval would be a useless run).

If `Path(args.validation_prompts).resolve() == Path(args.eval_prompts).resolve()`, the CLI refuses with a non-zero exit and an error message explaining the leakage risk. The user can override with `--allow-validation-eval-overlap`, which surfaces the choice in every frontier row's `validation_eval_overlap` field.

**Alternative considered**: silently allow same-path with a stderr warning. Rejected — warnings get ignored; the row-metadata surfacing forces downstream analysis to confront the choice. The leakage firewall is the *whole reason* for this proposal, so it has to be a first-class API constraint, not an advisory.

**Alternative considered**: hash the prompt content and compare hashes (catch users who copy the same prompts into two files). Rejected — too clever; users who want to share prompts will work around any check. The same-path refusal catches the dominant accidental case; the content-overlap case is the user's methodological problem.

### Decision 2 — `PATH` semantic flips under `--auto-materialise`

Without `--auto-materialise`: `--encoding LABEL:DIR` where `DIR` is a polygram-materialised directory.
With `--auto-materialise`: `--encoding LABEL:SAE` where `SAE` is a single uncompressed SAE-Lens checkpoint.

The flip is necessary because the two modes consume different artifacts. CLI help documents both; the dispatch in `_cmd_sweep_pareto` selects the interpretation based on whether `--auto-materialise` is present.

**Alternative considered**: separate flag names (`--encoding-dir` vs `--encoding-sae`). Rejected — doubles the flag surface and forces users to remember which is which. The mode flag (`--auto-materialise`) is the single switch; `--encoding`'s interpretation follows.

**Alternative considered**: detect the path's nature (file vs directory) and dispatch on that. Rejected — fragile (a user could pass a single-file pre-materialised K) and harder to error-message clearly.

### Decision 3 — Encoding class is per-encoding, not global

`--encoding-class LABEL:CLASS_NAME` is repeatable and matches by label. The Axis-4 use case is cross-encoding comparison, so a global `--encoding-class` would force all encodings to use the same class — defeating the comparison.

**Alternative considered**: encode the class in the encoding spec: `--encoding mps:sae.safetensors:MPSRung1`. Rejected — three-field colon-delimited specs are hard to read at a glance, and parsing breaks on paths containing colons (Windows drives). Two repeatable flags pair by label cleanly.

**Default**: `MPSRung1`. Same default as polygram's `from_sae_lens`.

### Decision 4 — Materialisation cache lives under `<output-dir>/_materialised/`

Cache directory layout:
```
<output-dir>/
  frontier.jsonl
  _materialised/
    <label>/
      auto_materialise_meta.json   # cache key + provenance
      validation_report.json       # polygram's ValidationReport.to_json
      pareto.json                  # polygram's ParetoReport.to_json
      pareto/
        k_<K>.safetensors          # one per requested K
  <label>/
    k_<K>/                         # per-K forge outputs (same as existing sweep)
```

`auto_materialise_meta.json` carries:
- `sae_checkpoint_sha256`, `sae_checkpoint_path`
- `validation_prompts_sha256`, `validation_prompts_path`
- `validation_threshold`, `jaccard_threshold`, `min_firing_rate`, `min_both_fire`
- `encoding_class`, `encoding_kwargs` (e.g. `{"n_qubits": 5}` for HEA_Rung2)
- `layer`, `model_name`
- `targets`, `score_field`, `rep_selection`

Cache hit: meta matches → skip the (expensive) validator + planner + apply chain. Cache miss: regenerate the whole materialised tree for that label.

**Why under the run's `output-dir`**: keeps everything one run produces in one tree; resumability semantic for the materialise step is the same as for the forge step (file-presence-driven). A global cache would invite cross-run staleness footguns.

**Alternative considered**: hash the cache key into a content-addressed directory name (e.g. `_materialised/<hash>/`). Rejected — opaque to inspect; the label is already user-meaningful.

### Decision 5 — Three new `ParetoFrontierRow` fields, all default `None`

`validation_threshold: float | None`, `encoding_class: str | None`, `validation_eval_overlap: bool | None`. Default `None` in the existing pre-materialised flow; populated under `--auto-materialise`.

This widens the row schema in a backwards-compatible way: existing `frontier.jsonl` consumers see `null` for the new fields. `from_json_dict` accepts missing keys (existing behaviour for forward-compatibility).

**Alternative considered**: a nested `provenance: dict | None` field. Rejected — analysts using `jq` are well-served by flat keys; nested dicts add ceremony.

### Decision 6 — Refuse `--validation-threshold` etc. without `--auto-materialise`

Validator-tuning flags only make sense in auto-materialise mode. If the user passes `--validation-threshold` without `--auto-materialise`, the CLI refuses with a message pointing at `polygram compress` as the place to tune thresholds for pre-materialised flows. Prevents silent mis-config.

### Decision 7 — `--force-rematerialise` and `--plan-only` escape hatches

Two operational flags, both opt-in, both cheap:

- **`--force-rematerialise`** bypasses the cache regardless of whether `auto_materialise_meta.json` matches. The cache key fingerprints validator-influencing inputs (SAE checkpoint content, validation prompts content, threshold, encoding, layer, targets) but cannot detect every drift mode — a user who edits polygram's source code in a sibling checkout, for example, won't invalidate the cache. The flag is the documented escape hatch for "I know the cache is stale, just rebuild it." It does NOT clear the cache directory before writing; existing files are overwritten in place so partial-write recovery still works.

- **`--plan-only`** prints the per-encoding cache-hit/miss decision plus the validator's prompt-fingerprint and target list, then exits 0 without doing any expensive work. Inspired by `terraform plan`: lets the user verify "yes, this is the run I think it is" before paying for it. Mutually exclusive with `--frontier-only` (the two modes overlap conceptually — both skip the forge — but `--frontier-only` still reads materialised manifests, while `--plan-only` skips even those).

**Alternative considered for `--force-rematerialise`**: a `rm -rf` of the encoding's cache directory before the run. Rejected — partial-write footgun (a Ctrl-C during cache-clear could leave the directory half-deleted), and overwriting in place is idempotent.

**Alternative considered for `--plan-only`**: fold into `--frontier-only`. Rejected — `--frontier-only` is for "show me the per-K manifest data without forging," which presumes materialisation already happened. `--plan-only` is for "show me what materialisation would do without doing it." Different lifecycle stages; different flag.

### Decision 8 — `BehaviouralValidator` dictionary-type gotcha

`BehaviouralValidator` accepts a `Dictionary` (not `ClusteredDictionary`), so `from_sae_lens(records, ids, clustered=True)` is incompatible with auto-materialise. Live evidence from the N=32 smoke during PR #33 implementation.

The driver SHALL only use polygram encodings whose `from_sae_lens` path returns a plain `Dictionary`: `MPSRung1`, `Rung3`, `Rung4`, `HEA_Rung2`. The CLI does not expose a `--clustered` flag. Larger feature counts that don't fit MPSRung1's cap of 8 are addressed by `--encoding-class HEA_Rung2 --encoding-qubits LABEL:N` (cap = 2^N).

**Alternative considered**: pass `ClusteredDictionary` through anyway and let polygram raise. Rejected — bad UX; the refusal at CLI-parse time names the right fix.

## Risks / Trade-offs

- **Auto-materialise hides the polygram-side knob layer.** Power users tuning the long tail (`min_firing_rate`, `min_both_fire`, `allow_layer_zero`, `rep_selection="n_fires"` vs `"scale_aware"`, encoding-class kwargs beyond `n_qubits`) will hit the surface limit of the CLI flags and have to drop back to the two-tool flow. That's acceptable — the auto-materialise CLI is for "I want a one-tool Axis-4 sweep with reasonable defaults," not "I'm doing exotic validator tuning." The full polygram surface stays available via the existing pre-materialised flow.

- **Materialisation cache makes wrong-defaults sticky.** If a user runs once with `--validation-threshold=0.7` and gets a degenerate frontier, then reruns with `--validation-threshold=0.95`, the cache invalidates correctly. But if they rerun WITHOUT changing the threshold flag (e.g. they edited `eval.jsonl` instead), the cache stays hot and the materialised SAEs are stale relative to the new eval. The new eval doesn't affect materialisation though, so this is fine — the cache key is over the validator-influencing inputs only. Documented in `auto_materialise_meta.json` so the user can inspect.

- **Same-path refusal is the only built-in leakage guard.** Users who copy the same prompts into two differently-named files defeat the check. We accept this — the file-system-level check catches the dominant accidental case; content-overlap is a methodological discipline issue the tool can't fully police.

- **One-shot validator pass per encoding can be expensive.** A 6B host × 1000-prompt validator run is multiple minutes on a single GPU. Caching addresses reruns, but the first run pays the full cost. The trade-off vs the two-tool flow is identical here (polygram pays the same cost in step 1); auto-materialise just makes the cost more visible by attributing it to the sae-forge invocation.
