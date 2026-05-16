## Context

`add-auto-materialise-sweep` shipped the one-tool Axis-4 workflow: validate → plan_pareto → apply → sweep → frontier, with per-encoding cache keyed on SAE-content SHA + threshold + encoding class + kwargs + layer + targets. Subsequent ships extended the per-encoding knob set:

- `polygram-0.7.0-rung5` added `Rung5(n_amp_qubits=k)` (cap `8·2^k`) — a Rung whose capacity is a CLI knob rather than a fixed cap. `--encoding-class LABEL:Rung5 --encoding-amp-qubits LABEL:k`.
- `polygram-0.8.0-learned-axis` added `--learn-axis-assignment`, a global boolean piped into polygram's `from_sae_lens` as `learn_axis_assignment=True`. Today it applies to every feature in the dictionary uniformly.

Both knobs are uniform: one capacity choice and one axis-assignment policy per dictionary. The frontier rows from `add-forge-quality-diagnostics` plus the `quality_tier="degenerate"` evidence suggest the uniform assumption is leaving signal on the table — heavy features and tail features want different substrate, and a single global value of either knob is the worst-case compromise.

This proposal makes the partition explicit: an off-line JSON manifest splits the SAE's features into blocks, each block carries its own `(encoding_class, encoding_kwargs, learn_axis_assignment)` triple, and the materialise step composes them into one block-structured dictionary via the polygram capability gated under `add-encoding-partition`.

## Goals / Non-Goals

**Goals:**
- A single new CLI surface (`--encoding-partition LABEL:PATH`) replaces the per-label single-encoding flags when supplied. No flag explosion across the existing per-encoding knobs.
- The partition is a deterministic, content-hashed input: same manifest content → same cache hit. Renaming or moving the manifest file is irrelevant; editing its contents invalidates.
- Provenance rich enough that a downstream frontier-consumer can attribute a row's KL movement to the partition shape without re-reading the manifest.
- Byte-identical behaviour when `--encoding-partition` is absent.
- The polygram-side contract is named in proposal.md but the design here pre-commits to it only at the API surface (`from_sae_lens(..., encoding_partition=...)`), not at the algorithm.

**Non-Goals:**
- Owning polygram's per-block axis-fit algorithm. Whether `learn_axis_assignment=True` on the heavy block uses a different optimiser, a different initialisation, or a different stopping rule is polygram's decision.
- Auto-generating the manifest. The firing-geometry helper is a polygram-side utility a user invokes once before the sweep, not a magic mode of the sweep CLI.
- Sub-block tuning. Per-block thresholds, per-block rep-selection, per-block score-fields are not in scope. The block carries `(encoding_class, encoding_kwargs, learn_axis_assignment, feature_ids)`. Anything else is global.
- Mixing partitioned and non-partitioned encodings in one invocation across different labels. If one label has a partition, all labels using `--auto-materialise` may have partitions or may keep single-encoding. The constraint is per-label exclusivity, not per-invocation.

## Decisions

### Decision 1 — Manifest is per-block kwargs, not flat repeat-flag CLI

The natural alternative is "extend the existing repeat-flag pattern": `--block-encoding-class mps:heavy:Rung5`, `--block-encoding-amp-qubits mps:heavy:4`, `--block-learn-axis-assignment mps:heavy`, `--block-feature-ids mps:heavy:3,17,22,...`. Rejected — the feature-id list is the wrong shape for a CLI flag (hundreds to thousands of integers per block), and the colon-delimited three-level addressing (`encoding:block:value`) is fragile.

A JSON manifest passed by path is the right shape for a heavyweight, edit-on-disk configuration. It also composes cleanly with the firing-geometry helper polygram ships (the helper writes a manifest the user then passes to `--encoding-partition`).

### Decision 2 — Partition is pre-materialisation, frozen, content-hashed

The cache key contribution is the SHA-256 of the manifest file's bytes. Not the SHA-256 of a normalised representation — bytes — so a user who edits whitespace or key ordering gets a cache miss (intentional: it surfaces the edit), but a user who renames or moves the file gets a cache hit (intentional: the cache is content-addressed, not path-addressed). This matches the existing `sae_checkpoint_sha256` / `validation_prompts_sha256` policy from `add-auto-materialise-sweep`.

**Alternative considered**: normalise the manifest (sort keys, strip whitespace, sort feature_ids within each block) before hashing. Rejected — gives the analyst a worse cache-miss diagnosis ("I changed the comment but the cache still missed? oh, the comment is part of the bytes"). The current rule is simpler and the analyst can re-sort feature_ids manually if they want a stable hash.

### Decision 3 — Per-block `learn_axis_assignment`, not global

The current `--learn-axis-assignment` is a global boolean. Under `--encoding-partition`, the global flag is refused for that label (mutually exclusive); the per-block flag in the manifest takes over. A manifest with no `learn_axis_assignment` key on a block defaults to `false`, matching polygram's own default.

**Alternative considered**: keep `--learn-axis-assignment` as a global override that applies to all blocks under partition (i.e., a default the manifest can override per block). Rejected — leaves two sources of truth for the same knob. The exclusivity rule makes the manifest the sole owner.

### Decision 4 — Cross-block pair-scoring runs once over the union (load-bearing)

Polygram's Compressor pair-scoring identifies feature pairs that should cancel under merge. If we ran pair-scoring per block, two features in different blocks that *would* cancel under uniform encoding wouldn't be scored against each other under partition — the partition would silently fragment the cancellation signal.

The polygram-side contract is therefore: pair-scoring runs once over the union of all blocks' features with a single threshold, and only the per-block dictionary materialisation differs. This is the load-bearing claim in the proposal's Risk note; if polygram's implementation reveals the union-mode pair-scoring needs per-block-aware similarity metrics, the polygram-side proposal must surface the change and the sae-forge cache key must extend accordingly. This sae-forge design does not pre-commit to either path.

### Decision 5 — `partition_label` is a deterministic format from manifest content

Format: per block, `block_id` + `:` + `encoding_class` + optional `(...)` carrying kwargs and the `learn` marker when set, joined by `+` across blocks in manifest order.

Examples:
- `heavy:Rung5(k=4,learn)+tail:MPSRung1`
- `top10:HEA_Rung2(n=5,learn)+mid:Rung4+long_tail:MPSRung1`
- `a:Rung4+b:Rung4` (two blocks of the same encoding, distinguished only by `block_id`)

The label is regenerated from the manifest at every materialise call. Two manifests that produce the same label are guaranteed to produce the same materialised dictionary up to feature-id ordering within a block (which polygram is free to canonicalise). The label is therefore safe to use as a row provenance field — analysts comparing frontier rows can read shape directly.

**Alternative considered**: include the manifest SHA in the label. Rejected — the SHA is already in the cache key meta; adding it to the row would clutter the schema and force consumers to look up the manifest to interpret the row. The human-readable shape is the load-bearing field; the SHA is the audit trail.

### Decision 6 — `--encoding-partition` is per-label and mutually exclusive with that label's single-encoding flags

Per-label exclusivity, not global. If a sweep runs `--auto-materialise --encoding mps:sae.safetensors --encoding hea:sae.safetensors`, the user may pass `--encoding-partition mps:manifest.json` without supplying one for `hea`. The `hea` label continues to consume `--encoding-class hea:HEA_Rung2`, `--encoding-qubits hea:5`, etc.

For any label that has `--encoding-partition LABEL:PATH`, the CLI refuses these flags scoped to that label:
- `--encoding-class LABEL:CLASS`
- `--encoding-amp-qubits LABEL:K`
- `--encoding-qubits LABEL:N`
- `--learn-axis-assignment` (if it scopes to this label; current implementation is global — see Risk 2 below).

**Alternative considered**: global exclusivity (any `--encoding-partition` excludes all single-encoding flags across all labels). Rejected — would force the user to convert single-encoding labels into trivial single-block manifests just to mix modes in one sweep. Per-label is the minimum useful exclusivity.

**Worked example — mixed sweep** (one partitioned label, one single-encoding label, with the global `--learn-axis-assignment` flag):

```
saeforge sweep-pareto \
  --auto-materialise \
  --encoding mps:gpt2_l8.safetensors \
  --encoding hea:gpt2_l8.safetensors \
  --encoding-partition mps:manifests/heavy_rung5_tail_mps.json \
  --encoding-class hea:HEA_Rung2 \
  --encoding-qubits hea:5 \
  --learn-axis-assignment \
  --validation-prompts vp.jsonl --eval-prompts ep.jsonl \
  --pareto 25,50,100,211 --layer 8 \
  --host-model gpt2 --output-dir out/
```

Behaviour:
- The `mps` label materialises via the partition manifest. Each block's `learn_axis_assignment` is owned by the manifest; the global `--learn-axis-assignment` flag does NOT override any block.
- The `hea` label materialises via single-encoding `HEA_Rung2(n_qubits=5)`. The global `--learn-axis-assignment` flag applies → polygram is called with `learn_axis_assignment=True` for that label.
- Frontier rows: `mps` rows carry `encoding_class="BlockStructured"`, `partition_label="heavy:Rung5(k=4,learn)+tail:MPSRung1"`. `hea` rows carry `encoding_class="HEA_Rung2"`, `partition_label=null`. Both row sets share the same eval prompts so KL is directly comparable across them.

If the user converts the `hea` label to a partition too (`--encoding-partition hea:hea_manifest.json` and removes `--encoding-class hea:... --encoding-qubits hea:...`), the global `--learn-axis-assignment` flag is then refused at parse time per Refusal 4 — every label owns its own policy and the global flag is meaningless.

## Risks

1. **Polygram-side scope creep.** If polygram's `BlockStructuredDictionary` ends up needing per-block thresholds (e.g. the cancellation pair-score threshold has to vary across blocks because the heavy block's features are more cancellable), this sae-forge proposal's manifest schema is too thin. Mitigation: the manifest is a JSON object with named keys; adding `compression_threshold` per block is forward-compatible. Existing manifests stay valid.

2. **Global `--learn-axis-assignment` exclusivity edge case.** The current `--learn-axis-assignment` flag is a single global boolean, not per-label. The exclusivity rule says "`--encoding-partition LABEL:...` refuses `--learn-axis-assignment` for that label" — but the flag has no label. The CLI implementation treats the global flag as scoped to "every label without a partition", refused at parse time only if **every** label has a partition (in which case the global flag is meaningless). For mixed sweeps (some labels partitioned, some single-encoding), the global flag applies only to the non-partitioned labels. This is documented in the `--help` text and in the refusal error.

3. **Manifest authoring is heavyweight.** A 50,000-feature SAE produces a manifest with two ~25k-id arrays. The user is expected to generate this via polygram's `partition_by_firing_geometry` helper, not hand-author it. The CLI does not ship a `--generate-partition` mode (that's polygram's job). Documented in the proposal under "Out of scope".

4. **Partition validity at materialise time, not parse time.** The manifest parser validates structure (disjoint, complete coverage of the id range declared in the manifest). It cannot validate that the declared id range matches the SAE checkpoint's actual feature count until the materialise step reads the SAE. A mismatch surfaces as a cache-miss + materialise-time error with a clear message naming the expected vs declared id count.

## Migration

No migration. New CLI surface, additive row field, gated polygram capability. Pre-existing frontier.jsonl files have `partition_label = null` when re-loaded by post-change `from_json_dict`. Pre-existing `_materialised/<label>/` directories without `partition_label` in their cache key meta continue to hit the cache as before for non-partitioned reruns; the first partitioned rerun against the same `<label>` is a cache miss (correct, since the manifest SHA wasn't in the prior meta).
