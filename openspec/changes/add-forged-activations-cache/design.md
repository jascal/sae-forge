## Context

PR #77 shipped the host-extraction cache (`HostExtractionCache`), which keys on `(host_model_id, sequences_hash, aggregator, max_seq_len, feed)` and stores host activations per `output_dir`. Stage K+1 of a progressive sweep reuses stage K's host activations for the overlapping prefix.

What stage K+1 does NOT reuse: **forged activations per cell**. If stage K and stage K+1 both have width `n=16` in their active set (very common — plateau membership tends to persist), stage K+1 currently re-extracts forged activations for that cell from scratch over its full protein subset. Even on the overlapping protein prefix where the inputs are identical and the forge construction is deterministic, the forward pass runs again.

Per-cell forge extraction is the dominant cost in the progressive sweep at scale: an 800-protein single cell at ~5ms/protein on CPU is ~4s; the sweep's host extraction is the same shape but happens once per stage, not per cell. Caching forged outputs at the cell granularity is the next-cheapest cost to eliminate.

## Goals / Non-Goals

**Goals:**
- File-backed forged-activations cache, matching the host-cache contract (content-addressed key, opt-out, loud invalidation, no in-library telemetry).
- Cache key includes the basis config (kept feature ids + scale_boost + encoding label) so genuinely-different forges don't collide.
- Default-on, opt-out via `cache_forged=False` parameter / `--no-forge-cache` CLI flag.
- Falsifiable wall-time gates against bio-sae's existing slow fixtures.

**Non-Goals:**
- Cross-`host_model_id` cache sharing.
- Compression of cached tensors (downstream concern; safetensors is the wire format).
- Caching of partial-cell state (forge extraction is atomic; partial writes corrupt the cache, prevented by safetensors' write-then-rename pattern).
- Caching during fine-tuning (the progressive wrapper doesn't fine-tune; if a future revision adds it, the warm-start story attaches to that change, not this one).

## Decisions

### Decision 1 — Cache key includes `basis_config_hash`, not just `(kept_ids, scale_boost)`

The forge's behaviour depends on:
- The set of kept feature ids (`basis.kept_ids`).
- The scale_boost value (or "auto").
- The encoding label (currently informational — `raw_slice` — but a future revision adds polygram encodings; the cache key must distinguish).

Hashing these three together as `basis_config_hash` makes the cache **forward-compatible with the encoding axis** without forcing a schema migration when the encoding becomes load-bearing.

`kept_ids` is hashed by its byte representation (`bytes(kept_ids.astype(np.int64).tobytes())`) — length-stable across runs, fast to compute.

### Decision 2 — File format mirrors host cache exactly

`forge_<digest>.safetensors` + `forge_<digest>.meta.json` under `output_dir/forge_cache/`. The `.safetensors` carries a single tensor under key `"forged_activations"`; the `.meta.json` carries the full cache-key payload for the loud-invalidation check.

Mirroring the host-cache file shape is deliberate: any operator script that runs against the host cache (debugging, manual invalidation, disk usage audit) Just Works on the forge cache too. The only difference is the directory name (`host_cache/` vs `forge_cache/`).

### Decision 3 — Cache is per-sweep-output-dir, NOT global

The cache lives under `output_dir/forge_cache/`. Reusing the cache across `sweep_pareto_capability(...)` calls with the same `output_dir` is the dominant intended use case (progressive sweep stages). Reusing across different `output_dir` calls is NOT supported — each top-level sweep call gets its own cache.

This is a deliberate **conservative scope choice**. A global cache (e.g. `~/.cache/sae-forge/forge_activations/`) would save more compute across independent users running the same `(host, SAE, sequences, basis_config)` combination, but introduces real complexity:

- Stale-cache risk when the sae-forge version changes the forge's numerical output (a future regression).
- Disk-usage hygiene (no automatic eviction policy v1).
- Security: a shared cache directory shared across users on multi-tenant hosts.

The per-output-dir cache satisfies the primary use case (progressive sweep cache hits across stages) without any of the above. If a global cache becomes valuable, that's its own openspec.

### Decision 4 — Atomic writes via safetensors' load_file / save_file

`safetensors.torch.save_file` writes to the target path directly without an intermediate atomic-rename. For our use case (single-process sweep, deterministic forge construction) this is acceptable: if the process crashes mid-write, the partial `.safetensors` file fails to parse on the next load and we re-extract. The `.meta.json` is written AFTER the `.safetensors` so a partial-write of the tensor file means the meta is absent, which `cache.has()` correctly detects as "not cached."

This is the same partial-write story as the host cache — not new risk.

### Decision 5 — Default-on with documented opt-out

Default `cache_forged=True` (matches host cache). Opt-out paths:

- Programmatic: `sweep_pareto_capability(..., cache_forged=False)`.
- CLI: `sae-forge sweep-capability --no-forge-cache` (and its progressive cousin).

Reasons to opt out:
- Non-deterministic forge paths (theoretical; none ship today, but a future fine-tune step would qualify).
- Disk-scarce environments (large `output_dir` + many cells).
- Debugging a suspected cache-key mismatch.

## Risks / Trade-offs

- **Disk footprint.** A 1000-protein × d_model=320 fp32 tensor is ~1.3MB per cell; at 8 widths × 4 stages = 32 cached cells per sweep, ~40MB. Tractable at the bio-sae scale; could grow at 100k+ proteins. Mitigation: the CLI `--no-forge-cache` flag is documented in `--help`; large-scale sweeps can opt out.
- **Forward-compatibility burden.** Once the cache ships, changes to the forge's numerical output (e.g. a refactor that changes the projection algebra by an epsilon) invalidate every previously-cached cell. Mitigation: `basis_config_hash` includes a version byte (`b"\x01"` initially); bumping it on any output-shape-changing refactor causes a clean cache miss everywhere.
- **Subtle cache poisoning.** If a user manually edits the cache files between sweep calls and the meta.json still validates, the load returns wrong data. Mitigation: the meta-mismatch check raises `RuntimeError` on any payload divergence; a user editing the .safetensors but NOT the meta is doing something unsupported.
- **Test coverage cost.** Two new slow gates (the warm/cold wall-time ratio tests) join the existing 3-slow-test set, lifting total slow-suite runtime from ~7 min to ~10-12 min. Still tractable; not run by default.
