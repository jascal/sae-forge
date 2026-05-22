# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: `ForgeExtractionCache` + `ForgeCacheKey` in `saeforge.datasets._forge_cache`

`saeforge.datasets._forge_cache.ForgeCacheKey` SHALL be a frozen dataclass with:

- `host_model_id: str`
- `sequences_hash: str` — SHA-256 of the newline-joined, length-prefixed sequence list.
- `basis_config_hash: str` — SHA-256 of `(kept_ids_bytes, scale_boost_repr, encoding_label, version_byte)`. The version byte (initially `0x01`) makes the cache forward-compatible with future forge-output-changing refactors.
- `feed: str` — `"pooled"` or `"residue"`.
- `max_seq_len: int`

Plus a `from_inputs(host_model_id, sequences, basis_config, feed, max_seq_len)` classmethod, a `digest()` method returning the 16-hex-char filename suffix, and a `to_meta_dict()` method returning the full payload for loud invalidation.

`saeforge.datasets._forge_cache.ForgeExtractionCache` SHALL be a file-backed cache with:

- `__init__(cache_dir, *, enabled=True)`.
- `has(key) -> bool`.
- `load(key) -> torch.Tensor` — raises `RuntimeError` on meta-mismatch.
- `save(key, tensor) -> None`.

Storage layout under `cache_dir/`:

- `forge_<digest>.safetensors` — single tensor under key `"forged_activations"`.
- `forge_<digest>.meta.json` — full cache-key payload as a JSON dict.

### Requirement: `sweep_pareto_capability(cache_forged=True)` parameter

`sweep_pareto_capability(...)` SHALL accept a `cache_forged: bool = True` keyword argument. When `True`:

1. The wrapper SHALL construct a `ForgeExtractionCache` rooted at `output_dir/forge_cache/`.
2. Before calling `_extract_forged_activations` in each cell's runner, the wrapper SHALL build a `ForgeCacheKey` for the (host, sequences, basis_config, feed) tuple, check `cache.has(key)`, and load from disk on hit.
3. On miss, extraction proceeds as today; the result SHALL be saved to the cache before being passed to the scoring step.

When `False`: the wrapper SHALL bypass cache lookups and writes entirely. Forge extraction proceeds as in v0.8.x.

### Requirement: `sweep_pareto_capability_progressive(cache_forged=True)` passthrough

The progressive wrapper SHALL accept a `cache_forged: bool = True` keyword argument and pass it through to each per-stage `sweep_pareto_capability` call. All stages SHALL share the same `output_dir/forge_cache/` directory so stage K+1 reads stage K's cached forge outputs for overlapping cells.

### Requirement: `sae-forge sweep-capability --no-forge-cache` CLI flag

Both `sae-forge sweep-capability` and `sae-forge sweep-capability-progressive` SHALL accept a `--no-forge-cache` flag that passes `cache_forged=False` to the underlying wrapper. The flag's `--help` text SHALL document the rationale (non-deterministic forge paths, disk-scarce environments, debugging suspected cache-key mismatches).

### Requirement: Falsifiable wall-time acceptance gate

The `add-forged-activations-cache` change SHALL include slow integration tests:

1. `tests/test_progressive_acceptance_gate.py::test_residue_warm_cache_is_faster` (`@pytest.mark.slow`): run the residue-regime progressive sweep twice — once cold-cache (delete `forge_cache/` between runs), once warm-cache. Assert `warm_wall_time / cold_wall_time <= 0.65`.

2. `tests/test_progressive_acceptance_gate.py::test_pooled_warm_cache_is_faster` (`@pytest.mark.slow`): same shape, pooled regime. Assert `warm_wall_time / cold_wall_time <= 0.70`.

Both bounds are slack enough to absorb per-process noise on commodity CPUs but tight enough to catch a cache-key mismatch or disabled-by-accident-on-write bug.

### Requirement: Cache key is per-sweep-output-dir, NOT global

The cache directory SHALL be rooted at the wrapper's `output_dir/forge_cache/`. Two separate `sweep_pareto_capability(...)` calls with different `output_dir` arguments SHALL NOT share cached cells, even when the `(host, sequences, basis_config, feed, max_seq_len)` tuple matches exactly.

This is a deliberate v1 scope choice (per design.md Decision 3) — a global cache satisfies broader workflows but introduces stale-cache / disk-hygiene / multi-tenant-security risks out of scope for this change.
