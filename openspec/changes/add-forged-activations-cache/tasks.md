# Implementation tasks

## 1. `saeforge/datasets/_forge_cache.py` — new module

- [ ] 1.1 Mirror `_host_cache.py`'s `HostCacheKey` / `HostExtractionCache` shape: `ForgeCacheKey` (frozen dataclass with `host_model_id`, `sequences_hash`, `basis_config_hash`, `feed`, `max_seq_len`) + `ForgeExtractionCache` (file-backed with `has` / `load` / `save`).
- [ ] 1.2 `ForgeCacheKey.from_inputs(host_model_id, sequences, basis_config, feed, max_seq_len)` — basis_config is a tuple `(kept_ids, scale_boost, encoding_label)`. Hash with a version byte (`b"\x01"` initially) so a future forge-output-changing refactor can bump it cleanly.
- [ ] 1.3 Storage layout: `cache_dir/forge_<digest>.safetensors` + `forge_<digest>.meta.json`. Single tensor under key `"forged_activations"`. `meta.json` payload carries the full cache-key fields for the loud-invalidation check.
- [ ] 1.4 Loud invalidation: `cache.load(key)` SHALL raise `RuntimeError` when the on-disk meta payload doesn't match the requested key's `to_meta_dict()`. Mirrors the host cache's behaviour.

## 2. Wire into `sweep_pareto_capability._run_capability_cell`

- [ ] 2.1 Construct `ForgeExtractionCache` once per sweep (top-level wrapper), pass through to per-cell runner.
- [ ] 2.2 In `_run_capability_cell`: build `ForgeCacheKey` for the cell, check `cache.has(key)`, load if present, otherwise extract + save.
- [ ] 2.3 New parameter `cache_forged: bool = True` on `sweep_pareto_capability(...)` (default-on; opt-out for non-deterministic forge paths).

## 3. Wire through `sweep_pareto_capability_progressive`

- [ ] 3.1 New parameter `cache_forged: bool = True` on the progressive wrapper; passes through to each per-stage `sweep_pareto_capability` call. The forge cache is shared across stages (single `output_dir/forge_cache/`).

## 4. CLI surface

- [ ] 4.1 `sae-forge sweep-capability --no-forge-cache` — passes `cache_forged=False` to the wrapper.
- [ ] 4.2 `sae-forge sweep-capability-progressive --no-forge-cache` — same passthrough.
- [ ] 4.3 `--help` text on both subcommands documents the flag + the rationale (non-deterministic forge / disk-scarce / debugging).

## 5. Unit tests

- [ ] 5.1 `tests/test_forge_cache.py`:
  - `test_cache_key_is_content_addressed`: same inputs → same key; different inputs → different key. Specifically verify the basis_config components (kept_ids, scale_boost, encoding_label) all participate.
  - `test_cache_hit_miss_round_trip`: save then load returns identical tensor.
  - `test_cache_opt_out_skips_io`: `enabled=False` constructor → `has` always returns False, `save` is a no-op.
  - `test_cache_corrupted_meta_raises`: hand-edited meta.json → `load` raises `RuntimeError`.

## 6. Tests covering the integration

- [ ] 6.1 `tests/test_sweep_pareto_capability.py`: extend with `test_forge_cache_hits_on_overlapping_cells` — call `sweep_pareto_capability` twice against the synthetic ESM fixture with the same args + `cache_forged=True`; assert the second call's wall time is ≤ 50 % of the first (cache hit on every cell).
- [ ] 6.2 `tests/test_sweep_progressive.py`: extend with `test_progressive_reuses_forge_cache_across_stages` — assert that progressive's stage 1 reads the forge cache emitted by stage 0 for the overlapping cell (specifically: stage 0 produces N cache files; stage 1 reads them without overwriting; stage 1 produces only the NEW cells' files).

## 7. Falsifiable acceptance gate (slow)

- [ ] 7.1 `tests/test_progressive_acceptance_gate.py::test_residue_warm_cache_is_faster` (slow): run the residue regime once cold-cache; run again warm-cache; assert warm/cold wall-time ratio ≤ 0.65.
- [ ] 7.2 `tests/test_progressive_acceptance_gate.py::test_pooled_warm_cache_is_faster` (slow): same shape, pooled regime; assert ratio ≤ 0.70.

## 8. Documentation

- [ ] 8.1 README: extend the "Capability-aware forge tuning" + "Progressive capability sweep" sections with one paragraph each on the forge cache (where it lives, how to opt out, disk-usage expectations).
- [ ] 8.2 CHANGELOG entry under `[Unreleased]`.

## 9. Release

- [ ] 9.1 Bump `__version__` to `0.9.0` (new public surface: `cache_forged` parameter + the cache module). Promote `[Unreleased]` → `[0.9.0]`.
- [ ] 9.2 Tag `v0.9.0` on the merge commit.
