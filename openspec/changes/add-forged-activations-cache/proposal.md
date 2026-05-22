# Forged-Activations Cache — the real "warm-start" for progressive sweeps

Add a per-cell forged-activations cache so re-running the same `(basis_config, protein_subset)` combination across progressive-sweep stages is a memmap read instead of a re-extraction. Matches the existing host-extraction cache's contract (content-addressed, opt-out, loud invalidation) but at the forge-output layer.

## Why

The progressive sweep's cumulative-subsample design (PR #83 design Decision 1) means **the same sweep cell often runs more than once** across stages. Concretely, after the openspec's `[10, 50, 200]` schedule for the residue regime:

| stage | proteins | active widths | n=16 cell | n=32 cell | n=64 cell |
|---|---|---|---|---|---|
| 0 | 10 | {4, 8, 16, 32, 64, 128, 256} | ✓ extracted | ✓ extracted | ✓ extracted |
| 1 | 50 | {8, 16, 32, 64, 96} (plateau + neighbours) | ✓ extracted | ✓ extracted | ✓ extracted |
| 2 | 200 | {16, 32, 48, 64} | ✓ extracted | ✓ extracted | ✓ extracted |

The host activations for `dataset.sequences[:10]` already live on disk after stage 0 (host-extraction cache). The **forged activations for `(n=16, sequences[:10])`** also already exist after stage 0 but are thrown away — stage 1 re-extracts them as part of `sequences[:50]` instead.

For the bio-sae 5000-protein pooled SAE at the openspec's predicted `[1000, 5000]` schedule with 8 candidate widths, the total per-cell-pair forge-extraction overhead dominates the sweep's wall-time. **Caching forged activations the same way we cache host activations cuts the dominant cost by 30-50% on typical schedules.** This is the reciprocal-architecture-aware version of the "warm-start" idea — there's no model state to warm-start because the forge is closed-form, but there IS extraction state worth caching.

## What

### `saeforge/datasets/_forge_cache.py` — new module

Mirrors `_host_cache.py`'s shape:

```python
@dataclass(frozen=True)
class ForgeCacheKey:
    host_model_id: str
    sequences_hash: str          # SHA-256 of sequence list (cumulative-stable)
    basis_config_hash: str       # SHA-256 of (kept_ids, scale_boost, encoding_label)
    feed: str                    # "pooled" / "residue"
    max_seq_len: int

class ForgeExtractionCache:
    """File-backed cache for forged activations across sweep cells."""

    def has(self, key: ForgeCacheKey) -> bool: ...
    def load(self, key: ForgeCacheKey) -> Tensor: ...
    def save(self, key: ForgeCacheKey, tensor: Tensor) -> None: ...
```

Storage: `output_dir/forge_cache/forge_<digest>.safetensors` + `forge_<digest>.meta.json` (same shape as the host cache). Single tensor per file: `{"forged_activations": tensor}`.

### `sweep_pareto_capability._run_capability_cell` hook

Inside the per-cell loop, before calling `_extract_forged_activations`:

```python
forge_cache = ForgeExtractionCache(
    output_dir / "forge_cache",
    enabled=cache_forged,
)
forge_key = ForgeCacheKey.from_inputs(
    host_model_id=host_model_id,
    sequences=stage_sequences,
    basis_config=(kept, cell.scale_boost, cell.encoding_label),
    feed=dataset.feed,
    max_seq_len=max_seq_len,
)
if forge_cache.has(forge_key):
    forged_h = forge_cache.load(forge_key)
else:
    forged_h = _extract_forged_activations(...)
    forge_cache.save(forge_key, forged_h)
```

### `sweep_pareto_capability(cache_forged=True)` parameter

Default `True` (consistent with host-cache default). Opt-out for non-deterministic forge paths or disk-scarce environments.

### CLI passthrough

`sae-forge sweep-capability` and `sae-forge sweep-capability-progressive` both gain `--no-forge-cache` (alongside the existing `--no-host-cache`).

## Falsifiable acceptance gate

Two predictions against the bio-sae acceptance gate's fixtures (slice 3/N's slow tests, run twice — once cold-cache, once warm):

| fixture | warm/cold wall-time ratio prediction |
|---|---|
| `runs/uniref50_small/residue`, schedule `[10, 50, 100]` | warm-cache run SHALL complete in ≤ 65 % of cold-cache wall time on the second invocation of the same fixture (overlapping cells across stages account for the largest slice of forge-extraction cost) |
| `runs/uniref50_n5000/pooled_w1024_k64`, schedule `[200, 500]` | warm-cache run SHALL complete in ≤ 70 % of cold-cache wall time (slightly weaker because each stage's protein set is much larger than the residue case, so the per-cell new-protein extraction cost is a bigger share) |

Falsifies if either ratio exceeds the stated bound — would mean the cache lookup isn't actually saving the time it should (likely a key-mismatch bug or a too-aggressive invalidation policy).

## Scope (v1)

- **In:** per-cell forge cache, same shape as host cache, on by default.
- **Out:**
  - Cross-host caching (different `host_model_id` → different key by design).
  - Compression of the cached tensors (safetensors raw fp32; users can post-compress).
  - Partial-cell caching (if a sweep crashes mid-cell, the cell's forge activations aren't half-written — atomic save only).

## Why this is NOT the warm-start the original proposal asked for

The original proposal (pasted to the maintainer 2026-05-22) suggested loading "SAE / forged model from previous stage" + "reset or scale learning rate (default: cosine restart at 30-50 % of previous peak LR)." That maps to a fine-tune-per-stage architecture; the current progressive wrapper has no fine-tune step, so there's no model state to warm-start.

The compute waste the proposal correctly identified — "earlier converged models are excellent initializations for larger data regimes" — manifests in *this* codebase as **redundant forward passes through the closed-form forge across overlapping cells**, not as redundant gradient updates. This openspec ships the architectural-aware version of that win.

If a future change adds optional fine-tuning to the progressive cell, the original proposal's warm-start ideas apply directly — that's a separate openspec (`add-progressive-finetune`).

## Related

- Host cache it mirrors: `saeforge.datasets._host_cache.HostExtractionCache` (PR #77).
- Cumulative subsampling that creates the cache-hit opportunities: design.md Decision 1 of `add-progressive-capability-sweep` (PR #82).
- Companion proposal: `add-scaling-summary-emitter` (the reporting layer the original proposal also asked for).
