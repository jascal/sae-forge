## Context

`ForgePipeline` today carries a mixed bag of polygram-related kwargs:

- Two flat fields plumbed through to `Compressor` (`compression_strategy`, `rep_selection`) — covered.
- Eight `EpochCompressor` knobs (`coverage_target`, `cosine_threshold`, `n_visits_per_feature`, `max_iterations`, `polygram_overlap_threshold`, `jaccard_threshold`, `min_both_fire`, `quality_delta_multiplier`) — none plumbed; only reachable by editing `examples/forge_gpt2_real_sae.py:125-138`.
- Five `Regrower` kwargs (`regrow_strategy`, `regrow_layer`, `regrow_seed`, `regrow_prompts`, `model_name`) — read from FSM context inside `saeforge/actions/__init__.py:118-124` with hard-coded fallbacks (`layer=10`, `model_name="gpt2"`).

The companion polygram change `polygram-tuning-config` introduces typed config dataclasses (`CompressionConfig`, `EpochCompressionConfig`, `RegrowConfig`, `SAEImportConfig`, `CancellationConfig`, `ValidationConfig`) and accepts a `config=` kwarg on every relevant constructor. It also drops the GPT-2 defaults from `Regrower.from_compression_report` — so without this change, the regrow path here breaks at runtime.

This change is the sae-forge-side passthrough: own the polygram tuning surface as typed fields on `ForgePipeline`, ship a clean handoff into FSM context, and make the regrow path loud-fail when the host model isn't specified.

## Goals / Non-Goals

**Goals:**
- One typed home per polygram tuning concern on `ForgePipeline`: `compression`, `epoch_compression`, `regrow`.
- FSM context carries plain dicts (round-tripped via `cfg.to_dict()` / `cfg.from_dict()`) so the existing JSON-friendly trace tooling keeps working unchanged.
- High-frequency knobs reachable from CLI (`--coverage-target`, `--cosine-threshold`, `--max-compress-iterations`, `--regrow-layer`, `--regrow-strategy`); long tail stays Python-API-only.
- Regrow path requires explicit `RegrowConfig` (carrying `model_name` and `layer`) when `regrow_count > 0`. No GPT-2 fallback.
- `examples/forge_gpt2_real_sae.py` shrinks to one line of compression config.

**Non-Goals:**
- No YAML loader. `ForgePipeline.from_dict` covers it; YAML parsing belongs to whoever calls the loader (one-line `yaml.safe_load`).
- No env-var fallback. Tuning is explicit.
- No back-compat shim for `compression_strategy=` / `rep_selection=` flat kwargs. The migration is one line per caller; we'd rather break loudly at construction than maintain two surfaces.
- Not changing `ForgePipeline`'s finetune fields. Those are sae-forge-internal training knobs, not polygram-related; they stay flat for now.
- Not touching `SAEImportConfig`, `CancellationConfig`, `ValidationConfig`. sae-forge doesn't call `from_sae_lens` or `Cancellation` directly. Embedded `ValidationConfig` reaches polygram via `EpochCompressionConfig.validation`.

## Decisions

### Decision 1 — Three polygram fields on ForgePipeline, not one mega-config

```python
@dataclass
class ForgePipeline:
    ...
    compression: CompressionConfig | None = None
    epoch_compression: EpochCompressionConfig | None = None
    regrow: RegrowConfig | None = None
    ...
```

**Why three?** They map 1:1 to the three polygram entry points sae-forge calls (`Compressor`, `EpochCompressor`, `Regrower`). One mega-config would conflate independent concerns and force callers using only one path to fill in unrelated fields. Polygram already split them; we mirror that split.

**Why optional with `= None`?** Callers who want polygram defaults pass nothing. The action layer interprets `None` as "construct the default config" → calls `Compressor(...)` without `config=`. This matches the today behaviour for callers who omit the flat kwargs.

**Alternatives considered:**
- One `polygram: PolygramConfig` field bundling all three — rejected: same conflation argument as polygram's own design doc.
- Keep flat fields and add config fields alongside — rejected: two ways to set the same knob is the exact mess we're trying to clean up.

### Decision 2 — FSM context carries dicts, not dataclasses

```python
ctx["compression"] = self.compression.to_dict() if self.compression else None
# inside compress_with_polygram:
cfg = CompressionConfig.from_dict(ctx["compression"]) if ctx.get("compression") else None
Compressor(..., config=cfg)
```

**Why dicts on ctx?** The orca-lang FSM context is JSON-serialised for tracing and machine verification. A frozen dataclass with tuples isn't JSON-trivially-serialisable. Polygram's `to_dict` / `from_dict` already handle the tuple↔list coercion and the unknown-key warning policy — we just call them.

**Why not pass dataclasses on ctx anyway?** orca's runtime doesn't care, but the JSON trace dump would lose information or crash. Sticking to dicts keeps the FSM contract clean.

**Why round-trip every entry?** Idempotent. If ctx came from a YAML loader (already a dict), we convert via `CompressionConfig.from_dict` → re-validate → use; if ctx came from `ForgePipeline._build_context`, we already converted via `to_dict`. Either input path lands in the same place.

### Decision 3 — Regrow path requires explicit config; no GPT-2 fallback

The action today reads:
```python
layer=ctx.get("regrow_layer", 10),
model_name=ctx.get("host_model_id") or "gpt2",
```

After this change:
```python
if ctx.get("regrow_count", 0) > 0:
    if not ctx.get("regrow"):
        raise ValueError("regrow_count > 0 requires ForgePipeline(regrow=RegrowConfig(...))")
    cfg = RegrowConfig.from_dict(ctx["regrow"])
    Regrower.from_compression_report(report, ..., config=cfg)
```

`ForgePipeline.__post_init__` enforces the same rule at construction so callers get the error before the FSM kicks off.

**Why no warn-then-default?** A silently-wrong layer index for a non-GPT-2 host produces nonsense regrowth. The error is cheap to fix (one line per caller), the bug is expensive to diagnose. Fail at construction.

### Decision 4 — CLI exposes 5 knobs; the rest are Python-API only

Picked by frequency-of-use heuristic (asked: "if I were tuning this from a shell script, what would I touch?"):

- `--coverage-target` (compression aggressiveness)
- `--cosine-threshold` (clustering sensitivity)
- `--max-compress-iterations` (cost cap)
- `--regrow-layer` (host-model-specific)
- `--regrow-strategy` (algorithm choice)

Everything else (`polygram_overlap_threshold`, `jaccard_threshold`, `min_both_fire`, `quality_delta_multiplier`, `regrow_seed`, `regrow_prompts`, `regrow_n_init`, `merge_mode`, `confirmer`) stays Python-API-only. CLI users who need them can `--config-file path/to/forge.yaml`.

**Why not all of them?** A 25-flag CLI is unreadable. The dataclass+YAML path covers the long tail; the CLI covers the daily-driver knobs.

### Decision 5 — `ForgePipeline.from_dict` for YAML/JSON config

Add a thin classmethod that:
- Pops `compression`, `epoch_compression`, `regrow` keys and feeds them through the matching `<Config>.from_dict`.
- Passes the rest as `ForgePipeline(**rest)`.
- Surfaces unknown top-level keys as `UserWarning` (mirroring polygram's policy).

**Why classmethod, not freestanding loader?** Discoverability: `ForgePipeline.from_dict` is the natural place to look. A freestanding `load_pipeline` adds a new symbol with no advantage.

## Risks / Trade-offs

- **[Risk] Callers passing `compression_strategy=` / `rep_selection=` get a hard `TypeError`.** → **Mitigation:** the migration is one substitution; document in README and CHANGELOG. Add a one-liner check in `ForgePipeline.__init_subclass__`/`__post_init__`? Considered — overkill; the dataclass `__init__` already raises `TypeError: unexpected keyword argument` cleanly.
- **[Risk] Callers with `regrow_count > 0` and no `regrow` config will trip __post_init__.** → **Mitigation:** intentional. Error message names the field and shows a one-line fix (`regrow=RegrowConfig(model_name=..., layer=...)`).
- **[Risk] The polygram dependency must land first.** → **Mitigation:** pin `polygram>=<version-with-tuning-config>` in `pyproject.toml`; add a `try/except ImportError` at the action-import site that points users at the polygram upgrade.
- **[Risk] FSM trace files now contain serialised configs — bigger payloads.** → **Mitigation:** trivial in size (≤2KB per config bundle); no observable cost.
- **[Trade-off] Removing `compression_strategy` / `rep_selection` flat fields is breaking.** Could deprecate-with-warning for one cycle. Not worth it: sae-forge is pre-v0.1 and has zero external users that we know of; CHANGELOG entry suffices.

## Migration Plan

1. Bump `polygram` minimum version in `pyproject.toml` to the version that ships `polygram-tuning-config`.
2. Add `compression` / `epoch_compression` / `regrow` fields and `from_dict` to `ForgePipeline`. Keep flat `compression_strategy` / `rep_selection` working for now (read them, build a `CompressionConfig` if `compression is None`).
3. Switch ctx serialisation in `_build_context` to use `to_dict()` for the three configs.
4. Update `compress_with_polygram` to read `ctx["compression"]` + `ctx["epoch_compression"]` and pass `config=`.
5. Update `perform_regrowth` to require `ctx["regrow"]` when `regrow_count > 0`; remove GPT-2 fallbacks.
6. Update `saeforge/machines/sae_forge.orca.md` context schema; remove the per-field legacy keys.
7. Update CLI to convert flags into config dataclasses.
8. Update `examples/forge_gpt2_real_sae.py` to use `EpochCompressionConfig` directly.
9. Drop the flat `compression_strategy` / `rep_selection` fields. Update CHANGELOG.

Steps 1–8 land in one PR; step 9 in a follow-up so we get a clean "before / after" for any external user.

Rollback: each commit is independent; reverting steps 5/9 individually is safe. The polygram-side `Regrower` default-removal is the only thing that genuinely can't be rolled back without a polygram patch — coordinate the merges accordingly.

## Open Questions

- Should `ForgePipeline.from_dict` also recurse into `finetune_*` flat fields by accepting a nested `finetune: dict`? Out of scope here; tracked as follow-up after this change settles.
- Do we want a `ForgePipeline.fast()` / `.thorough()` preset mirroring `EpochCompressor.fast() / .thorough()`? Probably yes once we see how many callers configure all three polygram fields the same way; defer until that pattern shows up.
