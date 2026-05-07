# Contributing to sae-forge

sae-forge is OpenSpec-driven: non-trivial work is staged as a change in
`openspec/changes/<change-name>/` before any code lands. The flow:

1. Open an OpenSpec change with `proposal.md` (why + scope), `tasks.md`
   (checklist), and `specs/<capability>/spec.md` (delta requirements +
   scenarios). Validate with `openspec validate <change-name>`.
2. Implement against the tasks. Keep diffs tight to the change scope.
3. When done, archive with `openspec archive <change-name>` and add a
   line to `CHANGELOG.md`.

## Local dev

```bash
pip install -e ".[dev,torch,polygram]"
pytest
ruff check saeforge tests examples
```

## House rules

- Only `numpy` and `safetensors` are mandatory runtime deps. Lazy-import
  torch / transformers / polygram inside modules that need them so
  `import saeforge` works on a no-extras install.
- No emojis in code or generated artifacts.
- Default to no comments unless the *why* is non-obvious. Don't explain
  what the code does; explain hidden constraints.
- Match the polygram CLI style — verbs first, file paths positional.
- Hardware-sensitive behaviour (per-layer streaming, dtype) goes behind
  explicit `ForgePipeline` knobs, never auto-detection.

## Testing

- Pure-numpy paths (basis load, projector math on synthetic shapes) run
  in the default CI matrix.
- Torch-dependent tests live behind `pytest.importorskip("torch")` and
  run on the `[torch]` extras job.
- Each public class needs a smoke test that constructs it and exercises
  one round-trip: load → inspect, project → shape check, forge → tiny
  forward pass.

## Q-Orca integration

sae-forge does not generate Q-Orca artifacts directly. Quantum-flavoured
analysis of the basis (interference, cancellation, tier separation)
flows through Polygram, which is the canonical Q-Orca emitter. If you
want to investigate a forged basis quantum-mechanically, build a
Polygram `Dictionary` from `basis.W_dec[:n]` with `from_sae_lens` and
hand it to Polygram's `Experiment` / `Cancellation` / `EpochCompressor`.
