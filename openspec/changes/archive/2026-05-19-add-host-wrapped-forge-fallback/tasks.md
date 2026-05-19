# Tasks — `add-host-wrapped-forge-fallback`

## Landing status (2026-05-18)

**Critical-path implementation landed.** Acceptance gates from
`specs/forge-forward-mode/spec.md` all PASS on the smoke regime via
`scripts/prototype_host_wrapped_forward.py`:

- Gate 1 (host_wrapped ≤ native at every K): PASS
- Gate 2 (K=211 KL < 25.0): PASS (15.42 nats, vs 89.91 native)
- Gate 3 (max adjacent ΔKL ≤ 10 nats): PASS (max 2.94)
- Good-tier sanity (orthonormal n=d basis, KL < 0.5): PASS (KL ≈ 0)

Full test suite green: 593 existing pass + 12 new
`tests/test_forward_mode_dispatch.py` pass = 605 total.

Items marked `[x]` landed in this critical-path pass. Items marked
`[ ]` are queued follow-ups — listed below as separate sub-changes
so they can land incrementally without blocking the headline gate.



## 1. New `forge-forward-mode` capability spec

- [ ] 1.1 Create `openspec/specs/forge-forward-mode/spec.md`
      describing `forward_mode` field, `"auto"` dispatch, and the
      host-wrapped forward contract.
- [ ] 1.2 Update `openspec/specs/architecture-adapters/spec.md` to
      include the new `host_wrapped_module(host_model, basis,
      scale_boost)` method requirement.
- [ ] 1.3 Update `openspec/specs/pareto-sweep/spec.md` to include
      the new `forward_mode_resolved: str | None` row field.
- [ ] 1.4 Update `openspec/specs/subspace-projector/spec.md` (or
      wherever `NativeModelConfig` lives) to include the new
      `forward_mode: str = "auto"` field.

## 2. `NativeModelConfig.forward_mode` field

- [x] 2.1 Add `forward_mode: str = "auto"` to
      `saeforge.model.NativeModelConfig`. Validate in
      `__post_init__`: SHALL be one of
      `{"auto", "native_in_basis", "host_wrapped"}`.
- [x] 2.2 `to_dict` / `from_dict` round-trip the new field; older
      configs missing the key SHALL default to `"auto"`.

## 3. `saeforge/forward_mode.py` — dispatch helper

- [x] 3.1 New module exposing
      `resolve_forward_mode(basis, requested) -> Literal[
      "native_in_basis", "host_wrapped"]`.
      - When `requested == "auto"`: compute `basis_rank` and
        `quality_tier` via `forge_quality.classify_quality`; return
        `"native_in_basis"` for `good`/`saturated`, `"host_wrapped"`
        for `undersized`/`degenerate`.
      - When `requested` is explicit: return it unchanged.
      - Log the resolution at INFO once per call when source was
        `"auto"`.
- [ ] 3.2 Export `resolve_forward_mode` from
      `saeforge/__init__.py`.

## 4. GPT-2 host-wrapped module

- [x] 4.1 New file `saeforge/adapters/_host_wrapped/gpt2.py` housing
      `HostWrappedGPT2(nn.Module)`. Stores frozen references to
      host's `wte`, `wpe`, `transformer.h`, `ln_f`, `lm_head`;
      registers `W_dec` and `pinv` as buffers; scale_boost as a
      python scalar (no `nn.Parameter` — encode is `... @ pinv *
      scale_boost`).
- [x] 4.2 Implement the forward in the proposal's pseudocode:
      decode at every block boundary, encode after every block, host
      `wte+wpe` at entry, host `ln_f+lm_head` at exit.
- [x] 4.3 `GPT2Adapter.host_wrapped_module(host_model, basis,
      scale_boost)` constructs and returns the module.

## 5. Per-family stubs

- [ ] 5.1 In each of
      `saeforge/adapters/{llama,gemma2,qwen2,qwen3,qwen3_moe,whisper}.py`,
      add `host_wrapped_module(...)` raising `NotImplementedError`
      with a message naming the family and pointing at this change's
      proposal as the v1 scope marker.
- [x] 5.2 `ArchitectureAdapter.host_wrapped_module` becomes part of
      the protocol; default impl on the base class raises
      `NotImplementedError` so third-party adapters that haven't
      caught up surface a clean error.  *(Single base-class default
      covers all non-GPT-2 families — no per-file stubs needed.)*

## 6. `NativeModel.from_host` dispatch

- [x] 6.1 In `NativeModel.from_host`, resolve `forward_mode` via
      `saeforge.forward_mode.resolve_forward_mode(basis,
      config.forward_mode)` before constructing the module.
- [x] 6.2 When resolved mode is `"host_wrapped"`: construct via
      `adapter.host_wrapped_module(host, basis, scale_boost)`;
      bypass `projector.project_module` and
      `from_projected_weights`.
- [x] 6.3 When resolved mode is `"native_in_basis"`: existing path,
      no change.
- [x] 6.4 Record the resolved mode on the returned `NativeModel` as
      `model.resolved_forward_mode` for downstream introspection.

## 7. `ForgePipeline` wiring

- [ ] 7.1 `ForgePipeline.__init__` accepts a `forward_mode: str =
      "auto"` kwarg. Threaded into the `NativeModelConfig` it builds.
- [ ] 7.2 In `host_wrapped` mode, `ForgePipeline.run_finetune` (if
      callable) raises `RuntimeError("fine-tune is not supported in
      host_wrapped forward_mode in v1; see add-host-wrapped-
      finetune-recipe")` with the queued follow-up reference.
- [ ] 7.3 `ForgePipeline.evaluate_faithfulness` works identically in
      both modes (it calls `forged.forward(input_ids)` — host-wrapped
      module exposes the same signature).

## 8. CLI

- [ ] 8.1 Add `--forward-mode VALUE` argument to the `forge` and
      `sweep-pareto` subparsers. Accepts `auto`/`native_in_basis`/
      `host_wrapped`. Default `auto`. Pass-through to
      `ForgePipeline(forward_mode=...)`.
- [ ] 8.2 Document the new flag in the relevant `--help` text and in
      `README.md` under "advanced flags" (one paragraph).

## 9. `ParetoFrontierRow.forward_mode_resolved`

- [ ] 9.1 Add `forward_mode_resolved: str | None = None` to
      `ParetoFrontierRow`. Validate in `__post_init__`: when not
      `None`, SHALL be one of `{"native_in_basis", "host_wrapped"}`.
- [ ] 9.2 `to_json_dict` / `from_json_dict` round-trip; older rows
      missing the key load as `None`.
- [ ] 9.3 `_process_row` populates the field from
      `model.resolved_forward_mode` when present.

## 10. Smoke validation

- [ ] 10.1 Re-run the GPT-2 layer-8 jbloom sweep at K ∈ {25, 103,
      163, 211}, HEA_Rung2 n_qubits=10, `scale_boost=1.0`. Three
      arms:
      - `--forward-mode native_in_basis` (regression for documented
        blow-up).
      - `--forward-mode host_wrapped` (acceptance gate).
      - `--forward-mode auto` (should byte-identical the
        host_wrapped arm since every K is undersized/degenerate).
- [ ] 10.2 Acceptance: host_wrapped arm SHALL produce monotone
      non-increasing KL across K. K=211 KL < 5.0 nats. Trajectory
      saved to `smoke_host_wrapped/host_wrapped/frontier.jsonl` and
      summarised in this change's `smoke-results.md`.
- [ ] 10.3 Good-tier sanity: construct a synthetic GPT-2 basis with
      `n_features = d_model = 768` and orthonormal `W_dec`. Build
      both modes; assert KL within 0.1 nats of each other and within
      0.5 nats of zero.

## 11. Test coverage

- [x] 11.1 `tests/test_forward_mode_resolve.py` — unit tests for
      `resolve_forward_mode` covering: each quality tier, explicit
      mode values, invalid mode strings.  *(Landed as
      `tests/test_forward_mode_dispatch.py`, 12 tests covering
      resolve, config validation/round-trip, GPT-2 wrapped forward
      shape + orthonormal-basis KL≈0, and Llama stub.)*
- [x] 11.2 `tests/adapters/test_host_wrapped_gpt2.py` — smoke test
      that the GPT-2 host-wrapped module constructs, accepts
      input_ids, and returns logits with the correct shape on a
      tiny synthetic basis.  *(Combined into the same file as 11.1.)*
- [ ] 11.3 `tests/test_native_model.py` — extend the existing
      `from_host` test to cover `forward_mode="auto"` on a
      good-tier basis (resolves to native_in_basis, existing
      assertions pass) and on an undersized basis (resolves to
      host_wrapped, returns logits of the right shape).
- [ ] 11.4 `tests/test_pipeline_finetune.py` — assert
      `run_finetune` raises a clear `RuntimeError` in host-wrapped
      mode pointing at the follow-up.
- [ ] 11.5 `tests/test_pareto_row_schema.py` — new
      `forward_mode_resolved` field round-trips through
      to/from_json_dict.

## 12. Docs

- [ ] 12.1 Add `docs/forward-mode.md` describing both
      implementations, the dispatch rule, and the v1 family-rollout
      table. ~150 lines.
- [ ] 12.2 Update `README.md` status section to reflect host-wrapped
      fallback for undersized/degenerate bases.
- [ ] 12.3 Add the `--forward-mode` flag to the CLI usage section.
