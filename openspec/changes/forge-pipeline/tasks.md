## 1. ForgePipeline.run

- [x] 1.1 Implement `ForgePipeline.run(output_dir)`: load HF GPT-2 host via `from_pretrained`, project, derive config, build NativeModel, move to dtype/device, optionally run faithfulness KL, save_pretrained, write `forge_result.json`
- [x] 1.2 Implement `ForgePipeline.run_synthetic(host_model, output_dir, eval_input_ids=None)` taking a pre-loaded host and pre-tokenized input ids
- [x] 1.3 Raise `ValueError` from `run` when `host_model_id is None`
- [x] 1.4 Lazy-import torch + transformers via `require_extra`

## 2. faithfulness_kl

- [x] 2.1 Implement `faithfulness_kl(forged, host, prompts, tokenizer=None, max_length=32, device="cpu")` computing mean per-token KL(host || forged) over masked positions
- [x] 2.2 Auto-load the host's tokenizer via `transformers.AutoTokenizer` when `tokenizer is None`
- [x] 2.3 Add `_kl_from_input_ids` internal helper for the synthetic path

## 3. Toy example

- [x] 3.1 Replace `examples/forge_gpt2_toy.py` stub with a working CPU example (tiny in-memory GPT-2 + synthetic 8-feature basis + KL eval + JSON summary)

## 4. Tests

- [x] 4.1 End-to-end `run_synthetic` against `tiny_gpt2` produces a non-zero `n_params`, a non-negative KL, and the expected artifact tree
- [x] 4.2 `run` raises `ValueError` when `host_model_id is None`
- [x] 4.3 **Identity-basis sanity check**: KL(host || host-via-forged) is `< 1e-3` when the basis is `np.eye(d_model)`. This is the strongest projection-algebra correctness signal in v0.
- [x] 4.4 `faithfulness_kl` returns a `float` for a synthetic input
- [x] 4.5 `examples/forge_gpt2_toy.main(tmp_path)` returns a summary dict with the expected keys
- [x] 4.6 Update `bootstrap-package` spec scenario to mark `ForgePipeline.run` stub superseded

## 5. OpenSpec scaffolding

- [x] 5.1 `openspec/changes/forge-pipeline/proposal.md`
- [x] 5.2 `openspec/changes/forge-pipeline/tasks.md` (this file)
- [x] 5.3 `openspec/changes/forge-pipeline/specs/forge-pipeline/spec.md`
