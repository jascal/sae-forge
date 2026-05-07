# forge-pipeline Specification

## Purpose

Defines `ForgePipeline.run` — the v0 imperative orchestrator — and
`saeforge.eval.faithfulness.faithfulness_kl`, the per-token KL eval
that quantifies how close a forged model's distribution is to its
host's.

## Requirements

### Requirement: Identity basis is byte-equivalent forge

When the basis is the `d_model × d_model` identity, the forged model's
forward pass SHALL match the host's forward pass exactly up to
floating-point arithmetic, and the resulting `faithfulness_kl` SHALL
be `< 1e-3`. This is the v0 strongest correctness signal — it pins
the projection algebra in `SubspaceProjector` plus the in-tree
`NativeModel` together end-to-end.

#### Scenario: identity-basis KL is sub-millibit

- **GIVEN** a `FeatureBasis` with `W_dec = np.eye(d_model)` and a
  random-init tiny GPT-2 host
- **WHEN** the pipeline runs against the host on a `(1, 4)` random
  input id tensor
- **THEN** the resulting `ForgeResult.faithfulness_kl` is below `1e-3`

### Requirement: run raises when host_model_id is None

`ForgePipeline.run` SHALL raise `ValueError` whose message contains
`"host_model_id"` when `self.host_model_id is None`. The
`run_synthetic` path is the alternative for callers who construct the
host model in memory.

#### Scenario: missing host id

- **GIVEN** a `ForgePipeline(basis=..., projector=..., host_model_id=None)`
- **WHEN** `run("/tmp/whatever")` is called
- **THEN** `ValueError` is raised whose message contains `"host_model_id"`

### Requirement: Artifact tree is complete

After a successful run, `output_dir` SHALL contain:
- `forged/config.json` (from `NativeModel.save_pretrained`)
- `forged/model.safetensors` (from `NativeModel.save_pretrained`)
- `forge_result.json` with keys `host_model_id`, `n_params`,
  `faithfulness_kl`, `n_features`, `scale_compression_ratio`

#### Scenario: artifact tree after run_synthetic

- **GIVEN** a successful `run_synthetic` against `tiny_gpt2`
- **WHEN** the call returns
- **THEN** all four files exist
- **AND** `forge_result.json["n_params"]` equals the returned
  `ForgeResult.n_params`

### Requirement: faithfulness_kl returns a non-negative float

`faithfulness_kl` SHALL return a `float >= 0.0` (KL divergence is
non-negative). It SHALL average per-token KL(host || forged) across
masked positions in the encoded prompt batch.

#### Scenario: KL is non-negative

- **WHEN** `faithfulness_kl` is called on any `(forged, host, prompts)`
  triple
- **THEN** the returned value is a `float` and `>= 0.0`

### Requirement: faithfulness_kl auto-loads tokenizer when none supplied

When `tokenizer is None`, `faithfulness_kl` SHALL load
`transformers.AutoTokenizer.from_pretrained(host._name_or_path)`
falling back to `"gpt2"` when the host has no `_name_or_path`. When
the tokenizer has no `pad_token`, it SHALL be set to `eos_token`.

### Requirement: run is opt-in for tokenizer / network

`ForgePipeline.run_synthetic` SHALL NOT load any tokenizer or download
any host weights — it operates strictly on the in-memory host and
optional pre-tokenized input ids. This makes the pipeline testable in
hermetic environments.

#### Scenario: run_synthetic with no eval input

- **GIVEN** a `run_synthetic(host, tmp_path, eval_input_ids=None)` call
- **WHEN** the call returns
- **THEN** `result.faithfulness_kl is None` and the artifact tree is
  still written
